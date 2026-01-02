"""Scenario Runner.

Executes multi-drone simulation scenarios, applying edge cases and
generating decision logs that can be viewed in the dashboard.
"""

import asyncio
import json
import logging
import math
import random
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent.api_models import ActionType
from agent.server.critics.orchestrator import AuthorityModel, CriticOrchestrator
from agent.server.decision import Decision
from agent.server.events import Event, EventSeverity, EventType
from agent.server.goal_selector import GoalSelector
from agent.server.goals import Goal, GoalType
from agent.server.monitoring.explanation_agent import ExplanationAgent
from agent.server.risk_evaluator import RiskAssessment, RiskFactor, RiskLevel
from agent.server.scenarios import (
    DroneState,
    EnvironmentConditions,
    Scenario,
    ScenarioEvent,
    SimulatedDrone,
    get_scenario,
)
from agent.server.state import connection_manager, server_state
from agent.server.world_model import (
    Anomaly,
    Asset,
    AssetType,
    DockStatus,
    WorldModel,
    WorldSnapshot,
)
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSState,
    Position,
    VehicleHealth,
    VehicleState,
    Velocity,
)
from simulation.coordinate_utils import haversine_distance, initial_bearing

logger = logging.getLogger(__name__)


@dataclass
class DroneSimState:
    """Runtime state for a simulated drone during scenario execution."""

    drone: SimulatedDrone
    world_model: WorldModel
    current_goal: Goal | None = None
    decisions_made: int = 0
    last_decision_time: datetime | None = None

    # Movement tracking
    target_position: Position | None = None
    moving: bool = False
    target_asset_id: str | None = None
    target_altitude_agl: float | None = None
    target_is_dock: bool = False

    # Mission tracking
    inspections_completed: int = 0
    anomalies_found: int = 0


@dataclass
class ScenarioRunState:
    """Overall state of a running scenario."""

    scenario: Scenario
    start_time: datetime
    current_time: datetime
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    elapsed_seconds: float = 0.0

    # Drone states
    drone_states: dict[str, DroneSimState] = field(default_factory=dict)

    # Environment
    environment: EnvironmentConditions = field(default_factory=EnvironmentConditions)

    # Event tracking
    events_fired: list[ScenarioEvent] = field(default_factory=list)
    pending_events: list[ScenarioEvent] = field(default_factory=list)

    # Decision log
    decision_log: list[dict[str, Any]] = field(default_factory=list)

    # Status
    is_running: bool = True
    is_complete: bool = False
    abort_reason: str | None = None

    # Statistics for summary
    total_battery_consumed: float = 0.0
    anomalies_triggered: int = 0
    decisions_count: int = 0

    # Shared fleet coordination
    # Assets currently being targeted by any drone (asset_id -> drone_id)
    assets_in_progress: dict[str, str] = field(default_factory=dict)
    # Assets that have been inspected by any drone
    assets_inspected: set[str] = field(default_factory=set)


class ScenarioRunner:
    """Executes simulation scenarios with multiple drones.

    The runner:
    - Initializes world models for each drone
    - Runs a simulation loop at configurable speed
    - Applies edge case triggers (battery drain, GPS loss, etc.)
    - Makes decisions using the goal selector
    - Logs all decisions for dashboard viewing

    Example:
        runner = ScenarioRunner(seed=42)  # Deterministic mode
        await runner.load_scenario("battery_cascade_001")
        await runner.run(time_scale=10.0)  # 10x speed
        runner.save_decision_log("logs/")
    """

    def __init__(
        self,
        tick_interval_s: float = 1.0,
        decision_interval_s: float = 5.0,
        cruise_speed_mps: float = 6.0,
        log_dir: Path | None = None,
        seed: int | None = None,
        enable_critics: bool = True,
        critic_authority: str = "advisory",
        enable_explanations: bool = False,
    ) -> None:
        """Initialize scenario runner.

        Args:
            tick_interval_s: Simulation tick interval in seconds
            decision_interval_s: How often to make decisions per drone
            cruise_speed_mps: Cruise speed in meters per second
            log_dir: Directory to save decision logs
            seed: Random seed for deterministic simulation (None for random)
            enable_critics: Whether to enable critic validation of decisions
            critic_authority: Authority model ('advisory', 'blocking', 'escalation', 'hierarchical')
            enable_explanations: Whether to generate decision explanations
        """
        self.tick_interval_s = tick_interval_s
        self.decision_interval_s = decision_interval_s
        self.cruise_speed_mps = cruise_speed_mps
        self.log_dir = log_dir or Path("logs")
        self.run_state: ScenarioRunState | None = None
        self._seed = seed
        self._rng = random.Random(seed)
        self._running_task: asyncio.Task | None = None

        # Callbacks for external integration
        self.on_decision: Callable[[str, Goal, dict], None] | None = None
        self.on_event: Callable[[ScenarioEvent], None] | None = None
        self.on_tick: Callable[[ScenarioRunState], None] | None = None

        # Fleet coordination lock - prevents race conditions in asset assignment
        self._asset_lock = asyncio.Lock()

        # B.1: Initialize critic orchestrator for multi-agent validation
        self.enable_critics = enable_critics
        self.critic_orchestrator: CriticOrchestrator | None = None
        if enable_critics:
            try:
                authority = AuthorityModel(critic_authority)
                self.critic_orchestrator = CriticOrchestrator(
                    authority_model=authority,
                    enable_llm=False,  # Start with fast classical critics
                )
                logger.info(f"Critic orchestrator initialized: authority={authority.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize critic orchestrator: {e}")
                self.critic_orchestrator = None

        # B.3: Initialize explanation agent for audit trails
        self.enable_explanations = enable_explanations
        self.explanation_agent: ExplanationAgent | None = None
        if enable_explanations:
            try:
                self.explanation_agent = ExplanationAgent()
                logger.info("Explanation agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize explanation agent: {e}")

    @property
    def is_running(self) -> bool:
        """Check if a scenario is currently running."""
        return (
            self.run_state is not None
            and self.run_state.is_running
            and self._running_task is not None
            and not self._running_task.done()
        )

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        return self.run_state.run_id if self.run_state else None

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the random seed for deterministic replays.

        Args:
            seed: New seed value (None for random)
        """
        self._seed = seed
        self._rng = random.Random(seed)

    async def load_scenario(self, scenario_id: str) -> bool:
        """Load a scenario by ID.

        Args:
            scenario_id: ID of scenario to load

        Returns:
            True if loaded successfully
        """
        scenario = get_scenario(scenario_id)
        if not scenario:
            logger.error(f"Scenario not found: {scenario_id}")
            return False

        now = datetime.now()
        self.run_state = ScenarioRunState(
            scenario=scenario,
            start_time=now,
            current_time=now,
            environment=scenario.environment,
            pending_events=list(scenario.events),
        )

        # Initialize drone states
        for drone in scenario.drones:
            world_model = self._create_world_model(drone, scenario)
            self.run_state.drone_states[drone.drone_id] = DroneSimState(
                drone=drone,
                world_model=world_model,
            )

        logger.info(f"Loaded scenario: {scenario.name} with {len(scenario.drones)} drones")
        return True

    def _create_world_model(self, drone: SimulatedDrone, scenario: Scenario) -> WorldModel:
        """Create and initialize a world model for a drone."""
        world = WorldModel()

        # Set dock position (use first drone's position as dock)
        dock_pos = Position(
            latitude=scenario.drones[0].latitude,
            longitude=scenario.drones[0].longitude,
            altitude_msl=50.0,
            altitude_agl=0.0,
        )
        world.set_dock(dock_pos, DockStatus.AVAILABLE)

        # Add assets
        for asset in scenario.assets:
            world.add_asset(
                Asset(
                    asset_id=asset.asset_id,
                    name=asset.name,
                    asset_type=AssetType(asset.asset_type)
                    if asset.asset_type in [e.value for e in AssetType]
                    else AssetType.OTHER,
                    position=Position(
                        latitude=asset.latitude,
                        longitude=asset.longitude,
                        altitude_msl=float(asset.altitude_m or 0.0),
                    ),
                    priority=asset.priority,
                    inspection_altitude_agl=asset.inspection_altitude_agl,
                    orbit_radius_m=asset.orbit_radius_m,
                    dwell_time_s=asset.dwell_time_s,
                )
            )

            # Add anomalies for assets that have them
            if asset.has_anomaly:
                world.add_anomaly(
                    Anomaly(
                        anomaly_id=f"anom_{asset.asset_id}",
                        asset_id=asset.asset_id,
                        detected_at=datetime.now(),
                        severity=asset.anomaly_severity,
                        description=f"Pre-existing anomaly on {asset.name}",
                    )
                )

        # Start mission
        world.start_mission(
            mission_id=f"scenario_{scenario.scenario_id}",
            mission_name=scenario.name,
        )

        # Set initial vehicle state
        vehicle_state = self._drone_to_vehicle_state(drone)
        world.update_vehicle(vehicle_state)

        return world

    def _drone_to_vehicle_state(self, drone: SimulatedDrone) -> VehicleState:
        """Convert SimulatedDrone to VehicleState."""
        return VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=drone.latitude,
                longitude=drone.longitude,
                altitude_msl=drone.altitude_agl + 50.0,
                altitude_agl=drone.altitude_agl,
            ),
            velocity=Velocity(north=0.0, east=0.0, down=0.0),
            attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
            battery=BatteryState(
                voltage=22.2 + (drone.battery_percent / 100) * 3.0,
                current=5.0 if drone.in_air else 0.5,
                remaining_percent=drone.battery_percent,
            ),
            mode=FlightMode.GUIDED if drone.in_air else FlightMode.STABILIZE,
            armed=drone.armed,
            in_air=drone.in_air,
            gps=GPSState(
                fix_type=drone.gps_fix_type,
                satellites_visible=drone.satellites_visible,
                hdop=drone.gps_hdop,
                vdop=1.0,
            ),
            health=VehicleHealth(
                sensors_healthy=drone.sensors_healthy,
                gps_healthy=drone.gps_healthy,
                battery_healthy=drone.battery_percent > 10,
                motors_healthy=drone.motors_healthy,
                ekf_healthy=drone.ekf_healthy,
            ),
        )

    async def run(
        self,
        time_scale: float = 1.0,
        max_duration_s: float | None = None,
    ) -> bool:
        """Run the scenario simulation.

        This method is idempotent - if already running, returns False immediately.
        Handles cancellation gracefully and saves logs on exit.

        Args:
            time_scale: Speed multiplier (1.0 = real-time, 10.0 = 10x speed)
            max_duration_s: Maximum real-time duration to run

        Returns:
            True if completed successfully
        """
        if not self.run_state:
            logger.error("No scenario loaded")
            return False

        # Idempotent check - don't start if already running
        if self.is_running:
            logger.warning(f"Scenario already running (run_id={self.run_state.run_id})")
            return False

        logger.info(
            f"Starting scenario: {self.run_state.scenario.name} "
            f"(run_id={self.run_state.run_id}) at {time_scale}x speed"
        )

        # Record initial battery levels for consumption tracking
        initial_batteries = {
            drone_id: ds.drone.battery_percent
            for drone_id, ds in self.run_state.drone_states.items()
        }

        real_start = datetime.now()
        max_ticks: int | None = None
        if max_duration_s is not None and self._seed is not None:
            max_sim_seconds = max_duration_s * time_scale
            max_ticks = max(1, int(max_sim_seconds / self.tick_interval_s))
        ticks_run = 0
        scenario_duration_s = self.run_state.scenario.duration_minutes * 60
        self._running_task = asyncio.current_task()

        try:
            while self.run_state.is_running:
                # Check if scenario time exceeded
                if self.run_state.elapsed_seconds >= scenario_duration_s:
                    self.run_state.is_complete = True
                    self.run_state.is_running = False
                    logger.info(
                        f"Scenario completed: duration reached (run_id={self.run_state.run_id})"
                    )
                    break

                # Check deterministic tick budget (seeded runs only)
                if max_ticks is not None and ticks_run >= max_ticks:
                    logger.info(
                        f"Scenario stopped: deterministic tick limit reached "
                        f"(run_id={self.run_state.run_id})"
                    )
                    self.run_state.abort_reason = "sim_time_limit"
                    self.run_state.is_running = False
                    break

                # Check real-time limit
                if max_duration_s and max_ticks is None:
                    real_elapsed = (datetime.now() - real_start).total_seconds()
                    if real_elapsed >= max_duration_s:
                        logger.info(
                            f"Scenario stopped: real-time limit reached "
                            f"(run_id={self.run_state.run_id})"
                        )
                        self.run_state.abort_reason = "real_time_limit"
                        self.run_state.is_running = False
                        break

                # Execute one tick
                await self._tick()

                # Wait for next tick (scaled by time_scale)
                await asyncio.sleep(self.tick_interval_s / time_scale)

                # Advance simulation time
                self.run_state.elapsed_seconds += self.tick_interval_s
                self.run_state.current_time = self.run_state.start_time + timedelta(
                    seconds=self.run_state.elapsed_seconds
                )
                ticks_run += 1

        except asyncio.CancelledError:
            logger.info(f"Scenario cancelled (run_id={self.run_state.run_id})")
            self.run_state.is_running = False
            self.run_state.abort_reason = "cancelled"
            raise  # Re-raise to propagate cancellation

        finally:
            # Calculate statistics
            self._finalize_run_stats(initial_batteries)

            # Log run summary
            self._log_run_summary()

            # Clear running task reference
            self._running_task = None

        return self.run_state.is_complete

    def _finalize_run_stats(self, initial_batteries: dict[str, float]) -> None:
        """Calculate final statistics for the run."""
        if not self.run_state:
            return

        # Calculate total battery consumed
        total_consumed = 0.0
        for drone_id, ds in self.run_state.drone_states.items():
            initial = initial_batteries.get(drone_id, 100.0)
            consumed = initial - ds.drone.battery_percent
            total_consumed += max(0.0, consumed)

        self.run_state.total_battery_consumed = total_consumed

        # Count total decisions
        self.run_state.decisions_count = sum(
            ds.decisions_made for ds in self.run_state.drone_states.values()
        )

        # Count anomalies triggered (from events)
        self.run_state.anomalies_triggered = sum(
            1 for e in self.run_state.events_fired if e.event_type == "anomaly"
        )

    def _log_run_summary(self) -> None:
        """Log a summary entry at the end of the run."""
        if not self.run_state:
            return

        summary_entry = {
            "type": "run_summary",
            "run_id": self.run_state.run_id,
            "scenario_id": self.run_state.scenario.scenario_id,
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": self.run_state.elapsed_seconds,
            "is_complete": self.run_state.is_complete,
            "abort_reason": self.run_state.abort_reason,
            "total_decisions": self.run_state.decisions_count,
            "total_battery_consumed": round(self.run_state.total_battery_consumed, 2),
            "events_fired": len(self.run_state.events_fired),
            "anomalies_triggered": self.run_state.anomalies_triggered,
            "drones": {
                drone_id: {
                    "name": ds.drone.name,
                    "final_battery": round(ds.drone.battery_percent, 1),
                    "final_state": ds.drone.state.value,
                    "decisions_made": ds.decisions_made,
                }
                for drone_id, ds in self.run_state.drone_states.items()
            },
        }
        self._log_entry(summary_entry)

        logger.info(
            f"Run {self.run_state.run_id} complete: "
            f"{self.run_state.decisions_count} decisions, "
            f"{self.run_state.total_battery_consumed:.1f}% battery consumed, "
            f"{self.run_state.anomalies_triggered} anomalies"
        )

    async def _tick(self) -> None:
        """Execute one simulation tick."""
        if not self.run_state:
            return

        # Fire pending events
        await self._process_events()

        # Update environment
        self._update_environment()

        # Update each drone
        for _drone_id, drone_state in self.run_state.drone_states.items():
            # Apply physics/state changes
            await self._update_drone_state(drone_state)

            # Check edge case triggers
            self._check_triggers(drone_state)

            # Make decision if interval elapsed
            if self._should_decide(drone_state):
                await self._make_decision(drone_state)

        # Cleanup stale fleet state AFTER all decisions are made
        # This prevents premature removal of assets that are still being targeted
        await self._cleanup_stale_fleet_state()

        # Callback
        if self.on_tick:
            self.on_tick(self.run_state)

    async def _process_events(self) -> None:
        """Process any events that should fire at current time."""
        if not self.run_state:
            return

        events_to_fire = []
        remaining = []

        for event in self.run_state.pending_events:
            if event.timestamp_offset_s <= self.run_state.elapsed_seconds:
                events_to_fire.append(event)
            else:
                remaining.append(event)

        self.run_state.pending_events = remaining

        for event in events_to_fire:
            self.run_state.events_fired.append(event)
            logger.info(
                f"[{event.timestamp_offset_s:.0f}s] {event.event_type}: {event.description}"
            )

            # Log event
            self._log_entry({
                "type": "event",
                "timestamp": self.run_state.current_time.isoformat(),
                "elapsed_s": self.run_state.elapsed_seconds,
                "event_type": event.event_type,
                "description": event.description,
                "data": event.data,
            })

            if self.on_event:
                self.on_event(event)

    def _update_environment(self) -> None:
        """Update environmental conditions based on triggers."""
        if not self.run_state:
            return

        env = self.run_state.environment
        current = self.run_state.current_time

        # Wind increase trigger
        if env.wind_increase_at and current >= env.wind_increase_at:
            if env.wind_speed_ms < env.wind_increase_to:
                # Gradual increase
                env.wind_speed_ms = min(env.wind_speed_ms + 0.5, env.wind_increase_to)

        # Visibility drop trigger
        if env.visibility_drop_at and current >= env.visibility_drop_at:
            if env.visibility_m > env.visibility_drop_to:
                # Gradual decrease
                env.visibility_m = max(env.visibility_m - 500, env.visibility_drop_to)

    async def _update_drone_state(self, drone_state: DroneSimState) -> None:
        """Update drone physical state (battery, position, etc.)."""
        drone = drone_state.drone

        if drone_state.moving and drone_state.target_position:
            target = drone_state.target_position
            distance_m = haversine_distance(
                drone.latitude,
                drone.longitude,
                target.latitude,
                target.longitude,
            )
            step_m = self.cruise_speed_mps * self.tick_interval_s

            if distance_m <= max(step_m, 0.5):
                drone.latitude = target.latitude
                drone.longitude = target.longitude
                drone_state.moving = False
                drone_state.target_position = None

                if drone_state.target_is_dock:
                    drone.state = DroneState.DOCKED
                    drone.in_air = False
                    drone.armed = False
                    drone.altitude_agl = 0.0
                    logger.info(
                        f"{drone.name}: Docked successfully (battery: {drone.battery_percent:.1f}%)"
                    )
                    # Trigger handoff to another drone if this one needs to charge
                    if drone.battery_percent < 80:
                        self._trigger_drone_handoff(drone_state)
                else:
                    drone.state = DroneState.IDLE
                    drone_state.inspections_completed += 1
                    if drone_state.target_asset_id:
                        asset_id = drone_state.target_asset_id
                        drone_state.world_model.record_inspection(asset_id)

                        # Mark as inspected in shared fleet state (atomic)
                        await self._complete_asset_inspection(asset_id, drone.drone_id)
                        if self.run_state:
                            logger.info(
                                f"{drone.name}: Completed inspection of {asset_id} "
                                f"(fleet: {len(self.run_state.assets_inspected)} done, "
                                f"{len(self.run_state.assets_in_progress)} in progress)"
                            )

                        if asset_id in drone_state.world_model.get_anomaly_assets():
                            drone_state.anomalies_found += 1
                            drone_state.world_model.resolve_anomaly(f"anom_{asset_id}")

                        # Record vision capture for the inspection
                        if server_state.vision_enabled and server_state.vision_service:
                            asyncio.create_task(
                                server_state.vision_service.process_inspection_result(
                                    asset_id=asset_id,
                                    vehicle_state={
                                        "position": {
                                            "latitude": drone.latitude,
                                            "longitude": drone.longitude,
                                            "altitude_msl": drone.altitude_agl,
                                        },
                                        "altitude_agl": drone.altitude_agl,
                                        "heading_deg": drone.heading,
                                    },
                                )
                            )

                drone_state.target_asset_id = None
                drone_state.target_altitude_agl = None
                drone_state.target_is_dock = False
            else:
                bearing_deg = initial_bearing(
                    drone.latitude,
                    drone.longitude,
                    target.latitude,
                    target.longitude,
                )
                drone.latitude, drone.longitude = self._destination_point(
                    drone.latitude,
                    drone.longitude,
                    bearing_deg,
                    step_m,
                )

            if drone_state.target_altitude_agl is not None:
                drone.altitude_agl = drone_state.target_altitude_agl
                drone.in_air = True
                drone.armed = True

        # Battery drain (in air) or charge (docked)
        if drone.in_air:
            drain = drone.battery_drain_rate * (self.tick_interval_s / 60.0)
            drone.battery_percent = max(0.0, drone.battery_percent - drain)

            # CRITICAL SAFETY: Force return/land on low battery
            self._check_battery_failsafe(drone_state)

        elif drone.state == DroneState.DOCKED or drone.state == DroneState.CHARGING:
            # Charge at dock - approximately 2% per minute (full charge in ~50 mins)
            charge_rate = 2.0  # percent per minute
            charge = charge_rate * (self.tick_interval_s / 60.0)
            drone.battery_percent = min(100.0, drone.battery_percent + charge)

            # Update state to CHARGING if below 95%, otherwise mark as ready (DOCKED)
            if drone.battery_percent < 95.0:
                drone.state = DroneState.CHARGING
            else:
                drone.state = DroneState.DOCKED

        # Update world model with new state
        vehicle_state = self._drone_to_vehicle_state(drone)
        drone_state.world_model.update_vehicle(vehicle_state)

    def _destination_point(
        self,
        lat: float,
        lon: float,
        bearing_deg: float,
        distance_m: float,
    ) -> tuple[float, float]:
        """Compute destination point given start, bearing, and distance."""
        if distance_m <= 0.0:
            return lat, lon

        radius_m = 6371000.0
        bearing_rad = math.radians(bearing_deg)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        delta = distance_m / radius_m

        lat2 = math.asin(
            math.sin(lat1) * math.cos(delta)
            + math.cos(lat1) * math.sin(delta) * math.cos(bearing_rad)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing_rad) * math.sin(delta) * math.cos(lat1),
            math.cos(delta) - math.sin(lat1) * math.sin(lat2),
        )

        return math.degrees(lat2), math.degrees(lon2)

    def _check_battery_failsafe(self, drone_state: DroneSimState) -> None:
        """Force return or emergency land when battery is critically low.

        This is a safety failsafe that bypasses the goal selector to ensure
        drones don't run out of battery mid-flight.

        Thresholds:
        - < 25%: Force return to dock (if not already returning)
        - < 10%: Emergency land immediately
        - < 5%: Critical - force immediate landing
        """
        drone = drone_state.drone

        # Don't interfere if already docked/charging/emergency
        if drone.state in (DroneState.DOCKED, DroneState.CHARGING, DroneState.EMERGENCY):
            return

        # Emergency land threshold - too low to make it back
        if drone.battery_percent < 10:
            if drone.state != DroneState.EMERGENCY:
                logger.warning(
                    f"{drone.name}: EMERGENCY LAND - Battery critical at {drone.battery_percent:.1f}%"
                )
                drone.state = DroneState.EMERGENCY
                drone.in_air = False
                drone.armed = False
                drone_state.moving = False
                drone_state.target_position = None
                drone_state.target_is_dock = False

                # Log the emergency
                if self.run_state:
                    self._log_entry({
                        "type": "emergency",
                        "timestamp": self.run_state.current_time.isoformat(),
                        "elapsed_s": self.run_state.elapsed_seconds,
                        "drone_id": drone.drone_id,
                        "drone_name": drone.name,
                        "reason": f"Emergency land - battery at {drone.battery_percent:.1f}%",
                        "battery_percent": drone.battery_percent,
                    })

                # Trigger handoff to another drone if available
                self._trigger_drone_handoff(drone_state)
            return

        # Force return threshold
        if drone.battery_percent < 25:
            # If not already heading to dock, force return
            if drone.state != DroneState.RETURNING or not drone_state.target_is_dock:
                logger.warning(
                    f"{drone.name}: LOW BATTERY - Forcing return to dock at {drone.battery_percent:.1f}%"
                )
                drone.state = DroneState.RETURNING
                dock = drone_state.world_model.get_dock()
                if dock:
                    drone_state.target_position = dock.position
                    drone_state.target_altitude_agl = dock.approach_altitude_m
                    drone_state.target_is_dock = True
                    drone_state.moving = True
                    drone_state.target_asset_id = None

                    # Log the forced return
                    if self.run_state:
                        self._log_entry({
                            "type": "battery_return",
                            "timestamp": self.run_state.current_time.isoformat(),
                            "elapsed_s": self.run_state.elapsed_seconds,
                            "drone_id": drone.drone_id,
                            "drone_name": drone.name,
                            "reason": f"Forced return - battery at {drone.battery_percent:.1f}%",
                            "battery_percent": drone.battery_percent,
                        })

    def _trigger_drone_handoff(self, returning_drone_state: DroneSimState) -> None:
        """Trigger handoff to another drone when one returns to dock.

        If a drone is returning due to low battery or has landed, check if
        another drone is available to continue the mission.
        """
        if not self.run_state:
            return

        returning_drone = returning_drone_state.drone

        # Find another drone that's ready
        for drone_id, drone_state in self.run_state.drone_states.items():
            if drone_id == returning_drone.drone_id:
                continue

            drone = drone_state.drone

            # Check if this drone is ready to take over
            if (
                drone.state in (DroneState.DOCKED, DroneState.IDLE)
                and drone.battery_percent >= 80  # Well charged
                and not drone_state.moving
            ):
                logger.info(
                    f"HANDOFF: {drone.name} taking over mission from {returning_drone.name} "
                    f"(battery: {drone.battery_percent:.1f}%)"
                )

                # Log the handoff
                self._log_entry({
                    "type": "handoff",
                    "timestamp": self.run_state.current_time.isoformat(),
                    "elapsed_s": self.run_state.elapsed_seconds,
                    "from_drone_id": returning_drone.drone_id,
                    "from_drone_name": returning_drone.name,
                    "to_drone_id": drone.drone_id,
                    "to_drone_name": drone.name,
                    "reason": f"{returning_drone.name} at {returning_drone.battery_percent:.1f}% battery",
                })

                # The new drone will be picked up by the regular decision loop
                # Just mark it as ready and it'll get assigned a task
                drone.state = DroneState.IDLE
                drone.armed = True
                break

    def _check_triggers(self, drone_state: DroneSimState) -> None:
        """Check and apply edge case triggers."""
        if not self.run_state:
            return

        drone = drone_state.drone
        current = self.run_state.current_time

        # Battery failure trigger
        if drone.battery_failure_at and drone.battery_percent <= drone.battery_failure_at:
            if drone.sensors_healthy:  # Only trigger once
                drone.sensors_healthy = False
                logger.warning(
                    f"{drone.name}: Battery-triggered sensor failure at {drone.battery_percent:.1f}%"
                )

        # GPS loss trigger
        if drone.gps_loss_at and current >= drone.gps_loss_at:
            if drone.gps_healthy:
                drone.gps_healthy = False
                drone.gps_fix_type = 0
                drone.satellites_visible = 0
                logger.warning(f"{drone.name}: GPS signal lost")

        # Sensor failure trigger
        if drone.sensor_failure_at and current >= drone.sensor_failure_at:
            if drone.sensors_healthy:
                drone.sensors_healthy = False
                drone.ekf_healthy = False
                logger.warning(f"{drone.name}: Sensor failure")

        # Motor issue trigger
        if drone.motor_issue_at and current >= drone.motor_issue_at:
            if drone.motors_healthy:
                drone.motors_healthy = False
                logger.warning(f"{drone.name}: Motor issue detected")

    def _should_decide(self, drone_state: DroneSimState) -> bool:
        """Check if it's time to make a decision for this drone."""
        if not self.run_state:
            return False

        if drone_state.last_decision_time is None:
            return True

        elapsed = (self.run_state.current_time - drone_state.last_decision_time).total_seconds()
        return elapsed >= self.decision_interval_s

    def _sync_fleet_state_to_world_model(self, drone_state: DroneSimState) -> None:
        """Sync shared fleet inspection state to a drone's world model.

        This ensures all drones see which assets have already been inspected
        by other drones in the fleet.
        """
        if not self.run_state:
            return

        # Mark inspected assets in this drone's world model
        for asset_id in self.run_state.assets_inspected:
            drone_state.world_model.record_inspection(asset_id)

    def _filter_available_assets(self, snapshot: WorldSnapshot, drone_id: str) -> WorldSnapshot:
        """Filter snapshot to exclude assets being targeted by other drones.

        Args:
            snapshot: Original world snapshot
            drone_id: ID of the drone making the decision

        Returns:
            Modified snapshot with unavailable assets filtered out
        """
        if not self.run_state:
            return snapshot

        # Get assets that are being targeted by OTHER drones
        unavailable_assets = {
            asset_id
            for asset_id, targeting_drone_id in self.run_state.assets_in_progress.items()
            if targeting_drone_id != drone_id
        }

        # Also exclude already-inspected assets
        unavailable_assets.update(self.run_state.assets_inspected)

        if not unavailable_assets:
            return snapshot

        # Filter assets list
        filtered_assets = [
            asset for asset in snapshot.assets if asset.asset_id not in unavailable_assets
        ]

        if not filtered_assets and snapshot.assets:
            filtered_assets = [
                asset
                for asset in snapshot.assets
                if asset.asset_id not in self.run_state.assets_inspected
            ]
            if filtered_assets:
                logger.info(
                    "Fleet filtering fallback restored %d assets (in_progress=%s inspected=%s)",
                    len(filtered_assets),
                    list(self.run_state.assets_in_progress.keys()),
                    list(self.run_state.assets_inspected),
                )
            else:
                logger.info(
                    "All assets inspected; resetting inspection cycle " "(total=%d, inspected=%s)",
                    len(snapshot.assets),
                    list(self.run_state.assets_inspected),
                )
                # Clear the inspected set to allow re-inspection
                self.run_state.assets_inspected.clear()
                # Reset next_scheduled on all assets in snapshot
                for asset in snapshot.assets:
                    asset.next_scheduled = None
                # Reset all drone world models so assets become pending again
                for drone_state in self.run_state.drone_states.values():
                    drone_state.world_model.reset_inspection_cycle()
                filtered_assets = list(snapshot.assets)

        filtered_asset_ids = {asset.asset_id for asset in filtered_assets}

        # Create new snapshot with filtered assets
        return WorldSnapshot(
            timestamp=snapshot.timestamp,
            vehicle=snapshot.vehicle,
            assets=filtered_assets,
            anomalies=[a for a in snapshot.anomalies if a.asset_id in filtered_asset_ids],
            dock=snapshot.dock,
            environment=snapshot.environment,
            mission=snapshot.mission,
            overall_confidence=snapshot.overall_confidence,
        )

    async def _try_claim_asset(self, asset_id: str, drone_id: str) -> bool:
        """Atomically try to claim an asset for a drone.

        Returns True if claim succeeded, False if asset already claimed.
        """
        async with self._asset_lock:
            if not self.run_state:
                return False

            # Check if asset is already claimed by another drone
            current_owner = self.run_state.assets_in_progress.get(asset_id)
            if current_owner is not None and current_owner != drone_id:
                return False

            # Check if already inspected
            if asset_id in self.run_state.assets_inspected:
                return False

            # Claim it
            self.run_state.assets_in_progress[asset_id] = drone_id
            return True

    async def _release_asset(self, asset_id: str, drone_id: str) -> None:
        """Release an asset claim (only if owned by this drone)."""
        async with self._asset_lock:
            if not self.run_state:
                return

            current_owner = self.run_state.assets_in_progress.get(asset_id)
            if current_owner == drone_id:
                self.run_state.assets_in_progress.pop(asset_id, None)
                logger.debug(f"Drone {drone_id} released asset {asset_id}")

    async def _complete_asset_inspection(self, asset_id: str, drone_id: str) -> None:
        """Atomically mark an asset as inspected and remove from in-progress."""
        async with self._asset_lock:
            if not self.run_state:
                return

            self.run_state.assets_inspected.add(asset_id)
            self.run_state.assets_in_progress.pop(asset_id, None)

    async def _cleanup_stale_fleet_state(self) -> None:
        """Remove stale entries from fleet coordination state."""
        if not self.run_state:
            return

        async with self._asset_lock:
            active_targets = {
                state.target_asset_id
                for state in self.run_state.drone_states.values()
                if state.target_asset_id is not None
            }

            stale = set(self.run_state.assets_in_progress.keys()) - active_targets
            for asset_id in stale:
                self.run_state.assets_in_progress.pop(asset_id, None)
                logger.warning("Cleaned stale in-progress asset: %s", asset_id)

    async def _make_decision(self, drone_state: DroneSimState) -> None:
        """Make a decision for a drone using goal selector."""
        if not self.run_state:
            return

        drone = drone_state.drone

        # Sync shared inspection state to this drone's world model
        self._sync_fleet_state_to_world_model(drone_state)

        # Get world snapshot
        snapshot = drone_state.world_model.get_snapshot()
        if not snapshot:
            return

        # Filter out assets already being targeted by other drones
        available_snapshot = self._filter_available_assets(snapshot, drone.drone_id)
        pending_assets = available_snapshot.get_pending_assets()

        logger.info("[DECISION] Drone %s:", drone.name)
        logger.info("  - Total assets in world: %d", len(snapshot.assets))
        logger.info("  - Fleet in-progress: %s", list(self.run_state.assets_in_progress.keys()))
        logger.info("  - Fleet inspected: %s", list(self.run_state.assets_inspected))
        logger.info("  - Available after filter: %d", len(available_snapshot.assets))
        logger.info("  - Pending inspections: %s", [a.asset_id for a in pending_assets])

        # Create goal selector and make decision
        selector = GoalSelector()
        goal = await selector.select_goal(available_snapshot)

        decision_id = (
            f"sim_{self.run_state.run_id}_{drone.drone_id}_{drone_state.decisions_made + 1}"
        )

        # B.2: Validate with critics (if enabled and not a WAIT decision)
        critic_result: dict[str, Any] | None = None
        if self.critic_orchestrator and goal.goal_type != GoalType.WAIT:
            try:
                # Convert goal to decision and create risk assessment
                decision = self._goal_to_decision(goal)
                risk = self._create_risk_assessment(drone_state)

                # Run critic validation
                approved, escalation = await self.critic_orchestrator.validate_decision(
                    decision=decision,
                    world=available_snapshot,
                    risk=risk,
                )

                critic_result = {
                    "approved": approved,
                    "escalation": escalation.model_dump() if escalation else None,
                }

                if not approved:
                    logger.warning(
                        f"Critics rejected {goal.goal_type.value} for {drone.name}: "
                        f"{escalation.reason if escalation else 'unknown'}"
                    )
                    # Log the rejection but don't override the goal in advisory mode
                    if escalation and escalation.requires_human_review:
                        logger.info("Escalation requires human review")

                # B.5: Broadcast critic validation event via WebSocket
                try:
                    await connection_manager.broadcast(
                        Event(
                            event_type=EventType.CRITIC_VALIDATION,
                            timestamp=self.run_state.current_time,
                            data={
                                "agent_label": "Orchestration AG",
                                "drone_id": drone.drone_id,
                                "drone_name": drone.name,
                                "decision_id": decision_id,
                                "action": decision.action.value,
                                "approved": approved,
                                "escalation": escalation.model_dump() if escalation else None,
                                "risk_level": risk.overall_level.value,
                                "risk_score": risk.overall_score,
                            },
                            severity=EventSeverity.WARNING if not approved else EventSeverity.INFO,
                        )
                    )
                except Exception as exc:
                    logger.debug("Failed to broadcast critic event: %s", exc)

            except Exception as e:
                logger.error(f"Critic validation failed: {e}")
                critic_result = {"approved": True, "error": str(e)}

        # Atomic asset claiming - prevent race conditions in multi-drone assignment
        if (
            goal.goal_type in (GoalType.INSPECT_ASSET, GoalType.INSPECT_ANOMALY)
            and goal.target_asset
        ):
            claimed = await self._try_claim_asset(goal.target_asset.asset_id, drone.drone_id)
            if not claimed:
                # Asset was claimed by another drone - fall back to WAIT
                logger.info(
                    f"{drone.name}: Asset {goal.target_asset.asset_id} claimed by another drone, waiting"
                )
                goal = Goal(
                    goal_type=GoalType.WAIT,
                    reason="Target asset claimed by another drone",
                    confidence=0.5,
                    priority=0,
                )

        drone_state.current_goal = goal
        drone_state.decisions_made += 1
        drone_state.last_decision_time = self.run_state.current_time

        # Update drone state based on goal
        await self._apply_goal(drone_state, goal)

        # Build decision record
        drone = drone_state.drone
        decision_record: dict[str, Any] = {
            "type": "decision",
            "timestamp": self.run_state.current_time.isoformat(),
            "elapsed_s": self.run_state.elapsed_seconds,
            "decision_id": decision_id,
            "drone_id": drone.drone_id,
            "drone_name": drone.name,
            "agent_label": "Decision Agent",
            "action": goal.goal_type.value,
            "confidence": goal.confidence,
            "reason": goal.reason,
            "priority": goal.priority,
            "risk_score": self._calculate_risk(drone_state),
            "battery_percent": drone.battery_percent,
            "risk_level": self._risk_level(drone_state),
            "vehicle_position": {
                "lat": drone.latitude,
                "lon": drone.longitude,
            },
            "mode": "GUIDED" if drone.in_air else "STABILIZE",
            "armed": drone.armed,
            "gps_healthy": drone.gps_healthy,
            "sensors_healthy": drone.sensors_healthy,
            # Reasoning context for dashboard display
            "reasoning_context": {
                "available_assets": len(available_snapshot.assets),
                "pending_inspections": len(pending_assets),
                "fleet_in_progress": len(self.run_state.assets_in_progress),
                "fleet_completed": len(self.run_state.assets_inspected),
                "battery_ok": drone.battery_percent > 25,
                "battery_percent": drone.battery_percent,
                "weather_ok": self.run_state.environment.wind_speed_ms < 12,
                "wind_speed": self.run_state.environment.wind_speed_ms,
            },
            # Alternatives considered
            "alternatives": self._get_alternatives_considered(
                drone_state, available_snapshot, goal
            ),
        }

        # Include target asset info for flight execution
        if goal.target_asset:
            asset = goal.target_asset
            # Extract lat/lon from asset.position (Asset class has position: Position)
            position = getattr(asset, "position", None)
            lat = getattr(position, "latitude", None) if position else None
            lon = getattr(position, "longitude", None) if position else None
            decision_record["target_asset"] = {
                "asset_id": getattr(asset, "asset_id", None) or getattr(asset, "name", "unknown"),
                "name": getattr(asset, "name", "unknown"),
                "latitude": lat,
                "longitude": lon,
                "inspection_altitude_agl": getattr(asset, "inspection_altitude_agl", 30.0),
                "orbit_radius_m": getattr(asset, "orbit_radius_m", 20.0),
                "dwell_time_s": getattr(asset, "dwell_time_s", 30.0),
            }

        # B.2: Add critic validation results to decision record
        if critic_result:
            decision_record["critic_validation"] = critic_result
            critic_entry = {
                "type": "critic_validation",
                "timestamp": self.run_state.current_time.isoformat(),
                "elapsed_s": self.run_state.elapsed_seconds,
                "decision_id": decision_id,
                "agent_label": "Orchestration AG",
                "drone_id": drone.drone_id,
                "drone_name": drone.name,
                "action": "critic_validation",
                "reason": (critic_result.get("escalation", {}) or {}).get("reason")
                or (
                    "Critics approved decision"
                    if critic_result.get("approved")
                    else "Critics rejected decision"
                ),
                "confidence": goal.confidence,
                "risk_score": self._calculate_risk(drone_state),
                "battery_percent": drone.battery_percent,
                "risk_level": self._risk_level(drone_state),
                "critic_validation": critic_result,
            }
            self._log_entry(critic_entry)

        self._log_entry(decision_record)

        logger.info(
            f"[{self.run_state.elapsed_seconds:.0f}s] {drone.name}: "
            f"{goal.goal_type.value} - {goal.reason}"
        )
        logger.info("  - Selected: %s - %s", goal.goal_type.value, goal.reason)

        try:
            await connection_manager.broadcast(
                Event(
                    event_type=EventType.SERVER_DECISION,
                    timestamp=self.run_state.current_time,
                    data={
                        "agent_label": "Decision Agent",
                        "decision_id": decision_id,
                        "drone_id": drone.drone_id,
                        "drone_name": drone.name,
                        "action": goal.goal_type.value,
                        "reasoning": goal.reason,
                        "confidence": goal.confidence,
                        "risk_level": self._risk_level(drone_state),
                        "risk_score": self._calculate_risk(drone_state),
                        "battery_percent": drone.battery_percent,
                        "target_asset": decision_record.get("target_asset"),
                        "elapsed_s": self.run_state.elapsed_seconds,
                        # Rich reasoning context for dashboard
                        "reasoning_context": decision_record.get("reasoning_context"),
                        "alternatives": decision_record.get("alternatives"),
                        "critic_validation": decision_record.get("critic_validation"),
                    },
                    severity=EventSeverity.INFO,
                )
            )
        except Exception as exc:
            logger.debug("Failed to broadcast decision event: %s", exc)

        if self.on_decision:
            self.on_decision(drone.drone_id, goal, decision_record)

    async def _apply_goal(self, drone_state: DroneSimState, goal: Goal) -> None:
        """Apply goal effects to drone state."""
        drone = drone_state.drone

        # Release any in-progress asset if switching away from inspection
        if drone_state.target_asset_id and self.run_state:
            if goal.goal_type not in (GoalType.INSPECT_ASSET, GoalType.INSPECT_ANOMALY):
                # Drone is abandoning current target - release it for other drones
                released_asset = drone_state.target_asset_id
                await self._release_asset(released_asset, drone.drone_id)
                logger.info(
                    f"{drone.name}: Released asset {released_asset} "
                    f"(switching to {goal.goal_type.value})"
                )

        if goal.goal_type == GoalType.ABORT:
            drone.state = DroneState.EMERGENCY
            drone.in_air = False
            drone.armed = False
            drone_state.moving = False
            drone_state.target_position = None
            drone_state.target_asset_id = None
            drone_state.target_altitude_agl = None
            drone_state.target_is_dock = False

        elif goal.goal_type in (
            GoalType.RETURN_LOW_BATTERY,
            GoalType.RETURN_WEATHER,
            GoalType.RETURN_MISSION_COMPLETE,
        ):
            drone.state = DroneState.RETURNING
            dock = drone_state.world_model.get_dock()
            if dock:
                drone_state.target_position = dock.position
                drone_state.target_asset_id = None
                drone_state.target_altitude_agl = dock.approach_altitude_m
                drone_state.target_is_dock = True
                drone_state.moving = True

        elif goal.goal_type in (GoalType.INSPECT_ASSET, GoalType.INSPECT_ANOMALY):
            drone.state = DroneState.INSPECTING
            if not drone.in_air:
                drone.in_air = True
                drone.armed = True
            if goal.target_asset:
                target = goal.target_asset
                drone_state.target_position = target.position
                drone_state.target_asset_id = target.asset_id
                drone_state.target_altitude_agl = target.inspection_altitude_agl
                drone_state.target_is_dock = False
                drone_state.moving = True

                # Asset claim already happened atomically in _make_decision()
                logger.debug(
                    f"{drone.name} targeting asset {target.asset_id} "
                    f"(fleet has {len(self.run_state.assets_in_progress) if self.run_state else 0} in progress)"
                )

        elif goal.goal_type == GoalType.WAIT:
            if not drone.in_air:
                drone.state = DroneState.IDLE
            drone_state.moving = False
            drone_state.target_position = None
            drone_state.target_asset_id = None
            drone_state.target_altitude_agl = None
            drone_state.target_is_dock = False

    def _calculate_risk(self, drone_state: DroneSimState) -> float:
        """Calculate current risk score for a drone."""
        drone = drone_state.drone
        risk = 0.0

        # Battery risk
        if drone.battery_percent < 15:
            risk += 0.5
        elif drone.battery_percent < 25:
            risk += 0.3
        elif drone.battery_percent < 40:
            risk += 0.1

        # GPS risk
        if not drone.gps_healthy:
            risk += 0.3
        elif drone.gps_hdop > 2.0:
            risk += 0.15

        # Sensor risk
        if not drone.sensors_healthy:
            risk += 0.2
        if not drone.ekf_healthy:
            risk += 0.2
        if not drone.motors_healthy:
            risk += 0.25

        # Environment risk
        if self.run_state:
            env = self.run_state.environment
            if env.wind_speed_ms > 12:
                risk += 0.2
            elif env.wind_speed_ms > 8:
                risk += 0.1
            if env.visibility_m < 1000:
                risk += 0.15

        return min(1.0, risk)

    def _risk_level(self, drone_state: DroneSimState) -> str:
        """Get risk level string."""
        risk = self._calculate_risk(drone_state)
        if risk >= 0.7:
            return "CRITICAL"
        elif risk >= 0.5:
            return "HIGH"
        elif risk >= 0.3:
            return "MODERATE"
        else:
            return "LOW"

    def _get_alternatives_considered(
        self,
        drone_state: DroneSimState,
        snapshot: WorldSnapshot,
        selected_goal: Goal,
    ) -> list[dict[str, Any]]:
        """Generate list of alternative actions and why they weren't selected."""
        alternatives: list[dict[str, Any]] = []
        drone = drone_state.drone

        # Check why we didn't inspect
        if selected_goal.goal_type != GoalType.INSPECT_ASSET:
            pending = snapshot.get_pending_assets()
            if not pending:
                alternatives.append({
                    "action": "inspect_asset",
                    "rejected": True,
                    "reason": "No pending assets available",
                })
            elif drone.battery_percent < 25:
                alternatives.append({
                    "action": "inspect_asset",
                    "rejected": True,
                    "reason": f"Battery too low ({drone.battery_percent:.0f}%)",
                })
            elif self.run_state and self.run_state.environment.wind_speed_ms > 12:
                alternatives.append({
                    "action": "inspect_asset",
                    "rejected": True,
                    "reason": f"Wind too high ({self.run_state.environment.wind_speed_ms:.1f} m/s)",
                })

        # Check why we didn't return for battery
        if selected_goal.goal_type != GoalType.RETURN_LOW_BATTERY:
            if drone.battery_percent > 25:
                alternatives.append({
                    "action": "return_low_battery",
                    "rejected": True,
                    "reason": f"Battery sufficient ({drone.battery_percent:.0f}%)",
                })

        # Check why we didn't return for weather
        if selected_goal.goal_type != GoalType.RETURN_WEATHER:
            if self.run_state and self.run_state.environment.wind_speed_ms <= 12:
                alternatives.append({
                    "action": "return_weather",
                    "rejected": True,
                    "reason": f"Weather acceptable (wind {self.run_state.environment.wind_speed_ms:.1f} m/s)",
                })

        # Check why we didn't abort
        if selected_goal.goal_type != GoalType.ABORT:
            alternatives.append({
                "action": "abort",
                "rejected": True,
                "reason": "No emergency conditions detected",
            })

        return alternatives

    def _create_risk_assessment(self, drone_state: DroneSimState) -> RiskAssessment:
        """Create a RiskAssessment object for critic validation.

        Args:
            drone_state: Current drone simulation state

        Returns:
            RiskAssessment with detailed risk factors
        """
        drone = drone_state.drone
        overall_score = self._calculate_risk(drone_state)

        # Map score to level
        if overall_score >= 0.7:
            overall_level = RiskLevel.CRITICAL
        elif overall_score >= 0.5:
            overall_level = RiskLevel.HIGH
        elif overall_score >= 0.3:
            overall_level = RiskLevel.MODERATE
        else:
            overall_level = RiskLevel.LOW

        # Build risk factors
        factors = {}

        # Battery factor
        battery_risk = max(0, (40 - drone.battery_percent) / 40)
        factors["battery"] = RiskFactor(
            name="battery",
            value=battery_risk,
            threshold=0.5,  # 50% depleted = concerning
            critical=0.8,  # 80% depleted = critical
            description=f"Battery at {drone.battery_percent:.1f}%",
        )

        # GPS factor
        gps_risk = 0.0 if drone.gps_healthy else 0.8
        if drone.gps_hdop > 2.0:
            gps_risk = max(gps_risk, 0.4)
        factors["gps"] = RiskFactor(
            name="gps",
            value=gps_risk,
            threshold=0.3,
            critical=0.7,
            description=f"GPS HDOP: {drone.gps_hdop:.1f}",
        )

        # Weather factor
        weather_risk = 0.0
        if self.run_state:
            wind = self.run_state.environment.wind_speed_ms
            weather_risk = min(1.0, wind / 15.0)  # 15 m/s = max risk
        factors["weather"] = RiskFactor(
            name="weather",
            value=weather_risk,
            threshold=0.5,
            critical=0.8,
            description=f"Wind: {self.run_state.environment.wind_speed_ms:.1f} m/s"
            if self.run_state
            else "Unknown",
        )

        # Build warnings
        warnings = []
        if battery_risk > 0.5:
            warnings.append(f"Low battery: {drone.battery_percent:.1f}%")
        if not drone.gps_healthy:
            warnings.append("GPS unhealthy")
        if weather_risk > 0.5:
            warnings.append("High wind conditions")

        return RiskAssessment(
            overall_level=overall_level,
            overall_score=overall_score,
            factors=factors,
            abort_recommended=overall_level == RiskLevel.CRITICAL,
            abort_reason="Critical risk level" if overall_level == RiskLevel.CRITICAL else None,
            warnings=warnings,
        )

    def _goal_to_decision(self, goal: Goal) -> Decision:
        """Convert a Goal to a Decision for critic validation.

        Args:
            goal: The goal selected by the goal selector

        Returns:
            Decision object compatible with critic orchestrator
        """
        # Map GoalType to ActionType
        goal_to_action = {
            GoalType.INSPECT_ASSET: ActionType.INSPECT,
            GoalType.INSPECT_ANOMALY: ActionType.INSPECT,
            GoalType.RETURN_LOW_BATTERY: ActionType.RTL,
            GoalType.RETURN_MISSION_COMPLETE: ActionType.DOCK,
            GoalType.RETURN_WEATHER: ActionType.RTL,
            GoalType.WAIT: ActionType.WAIT,
            GoalType.ABORT: ActionType.ABORT,
            GoalType.RECHARGE: ActionType.RECHARGE,
            GoalType.NONE: ActionType.WAIT,
        }

        action = goal_to_action.get(goal.goal_type, ActionType.WAIT)

        # Build parameters from goal
        parameters: dict[str, Any] = {}
        if goal.target_asset:
            asset = goal.target_asset
            if hasattr(asset, "position") and asset.position:
                parameters["position"] = {
                    "latitude": asset.position.latitude,
                    "longitude": asset.position.longitude,
                    "altitude_agl": getattr(asset, "inspection_altitude_agl", 30.0),
                }
            parameters["asset_id"] = asset.asset_id

        return Decision(
            action=action,
            parameters=parameters,
            confidence=goal.confidence,
            reasoning=goal.reason,
        )

    def set_critic_authority(self, authority: str) -> None:
        """Change critic authority model at runtime.

        Args:
            authority: One of 'advisory', 'blocking', 'escalation', 'hierarchical'
        """
        if not self.critic_orchestrator:
            logger.warning("Critics not enabled - cannot change authority model")
            return

        authority_map = {
            "advisory": AuthorityModel.ADVISORY,
            "blocking": AuthorityModel.BLOCKING,
            "escalation": AuthorityModel.ESCALATION,
            "hierarchical": AuthorityModel.HIERARCHICAL,
        }

        if authority.lower() in authority_map:
            self.critic_orchestrator.authority_model = authority_map[authority.lower()]
            logger.info(f"Critic authority changed to: {authority}")
        else:
            logger.error(
                f"Unknown authority model: {authority}. Valid: {list(authority_map.keys())}"
            )

    def _log_entry(self, entry: dict[str, Any]) -> None:
        """Add entry to decision log with run_id."""
        if self.run_state:
            # Add run_id to every entry for correlation
            entry_with_run_id = {"run_id": self.run_state.run_id, **entry}
            self.run_state.decision_log.append(entry_with_run_id)

    def save_decision_log(self, log_dir: Path | None = None) -> Path:
        """Save decision log to file.

        Args:
            log_dir: Directory to save log (uses default if None)

        Returns:
            Path to saved log file
        """
        if not self.run_state:
            raise ValueError("No scenario has been run")

        save_dir = log_dir or self.log_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.run_state.start_time.strftime("%Y%m%d_%H%M%S")
        scenario_id = self.run_state.scenario.scenario_id
        filename = f"decisions_{scenario_id}_{timestamp}.jsonl"
        filepath = save_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for entry in self.run_state.decision_log:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Saved decision log to {filepath}")
        return filepath

    def get_summary(self) -> dict[str, Any]:
        """Get summary of scenario run."""
        if not self.run_state:
            return {}

        total_decisions = sum(ds.decisions_made for ds in self.run_state.drone_states.values())

        drone_summaries = []
        for drone_id, ds in self.run_state.drone_states.items():
            drone_summaries.append({
                "drone_id": drone_id,
                "name": ds.drone.name,
                "final_battery": ds.drone.battery_percent,
                "final_state": ds.drone.state.value,
                "decisions_made": ds.decisions_made,
                "current_goal": ds.current_goal.goal_type.value if ds.current_goal else None,
            })

        return {
            "run_id": self.run_state.run_id,
            "scenario_id": self.run_state.scenario.scenario_id,
            "scenario_name": self.run_state.scenario.name,
            "duration_s": self.run_state.elapsed_seconds,
            "is_complete": self.run_state.is_complete,
            "abort_reason": self.run_state.abort_reason,
            "total_decisions": total_decisions,
            "total_battery_consumed": round(self.run_state.total_battery_consumed, 2),
            "anomalies_triggered": self.run_state.anomalies_triggered,
            "events_fired": len(self.run_state.events_fired),
            "drones": drone_summaries,
        }

    def stop(self) -> None:
        """Stop the running scenario."""
        if self.run_state:
            self.run_state.is_running = False
            logger.info("Scenario stopped by request")


async def run_scenario_demo(scenario_id: str, time_scale: float = 10.0) -> None:
    """Run a scenario demonstration.

    Args:
        scenario_id: ID of scenario to run
        time_scale: Speed multiplier
    """
    runner = ScenarioRunner()

    if not await runner.load_scenario(scenario_id):
        logger.error("Failed to load scenario: %s", scenario_id)
        return

    divider = "=" * 60
    logger.info("%s", divider)
    logger.info("Running: %s", runner.run_state.scenario.name)
    logger.info("Drones: %s", len(runner.run_state.scenario.drones))
    logger.info("Duration: %s minutes", runner.run_state.scenario.duration_minutes)
    logger.info("Time scale: %sx", time_scale)
    logger.info("%s", divider)

    await runner.run(time_scale=time_scale, max_duration_s=60)

    # Save log
    log_path = runner.save_decision_log()

    # Print summary
    summary = runner.get_summary()
    logger.info("SCENARIO SUMMARY")
    logger.info("%s", divider)
    logger.info("Duration: %.1fs simulated", summary.get("duration_s", 0.0))
    logger.info("Total decisions: %s", summary.get("total_decisions", 0))
    logger.info("Events fired: %s", summary.get("events_fired", 0))
    logger.info("Drone Status:")
    for drone in summary.get("drones", []):
        logger.info(
            "  %s: %s (battery: %.1f%%, decisions: %s)",
            drone.get("name", "unknown"),
            drone.get("final_state", "unknown"),
            float(drone.get("final_battery", 0.0)),
            drone.get("decisions_made", 0),
        )
    logger.info("Decision log saved to: %s", log_path)


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal_ops_001"
    asyncio.run(run_scenario_demo(scenario))
