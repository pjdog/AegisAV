"""Scenario Runner.

Executes multi-drone simulation scenarios, applying edge cases and
generating decision logs that can be viewed in the dashboard.
"""

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent.server.goal_selector import GoalSelector
from agent.server.goals import Goal, GoalType
from agent.server.scenarios import (
    DroneState,
    EnvironmentConditions,
    Scenario,
    ScenarioEvent,
    SimulatedDrone,
    get_scenario,
)
from agent.server.world_model import (
    Anomaly,
    Asset,
    AssetType,
    DockStatus,
    WorldModel,
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

    # Mission tracking
    inspections_completed: int = 0
    anomalies_found: int = 0


@dataclass
class ScenarioRunState:
    """Overall state of a running scenario."""

    scenario: Scenario
    start_time: datetime
    current_time: datetime
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


class ScenarioRunner:
    """Executes simulation scenarios with multiple drones.

    The runner:
    - Initializes world models for each drone
    - Runs a simulation loop at configurable speed
    - Applies edge case triggers (battery drain, GPS loss, etc.)
    - Makes decisions using the goal selector
    - Logs all decisions for dashboard viewing

    Example:
        runner = ScenarioRunner()
        await runner.load_scenario("battery_cascade_001")
        await runner.run(time_scale=10.0)  # 10x speed
        runner.save_decision_log("logs/")
    """

    def __init__(
        self,
        tick_interval_s: float = 1.0,
        decision_interval_s: float = 5.0,
        log_dir: Path | None = None,
    ) -> None:
        """Initialize scenario runner.

        Args:
            tick_interval_s: Simulation tick interval in seconds
            decision_interval_s: How often to make decisions per drone
            log_dir: Directory to save decision logs
        """
        self.tick_interval_s = tick_interval_s
        self.decision_interval_s = decision_interval_s
        self.log_dir = log_dir or Path("logs")
        self.run_state: ScenarioRunState | None = None

        # Callbacks for external integration
        self.on_decision: Callable[[str, Goal, dict], None] | None = None
        self.on_event: Callable[[ScenarioEvent], None] | None = None
        self.on_tick: Callable[[ScenarioRunState], None] | None = None

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
            world.add_asset(Asset(
                asset_id=asset.asset_id,
                name=asset.name,
                asset_type=AssetType(asset.asset_type) if asset.asset_type in [e.value for e in AssetType] else AssetType.OTHER,
                position=Position(
                    latitude=asset.latitude,
                    longitude=asset.longitude,
                    altitude_msl=50.0,
                ),
                priority=asset.priority,
            ))

            # Add anomalies for assets that have them
            if asset.has_anomaly:
                world.add_anomaly(Anomaly(
                    anomaly_id=f"anom_{asset.asset_id}",
                    asset_id=asset.asset_id,
                    detected_at=datetime.now(),
                    severity=asset.anomaly_severity,
                    description=f"Pre-existing anomaly on {asset.name}",
                ))

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

        Args:
            time_scale: Speed multiplier (1.0 = real-time, 10.0 = 10x speed)
            max_duration_s: Maximum real-time duration to run

        Returns:
            True if completed successfully
        """
        if not self.run_state:
            logger.error("No scenario loaded")
            return False

        logger.info(f"Starting scenario: {self.run_state.scenario.name} at {time_scale}x speed")

        real_start = datetime.now()
        scenario_duration_s = self.run_state.scenario.duration_minutes * 60

        while self.run_state.is_running:
            # Check if scenario time exceeded
            if self.run_state.elapsed_seconds >= scenario_duration_s:
                self.run_state.is_complete = True
                self.run_state.is_running = False
                logger.info("Scenario completed: duration reached")
                break

            # Check real-time limit
            if max_duration_s:
                real_elapsed = (datetime.now() - real_start).total_seconds()
                if real_elapsed >= max_duration_s:
                    logger.info("Scenario stopped: real-time limit reached")
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

        return self.run_state.is_complete

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
            self._update_drone_state(drone_state)

            # Check edge case triggers
            self._check_triggers(drone_state)

            # Make decision if interval elapsed
            if self._should_decide(drone_state):
                await self._make_decision(drone_state)

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
            logger.info(f"[{event.timestamp_offset_s:.0f}s] {event.event_type}: {event.description}")

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
                env.wind_speed_ms = min(
                    env.wind_speed_ms + 0.5,
                    env.wind_increase_to
                )

        # Visibility drop trigger
        if env.visibility_drop_at and current >= env.visibility_drop_at:
            if env.visibility_m > env.visibility_drop_to:
                # Gradual decrease
                env.visibility_m = max(
                    env.visibility_m - 500,
                    env.visibility_drop_to
                )

    def _update_drone_state(self, drone_state: DroneSimState) -> None:
        """Update drone physical state (battery, position, etc.)."""
        drone = drone_state.drone

        # Battery drain
        if drone.in_air:
            drain = drone.battery_drain_rate * (self.tick_interval_s / 60.0)
            drone.battery_percent = max(0.0, drone.battery_percent - drain)

        # Update world model with new state
        vehicle_state = self._drone_to_vehicle_state(drone)
        drone_state.world_model.update_vehicle(vehicle_state)

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
                logger.warning(f"{drone.name}: Battery-triggered sensor failure at {drone.battery_percent:.1f}%")

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

    async def _make_decision(self, drone_state: DroneSimState) -> None:
        """Make a decision for a drone using goal selector."""
        if not self.run_state:
            return

        # Get world snapshot
        snapshot = drone_state.world_model.get_snapshot()
        if not snapshot:
            return

        # Create goal selector and make decision
        selector = GoalSelector()
        goal = await selector.select_goal(snapshot)

        drone_state.current_goal = goal
        drone_state.decisions_made += 1
        drone_state.last_decision_time = self.run_state.current_time

        # Update drone state based on goal
        self._apply_goal(drone_state, goal)

        # Build decision record
        drone = drone_state.drone
        decision_record = {
            "type": "decision",
            "timestamp": self.run_state.current_time.isoformat(),
            "elapsed_s": self.run_state.elapsed_seconds,
            "drone_id": drone.drone_id,
            "drone_name": drone.name,
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
        }

        self._log_entry(decision_record)

        logger.info(
            f"[{self.run_state.elapsed_seconds:.0f}s] {drone.name}: "
            f"{goal.goal_type.value} - {goal.reason}"
        )

        if self.on_decision:
            self.on_decision(drone.drone_id, goal, decision_record)

    def _apply_goal(self, drone_state: DroneSimState, goal: Goal) -> None:
        """Apply goal effects to drone state."""
        drone = drone_state.drone

        if goal.goal_type == GoalType.ABORT:
            drone.state = DroneState.EMERGENCY
            drone.in_air = False
            drone.armed = False

        elif goal.goal_type in (GoalType.RETURN_LOW_BATTERY, GoalType.RETURN_WEATHER, GoalType.RETURN_MISSION_COMPLETE):
            drone.state = DroneState.RETURNING

        elif goal.goal_type in (GoalType.INSPECT_ASSET, GoalType.INSPECT_ANOMALY):
            drone.state = DroneState.INSPECTING
            if not drone.in_air:
                drone.in_air = True
                drone.armed = True

        elif goal.goal_type == GoalType.WAIT:
            if not drone.in_air:
                drone.state = DroneState.IDLE

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

    def _log_entry(self, entry: dict[str, Any]) -> None:
        """Add entry to decision log."""
        if self.run_state:
            self.run_state.decision_log.append(entry)

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

        total_decisions = sum(
            ds.decisions_made for ds in self.run_state.drone_states.values()
        )

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
            "scenario_id": self.run_state.scenario.scenario_id,
            "scenario_name": self.run_state.scenario.name,
            "duration_s": self.run_state.elapsed_seconds,
            "is_complete": self.run_state.is_complete,
            "total_decisions": total_decisions,
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
