"""Flight Controller.

Main orchestrator for autonomous flight operations.
Integrates flight backend, path planner, state estimator, and mission planner.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from autonomy.flight_backend import (
    AirSimBackendConfig,
    ArduPilotBackendConfig,
    BackendType,
    FlightBackend,
    FlightBackendConfig,
    create_backend,
)
from autonomy.mission_planner import (
    MissionPlan,
    MissionPlanner,
    MissionPlannerConfig,
    MissionWaypoint,
)
from autonomy.path_planner import FlightPath, PathPlanner, PathPlannerConfig
from autonomy.state_estimator import (
    EstimatedState,
    LocalizationMode,
    StateEstimator,
    StateEstimatorConfig,
)
from autonomy.vehicle_state import Position, VehicleState
from simulation.coordinate_utils import GeoReference

logger = logging.getLogger(__name__)


class FlightPhase(Enum):
    """Current phase of flight operations."""

    IDLE = "idle"  # Not flying, motors disarmed
    PREFLIGHT = "preflight"  # Running preflight checks
    TAKEOFF = "takeoff"  # Taking off
    ENROUTE = "enroute"  # Flying to waypoint
    INSPECTING = "inspecting"  # Performing inspection (orbiting/hovering)
    RETURNING = "returning"  # Returning to home
    LANDING = "landing"  # Landing
    EMERGENCY = "emergency"  # Emergency landing or abort


class AbortReason(Enum):
    """Reason for mission abort."""

    NONE = "none"
    USER_REQUESTED = "user_requested"
    LOW_BATTERY = "low_battery"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_LOST = "communication_lost"
    OBSTACLE_DETECTED = "obstacle_detected"
    GEOFENCE_VIOLATION = "geofence_violation"
    WEATHER = "weather"
    UNKNOWN = "unknown"


class FlightControllerEvent(BaseModel):
    """Event emitted by flight controller."""

    timestamp: datetime
    event_type: str
    phase: str
    message: str
    data: dict[str, Any] = {}


@dataclass
class FlightControllerConfig:
    """Configuration for flight controller."""

    # Backend selection
    backend_type: BackendType = BackendType.AIRSIM
    airsim_config: AirSimBackendConfig = field(default_factory=AirSimBackendConfig)
    ardupilot_config: ArduPilotBackendConfig = field(default_factory=ArduPilotBackendConfig)

    # Component configs
    path_planner_config: PathPlannerConfig = field(default_factory=PathPlannerConfig)
    state_estimator_config: StateEstimatorConfig = field(default_factory=StateEstimatorConfig)
    mission_planner_config: MissionPlannerConfig = field(default_factory=MissionPlannerConfig)

    # Flight parameters
    default_velocity_ms: float = 5.0
    inspection_velocity_ms: float = 2.0
    return_altitude_agl: float = 30.0
    waypoint_acceptance_radius_m: float = 3.0

    # Safety
    min_battery_percent: float = 20.0
    max_altitude_agl: float = 120.0
    geofence_radius_m: float = 500.0

    # Timing
    state_update_rate_hz: float = 10.0
    command_timeout_s: float = 60.0


class FlightController:
    """Main orchestrator for autonomous flight operations.

    Integrates all autonomy components:
    - FlightBackend: Low-level flight commands
    - PathPlanner: Obstacle-free path generation
    - StateEstimator: GPS/Visual sensor fusion
    - MissionPlanner: High-level mission management

    Provides a high-level interface for:
    - Executing single waypoints
    - Running complete missions
    - Handling emergencies and aborts

    Example:
        controller = FlightController(config)
        await controller.initialize()

        # Load and execute mission
        controller.load_mission("mission.yaml")
        await controller.execute_mission()

        # Or execute single goal
        await controller.goto_position(lat, lon, alt)

        await controller.shutdown()
    """

    def __init__(self, config: FlightControllerConfig | None = None) -> None:
        """Initialize flight controller.

        Args:
            config: Controller configuration
        """
        self._config = config or FlightControllerConfig()
        self._phase = FlightPhase.IDLE
        self._abort_reason = AbortReason.NONE

        # Components (initialized in initialize())
        self._backend: FlightBackend | None = None
        self._path_planner: PathPlanner | None = None
        self._state_estimator: StateEstimator | None = None
        self._mission_planner: MissionPlanner | None = None
        self._geo_ref: GeoReference | None = None

        # State
        self._current_mission: MissionPlan | None = None
        self._current_waypoint_idx: int = 0
        self._is_initialized = False
        self._abort_requested = False

        # Background tasks
        self._state_update_task: asyncio.Task | None = None
        self._running = False

        # Event callbacks
        self._event_callbacks: list[Callable[[FlightControllerEvent], None]] = []

    @property
    def phase(self) -> FlightPhase:
        """Current flight phase."""
        return self._phase

    @property
    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._is_initialized

    @property
    def is_flying(self) -> bool:
        """Check if currently in flight."""
        return self._phase in (
            FlightPhase.TAKEOFF,
            FlightPhase.ENROUTE,
            FlightPhase.INSPECTING,
            FlightPhase.RETURNING,
        )

    @property
    def current_mission(self) -> MissionPlan | None:
        """Get current mission plan."""
        return self._current_mission

    @property
    def backend(self) -> FlightBackend | None:
        """Get the flight backend."""
        return self._backend

    async def initialize(self) -> bool:
        """Initialize all components and connect to backend.

        Returns:
            True if initialization successful
        """
        if self._is_initialized:
            logger.warning("Already initialized")
            return True

        try:
            logger.info("Initializing FlightController...")

            # Create backend
            if self._config.backend_type == BackendType.AIRSIM:
                self._backend = create_backend(self._config.airsim_config)
            elif self._config.backend_type == BackendType.ARDUPILOT:
                self._backend = create_backend(self._config.ardupilot_config)
            else:
                self._backend = create_backend(FlightBackendConfig())

            # Connect to backend
            if not await self._backend.connect():
                logger.error("Failed to connect to flight backend")
                return False

            # Get initial state and set geo reference
            state = await self._backend.get_state()
            if state and state.home_position:
                self._geo_ref = GeoReference(
                    latitude=state.home_position.latitude,
                    longitude=state.home_position.longitude,
                    altitude=state.home_position.altitude_msl,
                )
            elif state:
                self._geo_ref = GeoReference(
                    latitude=state.position.latitude,
                    longitude=state.position.longitude,
                    altitude=state.position.altitude_msl,
                )

            # Initialize path planner
            self._path_planner = PathPlanner(
                geo_reference=self._geo_ref,
                config=self._config.path_planner_config,
            )

            # Initialize state estimator
            self._state_estimator = StateEstimator(
                config=self._config.state_estimator_config,
            )
            if self._geo_ref:
                self._state_estimator.set_reference(
                    self._geo_ref.latitude,
                    self._geo_ref.longitude,
                    self._geo_ref.altitude,
                )

            # Initialize mission planner
            self._mission_planner = MissionPlanner(
                config=self._config.mission_planner_config,
            )

            # Start state update loop
            self._running = True
            self._state_update_task = asyncio.create_task(self._state_update_loop())

            self._is_initialized = True
            self._emit_event("initialized", "FlightController initialized successfully")
            logger.info("FlightController initialized")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown controller and disconnect from backend."""
        logger.info("Shutting down FlightController...")

        self._running = False

        if self._state_update_task:
            self._state_update_task.cancel()
            try:
                await self._state_update_task
            except asyncio.CancelledError:
                pass

        if self._backend:
            await self._backend.disconnect()

        self._is_initialized = False
        self._emit_event("shutdown", "FlightController shut down")

    def load_mission(self, config_path: str | Path) -> bool:
        """Load mission configuration from YAML file.

        Args:
            config_path: Path to mission YAML file

        Returns:
            True if loaded successfully
        """
        if not self._mission_planner:
            logger.error("Mission planner not initialized")
            return False

        if not self._mission_planner.load_mission(config_path):
            return False

        # Update path planner with mission obstacles
        if self._mission_planner.path_planner:
            self._path_planner = self._mission_planner.path_planner

        # Update geo reference
        if self._mission_planner.geo_reference:
            self._geo_ref = self._mission_planner.geo_reference
            if self._state_estimator:
                self._state_estimator.set_reference(
                    self._geo_ref.latitude,
                    self._geo_ref.longitude,
                    self._geo_ref.altitude,
                )

        return True

    def load_mission_dict(self, config: dict) -> bool:
        """Load mission configuration from dictionary.

        Args:
            config: Mission configuration dictionary

        Returns:
            True if loaded successfully
        """
        if not self._mission_planner:
            logger.error("Mission planner not initialized")
            return False

        return self._mission_planner.load_mission_dict(config)

    async def execute_mission(self) -> bool:
        """Execute the loaded mission.

        Returns:
            True if mission completed successfully
        """
        if not self._is_initialized:
            logger.error("Controller not initialized")
            return False

        if not self._mission_planner:
            logger.error("No mission planner")
            return False

        # Create mission plan
        state = await self._backend.get_state() if self._backend else None
        current_pos = state.position if state else None

        plan = self._mission_planner.create_plan(
            current_position=current_pos,
            battery_percent=state.battery.remaining_percent if state else 100,
        )

        if not plan:
            logger.error("Failed to create mission plan")
            return False

        self._current_mission = plan
        self._current_waypoint_idx = 0
        self._abort_requested = False

        self._emit_event(
            "mission_started",
            f"Starting mission: {plan.mission_name}",
            {"mission_id": plan.mission_id, "waypoints": plan.num_waypoints},
        )

        # Execute waypoints
        for i, waypoint in enumerate(plan.waypoints):
            if self._abort_requested:
                logger.warning("Mission aborted")
                self._emit_event("mission_aborted", f"Mission aborted: {self._abort_reason.value}")
                return False

            self._current_waypoint_idx = i
            success = await self._execute_waypoint(waypoint)

            if not success:
                logger.error(f"Failed to execute waypoint {waypoint.waypoint_id}")
                await self._handle_waypoint_failure(waypoint)
                return False

            # Mark waypoint complete
            waypoint.completed = True

            # Update target status if inspection waypoint
            if waypoint.target_id:
                for target in plan.targets:
                    if target.target_id == waypoint.target_id:
                        target.inspected = True
                        target.last_inspection = datetime.now()
                        break

        self._emit_event(
            "mission_complete",
            f"Mission complete: {plan.completed_targets}/{plan.num_targets} targets inspected",
        )

        return True

    async def _execute_waypoint(self, waypoint: MissionWaypoint) -> bool:
        """Execute a single mission waypoint.

        Args:
            waypoint: Waypoint to execute

        Returns:
            True if waypoint executed successfully
        """
        if not self._backend:
            return False

        self._emit_event(
            "waypoint_started",
            f"Executing waypoint {waypoint.waypoint_id}: {waypoint.action}",
            {"waypoint_id": waypoint.waypoint_id, "action": waypoint.action},
        )

        try:
            if waypoint.action == "takeoff":
                self._set_phase(FlightPhase.TAKEOFF)
                altitude = waypoint.altitude_msl - (self._geo_ref.altitude if self._geo_ref else 0)
                success = await self._backend.takeoff(altitude_agl=altitude)

            elif waypoint.action == "land":
                self._set_phase(FlightPhase.LANDING)
                success = await self._backend.land()
                if success:
                    self._set_phase(FlightPhase.IDLE)

            elif waypoint.action == "inspect":
                self._set_phase(FlightPhase.ENROUTE)
                success = await self._fly_to_position(
                    waypoint.latitude,
                    waypoint.longitude,
                    waypoint.altitude_msl,
                )
                if success:
                    self._set_phase(FlightPhase.INSPECTING)
                    # Hover for dwell time
                    await self._backend.hover()
                    await asyncio.sleep(waypoint.dwell_time_s)

            elif waypoint.action == "flyover":
                self._set_phase(FlightPhase.ENROUTE)
                success = await self._fly_to_position(
                    waypoint.latitude,
                    waypoint.longitude,
                    waypoint.altitude_msl,
                )

            elif waypoint.action == "hover":
                self._set_phase(FlightPhase.INSPECTING)
                await self._backend.hover()
                await asyncio.sleep(waypoint.dwell_time_s)
                success = True

            else:
                logger.warning(f"Unknown waypoint action: {waypoint.action}")
                success = True

            if success:
                self._emit_event(
                    "waypoint_complete",
                    f"Waypoint {waypoint.waypoint_id} complete",
                )

            return success

        except Exception as e:
            logger.error(f"Waypoint execution error: {e}")
            return False

    async def _fly_to_position(
        self,
        latitude: float,
        longitude: float,
        altitude_msl: float,
    ) -> bool:
        """Fly to a GPS position with obstacle avoidance.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            altitude_msl: Target altitude MSL

        Returns:
            True if destination reached
        """
        if not self._backend:
            return False

        # Get current position
        state = await self._backend.get_state()
        if not state:
            logger.error("Cannot get current state")
            return False

        # Plan path with obstacle avoidance
        if self._path_planner:
            path = self._path_planner.plan_path_gps(
                state.position.latitude,
                state.position.longitude,
                state.position.altitude_msl,
                latitude,
                longitude,
                altitude_msl,
                velocity=self._config.default_velocity_ms,
            )

            if path.status != "success":
                logger.warning(f"Path planning failed: {path.status}")
                # Fall back to direct flight
                return await self._backend.goto_position_gps(
                    latitude, longitude, altitude_msl,
                    velocity=self._config.default_velocity_ms,
                )

            # Execute path waypoints
            if self._geo_ref:
                for wp_ned in path.waypoints[1:]:  # Skip start position
                    if self._abort_requested:
                        return False

                    wp_lat, wp_lon, wp_alt = self._geo_ref.ned_to_gps(*wp_ned)
                    success = await self._backend.goto_position_gps(
                        wp_lat, wp_lon, wp_alt,
                        velocity=self._config.default_velocity_ms,
                    )
                    if not success:
                        return False

            return True

        else:
            # Direct flight without path planning
            return await self._backend.goto_position_gps(
                latitude, longitude, altitude_msl,
                velocity=self._config.default_velocity_ms,
            )

    async def goto_position(
        self,
        latitude: float,
        longitude: float,
        altitude_msl: float,
        velocity: float | None = None,
    ) -> bool:
        """Fly to a GPS position (standalone command).

        Args:
            latitude: Target latitude
            longitude: Target longitude
            altitude_msl: Target altitude MSL
            velocity: Optional flight velocity

        Returns:
            True if destination reached
        """
        if not self._is_initialized or not self._backend:
            logger.error("Controller not initialized")
            return False

        velocity = velocity or self._config.default_velocity_ms
        self._set_phase(FlightPhase.ENROUTE)

        success = await self._fly_to_position(latitude, longitude, altitude_msl)

        if success:
            await self._backend.hover()

        return success

    async def takeoff(self, altitude_agl: float = 10.0) -> bool:
        """Take off to specified altitude.

        Args:
            altitude_agl: Target altitude above ground

        Returns:
            True if takeoff successful
        """
        if not self._is_initialized or not self._backend:
            return False

        self._set_phase(FlightPhase.TAKEOFF)
        success = await self._backend.takeoff(altitude_agl)

        if success:
            self._set_phase(FlightPhase.ENROUTE)
        else:
            self._set_phase(FlightPhase.IDLE)

        return success

    async def land(self) -> bool:
        """Land at current position.

        Returns:
            True if landing successful
        """
        if not self._is_initialized or not self._backend:
            return False

        self._set_phase(FlightPhase.LANDING)
        success = await self._backend.land()
        self._set_phase(FlightPhase.IDLE)

        return success

    async def return_to_home(self) -> bool:
        """Return to home position and land.

        Returns:
            True if RTH successful
        """
        if not self._is_initialized or not self._backend:
            return False

        self._set_phase(FlightPhase.RETURNING)
        success = await self._backend.return_to_home(
            altitude_agl=self._config.return_altitude_agl
        )
        self._set_phase(FlightPhase.IDLE)

        return success

    def request_abort(self, reason: AbortReason = AbortReason.USER_REQUESTED) -> None:
        """Request mission abort.

        Args:
            reason: Reason for abort
        """
        self._abort_requested = True
        self._abort_reason = reason
        self._emit_event("abort_requested", f"Abort requested: {reason.value}")
        logger.warning(f"Abort requested: {reason.value}")

    async def emergency_land(self) -> bool:
        """Perform emergency landing at current position.

        Returns:
            True if emergency landing initiated
        """
        if not self._backend:
            return False

        self._set_phase(FlightPhase.EMERGENCY)
        self._abort_requested = True
        self._abort_reason = AbortReason.USER_REQUESTED

        self._emit_event("emergency_landing", "Emergency landing initiated")
        return await self._backend.land()

    async def _handle_waypoint_failure(self, waypoint: MissionWaypoint) -> None:
        """Handle waypoint execution failure.

        Args:
            waypoint: The failed waypoint
        """
        logger.error(f"Handling waypoint failure: {waypoint.waypoint_id}")
        self._emit_event("waypoint_failed", f"Waypoint {waypoint.waypoint_id} failed")

        # Attempt to return home
        self._set_phase(FlightPhase.RETURNING)
        await self._backend.return_to_home() if self._backend else None

    async def _state_update_loop(self) -> None:
        """Background loop to update state and check safety."""
        interval = 1.0 / self._config.state_update_rate_hz

        while self._running:
            try:
                if self._backend and self._backend.is_connected:
                    state = await self._backend.get_state()
                    if state:
                        # Update state estimator
                        if self._state_estimator:
                            self._state_estimator.update_gps(state)

                        # Check safety constraints
                        await self._check_safety(state)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"State update error: {e}")
                await asyncio.sleep(1.0)

    async def _check_safety(self, state: VehicleState) -> None:
        """Check safety constraints and trigger abort if needed."""
        # Check battery
        if state.battery.remaining_percent < self._config.min_battery_percent:
            if self.is_flying and not self._abort_requested:
                logger.warning("Low battery - initiating RTH")
                self.request_abort(AbortReason.LOW_BATTERY)
                await self.return_to_home()

        # Check altitude
        if state.position.altitude_agl and state.position.altitude_agl > self._config.max_altitude_agl:
            logger.warning(f"Altitude {state.position.altitude_agl}m exceeds max {self._config.max_altitude_agl}m")

        # Check geofence (simple circular check)
        if self._geo_ref and self.is_flying:
            home = Position(
                latitude=self._geo_ref.latitude,
                longitude=self._geo_ref.longitude,
                altitude_msl=self._geo_ref.altitude,
            )
            distance = state.position.distance_to(home)
            if distance > self._config.geofence_radius_m:
                logger.warning(f"Geofence violation: {distance:.0f}m from home")

    def _set_phase(self, phase: FlightPhase) -> None:
        """Update flight phase and emit event."""
        if phase != self._phase:
            old_phase = self._phase
            self._phase = phase
            self._emit_event(
                "phase_changed",
                f"Phase: {old_phase.value} -> {phase.value}",
                {"old_phase": old_phase.value, "new_phase": phase.value},
            )

    def on_event(self, callback: Callable[[FlightControllerEvent], None]) -> None:
        """Register callback for controller events.

        Args:
            callback: Function to call on events
        """
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: str, message: str, data: dict | None = None) -> None:
        """Emit a controller event."""
        event = FlightControllerEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            phase=self._phase.value,
            message=message,
            data=data or {},
        )

        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_state_estimate(self) -> EstimatedState | None:
        """Get current fused state estimate."""
        if self._state_estimator:
            return self._state_estimator.get_estimate()
        return None

    async def get_vehicle_state(self) -> VehicleState | None:
        """Get current vehicle state from backend."""
        if self._backend:
            return await self._backend.get_state()
        return None
