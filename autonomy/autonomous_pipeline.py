"""Autonomous Flight Pipeline.

End-to-end autonomous system that:
1. Loads or generates a mission map with targets and obstacles
2. Uses visual localization for GPS-denied environments
3. Plans optimal paths around obstacles
4. Executes inspection missions autonomously

This is the main orchestrator that makes the drone actually fly.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from autonomy.flight_backend import FlightBackend, create_backend
from autonomy.flight_controller import (
    FlightController,
    FlightControllerConfig,
    FlightControllerEvent,
    FlightPhase,
)
from autonomy.mission_planner import (
    MissionPlanner,
    MissionPlannerConfig,
    MissionPlan,
    MissionConfig,
    InspectionTarget,
    load_mission_config,
)
from autonomy.path_planner import PathPlanner, PathPlannerConfig, Obstacle
from autonomy.state_estimator import StateEstimator, StateEstimatorConfig, LocalizationMode
from autonomy.vehicle_state import VehicleState, Position

if TYPE_CHECKING:
    from simulation.realtime_bridge import RealtimeAirSimBridge
    from simulation.coordinate_utils import GeoReference

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Autonomous pipeline state."""

    IDLE = "idle"
    LOADING_MAP = "loading_map"
    INITIALIZING = "initializing"
    PREFLIGHT = "preflight"
    EXECUTING = "executing"
    PAUSED = "paused"
    RETURNING = "returning"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class PipelineStatus:
    """Current status of the autonomous pipeline."""

    state: PipelineState = PipelineState.IDLE
    mission_name: str = ""
    current_target: str = ""
    targets_completed: int = 0
    targets_total: int = 0
    battery_percent: float = 100.0
    position: Position | None = None
    localization_mode: str = "unknown"
    flight_phase: str = "idle"
    error_message: str = ""
    start_time: datetime | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "mission_name": self.mission_name,
            "current_target": self.current_target,
            "targets_completed": self.targets_completed,
            "targets_total": self.targets_total,
            "battery_percent": round(self.battery_percent, 1),
            "position": {
                "lat": self.position.latitude,
                "lon": self.position.longitude,
                "alt_agl": self.position.altitude_agl,
            } if self.position else None,
            "localization_mode": self.localization_mode,
            "flight_phase": self.flight_phase,
            "error_message": self.error_message,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


@dataclass
class AutonomousPipelineConfig:
    """Configuration for the autonomous pipeline."""

    # Mission config file path (optional - will use default if not provided)
    mission_config_path: Path | None = None

    # Default mission parameters
    default_altitude_agl: float = 30.0
    default_velocity: float = 5.0
    default_orbit_radius: float = 20.0
    default_dwell_time: float = 30.0

    # Safety limits
    min_battery_percent: float = 20.0
    max_flight_time_minutes: float = 30.0
    geofence_radius_m: float = 500.0

    # Localization
    use_visual_odometry: bool = True
    gps_required: bool = False

    # Path planning
    enable_obstacle_avoidance: bool = True
    obstacle_buffer_m: float = 5.0

    # Auto-start
    auto_takeoff: bool = True
    auto_return_on_complete: bool = True


class AutonomousPipeline:
    """End-to-end autonomous flight pipeline.

    This class orchestrates:
    - Map/mission loading
    - Flight backend connection
    - State estimation with GPS/visual fusion
    - Path planning with obstacle avoidance
    - Mission execution

    Example:
        # Create pipeline
        pipeline = AutonomousPipeline(config)

        # Load mission from config file
        await pipeline.load_mission("configs/mission_config.yaml")

        # Or generate a default mission around current position
        await pipeline.generate_default_mission(home_lat, home_lon)

        # Start autonomous execution
        await pipeline.start()

        # Monitor status
        status = pipeline.get_status()
        print(f"State: {status.state}, Target: {status.current_target}")

        # Stop if needed
        await pipeline.stop()
    """

    def __init__(
        self,
        config: AutonomousPipelineConfig | None = None,
        backend: FlightBackend | None = None,
        geo_ref: "GeoReference | None" = None,
    ) -> None:
        """Initialize the autonomous pipeline.

        Args:
            config: Pipeline configuration
            backend: Optional pre-configured flight backend
            geo_ref: Optional geographic reference
        """
        self._config = config or AutonomousPipelineConfig()
        self._backend = backend
        self._geo_ref = geo_ref

        # Components (initialized lazily)
        self._flight_controller: FlightController | None = None
        self._state_estimator: StateEstimator | None = None
        self._path_planner: PathPlanner | None = None
        self._mission_planner: MissionPlanner | None = None

        # Mission state
        self._mission: MissionPlan | None = None
        self._mission_config_path: Path | None = None
        self._status = PipelineStatus()
        self._is_running = False
        self._run_task: asyncio.Task | None = None
        self._navigation_map: dict[str, Any] | None = None

        # Callbacks
        self._on_status_change: list[Callable[[PipelineStatus], None]] = []
        self._on_target_complete: list[Callable[[str], None]] = []
        self._on_error: list[Callable[[str], None]] = []

        logger.info("AutonomousPipeline created")

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running

    @property
    def status(self) -> PipelineStatus:
        """Get current pipeline status."""
        return self._status

    async def initialize(
        self,
        airsim_bridge: "RealtimeAirSimBridge | None" = None,
        geo_ref: "GeoReference | None" = None,
    ) -> bool:
        """Initialize pipeline components.

        Args:
            airsim_bridge: Optional AirSim bridge for simulation
            geo_ref: Geographic reference for coordinate conversion

        Returns:
            True if initialization successful
        """
        try:
            self._update_state(PipelineState.INITIALIZING)

            # Store geo reference
            if geo_ref:
                self._geo_ref = geo_ref

            # Create or use provided backend
            if self._backend is None:
                if airsim_bridge:
                    from autonomy.backends.airsim_backend import AirSimBackend
                    self._backend = AirSimBackend(airsim_bridge, self._geo_ref)
                else:
                    # Create mock backend for testing
                    from autonomy.backends.mock_backend import MockFlightBackend
                    self._backend = MockFlightBackend()
                    logger.warning("Using MockFlightBackend - no real flight")

            # Connect backend
            if not await self._backend.connect():
                logger.error("Failed to connect flight backend")
                self._update_state(PipelineState.ERROR, error="Backend connection failed")
                return False

            # Create mission planner (it creates its own path planner internally)
            self._mission_planner = MissionPlanner(
                MissionPlannerConfig(
                    path_planner_config=PathPlannerConfig(
                        obstacle_buffer=self._config.obstacle_buffer_m,
                        min_altitude=10.0,
                        max_altitude=120.0,
                    )
                )
            )

            # Create flight controller with config
            fc_config = FlightControllerConfig(
                path_planner_config=PathPlannerConfig(
                    obstacle_buffer=self._config.obstacle_buffer_m,
                    min_altitude=10.0,
                    max_altitude=120.0,
                ),
                default_velocity_ms=self._config.default_velocity,
                min_battery_percent=self._config.min_battery_percent,
                geofence_radius_m=self._config.geofence_radius_m,
            )
            self._flight_controller = FlightController(fc_config)

            # Store references that will be set after initialization
            self._state_estimator = None
            self._path_planner = None

            self._update_state(PipelineState.IDLE)
            logger.info("AutonomousPipeline initialized successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize pipeline: {e}")
            self._update_state(PipelineState.ERROR, error=str(e))
            return False

    async def load_mission(self, config_path: str | Path) -> bool:
        """Load mission from YAML config file.

        Args:
            config_path: Path to mission YAML config

        Returns:
            True if mission loaded successfully
        """
        try:
            self._update_state(PipelineState.LOADING_MAP)

            path = Path(config_path)
            if not path.exists():
                logger.error(f"Mission config not found: {path}")
                self._update_state(PipelineState.ERROR, error=f"Config not found: {path}")
                return False

            if self._flight_controller is None:
                logger.error("Pipeline not initialized")
                self._update_state(PipelineState.ERROR, error="Pipeline not initialized")
                return False

            # Store config path for later use by flight controller
            self._mission_config_path = path

            # Also load it with the mission planner for status tracking
            if self._mission_planner and self._mission_planner.load_mission(path):
                self._mission = self._mission_planner.create_plan()
                if self._mission:
                    self._status.mission_name = self._mission.mission_name
                    self._status.targets_total = len(self._mission.targets)
                    self._status.targets_completed = 0

            self._update_state(PipelineState.IDLE)
            logger.info(f"Mission config loaded: {path}")
            return True

        except Exception as e:
            logger.exception(f"Failed to load mission: {e}")
            self._update_state(PipelineState.ERROR, error=str(e))
            return False

    async def generate_default_mission(
        self,
        home_lat: float,
        home_lon: float,
        home_alt: float = 0.0,
        targets: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Generate a default mission around a home position.

        Args:
            home_lat: Home latitude
            home_lon: Home longitude
            home_alt: Home altitude MSL
            targets: Optional list of target dictionaries

        Returns:
            True if mission generated successfully
        """
        try:
            self._update_state(PipelineState.LOADING_MAP)

            if self._mission_planner is None:
                await self.initialize()

            # Create default mission config
            config = MissionConfig(
                name="auto_generated_mission",
                home_latitude=home_lat,
                home_longitude=home_lon,
                home_altitude=home_alt,
                default_altitude_agl=self._config.default_altitude_agl,
                default_velocity=self._config.default_velocity,
                default_orbit_radius=self._config.default_orbit_radius,
                default_dwell_time=self._config.default_dwell_time,
                min_battery_return=self._config.min_battery_percent,
                targets=[],
                obstacles=[],
            )

            # Add targets if provided
            if targets:
                for i, t in enumerate(targets):
                    target = InspectionTarget(
                        target_id=t.get("id", f"target_{i}"),
                        name=t.get("name", f"Target {i}"),
                        latitude=t.get("latitude", home_lat),
                        longitude=t.get("longitude", home_lon),
                        altitude_agl=t.get("altitude_agl", self._config.default_altitude_agl),
                        priority=t.get("priority", 1),
                        orbit_radius=t.get("orbit_radius", self._config.default_orbit_radius),
                        dwell_time=t.get("dwell_time", self._config.default_dwell_time),
                    )
                    config.targets.append(target)

            # If no targets, create a simple patrol pattern
            if not config.targets:
                import math
                radius = 50.0  # meters
                for i in range(4):
                    angle = math.radians(i * 90)
                    # Simple offset calculation (approximate for small distances)
                    dlat = (radius * math.cos(angle)) / 111000
                    dlon = (radius * math.sin(angle)) / (111000 * math.cos(math.radians(home_lat)))

                    config.targets.append(InspectionTarget(
                        target_id=f"patrol_{i}",
                        name=f"Patrol Point {i+1}",
                        latitude=home_lat + dlat,
                        longitude=home_lon + dlon,
                        altitude_agl=self._config.default_altitude_agl,
                        priority=1,
                        orbit_radius=10.0,
                        dwell_time=10.0,
                    ))

            # Plan mission
            self._mission = self._mission_planner.plan_mission(config)

            self._status.mission_name = config.name
            self._status.targets_total = len(self._mission.targets)
            self._status.targets_completed = 0

            self._update_state(PipelineState.IDLE)
            logger.info(f"Generated mission with {len(self._mission.targets)} targets")
            return True

        except Exception as e:
            logger.exception(f"Failed to generate mission: {e}")
            self._update_state(PipelineState.ERROR, error=str(e))
            return False

    async def start(self) -> bool:
        """Start autonomous mission execution.

        Returns:
            True if started successfully
        """
        if self._is_running:
            logger.warning("Pipeline already running")
            return False

        if self._mission is None:
            logger.error("No mission loaded")
            self._update_state(PipelineState.ERROR, error="No mission loaded")
            return False

        if self._flight_controller is None:
            logger.error("Pipeline not initialized")
            self._update_state(PipelineState.ERROR, error="Not initialized")
            return False

        self._is_running = True
        self._status.start_time = datetime.now()
        self._run_task = asyncio.create_task(self._run_mission())

        logger.info(f"Started mission: {self._status.mission_name}")
        return True

    async def stop(self) -> bool:
        """Stop mission execution and return to home.

        Returns:
            True if stopped successfully
        """
        if not self._is_running:
            return True

        self._is_running = False
        self._update_state(PipelineState.RETURNING)

        # Cancel run task
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        # Return to home
        if self._flight_controller and self._flight_controller.is_flying:
            await self._flight_controller.request_abort()

        self._update_state(PipelineState.IDLE)
        logger.info("Pipeline stopped")
        return True

    async def pause(self) -> bool:
        """Pause mission execution (hover in place)."""
        if not self._is_running:
            return False

        self._update_state(PipelineState.PAUSED)
        if self._backend:
            await self._backend.hover()

        logger.info("Mission paused")
        return True

    async def resume(self) -> bool:
        """Resume paused mission."""
        if self._status.state != PipelineState.PAUSED:
            return False

        self._update_state(PipelineState.EXECUTING)
        logger.info("Mission resumed")
        return True

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status."""
        # Update elapsed time
        if self._status.start_time:
            self._status.elapsed_seconds = (
                datetime.now() - self._status.start_time
            ).total_seconds()

        # Update from flight controller
        if self._flight_controller:
            self._status.flight_phase = self._flight_controller.phase.value

        # Update from state estimator
        if self._state_estimator:
            self._status.localization_mode = self._state_estimator.mode.value

        # Update from backend
        if self._backend:
            state = asyncio.get_event_loop().run_until_complete(
                self._backend.get_state()
            ) if asyncio.get_event_loop().is_running() else None
            if state:
                self._status.position = state.position
                self._status.battery_percent = state.battery.remaining_percent

        return self._status

    def on_status_change(self, callback: Callable[[PipelineStatus], None]) -> None:
        """Register callback for status changes."""
        self._on_status_change.append(callback)

    def on_target_complete(self, callback: Callable[[str], None]) -> None:
        """Register callback for target completion."""
        self._on_target_complete.append(callback)

    def on_error(self, callback: Callable[[str], None]) -> None:
        """Register callback for errors."""
        self._on_error.append(callback)

    def set_navigation_map(self, nav_map: dict[str, Any] | None) -> None:
        """Update navigation map and trigger replanning if active."""
        self._navigation_map = nav_map
        if self._mission_planner:
            self._mission_planner.set_navigation_map(nav_map)
        if self._flight_controller:
            self._flight_controller.set_navigation_map(nav_map)
            if self._is_running:
                self._flight_controller.request_replan("map_update")

    async def _run_mission(self) -> None:
        """Main mission execution loop."""
        try:
            self._update_state(PipelineState.PREFLIGHT)

            # Initialize flight controller
            if not await self._flight_controller.initialize():
                self._update_state(PipelineState.ERROR, error="Flight controller init failed")
                return

            # Load mission into flight controller
            if self._mission_config_path:
                if not self._flight_controller.load_mission(self._mission_config_path):
                    self._update_state(PipelineState.ERROR, error="Failed to load mission into flight controller")
                    return
            else:
                self._update_state(PipelineState.ERROR, error="No mission config path set")
                return

            # Execute mission
            self._update_state(PipelineState.EXECUTING)

            success = await self._flight_controller.execute_mission()

            if success:
                self._update_state(PipelineState.COMPLETE)
                logger.info("Mission completed successfully")
            else:
                self._update_state(PipelineState.ERROR, error="Mission execution failed")

        except asyncio.CancelledError:
            logger.info("Mission cancelled")
            raise

        except Exception as e:
            logger.exception(f"Mission error: {e}")
            self._update_state(PipelineState.ERROR, error=str(e))

        finally:
            self._is_running = False

    def _update_state(
        self,
        state: PipelineState,
        error: str = "",
    ) -> None:
        """Update pipeline state and notify callbacks."""
        self._status.state = state
        self._status.error_message = error

        for callback in self._on_status_change:
            try:
                callback(self._status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

        if error:
            for callback in self._on_error:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"Error callback error: {e}")

    def _handle_flight_event(self, event: FlightControllerEvent) -> None:
        """Handle flight controller events."""
        logger.info(f"Flight event: {event.event_type} - {event.message}")

        # Update status based on event
        if "target" in event.event_type.lower() and "complete" in event.event_type.lower():
            self._status.targets_completed += 1
            target_id = event.data.get("target_id", "unknown")
            self._status.current_target = ""

            for callback in self._on_target_complete:
                try:
                    callback(target_id)
                except Exception as e:
                    logger.error(f"Target complete callback error: {e}")

        elif "approaching" in event.message.lower():
            self._status.current_target = event.data.get("target_id", "unknown")


# Factory function for easy creation
async def create_autonomous_pipeline(
    airsim_bridge: "RealtimeAirSimBridge | None" = None,
    geo_ref: "GeoReference | None" = None,
    mission_config_path: str | Path | None = None,
    config: AutonomousPipelineConfig | None = None,
) -> AutonomousPipeline:
    """Create and initialize an autonomous pipeline.

    Args:
        airsim_bridge: Optional AirSim bridge for simulation
        geo_ref: Geographic reference for coordinate conversion
        mission_config_path: Optional path to mission YAML config
        config: Optional pipeline configuration

    Returns:
        Initialized AutonomousPipeline

    Example:
        pipeline = await create_autonomous_pipeline(
            airsim_bridge=bridge,
            geo_ref=geo_ref,
            mission_config_path="configs/mission_config.yaml"
        )
        await pipeline.start()
    """
    pipeline = AutonomousPipeline(config)

    if not await pipeline.initialize(airsim_bridge, geo_ref):
        raise RuntimeError("Failed to initialize pipeline")

    if mission_config_path:
        if not await pipeline.load_mission(mission_config_path):
            raise RuntimeError(f"Failed to load mission: {mission_config_path}")

    return pipeline
