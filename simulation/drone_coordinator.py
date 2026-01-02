"""Drone Coordinator for Scenario-AirSim Synchronization.

Bridges the gap between:
- Scenario-defined logical drones (with drone_id, positions, states)
- AirSim physical vehicles (with vehicle_name, 3D positions)

The coordinator:
1. Loads a scenario and extracts drone definitions
2. Maps scenario drones to available AirSim vehicles
3. Positions drones at their scenario-defined locations
4. Syncs state updates between scenario engine and AirSim
5. Handles multi-drone coordination (spacing, sequencing)
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

# Scenario types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.server.scenarios import (
    Scenario,
    SimulatedDrone,
    DroneState,
    DOCK_LATITUDE,
    DOCK_LONGITUDE,
)

from simulation.multi_vehicle_manager import (
    MultiVehicleManager,
    ManagedVehicle,
    VehicleState,
    get_multi_vehicle_manager,
)

logger = logging.getLogger(__name__)


# Geographic constants for coordinate conversion
METERS_PER_DEGREE_LAT = 111_320  # Approximate at equator
METERS_PER_DEGREE_LON = 85_390   # Approximate at 40° latitude


class CoordinatorState(str, Enum):
    """State of the drone coordinator."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DroneAssignment:
    """Assignment of a scenario drone to an AirSim vehicle."""

    scenario_drone: SimulatedDrone
    airsim_vehicle_name: str
    managed_vehicle: ManagedVehicle | None = None

    # NED coordinates derived from scenario lat/lon
    ned_x: float = 0.0  # North (meters from origin)
    ned_y: float = 0.0  # East (meters from origin)
    ned_z: float = 0.0  # Down (negative = altitude)

    # Sync state
    last_sync: datetime | None = None
    sync_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "drone_id": self.scenario_drone.drone_id,
            "drone_name": self.scenario_drone.name,
            "airsim_vehicle": self.airsim_vehicle_name,
            "scenario_state": self.scenario_drone.state.value,
            "position_ned": {
                "x": self.ned_x,
                "y": self.ned_y,
                "z": self.ned_z,
            },
            "position_geo": {
                "latitude": self.scenario_drone.latitude,
                "longitude": self.scenario_drone.longitude,
                "altitude_agl": self.scenario_drone.altitude_agl,
            },
            "battery_percent": self.scenario_drone.battery_percent,
            "in_air": self.scenario_drone.in_air,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
        }


class DroneCoordinator:
    """Coordinates scenario drones with AirSim vehicles.

    Example:
        coordinator = DroneCoordinator()
        await coordinator.connect()

        # Load and prepare scenario
        success = await coordinator.load_scenario(my_scenario)

        # Position drones at scenario start positions
        await coordinator.setup_initial_positions()

        # Start syncing drone positions from scenario to AirSim
        await coordinator.start_sync()

        # Update a drone position (from scenario engine)
        coordinator.update_drone_position("alpha", lat=37.77, lon=-122.42, alt=30.0)

        # Cleanup
        await coordinator.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        vehicle_mapping: dict[str, str] | None = None,
        origin_lat: float = DOCK_LATITUDE,
        origin_lon: float = DOCK_LONGITUDE,
        sync_interval_ms: float = 100.0,
    ) -> None:
        """Initialize the drone coordinator.

        Args:
            host: AirSim host address
            vehicle_mapping: Pre-defined scenario drone_id -> AirSim vehicle mapping
            origin_lat: Latitude origin for NED coordinate conversion
            origin_lon: Longitude origin for NED coordinate conversion
            sync_interval_ms: How often to sync state (milliseconds)
        """
        self.host = host
        self.vehicle_mapping = vehicle_mapping or {}
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.sync_interval_ms = sync_interval_ms

        self.state = CoordinatorState.IDLE
        self.current_scenario: Scenario | None = None

        # Vehicle manager
        self._vehicle_manager: MultiVehicleManager | None = None

        # Drone assignments
        self._assignments: dict[str, DroneAssignment] = {}  # drone_id -> assignment

        # Sync task
        self._sync_task: asyncio.Task | None = None
        self._sync_running = False

        # Callbacks
        self._on_drone_update: list[Callable[[DroneAssignment], Awaitable[None]]] = []
        self._on_state_change: list[Callable[[CoordinatorState], Awaitable[None]]] = []

        logger.info(f"DroneCoordinator initialized (origin: {origin_lat}, {origin_lon})")

    async def connect(self) -> bool:
        """Connect to AirSim via the vehicle manager.

        Returns:
            True if connected successfully.
        """
        try:
            self._vehicle_manager = get_multi_vehicle_manager(
                host=self.host,
                vehicle_mapping=self.vehicle_mapping,
            )

            success = await self._vehicle_manager.connect()
            if success:
                logger.info("DroneCoordinator connected to AirSim")
                self.state = CoordinatorState.IDLE
            else:
                logger.error("DroneCoordinator failed to connect")
                self.state = CoordinatorState.ERROR

            return success

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.state = CoordinatorState.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        await self.stop_sync()

        if self._vehicle_manager:
            await self._vehicle_manager.disconnect()
            self._vehicle_manager = None

        self._assignments.clear()
        self.current_scenario = None
        self.state = CoordinatorState.IDLE

        logger.info("DroneCoordinator disconnected")

    async def load_scenario(self, scenario: Scenario) -> bool:
        """Load a scenario and assign drones to AirSim vehicles.

        This:
        1. Extracts drones from the scenario
        2. Discovers available AirSim vehicles
        3. Creates assignments (scenario drone -> AirSim vehicle)
        4. Computes initial NED positions from lat/lon

        Args:
            scenario: The scenario to load

        Returns:
            True if at least one drone was assigned.
        """
        self.state = CoordinatorState.LOADING
        self._assignments.clear()
        self.current_scenario = scenario

        logger.info(f"Loading scenario: {scenario.name} ({len(scenario.drones)} drones)")

        if not self._vehicle_manager:
            logger.error("Not connected - cannot load scenario")
            self.state = CoordinatorState.ERROR
            return False

        # Discover vehicles
        available_vehicles = await self._vehicle_manager.discover_vehicles()
        logger.info(f"Available AirSim vehicles: {available_vehicles}")

        if not available_vehicles:
            logger.error("No AirSim vehicles available")
            self.state = CoordinatorState.ERROR
            return False

        # Get drone IDs from scenario
        drone_ids = [d.drone_id for d in scenario.drones]

        # Auto-assign drones to vehicles
        assignments = self._vehicle_manager.auto_assign_scenario_drones(drone_ids)
        logger.info(f"Drone assignments: {assignments}")

        # Create DroneAssignment objects
        for drone in scenario.drones:
            vehicle_name = assignments.get(drone.drone_id)
            if not vehicle_name:
                logger.warning(f"No vehicle for drone {drone.drone_id} - skipping")
                continue

            # Get the managed vehicle
            managed = self._vehicle_manager.get_vehicle(vehicle_name)

            # Compute NED coordinates from lat/lon
            ned_x, ned_y = self._geo_to_ned(drone.latitude, drone.longitude)
            ned_z = -drone.altitude_agl  # NED: negative Z is up

            assignment = DroneAssignment(
                scenario_drone=drone,
                airsim_vehicle_name=vehicle_name,
                managed_vehicle=managed,
                ned_x=ned_x,
                ned_y=ned_y,
                ned_z=ned_z,
            )

            self._assignments[drone.drone_id] = assignment
            logger.info(
                f"  {drone.drone_id} ({drone.name}) -> {vehicle_name} "
                f"@ NED({ned_x:.1f}, {ned_y:.1f}, {ned_z:.1f})"
            )

        if not self._assignments:
            logger.error("No drones could be assigned")
            self.state = CoordinatorState.ERROR
            return False

        self.state = CoordinatorState.READY
        logger.info(f"Scenario loaded: {len(self._assignments)} drones assigned")
        return True

    def _geo_to_ned(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert geographic coordinates to local NED (North-East-Down).

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Tuple of (north_meters, east_meters) from origin
        """
        dlat = lat - self.origin_lat
        dlon = lon - self.origin_lon

        # Approximate conversion (good for small areas)
        north_m = dlat * METERS_PER_DEGREE_LAT
        east_m = dlon * METERS_PER_DEGREE_LON * math.cos(math.radians(self.origin_lat))

        return north_m, east_m

    def _ned_to_geo(self, north_m: float, east_m: float) -> tuple[float, float]:
        """Convert local NED to geographic coordinates.

        Args:
            north_m: North distance in meters from origin
            east_m: East distance in meters from origin

        Returns:
            Tuple of (latitude, longitude)
        """
        lat = self.origin_lat + (north_m / METERS_PER_DEGREE_LAT)
        lon = self.origin_lon + (east_m / (METERS_PER_DEGREE_LON * math.cos(math.radians(self.origin_lat))))

        return lat, lon

    async def setup_initial_positions(self) -> dict[str, bool]:
        """Position all drones at their scenario-defined initial positions.

        Uses teleportation (simSetVehiclePose) to instantly place drones.

        Returns:
            Dict of drone_id -> success status
        """
        if self.state != CoordinatorState.READY:
            logger.warning(f"Cannot setup positions in state: {self.state}")
            return {}

        results = {}

        for drone_id, assignment in self._assignments.items():
            try:
                # Teleport to initial position
                success = await self._vehicle_manager.set_pose(
                    drone_id,
                    x=assignment.ned_x,
                    y=assignment.ned_y,
                    z=assignment.ned_z,
                    yaw_deg=0.0,
                )

                results[drone_id] = success

                if success:
                    logger.info(f"Positioned {drone_id} at NED({assignment.ned_x:.1f}, {assignment.ned_y:.1f}, {assignment.ned_z:.1f})")
                else:
                    logger.warning(f"Failed to position {drone_id}")

            except Exception as e:
                logger.error(f"Error positioning {drone_id}: {e}")
                results[drone_id] = False

        return results

    async def start_sync(self) -> None:
        """Start the position/state sync loop.

        This continuously syncs scenario drone positions to AirSim.
        """
        if self._sync_running:
            logger.warning("Sync already running")
            return

        if self.state != CoordinatorState.READY:
            logger.warning(f"Cannot start sync in state: {self.state}")
            return

        self._sync_running = True
        self.state = CoordinatorState.RUNNING
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Started drone sync loop")

    async def stop_sync(self) -> None:
        """Stop the sync loop."""
        self._sync_running = False

        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        self._sync_task = None

        if self.state == CoordinatorState.RUNNING:
            self.state = CoordinatorState.READY

        logger.info("Stopped drone sync loop")

    async def _sync_loop(self) -> None:
        """Main sync loop - runs at configured interval."""
        interval_s = self.sync_interval_ms / 1000.0

        while self._sync_running:
            try:
                await self._sync_all_drones()
                await asyncio.sleep(interval_s)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(0.5)  # Back off on error

    async def _sync_all_drones(self) -> None:
        """Sync all drone positions to AirSim."""
        for assignment in self._assignments.values():
            try:
                # Only sync if drone is in the air or at non-dock position
                drone = assignment.scenario_drone

                if drone.in_air:
                    # Update AirSim position to match scenario
                    await self._vehicle_manager.move_to_position(
                        drone.drone_id,
                        x=assignment.ned_x,
                        y=assignment.ned_y,
                        z=assignment.ned_z,
                        velocity=2.0,  # Slow for sync
                        timeout=5.0,
                    )

                assignment.last_sync = datetime.now()

                # Fire callbacks
                for callback in self._on_drone_update:
                    try:
                        await callback(assignment)
                    except Exception as e:
                        logger.debug(f"Callback error: {e}")

            except Exception as e:
                assignment.sync_errors += 1
                logger.debug(f"Sync error for {assignment.scenario_drone.drone_id}: {e}")

    # =========================================================================
    # Drone Updates (called by scenario engine)
    # =========================================================================

    def update_drone_position(
        self,
        drone_id: str,
        latitude: float | None = None,
        longitude: float | None = None,
        altitude_agl: float | None = None,
    ) -> bool:
        """Update a drone's position from the scenario engine.

        This updates the local state; the sync loop will push to AirSim.

        Args:
            drone_id: The scenario drone_id
            latitude, longitude, altitude_agl: New position values

        Returns:
            True if drone was found and updated.
        """
        assignment = self._assignments.get(drone_id)
        if not assignment:
            return False

        drone = assignment.scenario_drone

        # Update geo coordinates
        if latitude is not None:
            drone.latitude = latitude
        if longitude is not None:
            drone.longitude = longitude
        if altitude_agl is not None:
            drone.altitude_agl = altitude_agl

        # Recompute NED
        assignment.ned_x, assignment.ned_y = self._geo_to_ned(drone.latitude, drone.longitude)
        assignment.ned_z = -drone.altitude_agl

        return True

    def update_drone_state(
        self,
        drone_id: str,
        state: DroneState | None = None,
        battery_percent: float | None = None,
        in_air: bool | None = None,
        armed: bool | None = None,
    ) -> bool:
        """Update a drone's state from the scenario engine.

        Args:
            drone_id: The scenario drone_id
            state, battery_percent, in_air, armed: New state values

        Returns:
            True if drone was found and updated.
        """
        assignment = self._assignments.get(drone_id)
        if not assignment:
            return False

        drone = assignment.scenario_drone

        if state is not None:
            drone.state = state
        if battery_percent is not None:
            drone.battery_percent = battery_percent
        if in_air is not None:
            drone.in_air = in_air
        if armed is not None:
            drone.armed = armed

        return True

    # =========================================================================
    # Control Commands (from operator/agent)
    # =========================================================================

    async def command_takeoff(self, drone_id: str, altitude: float = 10.0) -> bool:
        """Command a drone to take off.

        Args:
            drone_id: The scenario drone_id
            altitude: Target altitude AGL in meters

        Returns:
            True if command succeeded.
        """
        assignment = self._assignments.get(drone_id)
        if not assignment:
            logger.warning(f"Unknown drone: {drone_id}")
            return False

        success = await self._vehicle_manager.takeoff(drone_id, altitude)

        if success:
            assignment.scenario_drone.in_air = True
            assignment.scenario_drone.armed = True
            assignment.scenario_drone.state = DroneState.TAKEOFF

        return success

    async def command_land(self, drone_id: str) -> bool:
        """Command a drone to land."""
        assignment = self._assignments.get(drone_id)
        if not assignment:
            return False

        success = await self._vehicle_manager.land(drone_id)

        if success:
            assignment.scenario_drone.in_air = False
            assignment.scenario_drone.state = DroneState.LANDING

        return success

    async def command_move_to(
        self,
        drone_id: str,
        latitude: float,
        longitude: float,
        altitude_agl: float,
        velocity: float = 5.0,
    ) -> bool:
        """Command a drone to move to a geographic position.

        Args:
            drone_id: The scenario drone_id
            latitude, longitude: Target geo coordinates
            altitude_agl: Target altitude AGL in meters
            velocity: Flight velocity in m/s

        Returns:
            True if command started successfully.
        """
        assignment = self._assignments.get(drone_id)
        if not assignment:
            return False

        # Convert to NED
        ned_x, ned_y = self._geo_to_ned(latitude, longitude)
        ned_z = -altitude_agl

        # Update assignment
        assignment.ned_x = ned_x
        assignment.ned_y = ned_y
        assignment.ned_z = ned_z
        assignment.scenario_drone.latitude = latitude
        assignment.scenario_drone.longitude = longitude
        assignment.scenario_drone.altitude_agl = altitude_agl

        # Send command
        success = await self._vehicle_manager.move_to_position(
            drone_id, ned_x, ned_y, ned_z, velocity
        )

        if success:
            assignment.scenario_drone.state = DroneState.INSPECTING

        return success

    async def command_return_to_dock(self, drone_id: str, velocity: float = 5.0) -> bool:
        """Command a drone to return to the dock/origin.

        Args:
            drone_id: The scenario drone_id
            velocity: Return flight velocity in m/s

        Returns:
            True if command started successfully.
        """
        assignment = self._assignments.get(drone_id)
        if not assignment:
            return False

        assignment.scenario_drone.state = DroneState.RETURNING

        # Move to origin (dock position)
        success = await self._vehicle_manager.move_to_position(
            drone_id,
            x=0.0,
            y=0.0,
            z=-5.0,  # Hover above dock
            velocity=velocity,
        )

        return success

    # =========================================================================
    # Fleet Commands
    # =========================================================================

    async def takeoff_all(self, altitude: float = 10.0) -> dict[str, bool]:
        """Take off all scenario drones.

        Returns:
            Dict of drone_id -> success status
        """
        results = {}
        for drone_id in self._assignments:
            results[drone_id] = await self.command_takeoff(drone_id, altitude)
        return results

    async def land_all(self) -> dict[str, bool]:
        """Land all scenario drones."""
        results = {}
        for drone_id in self._assignments:
            results[drone_id] = await self.command_land(drone_id)
        return results

    async def return_all_to_dock(self, velocity: float = 5.0) -> dict[str, bool]:
        """Return all scenario drones to dock."""
        results = {}
        for drone_id in self._assignments:
            results[drone_id] = await self.command_return_to_dock(drone_id, velocity)
        return results

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_assignment(self, drone_id: str) -> DroneAssignment | None:
        """Get assignment for a drone."""
        return self._assignments.get(drone_id)

    def get_all_assignments(self) -> list[DroneAssignment]:
        """Get all drone assignments."""
        return list(self._assignments.values())

    def get_drone_ids(self) -> list[str]:
        """Get all assigned drone IDs."""
        return list(self._assignments.keys())

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status summary."""
        return {
            "state": self.state.value,
            "scenario": self.current_scenario.name if self.current_scenario else None,
            "drone_count": len(self._assignments),
            "drones": [a.to_dict() for a in self._assignments.values()],
            "origin": {
                "latitude": self.origin_lat,
                "longitude": self.origin_lon,
            },
        }


# Singleton instance
_coordinator_instance: DroneCoordinator | None = None


def get_drone_coordinator(
    host: str = "127.0.0.1",
    vehicle_mapping: dict[str, str] | None = None,
) -> DroneCoordinator:
    """Get or create the global DroneCoordinator instance."""
    global _coordinator_instance

    if _coordinator_instance is None:
        _coordinator_instance = DroneCoordinator(host, vehicle_mapping)

    return _coordinator_instance


async def reset_drone_coordinator() -> None:
    """Reset the global coordinator (for testing or reconfiguration)."""
    global _coordinator_instance

    if _coordinator_instance is not None:
        await _coordinator_instance.disconnect()
        _coordinator_instance = None
