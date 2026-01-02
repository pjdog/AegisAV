"""Multi-Vehicle Manager for AirSim.

Manages multiple AirSim vehicles (drones) and provides:
- Vehicle discovery and registration
- Scenario drone_id to AirSim vehicle_name mapping
- Independent control of each vehicle
- Coordinated fleet operations

This bridges the gap between scenario-defined logical drones (alpha, bravo, etc.)
and AirSim's vehicle instances (Drone1, Drone2, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from concurrent.futures import ThreadPoolExecutor

try:
    import cosysairsim as airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    airsim = None

logger = logging.getLogger(__name__)


# Shared executor for all AirSim operations (thread-safety)
_multi_vehicle_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airsim_mv")
_airsim_client: Any = None
_client_lock = threading.Lock()


class VehicleState(str, Enum):
    """State of a managed vehicle."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"  # Discovered, not assigned
    ASSIGNED = "assigned"    # Assigned to a scenario drone
    ACTIVE = "active"        # Currently flying/operating
    RETURNING = "returning"  # Returning to dock
    LANDED = "landed"        # On ground
    ERROR = "error"          # Error state


@dataclass
class ManagedVehicle:
    """A vehicle managed by the MultiVehicleManager."""

    # AirSim identity
    airsim_name: str  # The name in AirSim (e.g., "Drone1", "Drone2")

    # Scenario mapping
    scenario_drone_id: str | None = None  # The scenario drone_id if assigned

    # Current state
    state: VehicleState = VehicleState.AVAILABLE

    # Position (NED coordinates, meters)
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0

    # Orientation (radians)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Status
    battery_percent: float = 100.0
    is_armed: bool = False
    in_air: bool = False
    api_control_enabled: bool = False

    # Timestamps
    last_update: datetime = field(default_factory=datetime.now)
    assigned_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "airsim_name": self.airsim_name,
            "scenario_drone_id": self.scenario_drone_id,
            "state": self.state.value,
            "position": {
                "x": self.position_x,
                "y": self.position_y,
                "z": self.position_z,
            },
            "orientation": {
                "yaw": math.degrees(self.yaw),
                "pitch": math.degrees(self.pitch),
                "roll": math.degrees(self.roll),
            },
            "battery_percent": self.battery_percent,
            "is_armed": self.is_armed,
            "in_air": self.in_air,
            "api_control_enabled": self.api_control_enabled,
            "last_update": self.last_update.isoformat(),
        }


class MultiVehicleManager:
    """Manages multiple AirSim vehicles for multi-drone scenarios.

    This class:
    1. Discovers available vehicles in AirSim
    2. Maps scenario drone_ids to AirSim vehicle names
    3. Provides vehicle-specific control methods
    4. Tracks state of all vehicles

    Example:
        manager = MultiVehicleManager()
        await manager.connect()

        # Discover vehicles
        vehicles = await manager.discover_vehicles()

        # Assign scenario drones
        manager.assign_vehicle("alpha", "Drone1")
        manager.assign_vehicle("bravo", "Drone2")

        # Control specific vehicle
        await manager.takeoff("alpha", altitude=10.0)
        await manager.move_to("bravo", x=50, y=0, z=-15)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        vehicle_mapping: dict[str, str] | None = None,
        auto_assign: bool = True,
    ) -> None:
        """Initialize the multi-vehicle manager.

        Args:
            host: AirSim host address
            vehicle_mapping: Pre-defined mapping of scenario drone_id -> AirSim vehicle name
            auto_assign: If True, automatically assign unmapped scenario drones to available vehicles
        """
        self.host = host
        self.connected = False
        self.auto_assign = auto_assign

        # Vehicle storage
        self._vehicles: dict[str, ManagedVehicle] = {}  # airsim_name -> ManagedVehicle

        # Mapping: scenario_drone_id -> airsim_name
        self._drone_to_vehicle: dict[str, str] = vehicle_mapping.copy() if vehicle_mapping else {}

        # Reverse mapping: airsim_name -> scenario_drone_id
        self._vehicle_to_drone: dict[str, str] = {}

        # Update reverse mapping from initial mapping
        for drone_id, vehicle_name in self._drone_to_vehicle.items():
            self._vehicle_to_drone[vehicle_name] = drone_id

        # Callbacks
        self._on_vehicle_state_change: list[Callable[[ManagedVehicle], Awaitable[None]]] = []

        logger.info(f"MultiVehicleManager initialized (host={host}, auto_assign={auto_assign})")

    async def connect(self) -> bool:
        """Connect to AirSim and discover vehicles.

        Returns:
            True if connected and at least one vehicle discovered.
        """
        global _airsim_client

        if not AIRSIM_AVAILABLE:
            logger.error("AirSim not available (cosysairsim not installed)")
            return False

        try:
            loop = asyncio.get_running_loop()

            def _connect():
                global _airsim_client
                with _client_lock:
                    client = airsim.MultirotorClient(ip=self.host)
                    client.confirmConnection()
                    _airsim_client = client
                    return True

            await loop.run_in_executor(_multi_vehicle_executor, _connect)
            self.connected = True
            logger.info(f"Connected to AirSim at {self.host}")

            # Discover vehicles
            vehicles = await self.discover_vehicles()

            if not vehicles:
                logger.warning("Connected but no vehicles found")
                return True  # Still connected, just no vehicles

            return True

        except Exception as e:
            logger.error(f"Failed to connect to AirSim: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from AirSim."""
        global _airsim_client

        try:
            if self.connected:
                loop = asyncio.get_running_loop()

                def _disconnect():
                    global _airsim_client
                    with _client_lock:
                        if _airsim_client is not None:
                            # Disable API control on all vehicles
                            for vehicle in self._vehicles.values():
                                if vehicle.api_control_enabled:
                                    try:
                                        _airsim_client.enableApiControl(False, vehicle.airsim_name)
                                    except Exception:
                                        pass
                            _airsim_client = None

                await loop.run_in_executor(_multi_vehicle_executor, _disconnect)

            self.connected = False
            self._vehicles.clear()
            logger.info("Disconnected from AirSim")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def discover_vehicles(self) -> list[str]:
        """Discover all available vehicles in AirSim.

        Returns:
            List of vehicle names discovered.
        """
        if not self.connected:
            logger.warning("Not connected - cannot discover vehicles")
            return []

        try:
            loop = asyncio.get_running_loop()

            def _list_vehicles():
                global _airsim_client
                with _client_lock:
                    if _airsim_client is None:
                        return []
                    try:
                        return _airsim_client.listVehicles()
                    except Exception as e:
                        logger.error(f"listVehicles failed: {e}")
                        return []

            vehicle_names = await loop.run_in_executor(_multi_vehicle_executor, _list_vehicles)

            # Create/update ManagedVehicle for each discovered vehicle
            for name in vehicle_names:
                if name not in self._vehicles:
                    self._vehicles[name] = ManagedVehicle(airsim_name=name)
                    logger.info(f"Discovered vehicle: {name}")

                    # Check if this vehicle was pre-mapped
                    if name in self._vehicle_to_drone:
                        drone_id = self._vehicle_to_drone[name]
                        self._vehicles[name].scenario_drone_id = drone_id
                        self._vehicles[name].state = VehicleState.ASSIGNED
                        logger.info(f"  -> Pre-assigned to scenario drone: {drone_id}")

            # Update state of all vehicles
            await self._update_all_vehicle_states()

            return vehicle_names

        except Exception as e:
            logger.error(f"Failed to discover vehicles: {e}")
            return []

    async def _update_all_vehicle_states(self) -> None:
        """Update state of all managed vehicles."""
        for vehicle in self._vehicles.values():
            await self._update_vehicle_state(vehicle)

    async def _update_vehicle_state(self, vehicle: ManagedVehicle) -> None:
        """Update state from AirSim for a single vehicle."""
        try:
            loop = asyncio.get_running_loop()

            def _get_state():
                global _airsim_client
                with _client_lock:
                    if _airsim_client is None:
                        return None
                    try:
                        return _airsim_client.getMultirotorState(vehicle_name=vehicle.airsim_name)
                    except Exception as e:
                        logger.debug(f"getMultirotorState failed for {vehicle.airsim_name}: {e}")
                        return None

            state = await loop.run_in_executor(_multi_vehicle_executor, _get_state)

            if state is not None:
                # Update position
                vehicle.position_x = state.kinematics_estimated.position.x_val
                vehicle.position_y = state.kinematics_estimated.position.y_val
                vehicle.position_z = state.kinematics_estimated.position.z_val

                # Update orientation (quaternion to euler)
                q = state.kinematics_estimated.orientation
                vehicle.yaw, vehicle.pitch, vehicle.roll = self._quaternion_to_euler(
                    q.w_val, q.x_val, q.y_val, q.z_val
                )

                # Update status
                vehicle.in_air = state.landed_state != airsim.LandedState.Landed

                vehicle.last_update = datetime.now()

        except Exception as e:
            logger.debug(f"Failed to update state for {vehicle.airsim_name}: {e}")

    @staticmethod
    def _quaternion_to_euler(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Convert quaternion to euler angles (yaw, pitch, roll) in radians."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw, pitch, roll

    def assign_vehicle(self, scenario_drone_id: str, airsim_name: str) -> bool:
        """Assign a scenario drone to an AirSim vehicle.

        Args:
            scenario_drone_id: The drone_id from the scenario (e.g., "alpha")
            airsim_name: The AirSim vehicle name (e.g., "Drone1")

        Returns:
            True if assignment successful.
        """
        if airsim_name not in self._vehicles:
            logger.warning(f"Vehicle {airsim_name} not found - cannot assign")
            return False

        # Check if already assigned
        vehicle = self._vehicles[airsim_name]
        if vehicle.scenario_drone_id is not None and vehicle.scenario_drone_id != scenario_drone_id:
            logger.warning(
                f"Vehicle {airsim_name} already assigned to {vehicle.scenario_drone_id}"
            )
            return False

        # Update mappings
        self._drone_to_vehicle[scenario_drone_id] = airsim_name
        self._vehicle_to_drone[airsim_name] = scenario_drone_id

        # Update vehicle
        vehicle.scenario_drone_id = scenario_drone_id
        vehicle.state = VehicleState.ASSIGNED
        vehicle.assigned_at = datetime.now()

        logger.info(f"Assigned scenario drone '{scenario_drone_id}' to AirSim vehicle '{airsim_name}'")
        return True

    def auto_assign_scenario_drones(self, drone_ids: list[str]) -> dict[str, str]:
        """Auto-assign scenario drone_ids to available AirSim vehicles.

        Assigns in order: first drone_id to first available vehicle, etc.

        Args:
            drone_ids: List of scenario drone_ids to assign

        Returns:
            Dict of successful assignments: drone_id -> airsim_name
        """
        assignments = {}

        # Get available vehicles (not yet assigned)
        available = [
            v for v in self._vehicles.values()
            if v.scenario_drone_id is None and v.state == VehicleState.AVAILABLE
        ]

        for drone_id in drone_ids:
            # Skip if already mapped
            if drone_id in self._drone_to_vehicle:
                assignments[drone_id] = self._drone_to_vehicle[drone_id]
                continue

            # Find next available vehicle
            if not available:
                logger.warning(f"No more vehicles available for drone '{drone_id}'")
                break

            vehicle = available.pop(0)
            if self.assign_vehicle(drone_id, vehicle.airsim_name):
                assignments[drone_id] = vehicle.airsim_name

        return assignments

    def get_vehicle_name(self, scenario_drone_id: str) -> str | None:
        """Get the AirSim vehicle name for a scenario drone_id.

        Returns:
            The vehicle name, or None if not assigned.
        """
        return self._drone_to_vehicle.get(scenario_drone_id)

    def get_drone_id(self, airsim_name: str) -> str | None:
        """Get the scenario drone_id for an AirSim vehicle.

        Returns:
            The drone_id, or None if not assigned.
        """
        return self._vehicle_to_drone.get(airsim_name)

    def get_vehicle(self, identifier: str) -> ManagedVehicle | None:
        """Get a managed vehicle by either drone_id or airsim_name.

        Args:
            identifier: Either a scenario drone_id or AirSim vehicle name

        Returns:
            The ManagedVehicle, or None if not found.
        """
        # Try as airsim_name first
        if identifier in self._vehicles:
            return self._vehicles[identifier]

        # Try as drone_id
        vehicle_name = self._drone_to_vehicle.get(identifier)
        if vehicle_name:
            return self._vehicles.get(vehicle_name)

        return None

    def get_all_vehicles(self) -> list[ManagedVehicle]:
        """Get all managed vehicles."""
        return list(self._vehicles.values())

    def get_assigned_vehicles(self) -> list[ManagedVehicle]:
        """Get all vehicles assigned to scenario drones."""
        return [v for v in self._vehicles.values() if v.scenario_drone_id is not None]

    # =========================================================================
    # Vehicle Control Methods
    # =========================================================================

    async def enable_api_control(self, identifier: str) -> bool:
        """Enable API control for a vehicle.

        Args:
            identifier: Scenario drone_id or AirSim vehicle name
        """
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            logger.warning(f"Vehicle not found: {identifier}")
            return False

        try:
            loop = asyncio.get_running_loop()

            def _enable():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        _airsim_client.enableApiControl(True, vehicle.airsim_name)
                        return True
                    return False

            result = await loop.run_in_executor(_multi_vehicle_executor, _enable)
            if result:
                vehicle.api_control_enabled = True
                logger.debug(f"Enabled API control for {vehicle.airsim_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to enable API control for {identifier}: {e}")
            return False

    async def arm(self, identifier: str) -> bool:
        """Arm a vehicle.

        Args:
            identifier: Scenario drone_id or AirSim vehicle name
        """
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return False

        try:
            loop = asyncio.get_running_loop()

            def _arm():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        _airsim_client.armDisarm(True, vehicle.airsim_name)
                        return True
                    return False

            result = await loop.run_in_executor(_multi_vehicle_executor, _arm)
            if result:
                vehicle.is_armed = True
            return result

        except Exception as e:
            logger.error(f"Failed to arm {identifier}: {e}")
            return False

    async def disarm(self, identifier: str) -> bool:
        """Disarm a vehicle."""
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return False

        try:
            loop = asyncio.get_running_loop()

            def _disarm():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        _airsim_client.armDisarm(False, vehicle.airsim_name)
                        return True
                    return False

            result = await loop.run_in_executor(_multi_vehicle_executor, _disarm)
            if result:
                vehicle.is_armed = False
            return result

        except Exception as e:
            logger.error(f"Failed to disarm {identifier}: {e}")
            return False

    async def takeoff(self, identifier: str, altitude: float = 10.0, timeout: float = 10.0) -> bool:
        """Take off a vehicle.

        Args:
            identifier: Scenario drone_id or AirSim vehicle name
            altitude: Target altitude in meters (positive = up)
            timeout: Timeout in seconds
        """
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return False

        # Ensure API control and armed
        await self.enable_api_control(identifier)
        await self.arm(identifier)

        try:
            loop = asyncio.get_running_loop()

            def _takeoff():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        future = _airsim_client.takeoffAsync(
                            timeout_sec=timeout,
                            vehicle_name=vehicle.airsim_name
                        )
                        future.join()
                        return True
                    return False

            result = await loop.run_in_executor(_multi_vehicle_executor, _takeoff)
            if result:
                vehicle.in_air = True
                vehicle.state = VehicleState.ACTIVE
            return result

        except Exception as e:
            logger.error(f"Failed to takeoff {identifier}: {e}")
            return False

    async def land(self, identifier: str, timeout: float = 30.0) -> bool:
        """Land a vehicle.

        Args:
            identifier: Scenario drone_id or AirSim vehicle name
            timeout: Timeout in seconds
        """
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return False

        try:
            loop = asyncio.get_running_loop()

            def _land():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        future = _airsim_client.landAsync(
                            timeout_sec=timeout,
                            vehicle_name=vehicle.airsim_name
                        )
                        future.join()
                        return True
                    return False

            result = await loop.run_in_executor(_multi_vehicle_executor, _land)
            if result:
                vehicle.in_air = False
                vehicle.state = VehicleState.LANDED
            return result

        except Exception as e:
            logger.error(f"Failed to land {identifier}: {e}")
            return False

    async def hover(self, identifier: str) -> bool:
        """Hover in place."""
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return False

        try:
            loop = asyncio.get_running_loop()

            def _hover():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        future = _airsim_client.hoverAsync(vehicle_name=vehicle.airsim_name)
                        future.join()
                        return True
                    return False

            return await loop.run_in_executor(_multi_vehicle_executor, _hover)

        except Exception as e:
            logger.error(f"Failed to hover {identifier}: {e}")
            return False

    async def move_to_position(
        self,
        identifier: str,
        x: float,
        y: float,
        z: float,
        velocity: float = 5.0,
        timeout: float = 60.0,
    ) -> bool:
        """Move vehicle to a position (NED coordinates).

        Args:
            identifier: Scenario drone_id or AirSim vehicle name
            x, y, z: Target position in NED (z is negative for altitude)
            velocity: Flight velocity in m/s
            timeout: Timeout in seconds
        """
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return False

        try:
            loop = asyncio.get_running_loop()

            def _move():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        future = _airsim_client.moveToPositionAsync(
                            x, y, z, velocity,
                            timeout_sec=timeout,
                            vehicle_name=vehicle.airsim_name
                        )
                        future.join()
                        return True
                    return False

            result = await loop.run_in_executor(_multi_vehicle_executor, _move)

            # Update position
            if result:
                vehicle.position_x = x
                vehicle.position_y = y
                vehicle.position_z = z

            return result

        except Exception as e:
            logger.error(f"Failed to move {identifier} to ({x}, {y}, {z}): {e}")
            return False

    async def set_pose(
        self,
        identifier: str,
        x: float,
        y: float,
        z: float,
        yaw_deg: float = 0.0,
        ignore_collision: bool = True,
    ) -> bool:
        """Teleport vehicle to a pose (position + yaw).

        Args:
            identifier: Scenario drone_id or AirSim vehicle name
            x, y, z: Position in NED coordinates
            yaw_deg: Yaw angle in degrees
            ignore_collision: If True, ignore collisions during teleport
        """
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return False

        try:
            loop = asyncio.get_running_loop()

            def _set_pose():
                global _airsim_client
                with _client_lock:
                    if _airsim_client:
                        # Create pose
                        pose = airsim.Pose(
                            airsim.Vector3r(x, y, z),
                            airsim.to_quaternion(0, 0, math.radians(yaw_deg))
                        )
                        _airsim_client.simSetVehiclePose(
                            pose, ignore_collision, vehicle.airsim_name
                        )
                        return True
                    return False

            result = await loop.run_in_executor(_multi_vehicle_executor, _set_pose)

            if result:
                vehicle.position_x = x
                vehicle.position_y = y
                vehicle.position_z = z
                vehicle.yaw = math.radians(yaw_deg)

            return result

        except Exception as e:
            logger.error(f"Failed to set pose for {identifier}: {e}")
            return False

    async def get_state(self, identifier: str) -> dict[str, Any] | None:
        """Get current state of a vehicle.

        Returns:
            Dict with position, orientation, velocity, etc.
        """
        vehicle = self.get_vehicle(identifier)
        if not vehicle:
            return None

        await self._update_vehicle_state(vehicle)
        return vehicle.to_dict()

    # =========================================================================
    # Fleet Operations
    # =========================================================================

    async def takeoff_all(self, altitude: float = 10.0) -> dict[str, bool]:
        """Take off all assigned vehicles.

        Returns:
            Dict of drone_id -> success status
        """
        results = {}
        tasks = []

        for vehicle in self.get_assigned_vehicles():
            if vehicle.scenario_drone_id:
                tasks.append(
                    (vehicle.scenario_drone_id, self.takeoff(vehicle.scenario_drone_id, altitude))
                )

        for drone_id, task in tasks:
            results[drone_id] = await task

        return results

    async def land_all(self) -> dict[str, bool]:
        """Land all assigned vehicles."""
        results = {}

        for vehicle in self.get_assigned_vehicles():
            if vehicle.scenario_drone_id:
                results[vehicle.scenario_drone_id] = await self.land(vehicle.scenario_drone_id)

        return results

    async def update_all_states(self) -> dict[str, dict[str, Any]]:
        """Update and return state of all vehicles.

        Returns:
            Dict of airsim_name -> state dict
        """
        await self._update_all_vehicle_states()
        return {v.airsim_name: v.to_dict() for v in self._vehicles.values()}


# Singleton instance
_manager_instance: MultiVehicleManager | None = None


def get_multi_vehicle_manager(
    host: str = "127.0.0.1",
    vehicle_mapping: dict[str, str] | None = None,
) -> MultiVehicleManager:
    """Get or create the global MultiVehicleManager instance."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = MultiVehicleManager(host, vehicle_mapping)

    return _manager_instance


async def reset_multi_vehicle_manager() -> None:
    """Reset the global manager (for testing or reconfiguration)."""
    global _manager_instance

    if _manager_instance is not None:
        await _manager_instance.disconnect()
        _manager_instance = None
