"""AegisAV High-Fidelity Simulation Package

Integrates with:
- AirSim (Unreal Engine) for photorealistic rendering
- ArduPilot SITL for rock-solid flight control

Features:
- Flight control integration (takeoff, land, move_to_position, orbit)
- Coordinate conversion utilities (GPS to NED)
- AirSim Action Executor (translates decisions to flight commands)
- Multi-vehicle management for multi-drone scenarios
- Drone coordinator for scenario-AirSim synchronization
- 3D asset management for realistic environment objects
"""

from simulation.airsim_bridge import AirSimBridge, AirSimCameraConfig

# Coordinate utilities (always available)
from simulation.coordinate_utils import GeoReference, haversine_distance, initial_bearing
from simulation.sitl_manager import SITLManager

# AirSim flight control (requires airsim package)
try:
    from simulation.airsim_action_executor import (
        AirSimActionExecutor,
        ExecutionResult,
        ExecutionStatus,
        FlightConfig,
    )
    from simulation.realtime_bridge import (
        RealtimeAirSimBridge,
        RealtimeBridgeConfig,
        TelemetryBroadcaster,
        TelemetryFrame,
        connect_all_bridges,
        create_multi_vehicle_bridges,
        disconnect_all_bridges,
    )

    AIRSIM_FLIGHT_AVAILABLE = True
except ImportError:
    AIRSIM_FLIGHT_AVAILABLE = False

# Multi-vehicle management
try:
    from simulation.drone_coordinator import (
        CoordinatorState,
        DroneAssignment,
        DroneCoordinator,
        get_drone_coordinator,
        reset_drone_coordinator,
    )
    from simulation.multi_vehicle_manager import (
        ManagedVehicle,
        MultiVehicleManager,
        VehicleState,
        get_multi_vehicle_manager,
        reset_multi_vehicle_manager,
    )

    MULTI_VEHICLE_AVAILABLE = True
except ImportError:
    MULTI_VEHICLE_AVAILABLE = False

# Asset management
try:
    from simulation.asset_manager import (
        AssetFormat,
        AssetLicense,
        AssetManager,
        AssetMetadata,
        AssetType,
        get_asset_manager,
    )

    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    ASSET_MANAGER_AVAILABLE = False

__all__ = [
    "AIRSIM_FLIGHT_AVAILABLE",
    "ASSET_MANAGER_AVAILABLE",
    "MULTI_VEHICLE_AVAILABLE",
    "AirSimActionExecutor",
    # Base bridge
    "AirSimBridge",
    "AirSimCameraConfig",
    "AssetFormat",
    "AssetLicense",
    # Asset management
    "AssetManager",
    "AssetMetadata",
    "AssetType",
    "CoordinatorState",
    "DroneAssignment",
    # Drone coordination
    "DroneCoordinator",
    "ExecutionResult",
    "ExecutionStatus",
    "FlightConfig",
    # Coordinate utilities
    "GeoReference",
    "ManagedVehicle",
    # Multi-vehicle management
    "MultiVehicleManager",
    # Flight control (conditionally available)
    "RealtimeAirSimBridge",
    "RealtimeBridgeConfig",
    "SITLManager",
    "TelemetryBroadcaster",
    "TelemetryFrame",
    "VehicleState",
    "connect_all_bridges",
    # Multi-vehicle bridge helpers
    "create_multi_vehicle_bridges",
    "disconnect_all_bridges",
    "get_asset_manager",
    "get_drone_coordinator",
    "get_multi_vehicle_manager",
    "haversine_distance",
    "initial_bearing",
    "reset_drone_coordinator",
    "reset_multi_vehicle_manager",
]
