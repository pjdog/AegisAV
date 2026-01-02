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
from simulation.sitl_manager import SITLManager

# Coordinate utilities (always available)
from simulation.coordinate_utils import GeoReference, haversine_distance, initial_bearing

# AirSim flight control (requires airsim package)
try:
    from simulation.realtime_bridge import (
        RealtimeAirSimBridge,
        RealtimeBridgeConfig,
        TelemetryBroadcaster,
        TelemetryFrame,
        create_multi_vehicle_bridges,
        connect_all_bridges,
        disconnect_all_bridges,
    )
    from simulation.airsim_action_executor import (
        AirSimActionExecutor,
        ExecutionResult,
        ExecutionStatus,
        FlightConfig,
    )
    AIRSIM_FLIGHT_AVAILABLE = True
except ImportError:
    AIRSIM_FLIGHT_AVAILABLE = False

# Multi-vehicle management
try:
    from simulation.multi_vehicle_manager import (
        MultiVehicleManager,
        ManagedVehicle,
        VehicleState,
        get_multi_vehicle_manager,
        reset_multi_vehicle_manager,
    )
    from simulation.drone_coordinator import (
        DroneCoordinator,
        DroneAssignment,
        CoordinatorState,
        get_drone_coordinator,
        reset_drone_coordinator,
    )
    MULTI_VEHICLE_AVAILABLE = True
except ImportError:
    MULTI_VEHICLE_AVAILABLE = False

# Asset management
try:
    from simulation.asset_manager import (
        AssetManager,
        AssetMetadata,
        AssetType,
        AssetFormat,
        AssetLicense,
        get_asset_manager,
    )
    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    ASSET_MANAGER_AVAILABLE = False

__all__ = [
    # Base bridge
    "AirSimBridge",
    "AirSimCameraConfig",
    "SITLManager",
    # Coordinate utilities
    "GeoReference",
    "haversine_distance",
    "initial_bearing",
    # Flight control (conditionally available)
    "RealtimeAirSimBridge",
    "RealtimeBridgeConfig",
    "TelemetryBroadcaster",
    "TelemetryFrame",
    "AirSimActionExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "FlightConfig",
    "AIRSIM_FLIGHT_AVAILABLE",
    # Multi-vehicle bridge helpers
    "create_multi_vehicle_bridges",
    "connect_all_bridges",
    "disconnect_all_bridges",
    # Multi-vehicle management
    "MultiVehicleManager",
    "ManagedVehicle",
    "VehicleState",
    "get_multi_vehicle_manager",
    "reset_multi_vehicle_manager",
    "MULTI_VEHICLE_AVAILABLE",
    # Drone coordination
    "DroneCoordinator",
    "DroneAssignment",
    "CoordinatorState",
    "get_drone_coordinator",
    "reset_drone_coordinator",
    # Asset management
    "AssetManager",
    "AssetMetadata",
    "AssetType",
    "AssetFormat",
    "AssetLicense",
    "get_asset_manager",
    "ASSET_MANAGER_AVAILABLE",
]
