"""AegisAV High-Fidelity Simulation Package

Integrates with:
- AirSim (Unreal Engine) for photorealistic rendering
- ArduPilot SITL for rock-solid flight control

New in this version:
- Flight control integration (takeoff, land, move_to_position, orbit)
- Coordinate conversion utilities (GPS to NED)
- AirSim Action Executor (translates decisions to flight commands)
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
]
