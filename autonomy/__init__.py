"""
AegisAV Autonomy Package

Platform-agnostic flight interface supporting MAVLink-based autopilots.
"""

# Core types that don't require external dependencies
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSState,
    GPSInfo,
    Position,
    VehicleHealth,
    VehicleState,
    Velocity,
)

__all__ = [
    # Vehicle state
    "Position",
    "Velocity",
    "Attitude",
    "BatteryState",
    "FlightMode",
    "GPSState",
    "GPSInfo",
    "VehicleHealth",
    "VehicleState",
]

# Optional imports that require pymavlink
try:
    from autonomy.mavlink_interface import (
        ConnectionState,
        MAVLinkConfig,
        MAVLinkInterface,
    )
    from autonomy.mission_primitives import (
        MissionPrimitives,
        PrimitiveConfig,
        PrimitiveResult,
    )
    __all__.extend([
        "ConnectionState",
        "MAVLinkConfig",
        "MAVLinkInterface",
        "MissionPrimitives",
        "PrimitiveConfig",
        "PrimitiveResult",
    ])
except ImportError:
    # pymavlink not installed - MAVLink functionality not available
    pass
