"""AegisAV Autonomy Package.

Platform-agnostic flight interface supporting MAVLink-based autopilots.
"""

# Core types that don't require external dependencies
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSInfo,
    GPSState,
    Position,
    VehicleHealth,
    VehicleState,
    Velocity,
)

__all__ = [
    "Attitude",
    "BatteryState",
    "FlightMode",
    "GPSInfo",
    "GPSState",
    # Vehicle state
    "Position",
    "VehicleHealth",
    "VehicleState",
    "Velocity",
]

# Optional imports that require pymavlink
try:
    from autonomy.mavlink_interface import (  # noqa: F401
        ConnectionState,
        MAVLinkConfig,
        MAVLinkInterface,
    )
    from autonomy.mission_primitives import (  # noqa: F401
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
