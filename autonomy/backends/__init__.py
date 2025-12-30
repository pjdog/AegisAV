"""Flight Backend Implementations.

This package contains concrete implementations of the FlightBackend protocol
for different flight platforms:

- AirSimBackend: For AirSim/Unreal simulation
- ArduPilotBackend: For real ArduPilot hardware via MAVLink
- MockBackend: For testing without hardware/simulation
"""

from autonomy.flight_backend import (
    AirSimBackendConfig,
    ArduPilotBackendConfig,
    BackendType,
    ConnectionStatus,
    FlightBackend,
    FlightBackendBase,
    FlightBackendConfig,
    WaypointGPS,
    WaypointNED,
    create_backend,
)

__all__ = [
    "AirSimBackendConfig",
    "ArduPilotBackendConfig",
    "BackendType",
    "ConnectionStatus",
    "FlightBackend",
    "FlightBackendBase",
    "FlightBackendConfig",
    "WaypointGPS",
    "WaypointNED",
    "create_backend",
]
