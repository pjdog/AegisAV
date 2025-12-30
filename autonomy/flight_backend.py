"""Flight Backend Protocol.

Defines the interface that all flight backends (AirSim, ArduPilot, etc.)
must implement. This abstraction allows the FlightController to work with
different platforms using the same API.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from autonomy.vehicle_state import Position, VehicleState


class BackendType(Enum):
    """Supported flight backend types."""

    AIRSIM = "airsim"
    ARDUPILOT = "ardupilot"
    MOCK = "mock"


class ConnectionStatus(Enum):
    """Backend connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class FlightBackendConfig:
    """Base configuration for flight backends."""

    backend_type: BackendType = BackendType.AIRSIM
    connection_timeout_s: float = 30.0
    command_timeout_s: float = 60.0
    telemetry_rate_hz: float = 10.0


@dataclass
class AirSimBackendConfig(FlightBackendConfig):
    """Configuration for AirSim backend."""

    backend_type: BackendType = field(default=BackendType.AIRSIM, init=False)
    host: str = "127.0.0.1"
    vehicle_name: str = "Drone1"
    camera_name: str = "front_center"


@dataclass
class ArduPilotBackendConfig(FlightBackendConfig):
    """Configuration for ArduPilot backend."""

    backend_type: BackendType = field(default=BackendType.ARDUPILOT, init=False)
    connection_string: str = "udp:127.0.0.1:14550"
    source_system: int = 255
    source_component: int = 0


class WaypointNED(BaseModel):
    """A waypoint in NED (North-East-Down) coordinates relative to home."""

    north: float  # meters north of home
    east: float  # meters east of home
    down: float  # meters down from home (negative = up/altitude)

    @property
    def altitude_agl(self) -> float:
        """Altitude above ground (positive up)."""
        return -self.down


class WaypointGPS(BaseModel):
    """A waypoint in GPS coordinates."""

    latitude: float
    longitude: float
    altitude_msl: float  # meters above sea level

    @classmethod
    def from_position(cls, pos: Position) -> WaypointGPS:
        """Create waypoint from Position."""
        return cls(
            latitude=pos.latitude,
            longitude=pos.longitude,
            altitude_msl=pos.altitude_msl,
        )


@runtime_checkable
class FlightBackend(Protocol):
    """Protocol defining the flight backend interface.

    All flight platforms (AirSim, ArduPilot, etc.) must implement this
    interface to work with the FlightController.

    The interface provides:
    - Connection management
    - Basic flight commands (arm, takeoff, land, hover)
    - Position commands in both GPS and NED coordinates
    - State retrieval
    - Property accessors for status

    Example:
        backend: FlightBackend = AirSimBackend(config)
        await backend.connect()
        await backend.arm()
        await backend.takeoff(altitude_agl=10.0)
        await backend.goto_position_gps(lat, lon, alt, velocity=5.0)
        await backend.land()
    """

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> bool:
        """Establish connection to the flight platform.

        Returns:
            True if connection successful, False otherwise
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the flight platform."""
        ...

    # -------------------------------------------------------------------------
    # Basic Flight Commands
    # -------------------------------------------------------------------------

    async def arm(self) -> bool:
        """Arm the vehicle motors.

        Returns:
            True if arming successful
        """
        ...

    async def disarm(self) -> bool:
        """Disarm the vehicle motors.

        Returns:
            True if disarming successful
        """
        ...

    async def takeoff(self, altitude_agl: float, timeout_s: float = 30.0) -> bool:
        """Take off to specified altitude.

        Args:
            altitude_agl: Target altitude in meters above ground level
            timeout_s: Maximum time to wait for takeoff completion

        Returns:
            True if takeoff completed successfully
        """
        ...

    async def land(self, timeout_s: float = 60.0) -> bool:
        """Land at current position.

        Args:
            timeout_s: Maximum time to wait for landing

        Returns:
            True if landing completed successfully
        """
        ...

    async def hover(self) -> bool:
        """Hold current position (loiter/hover in place).

        Returns:
            True if hover command sent successfully
        """
        ...

    async def emergency_stop(self) -> bool:
        """Immediately stop all motors (emergency only).

        Warning: This will cause the vehicle to fall from the sky.

        Returns:
            True if stop command sent
        """
        ...

    # -------------------------------------------------------------------------
    # Position Commands
    # -------------------------------------------------------------------------

    async def goto_position_gps(
        self,
        latitude: float,
        longitude: float,
        altitude_msl: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        """Fly to a GPS position.

        Args:
            latitude: Target latitude in degrees
            longitude: Target longitude in degrees
            altitude_msl: Target altitude in meters above sea level
            velocity: Flight speed in m/s
            timeout_s: Maximum time to reach destination

        Returns:
            True if destination reached
        """
        ...

    async def goto_position_ned(
        self,
        north: float,
        east: float,
        down: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        """Fly to a position in NED coordinates relative to home.

        Args:
            north: Meters north of home position
            east: Meters east of home position
            down: Meters down from home (negative = altitude)
            velocity: Flight speed in m/s
            timeout_s: Maximum time to reach destination

        Returns:
            True if destination reached
        """
        ...

    async def return_to_home(self, altitude_agl: float = 30.0) -> bool:
        """Return to home/launch position and land.

        Args:
            altitude_agl: Altitude to fly at during return

        Returns:
            True if RTH completed
        """
        ...

    async def orbit(
        self,
        center_lat: float,
        center_lon: float,
        altitude_msl: float,
        radius: float = 20.0,
        velocity: float = 3.0,
        duration_s: float = 30.0,
        clockwise: bool = True,
    ) -> bool:
        """Orbit around a GPS position.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            altitude_msl: Orbit altitude in meters MSL
            radius: Orbit radius in meters
            velocity: Tangential velocity in m/s
            duration_s: Duration of orbit in seconds
            clockwise: True for clockwise orbit

        Returns:
            True if orbit completed
        """
        ...

    # -------------------------------------------------------------------------
    # State and Properties
    # -------------------------------------------------------------------------

    async def get_state(self) -> VehicleState | None:
        """Get current vehicle state.

        Returns:
            VehicleState if available, None if not connected or data unavailable
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected to the platform."""
        ...

    @property
    def is_armed(self) -> bool:
        """Check if vehicle is armed."""
        ...

    @property
    def is_flying(self) -> bool:
        """Check if vehicle is currently in flight."""
        ...

    @property
    def home_position(self) -> Position | None:
        """Get home/launch position."""
        ...

    @property
    def connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        ...


class FlightBackendBase(ABC):
    """Base class for flight backend implementations.

    Provides common functionality and state tracking that all backends need.
    Concrete backends should inherit from this class.
    """

    def __init__(self, config: FlightBackendConfig) -> None:
        """Initialize the backend.

        Args:
            config: Backend configuration
        """
        self._config = config
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._armed = False
        self._flying = False
        self._home_position: Position | None = None
        self._last_state: VehicleState | None = None
        self._last_state_time: datetime | None = None

    @property
    def is_connected(self) -> bool:
        return self._connection_status == ConnectionStatus.CONNECTED

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def is_flying(self) -> bool:
        return self._flying

    @property
    def home_position(self) -> Position | None:
        return self._home_position

    @property
    def connection_status(self) -> ConnectionStatus:
        return self._connection_status

    # Subclasses must implement these abstract methods
    @abstractmethod
    async def connect(self) -> bool:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def arm(self) -> bool:
        ...

    @abstractmethod
    async def disarm(self) -> bool:
        ...

    @abstractmethod
    async def takeoff(self, altitude_agl: float, timeout_s: float = 30.0) -> bool:
        ...

    @abstractmethod
    async def land(self, timeout_s: float = 60.0) -> bool:
        ...

    @abstractmethod
    async def hover(self) -> bool:
        ...

    @abstractmethod
    async def emergency_stop(self) -> bool:
        ...

    @abstractmethod
    async def goto_position_gps(
        self,
        latitude: float,
        longitude: float,
        altitude_msl: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        ...

    @abstractmethod
    async def goto_position_ned(
        self,
        north: float,
        east: float,
        down: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        ...

    @abstractmethod
    async def return_to_home(self, altitude_agl: float = 30.0) -> bool:
        ...

    @abstractmethod
    async def orbit(
        self,
        center_lat: float,
        center_lon: float,
        altitude_msl: float,
        radius: float = 20.0,
        velocity: float = 3.0,
        duration_s: float = 30.0,
        clockwise: bool = True,
    ) -> bool:
        ...

    @abstractmethod
    async def get_state(self) -> VehicleState | None:
        ...


def create_backend(config: FlightBackendConfig) -> FlightBackend:
    """Factory function to create the appropriate backend.

    Args:
        config: Backend configuration

    Returns:
        Configured FlightBackend instance

    Raises:
        ValueError: If backend type is not supported
    """
    if config.backend_type == BackendType.AIRSIM:
        from autonomy.backends.airsim_backend import AirSimBackend

        if not isinstance(config, AirSimBackendConfig):
            config = AirSimBackendConfig()
        return AirSimBackend(config)

    elif config.backend_type == BackendType.ARDUPILOT:
        from autonomy.backends.ardupilot_backend import ArduPilotBackend

        if not isinstance(config, ArduPilotBackendConfig):
            config = ArduPilotBackendConfig()
        return ArduPilotBackend(config)

    elif config.backend_type == BackendType.MOCK:
        from autonomy.backends.mock_backend import MockBackend

        return MockBackend(config)

    else:
        raise ValueError(f"Unsupported backend type: {config.backend_type}")
