"""
Vehicle State Data Models

Core models representing vehicle telemetry and state information.
These models are platform-agnostic and used throughout the system.
"""

import math
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FlightMode(Enum):
    """ArduPilot/PX4 flight modes supported by AegisAV."""

    STABILIZE = "STABILIZE"
    ALT_HOLD = "ALT_HOLD"
    LOITER = "LOITER"
    GUIDED = "GUIDED"
    AUTO = "AUTO"
    RTL = "RTL"
    LAND = "LAND"
    POSHOLD = "POSHOLD"
    BRAKE = "BRAKE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, mode_str: str) -> "FlightMode":
        """Convert string to FlightMode, defaulting to UNKNOWN."""
        try:
            return cls(mode_str.upper())
        except ValueError:
            return cls.UNKNOWN


class Position(BaseModel):
    """
    Geographic position in WGS84 coordinates.

    Attributes:
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
        altitude_msl: Altitude above mean sea level in meters
        altitude_agl: Altitude above ground level in meters (optional)
    """

    model_config = ConfigDict(frozen=True)

    latitude: float
    longitude: float
    altitude_msl: float
    altitude_agl: float | None = None

    def distance_to(self, other: "Position") -> float:
        """
        Calculate approximate distance to another position in meters.
        Uses haversine formula for accuracy.
        """
        earth_radius_m = 6371000  # Earth radius in meters

        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlat = math.radians(other.latitude - self.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        horizontal = earth_radius_m * c
        vertical = abs(self.altitude_msl - other.altitude_msl)

        return math.sqrt(horizontal**2 + vertical**2)


class Velocity(BaseModel):
    """
    Velocity in NED (North-East-Down) frame.

    Attributes:
        north: Velocity north in m/s
        east: Velocity east in m/s
        down: Velocity down in m/s (positive = descending)
    """

    model_config = ConfigDict(frozen=True)

    north: float
    east: float
    down: float

    @property
    def ground_speed(self) -> float:
        """Horizontal ground speed in m/s."""
        return math.sqrt(self.north**2 + self.east**2)

    @property
    def speed_3d(self) -> float:
        """Total 3D speed in m/s."""
        return math.sqrt(self.north**2 + self.east**2 + self.down**2)

    @property
    def climb_rate(self) -> float:
        """Vertical climb rate in m/s (positive = climbing)."""
        return -self.down


class Attitude(BaseModel):
    """
    Vehicle attitude in radians.

    Attributes:
        roll: Roll angle in radians (positive = right wing down)
        pitch: Pitch angle in radians (positive = nose up)
        yaw: Yaw angle in radians (0 = north, positive = clockwise)
    """

    model_config = ConfigDict(frozen=True)

    roll: float
    pitch: float
    yaw: float

    @property
    def roll_deg(self) -> float:
        """Roll in degrees."""
        return math.degrees(self.roll)

    @property
    def pitch_deg(self) -> float:
        """Pitch in degrees."""
        return math.degrees(self.pitch)

    @property
    def yaw_deg(self) -> float:
        """Yaw/heading in degrees."""
        return math.degrees(self.yaw) % 360


class BatteryState(BaseModel):
    """
    Battery status information.

    Attributes:
        voltage: Battery voltage in volts
        current: Current draw in amps
        remaining_percent: Remaining capacity (0-100)
        remaining_mah: Remaining capacity in mAh (optional)
        time_remaining_s: Estimated time remaining in seconds (optional)
    """

    model_config = ConfigDict(frozen=True)

    voltage: float
    current: float
    remaining_percent: float
    remaining_mah: float | None = None
    time_remaining_s: int | None = None

    @property
    def is_critical(self) -> bool:
        """Check if battery is at critical level (<20%)."""
        return self.remaining_percent < 20.0

    @property
    def is_low(self) -> bool:
        """Check if battery is low (<30%)."""
        return self.remaining_percent < 30.0


class GPSState(BaseModel):
    """GPS quality information."""

    fix_type: int  # 0=no fix, 2=2D, 3=3D, 4=DGPS, 5=RTK
    satellites_visible: int
    hdop: float
    vdop: float = 99.9

    @property
    def has_fix(self) -> bool:
        """Check if GPS has a valid fix."""
        return self.fix_type >= 3

    @property
    def is_good(self) -> bool:
        """Check if GPS quality is good for autonomous operations."""
        return self.has_fix and self.hdop < 2.0 and self.satellites_visible >= 6


# Alias for backwards compatibility
GPSInfo = GPSState


class VehicleHealth(BaseModel):
    """Vehicle health status."""

    sensors_healthy: bool = True
    gps_healthy: bool = True
    battery_healthy: bool = True
    motors_healthy: bool = True
    ekf_healthy: bool = True
    error_messages: list = Field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if all critical systems are healthy."""
        return all([
            self.sensors_healthy,
            self.gps_healthy,
            self.battery_healthy,
            self.motors_healthy,
            self.ekf_healthy,
        ])


class VehicleState(BaseModel):
    """
    Complete vehicle state snapshot.

    This is the primary telemetry object passed between components.
    It aggregates all relevant vehicle information at a point in time.
    """

    timestamp: datetime
    position: Position
    velocity: Velocity
    attitude: Attitude
    battery: BatteryState
    mode: FlightMode
    armed: bool

    # Health indicators
    gps: GPSState | None = None
    health: VehicleHealth | None = None

    # Additional status
    in_air: bool = False
    home_position: Position | None = None

    # Telemetry quality
    last_heartbeat: datetime | None = None

    def age_seconds(self) -> float:
        """Time since this state was captured in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()

    def distance_to_home(self) -> float | None:
        """Distance to home position in meters, if home is set."""
        if self.home_position:
            return self.position.distance_to(self.home_position)
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "position": {
                "latitude": self.position.latitude,
                "longitude": self.position.longitude,
                "altitude_msl": self.position.altitude_msl,
                "altitude_agl": self.position.altitude_agl,
            },
            "velocity": {
                "north": self.velocity.north,
                "east": self.velocity.east,
                "down": self.velocity.down,
                "ground_speed": self.velocity.ground_speed,
            },
            "attitude": {
                "roll": self.attitude.roll_deg,
                "pitch": self.attitude.pitch_deg,
                "yaw": self.attitude.yaw_deg,
            },
            "battery": {
                "voltage": self.battery.voltage,
                "current": self.battery.current,
                "remaining_percent": self.battery.remaining_percent,
            },
            "mode": self.mode.value,
            "armed": self.armed,
            "healthy": self.health.is_healthy if self.health else True,
            "in_air": self.in_air,
        }
