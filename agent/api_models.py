"""Pydantic models for API validation and serialization.

These models complement the dataclasses in vehicle_state.py by providing
validation, serialization, and API request/response models.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSState,
    Position,
    VehicleHealth,
    VehicleState,
    Velocity,
)


class FlightModeEnum(str, Enum):
    """Flight mode enum for API validation."""

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


class PositionModel(BaseModel):
    """Pydantic model for Position validation."""

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")
    altitude_msl: float = Field(..., ge=-1000.0, le=50000.0, description="Altitude MSL in meters")
    altitude_agl: float | None = Field(
        None, ge=-100.0, le=1000.0, description="Altitude AGL in meters"
    )

    @field_validator("latitude", "longitude")
    @classmethod
    def validate_coordinates(cls, v: float) -> float:
        """Validate coordinate ranges."""
        return float(v)

    def to_dataclass(self) -> Position:
        """Convert to dataclass."""
        return Position(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude_msl=self.altitude_msl,
            altitude_agl=self.altitude_agl,
        )

    @classmethod
    def from_dataclass(cls, pos: Position) -> "PositionModel":
        """Create from dataclass."""
        return cls(
            latitude=pos.latitude,
            longitude=pos.longitude,
            altitude_msl=pos.altitude_msl,
            altitude_agl=pos.altitude_agl,
        )


class VelocityModel(BaseModel):
    """Pydantic model for Velocity validation."""

    north: float = Field(..., description="North velocity in m/s")
    east: float = Field(..., description="East velocity in m/s")
    down: float = Field(..., description="Down velocity in m/s")

    def to_dataclass(self) -> Velocity:
        """Convert to dataclass."""
        return Velocity(north=self.north, east=self.east, down=self.down)

    @classmethod
    def from_dataclass(cls, vel: Velocity) -> "VelocityModel":
        """Create from dataclass."""
        return cls(north=vel.north, east=vel.east, down=vel.down)


class AttitudeModel(BaseModel):
    """Pydantic model for Attitude validation."""

    roll: float = Field(..., ge=-3.14159, le=3.14159, description="Roll angle in radians")
    pitch: float = Field(..., ge=-3.14159, le=3.14159, description="Pitch angle in radians")
    yaw: float = Field(..., ge=0, le=6.28318, description="Yaw angle in radians")

    def to_dataclass(self) -> Attitude:
        """Convert to dataclass."""
        return Attitude(roll=self.roll, pitch=self.pitch, yaw=self.yaw)

    @classmethod
    def from_dataclass(cls, att: Attitude) -> "AttitudeModel":
        """Create from dataclass."""
        return cls(roll=att.roll, pitch=att.pitch, yaw=att.yaw)


class BatteryModel(BaseModel):
    """Pydantic model for BatteryState validation."""

    voltage: float = Field(..., gt=0, le=50.0, description="Battery voltage in volts")
    current: float = Field(..., ge=-100, le=100, description="Current in amps")
    remaining_percent: float = Field(..., ge=0, le=100, description="Remaining capacity percentage")
    remaining_mah: float | None = Field(None, ge=0, description="Remaining capacity in mAh")
    time_remaining_s: int | None = Field(None, ge=0, description="Time remaining in seconds")

    @field_validator("voltage")
    @classmethod
    def validate_voltage(cls, v: float) -> float:
        """Validate reasonable voltage range."""
        if v < 10 or v > 30:  # Typical drone battery range
            raise ValueError("Voltage outside typical drone battery range")
        return v

    def to_dataclass(self) -> BatteryState:
        """Convert to dataclass."""
        return BatteryState(
            voltage=self.voltage,
            current=self.current,
            remaining_percent=self.remaining_percent,
            remaining_mah=self.remaining_mah,
            time_remaining_s=self.time_remaining_s,
        )

    @classmethod
    def from_dataclass(cls, batt: BatteryState) -> "BatteryModel":
        """Create from dataclass."""
        return cls(
            voltage=batt.voltage,
            current=batt.current,
            remaining_percent=batt.remaining_percent,
            remaining_mah=batt.remaining_mah,
            time_remaining_s=batt.time_remaining_s,
        )


class GPSModel(BaseModel):
    """Pydantic model for GPSState validation."""

    fix_type: int = Field(..., ge=0, le=5, description="GPS fix type")
    satellites_visible: int = Field(..., ge=0, le=50, description="Number of visible satellites")
    hdop: float = Field(..., ge=0, le=99.9, description="Horizontal dilution of precision")
    vdop: float = Field(99.9, ge=0, le=99.9, description="Vertical dilution of precision")

    def to_dataclass(self) -> GPSState:
        """Convert to dataclass."""
        return GPSState(
            fix_type=self.fix_type,
            satellites_visible=self.satellites_visible,
            hdop=self.hdop,
            vdop=self.vdop,
        )

    @classmethod
    def from_dataclass(cls, gps: GPSState) -> "GPSModel":
        """Create from dataclass."""
        return cls(
            fix_type=gps.fix_type,
            satellites_visible=gps.satellites_visible,
            hdop=gps.hdop,
            vdop=gps.vdop,
        )


class HealthModel(BaseModel):
    """Pydantic model for VehicleHealth validation."""

    sensors_healthy: bool = True
    gps_healthy: bool = True
    battery_healthy: bool = True
    motors_healthy: bool = True
    ekf_healthy: bool = True
    error_messages: list[str] = Field(default_factory=list)

    def to_dataclass(self) -> VehicleHealth:
        """Convert to dataclass."""
        return VehicleHealth(
            sensors_healthy=self.sensors_healthy,
            gps_healthy=self.gps_healthy,
            battery_healthy=self.battery_healthy,
            motors_healthy=self.motors_healthy,
            ekf_healthy=self.ekf_healthy,
            error_messages=self.error_messages,
        )

    @classmethod
    def from_dataclass(cls, health: VehicleHealth) -> "HealthModel":
        """Create from dataclass."""
        return cls(
            sensors_healthy=health.sensors_healthy,
            gps_healthy=health.gps_healthy,
            battery_healthy=health.battery_healthy,
            motors_healthy=health.motors_healthy,
            ekf_healthy=health.ekf_healthy,
            error_messages=health.error_messages,
        )


class VehicleStateRequest(BaseModel):
    """API request model for vehicle state updates."""

    timestamp: datetime
    position: PositionModel
    velocity: VelocityModel
    attitude: AttitudeModel
    battery: BatteryModel
    mode: FlightModeEnum
    armed: bool
    in_air: bool = False
    gps: GPSModel | None = None
    health: HealthModel | None = None
    home_position: PositionModel | None = None
    last_heartbeat: datetime | None = None

    def to_dataclass(self) -> VehicleState:
        """Convert to VehicleState dataclass."""
        return VehicleState(
            timestamp=self.timestamp,
            position=self.position.to_dataclass(),
            velocity=self.velocity.to_dataclass(),
            attitude=self.attitude.to_dataclass(),
            battery=self.battery.to_dataclass(),
            mode=FlightMode(self.mode.value),
            armed=self.armed,
            in_air=self.in_air,
            gps=self.gps.to_dataclass() if self.gps else None,
            health=self.health.to_dataclass() if self.health else None,
            home_position=self.home_position.to_dataclass() if self.home_position else None,
            last_heartbeat=self.last_heartbeat,
        )

    @classmethod
    def from_dataclass(cls, state: VehicleState) -> "VehicleStateRequest":
        """Create from VehicleState dataclass."""
        return cls(
            timestamp=state.timestamp,
            position=PositionModel.from_dataclass(state.position),
            velocity=VelocityModel.from_dataclass(state.velocity),
            attitude=AttitudeModel.from_dataclass(state.attitude),
            battery=BatteryModel.from_dataclass(state.battery),
            mode=FlightModeEnum(state.mode.value),
            armed=state.armed,
            in_air=state.in_air,
            gps=GPSModel.from_dataclass(state.gps) if state.gps else None,
            health=HealthModel.from_dataclass(state.health) if state.health else None,
            home_position=PositionModel.from_dataclass(state.home_position)
            if state.home_position
            else None,
            last_heartbeat=state.last_heartbeat,
        )


# Decision and Action Models


class ActionType(str, Enum):
    """Types of actions the agent can command."""

    # Movement actions
    GOTO = "goto"  # Fly to a position
    TAKEOFF = "takeoff"  # Take off to altitude
    LAND = "land"  # Land at current position
    RTL = "rtl"  # Return to launch

    # Mission actions
    INSPECT = "inspect"  # Perform asset inspection
    ORBIT = "orbit"  # Orbit around a point

    # Dock actions
    DOCK = "dock"  # Return to dock and land
    RECHARGE = "recharge"  # Recharge at dock
    UNDOCK = "undock"  # Take off from dock

    # Control actions
    WAIT = "wait"  # Hold position, wait for condition
    ABORT = "abort"  # Emergency abort mission
    RETURN = "return"  # General return command

    # No action
    NONE = "none"  # No action required


class DecisionResponse(BaseModel):
    """API response model for agent decisions."""

    decision_id: str = Field(..., description="Unique ID for correlating execution feedback")
    action: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1, max_length=1000)
    risk_assessment: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime

    # Optional execution info
    execution_time_s: float | None = None
    waypoint_index: int | None = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Server status: healthy, degraded, or error")
    timestamp: datetime
    uptime_seconds: float
    decisions_made: int = 0
    last_state_update: str | None = None

    # Optional system info
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None


# Configuration Models


class MAVLinkConfig(BaseModel):
    """MAVLink configuration model."""

    connection: str = Field(..., description="MAVLink connection string")
    system_id: int = Field(1, ge=1, le=255)
    component_id: int = Field(1, ge=1, le=255)
    timeout_ms: int = Field(1000, ge=100, le=10000)
    heartbeat_interval_s: float = Field(1.0, ge=0.1, le=10.0)


class ServerConfig(BaseModel):
    """Server configuration model."""

    host: str = Field("127.0.0.1", description="Server host address")
    port: int = Field(8080, ge=1024, le=65535, description="Server port")
    workers: int = Field(1, ge=1, le=10, description="Number of worker processes")
    log_level: str = Field("INFO", description="Logging level")


class AgentConfig(BaseModel):
    """Main agent configuration model."""

    name: str = Field("aegis-agent", description="Agent name")
    loop_rate_hz: float = Field(10.0, ge=1.0, le=100.0, description="Decision loop rate")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Confidence threshold")
    max_replan_attempts: int = Field(3, ge=1, le=10, description="Maximum replan attempts")

    mavlink: MAVLinkConfig
    server: ServerConfig

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AgentConfig":
        """Create from configuration dictionary."""
        mavlink_config = config_dict.get("mavlink", {})
        server_config = config_dict.get("server", {})

        return cls(
            name=config_dict.get("agent", {}).get("name", "aegis-agent"),
            loop_rate_hz=config_dict.get("agent", {}).get("loop_rate_hz", 10.0),
            confidence_threshold=config_dict.get("decision", {}).get("confidence_threshold", 0.7),
            max_replan_attempts=config_dict.get("decision", {}).get("max_replan_attempts", 3),
            mavlink=MAVLinkConfig(**mavlink_config),
            server=ServerConfig(**server_config),
        )
