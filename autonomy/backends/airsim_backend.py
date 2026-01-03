"""AirSim Flight Backend.

Implements the FlightBackend protocol for AirSim simulation.
Wraps the RealtimeAirSimBridge for flight control.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from autonomy.flight_backend import (
    AirSimBackendConfig,
    ConnectionStatus,
    FlightBackendBase,
)
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
from simulation.coordinate_utils import GeoReference

logger = logging.getLogger(__name__)

# Default home position for AirSim (Zurich area, commonly used in AirSim)
DEFAULT_HOME_LAT = 47.397742
DEFAULT_HOME_LON = 8.545594
DEFAULT_HOME_ALT = 488.0  # meters MSL


class AirSimBackend(FlightBackendBase):
    """AirSim implementation of FlightBackend.

    Uses RealtimeAirSimBridge for actual communication with AirSim.
    Provides GPS and NED coordinate interfaces for flight control.

    Example:
        config = AirSimBackendConfig(host="127.0.0.1")
        backend = AirSimBackend(config)
        await backend.connect()
        await backend.arm()
        await backend.takeoff(altitude_agl=10.0)
    """

    def __init__(self, config: AirSimBackendConfig) -> None:
        """Initialize AirSim backend.

        Args:
            config: AirSim configuration
        """
        super().__init__(config)
        self._config: AirSimBackendConfig = config
        self._bridge = None  # RealtimeAirSimBridge instance
        self._geo_ref: GeoReference | None = None

        # Set default home position (will be updated on connect)
        self._home_position = Position(
            latitude=DEFAULT_HOME_LAT,
            longitude=DEFAULT_HOME_LON,
            altitude_msl=DEFAULT_HOME_ALT,
            altitude_agl=0.0,
        )
        self._geo_ref = GeoReference(
            latitude=DEFAULT_HOME_LAT,
            longitude=DEFAULT_HOME_LON,
            altitude=DEFAULT_HOME_ALT,
        )

    async def connect(self) -> bool:
        """Connect to AirSim simulator."""
        try:
            self._connection_status = ConnectionStatus.CONNECTING
            logger.info(f"Connecting to AirSim at {self._config.host}...")

            # Import and create bridge
            from simulation.realtime_bridge import (
                RealtimeAirSimBridge,
                RealtimeBridgeConfig,
            )

            bridge_config = RealtimeBridgeConfig(
                host=self._config.host,
                vehicle_name=self._config.vehicle_name,
                camera_name=self._config.camera_name,
                telemetry_hz=int(self._config.telemetry_rate_hz),
            )

            self._bridge = RealtimeAirSimBridge(bridge_config)
            success = await self._bridge.connect()

            if success:
                self._connection_status = ConnectionStatus.CONNECTED
                logger.info("Connected to AirSim")

                # Get initial state to determine home position
                state = await self._bridge.get_synchronized_state()
                if state and state.pose:
                    # Store NED origin as home
                    # AirSim NED origin corresponds to the GPS home
                    pass  # Keep default home, AirSim origin is at (0,0,0) NED

                return True
            else:
                self._connection_status = ConnectionStatus.ERROR
                return False

        except ImportError as e:
            logger.error(f"AirSim not available: {e}")
            self._connection_status = ConnectionStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to connect to AirSim: {e}")
            self._connection_status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from AirSim."""
        if self._bridge:
            await self._bridge.disconnect()
            self._bridge = None

        self._connection_status = ConnectionStatus.DISCONNECTED
        self._armed = False
        self._flying = False
        logger.info("Disconnected from AirSim")

    async def arm(self) -> bool:
        """Arm the vehicle."""
        if not self._bridge or not self.is_connected:
            return False

        try:
            success = await self._bridge.arm()
            if success:
                self._armed = True
                logger.info("Vehicle armed")
            return success
        except Exception as e:
            logger.error(f"Arm failed: {e}")
            return False

    async def disarm(self) -> bool:
        """Disarm the vehicle."""
        if not self._bridge or not self.is_connected:
            return False

        try:
            success = await self._bridge.disarm()
            if success:
                self._armed = False
                self._flying = False
                logger.info("Vehicle disarmed")
            return success
        except Exception as e:
            logger.error(f"Disarm failed: {e}")
            return False

    async def takeoff(self, altitude_agl: float, timeout_s: float = 30.0) -> bool:
        """Take off to specified altitude AGL."""
        if not self._bridge or not self.is_connected:
            return False

        try:
            # AirSim takeoff uses altitude parameter
            success = await self._bridge.takeoff(altitude=altitude_agl, timeout=timeout_s)
            if success:
                self._flying = True
                self._armed = True
                logger.info(f"Takeoff to {altitude_agl}m complete")
            return success
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False

    async def land(self, timeout_s: float = 60.0) -> bool:
        """Land at current position."""
        if not self._bridge or not self.is_connected:
            return False

        try:
            success = await self._bridge.land(timeout=timeout_s)
            if success:
                self._flying = False
                logger.info("Landing complete")
            return success
        except Exception as e:
            logger.error(f"Land failed: {e}")
            return False

    async def hover(self) -> bool:
        """Hold current position."""
        if not self._bridge or not self.is_connected:
            return False

        try:
            success = await self._bridge.hover()
            return success
        except Exception as e:
            logger.error(f"Hover failed: {e}")
            return False

    async def emergency_stop(self) -> bool:
        """Emergency motor stop.

        Warning: Will cause immediate fall.
        """
        if not self._bridge or not self.is_connected:
            return False

        try:
            # AirSim doesn't have a direct emergency stop
            # Best we can do is disarm
            await self._bridge.disarm()
            self._armed = False
            self._flying = False
            logger.warning("Emergency stop executed - motors disabled")
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

    async def goto_position_gps(
        self,
        latitude: float,
        longitude: float,
        altitude_msl: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        """Fly to GPS position.

        Converts GPS to NED and delegates to move_to_position.
        """
        if not self._bridge or not self.is_connected or not self._geo_ref:
            return False

        try:
            # Convert GPS to NED
            north, east, down = self._geo_ref.gps_to_ned(latitude, longitude, altitude_msl)

            logger.info(
                f"Goto GPS ({latitude:.6f}, {longitude:.6f}, {altitude_msl:.1f}m) "
                f"-> NED ({north:.1f}, {east:.1f}, {down:.1f})"
            )

            # Use NED command
            success = await self._bridge.move_to_position(
                x=north, y=east, z=down, velocity=velocity, timeout=timeout_s
            )
            return success

        except Exception as e:
            logger.error(f"Goto GPS failed: {e}")
            return False

    async def goto_position_ned(
        self,
        north: float,
        east: float,
        down: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        """Fly to NED position."""
        if not self._bridge or not self.is_connected:
            return False

        try:
            logger.info(f"Goto NED ({north:.1f}, {east:.1f}, {down:.1f}) at {velocity} m/s")

            success = await self._bridge.move_to_position(
                x=north, y=east, z=down, velocity=velocity, timeout=timeout_s
            )
            return success

        except Exception as e:
            logger.error(f"Goto NED failed: {e}")
            return False

    async def return_to_home(self, altitude_agl: float = 30.0) -> bool:
        """Return to home and land."""
        if not self._bridge or not self.is_connected:
            return False

        try:
            logger.info(f"Returning to home at {altitude_agl}m AGL")
            success = await self._bridge.return_to_launch(altitude=altitude_agl)
            if success:
                self._flying = False
            return success

        except Exception as e:
            logger.error(f"Return to home failed: {e}")
            return False

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
        """Orbit around a GPS position."""
        if not self._bridge or not self.is_connected or not self._geo_ref:
            return False

        try:
            # Convert center to NED
            center_n, center_e, center_d = self._geo_ref.gps_to_ned(
                center_lat, center_lon, altitude_msl
            )

            logger.info(
                f"Orbit around GPS ({center_lat:.6f}, {center_lon:.6f}) "
                f"radius={radius}m for {duration_s}s"
            )

            success = await self._bridge.orbit(
                center_x=center_n,
                center_y=center_e,
                center_z=center_d,
                radius=radius,
                velocity=velocity,
                duration=duration_s,
                clockwise=clockwise,
            )
            return success

        except Exception as e:
            logger.error(f"Orbit failed: {e}")
            return False

    async def get_state(self) -> VehicleState | None:
        """Get current vehicle state."""
        if not self._bridge or not self.is_connected:
            return None

        try:
            frame = await self._bridge.get_synchronized_state()
            if not frame or not frame.pose:
                return None

            # Convert NED position to GPS
            if self._geo_ref:
                lat, lon, alt_msl = self._geo_ref.ned_to_gps(
                    frame.pose.position.x,
                    frame.pose.position.y,
                    frame.pose.position.z,
                )
                alt_agl = -frame.pose.position.z  # Down is negative altitude
            else:
                lat, lon, alt_msl = DEFAULT_HOME_LAT, DEFAULT_HOME_LON, DEFAULT_HOME_ALT
                alt_agl = -frame.pose.position.z

            position = Position(
                latitude=lat,
                longitude=lon,
                altitude_msl=alt_msl,
                altitude_agl=alt_agl,
            )

            velocity = Velocity(
                north=frame.pose.linear_velocity.x,
                east=frame.pose.linear_velocity.y,
                down=frame.pose.linear_velocity.z,
            )

            # Convert quaternion to Euler (simplified)
            qw = frame.pose.orientation.w
            qx = frame.pose.orientation.x
            qy = frame.pose.orientation.y
            qz = frame.pose.orientation.z

            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)

            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            attitude = Attitude(roll=roll, pitch=pitch, yaw=yaw)

            battery = BatteryState(
                voltage=16.8,  # Simulated full 4S battery
                current=5.0,
                remaining_percent=frame.battery_percent,
            )

            # Update internal state
            self._flying = frame.landed_state == "flying"
            self._last_state = VehicleState(
                timestamp=datetime.now(),
                position=position,
                velocity=velocity,
                attitude=attitude,
                battery=battery,
                mode=FlightMode.GUIDED if self._flying else FlightMode.STABILIZE,
                armed=self._armed,
                in_air=self._flying,
                gps=GPSState(fix_type=3, satellites_visible=12, hdop=1.0, vdop=1.0),
                health=VehicleHealth(),
                home_position=self._home_position,
            )

            return self._last_state

        except Exception as e:
            logger.error(f"Get state failed: {e}")
            return None

    def set_geo_reference(self, latitude: float, longitude: float, altitude_msl: float) -> None:
        """Set the geographic reference point for NED conversions.

        Args:
            latitude: Reference latitude in degrees
            longitude: Reference longitude in degrees
            altitude_msl: Reference altitude in meters MSL
        """
        self._geo_ref = GeoReference(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude_msl,
        )
        self._home_position = Position(
            latitude=latitude,
            longitude=longitude,
            altitude_msl=altitude_msl,
            altitude_agl=0.0,
        )
        logger.info(f"Geo reference set: ({latitude:.6f}, {longitude:.6f}, {altitude_msl:.1f}m)")
