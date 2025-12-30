"""Mock Flight Backend for Testing.

Provides a simulated flight backend for testing without hardware or simulation.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime

from autonomy.flight_backend import (
    ConnectionStatus,
    FlightBackendBase,
    FlightBackendConfig,
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

# Default mock home position
MOCK_HOME_LAT = 47.397742
MOCK_HOME_LON = 8.545594
MOCK_HOME_ALT = 488.0


class MockBackend(FlightBackendBase):
    """Mock implementation of FlightBackend for testing.

    Simulates basic flight behavior without actual hardware.
    Useful for unit testing and development.

    Example:
        backend = MockBackend(FlightBackendConfig())
        await backend.connect()
        await backend.arm()
        await backend.takeoff(10.0)
        state = await backend.get_state()
    """

    def __init__(self, config: FlightBackendConfig) -> None:
        """Initialize mock backend."""
        super().__init__(config)

        # Simulated position (NED)
        self._position_ned = [0.0, 0.0, 0.0]  # north, east, down
        self._velocity_ned = [0.0, 0.0, 0.0]
        self._attitude = [0.0, 0.0, 0.0]  # roll, pitch, yaw

        # Home position
        self._home_position = Position(
            latitude=MOCK_HOME_LAT,
            longitude=MOCK_HOME_LON,
            altitude_msl=MOCK_HOME_ALT,
            altitude_agl=0.0,
        )
        self._geo_ref = GeoReference(
            latitude=MOCK_HOME_LAT,
            longitude=MOCK_HOME_LON,
            altitude=MOCK_HOME_ALT,
        )

        # Simulation speed (m/s)
        self._sim_velocity = 5.0
        self._target_ned: list[float] | None = None

    async def connect(self) -> bool:
        """Simulate connection."""
        logger.info("Mock backend connecting...")
        await asyncio.sleep(0.1)
        self._connection_status = ConnectionStatus.CONNECTED
        logger.info("Mock backend connected")
        return True

    async def disconnect(self) -> None:
        """Simulate disconnect."""
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._armed = False
        self._flying = False
        logger.info("Mock backend disconnected")

    async def arm(self) -> bool:
        """Simulate arming."""
        if not self.is_connected:
            return False

        await asyncio.sleep(0.1)
        self._armed = True
        logger.info("Mock: Armed")
        return True

    async def disarm(self) -> bool:
        """Simulate disarming."""
        if not self.is_connected:
            return False

        await asyncio.sleep(0.1)
        self._armed = False
        self._flying = False
        logger.info("Mock: Disarmed")
        return True

    async def takeoff(self, altitude_agl: float, timeout_s: float = 30.0) -> bool:
        """Simulate takeoff."""
        if not self.is_connected:
            return False

        if not self._armed:
            await self.arm()

        logger.info(f"Mock: Taking off to {altitude_agl}m")

        # Simulate climb
        target_down = -altitude_agl
        climb_rate = 2.0  # m/s

        while self._position_ned[2] > target_down + 0.5:
            self._position_ned[2] -= climb_rate * 0.1
            self._velocity_ned[2] = -climb_rate
            await asyncio.sleep(0.1)

        self._position_ned[2] = target_down
        self._velocity_ned = [0.0, 0.0, 0.0]
        self._flying = True
        logger.info(f"Mock: Takeoff complete at {-self._position_ned[2]:.1f}m")
        return True

    async def land(self, timeout_s: float = 60.0) -> bool:
        """Simulate landing."""
        if not self.is_connected:
            return False

        logger.info("Mock: Landing")

        descent_rate = 1.5  # m/s

        while self._position_ned[2] < -0.5:
            self._position_ned[2] += descent_rate * 0.1
            self._velocity_ned[2] = descent_rate
            await asyncio.sleep(0.1)

        self._position_ned[2] = 0.0
        self._velocity_ned = [0.0, 0.0, 0.0]
        self._flying = False
        logger.info("Mock: Landing complete")
        return True

    async def hover(self) -> bool:
        """Simulate hover."""
        if not self.is_connected:
            return False

        self._velocity_ned = [0.0, 0.0, 0.0]
        self._target_ned = None
        logger.info("Mock: Hovering")
        return True

    async def emergency_stop(self) -> bool:
        """Simulate emergency stop."""
        self._armed = False
        self._flying = False
        self._velocity_ned = [0.0, 0.0, 0.0]
        logger.warning("Mock: Emergency stop")
        return True

    async def goto_position_gps(
        self,
        latitude: float,
        longitude: float,
        altitude_msl: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        """Simulate GPS goto."""
        if not self.is_connected or not self._geo_ref:
            return False

        # Convert to NED
        north, east, down = self._geo_ref.gps_to_ned(latitude, longitude, altitude_msl)
        return await self.goto_position_ned(north, east, down, velocity, timeout_s)

    async def goto_position_ned(
        self,
        north: float,
        east: float,
        down: float,
        velocity: float = 5.0,
        timeout_s: float = 120.0,
    ) -> bool:
        """Simulate NED goto."""
        if not self.is_connected:
            return False

        logger.info(f"Mock: Flying to NED ({north:.1f}, {east:.1f}, {down:.1f})")
        self._sim_velocity = velocity
        self._target_ned = [north, east, down]

        # Simulate flight
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout_s:
            # Calculate distance to target
            dx = self._target_ned[0] - self._position_ned[0]
            dy = self._target_ned[1] - self._position_ned[1]
            dz = self._target_ned[2] - self._position_ned[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance < 1.0:
                self._position_ned = list(self._target_ned)
                self._velocity_ned = [0.0, 0.0, 0.0]
                self._target_ned = None
                logger.info("Mock: Destination reached")
                return True

            # Move towards target
            step = min(velocity * 0.1, distance)
            self._position_ned[0] += (dx / distance) * step
            self._position_ned[1] += (dy / distance) * step
            self._position_ned[2] += (dz / distance) * step

            self._velocity_ned[0] = (dx / distance) * velocity
            self._velocity_ned[1] = (dy / distance) * velocity
            self._velocity_ned[2] = (dz / distance) * velocity

            await asyncio.sleep(0.1)

        logger.warning("Mock: Goto timeout")
        return False

    async def return_to_home(self, altitude_agl: float = 30.0) -> bool:
        """Simulate RTH."""
        if not self.is_connected:
            return False

        logger.info(f"Mock: Returning to home at {altitude_agl}m")

        # Climb to RTH altitude
        await self.goto_position_ned(
            self._position_ned[0], self._position_ned[1], -altitude_agl
        )

        # Fly to home
        await self.goto_position_ned(0.0, 0.0, -altitude_agl)

        # Land
        await self.land()

        return True

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
        """Simulate orbit."""
        if not self.is_connected or not self._geo_ref:
            return False

        logger.info(f"Mock: Orbiting at radius {radius}m for {duration_s}s")

        # Convert center to NED
        center_n, center_e, center_d = self._geo_ref.gps_to_ned(
            center_lat, center_lon, altitude_msl
        )

        # Calculate angular velocity
        angular_velocity = velocity / radius
        if not clockwise:
            angular_velocity = -angular_velocity

        start_angle = math.atan2(
            self._position_ned[1] - center_e,
            self._position_ned[0] - center_n,
        )

        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration_s:
            elapsed = asyncio.get_event_loop().time() - start_time
            angle = start_angle + angular_velocity * elapsed

            # Calculate position on orbit
            target_n = center_n + radius * math.cos(angle)
            target_e = center_e + radius * math.sin(angle)

            self._position_ned[0] = target_n
            self._position_ned[1] = target_e
            self._position_ned[2] = center_d

            await asyncio.sleep(0.1)

        logger.info("Mock: Orbit complete")
        return True

    async def get_state(self) -> VehicleState | None:
        """Get simulated vehicle state."""
        if not self.is_connected:
            return None

        # Convert NED to GPS
        lat, lon, alt_msl = self._geo_ref.ned_to_gps(
            self._position_ned[0],
            self._position_ned[1],
            self._position_ned[2],
        )

        position = Position(
            latitude=lat,
            longitude=lon,
            altitude_msl=alt_msl,
            altitude_agl=-self._position_ned[2],
        )

        velocity = Velocity(
            north=self._velocity_ned[0],
            east=self._velocity_ned[1],
            down=self._velocity_ned[2],
        )

        attitude = Attitude(
            roll=self._attitude[0],
            pitch=self._attitude[1],
            yaw=self._attitude[2],
        )

        battery = BatteryState(
            voltage=16.8,
            current=5.0,
            remaining_percent=85.0,
        )

        return VehicleState(
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

    def set_geo_reference(self, latitude: float, longitude: float, altitude_msl: float) -> None:
        """Set geographic reference."""
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
