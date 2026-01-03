"""ArduPilot Flight Backend.

Implements the FlightBackend protocol for real ArduPilot hardware.
Wraps the MAVLinkInterface for flight control via MAVLink.
"""

from __future__ import annotations

import asyncio
import logging
import math

from autonomy.flight_backend import (
    ArduPilotBackendConfig,
    ConnectionStatus,
    FlightBackendBase,
)
from autonomy.mavlink_interface import ConnectionState, MAVLinkConfig, MAVLinkInterface
from autonomy.vehicle_state import (
    Position,
    VehicleState,
)
from simulation.coordinate_utils import GeoReference

logger = logging.getLogger(__name__)


class ArduPilotBackend(FlightBackendBase):
    """ArduPilot implementation of FlightBackend.

    Uses MAVLinkInterface for communication with ArduPilot hardware.
    Supports both SITL (UDP) and real hardware (serial) connections.

    Example:
        config = ArduPilotBackendConfig(connection_string="udp:127.0.0.1:14550")
        backend = ArduPilotBackend(config)
        await backend.connect()
        await backend.set_mode("GUIDED")
        await backend.arm()
        await backend.takeoff(altitude_agl=10.0)
    """

    def __init__(self, config: ArduPilotBackendConfig) -> None:
        """Initialize ArduPilot backend.

        Args:
            config: ArduPilot configuration
        """
        super().__init__(config)
        self._config: ArduPilotBackendConfig = config
        self._mavlink: MAVLinkInterface | None = None
        self._geo_ref: GeoReference | None = None
        self._state_update_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        """Connect to ArduPilot via MAVLink."""
        try:
            self._connection_status = ConnectionStatus.CONNECTING
            logger.info(f"Connecting to ArduPilot at {self._config.connection_string}...")

            # Create MAVLink interface
            mav_config = MAVLinkConfig(
                connection_string=self._config.connection_string,
                source_system=self._config.source_system,
                source_component=self._config.source_component,
            )

            self._mavlink = MAVLinkInterface(mav_config)

            # Register state callback
            self._mavlink.on_state_update(self._on_state_update)
            self._mavlink.on_connection_change(self._on_connection_change)

            # Connect
            success = await self._mavlink.connect()

            if success:
                self._connection_status = ConnectionStatus.CONNECTED
                logger.info("Connected to ArduPilot")

                # Wait for initial state
                await asyncio.sleep(1.0)

                # Get home position if available
                state = self._mavlink.get_current_state()
                if state and state.home_position:
                    self._home_position = state.home_position
                    self._geo_ref = GeoReference(
                        latitude=state.home_position.latitude,
                        longitude=state.home_position.longitude,
                        altitude=state.home_position.altitude_msl,
                    )
                    logger.info(f"Home position set: {state.home_position}")

                return True
            else:
                self._connection_status = ConnectionStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Failed to connect to ArduPilot: {e}")
            self._connection_status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from ArduPilot."""
        if self._mavlink:
            await self._mavlink.disconnect()
            self._mavlink = None

        self._connection_status = ConnectionStatus.DISCONNECTED
        self._armed = False
        self._flying = False
        logger.info("Disconnected from ArduPilot")

    def _on_state_update(self, state: VehicleState) -> None:
        """Handle state updates from MAVLink."""
        self._last_state = state
        self._armed = state.armed
        self._flying = state.in_air
        if state.home_position and not self._home_position:
            self._home_position = state.home_position

    def _on_connection_change(self, state: ConnectionState) -> None:
        """Handle connection state changes."""
        if state == ConnectionState.CONNECTED:
            self._connection_status = ConnectionStatus.CONNECTED
        elif state == ConnectionState.LOST:
            self._connection_status = ConnectionStatus.ERROR
        elif state == ConnectionState.DISCONNECTED:
            self._connection_status = ConnectionStatus.DISCONNECTED

    async def set_mode(self, mode: str) -> bool:
        """Set flight mode.

        Args:
            mode: Mode name (e.g., "GUIDED", "AUTO", "RTL")

        Returns:
            True if mode change command sent
        """
        if not self._mavlink or not self.is_connected:
            return False

        return await self._mavlink.set_mode(mode)

    async def arm(self) -> bool:
        """Arm the vehicle."""
        if not self._mavlink or not self.is_connected:
            return False

        try:
            # Ensure we're in GUIDED mode for arming
            await self.set_mode("GUIDED")
            await asyncio.sleep(0.5)

            success = await self._mavlink.arm()
            if success:
                # Wait for arm confirmation
                for _ in range(20):
                    state = self._mavlink.get_current_state()
                    if state and state.armed:
                        self._armed = True
                        logger.info("Vehicle armed")
                        return True
                    await asyncio.sleep(0.1)

            return self._armed

        except Exception as e:
            logger.error(f"Arm failed: {e}")
            return False

    async def disarm(self) -> bool:
        """Disarm the vehicle."""
        if not self._mavlink or not self.is_connected:
            return False

        try:
            success = await self._mavlink.disarm()
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
        if not self._mavlink or not self.is_connected:
            return False

        try:
            # Ensure armed and in GUIDED mode
            if not self._armed:
                await self.arm()

            await self.set_mode("GUIDED")
            await asyncio.sleep(0.2)

            # Send takeoff command
            success = await self._mavlink.takeoff(altitude_agl)
            if not success:
                return False

            # Wait for takeoff completion
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < timeout_s:
                state = self._mavlink.get_current_state()
                if state and state.position.altitude_agl:
                    if state.position.altitude_agl >= altitude_agl * 0.9:
                        self._flying = True
                        logger.info(f"Takeoff complete at {state.position.altitude_agl:.1f}m")
                        return True
                await asyncio.sleep(0.5)

            logger.warning("Takeoff timeout")
            return False

        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False

    async def land(self, timeout_s: float = 60.0) -> bool:
        """Land at current position."""
        if not self._mavlink or not self.is_connected:
            return False

        try:
            success = await self._mavlink.land()
            if not success:
                return False

            # Wait for landing
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < timeout_s:
                state = self._mavlink.get_current_state()
                if state:
                    if not state.in_air or (
                        state.position.altitude_agl and state.position.altitude_agl < 0.5
                    ):
                        self._flying = False
                        logger.info("Landing complete")
                        return True
                await asyncio.sleep(0.5)

            logger.warning("Landing timeout")
            return False

        except Exception as e:
            logger.error(f"Land failed: {e}")
            return False

    async def hover(self) -> bool:
        """Hold current position (LOITER mode)."""
        if not self._mavlink or not self.is_connected:
            return False

        try:
            return await self.set_mode("LOITER")
        except Exception as e:
            logger.error(f"Hover failed: {e}")
            return False

    async def emergency_stop(self) -> bool:
        """Emergency motor stop.

        Warning: Will cause immediate fall.
        """
        if not self._mavlink or not self.is_connected:
            return False

        try:
            # Force disarm (dangerous!)
            await self._mavlink.disarm()
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
        """Fly to GPS position."""
        if not self._mavlink or not self.is_connected:
            return False

        try:
            # Ensure GUIDED mode
            await self.set_mode("GUIDED")
            await asyncio.sleep(0.2)

            logger.info(f"Goto GPS ({latitude:.6f}, {longitude:.6f}, {altitude_msl:.1f}m)")

            # Send goto command
            success = await self._mavlink.goto(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude_msl,
                ground_speed=velocity,
            )

            if not success:
                return False

            # Wait to reach destination
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < timeout_s:
                state = self._mavlink.get_current_state()
                if state:
                    # Check if we're close to target
                    target = Position(
                        latitude=latitude,
                        longitude=longitude,
                        altitude_msl=altitude_msl,
                    )
                    distance = state.position.distance_to(target)
                    if distance < 2.0:  # Within 2 meters
                        logger.info(f"Reached GPS target (distance: {distance:.1f}m)")
                        return True

                await asyncio.sleep(0.5)

            logger.warning("Goto GPS timeout")
            return False

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
        """Fly to NED position relative to home."""
        if not self._mavlink or not self.is_connected:
            return False

        if not self._geo_ref:
            # Try to get home from current state
            state = self._mavlink.get_current_state()
            if state and state.home_position:
                self._geo_ref = GeoReference(
                    latitude=state.home_position.latitude,
                    longitude=state.home_position.longitude,
                    altitude=state.home_position.altitude_msl,
                )
            else:
                logger.error("No geo reference available for NED conversion")
                return False

        try:
            # Convert NED to GPS
            lat, lon, alt_msl = self._geo_ref.ned_to_gps(north, east, down)

            logger.info(
                f"Goto NED ({north:.1f}, {east:.1f}, {down:.1f}) "
                f"-> GPS ({lat:.6f}, {lon:.6f}, {alt_msl:.1f}m)"
            )

            return await self.goto_position_gps(lat, lon, alt_msl, velocity, timeout_s)

        except Exception as e:
            logger.error(f"Goto NED failed: {e}")
            return False

    async def return_to_home(self, altitude_agl: float = 30.0) -> bool:
        """Return to home and land (RTL mode)."""
        if not self._mavlink or not self.is_connected:
            return False

        try:
            logger.info("Returning to home (RTL)")
            success = await self._mavlink.return_to_launch()

            if success:
                # Wait for RTL to complete (simplified - just wait for landing)
                for _ in range(120):  # 60 seconds max
                    state = self._mavlink.get_current_state()
                    if state and not state.in_air:
                        self._flying = False
                        logger.info("RTL complete - landed")
                        return True
                    await asyncio.sleep(0.5)

            return False

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
        """Orbit around a GPS position.

        Implemented by flying waypoints in a circle.
        """
        if not self._mavlink or not self.is_connected:
            return False

        try:
            await self.set_mode("GUIDED")

            logger.info(
                f"Orbit around ({center_lat:.6f}, {center_lon:.6f}) "
                f"radius={radius}m for {duration_s}s"
            )

            # Create a simple orbit by flying to points on the circle
            num_waypoints = max(8, int(duration_s / 3))  # Roughly 3s per waypoint
            angular_step = (2 * math.pi / num_waypoints) * (1 if clockwise else -1)

            # Calculate starting angle from current position
            state = self._mavlink.get_current_state()
            if state:
                start_geo = GeoReference(
                    latitude=center_lat,
                    longitude=center_lon,
                    altitude=altitude_msl,
                )
                current_n, current_e, _ = start_geo.gps_to_ned(
                    state.position.latitude,
                    state.position.longitude,
                    state.position.altitude_msl,
                )
                start_angle = math.atan2(current_e, current_n)
            else:
                start_angle = 0.0

            # Fly to each waypoint
            for i in range(num_waypoints):
                angle = start_angle + angular_step * i

                # Calculate waypoint in NED from center
                wp_n = radius * math.cos(angle)
                wp_e = radius * math.sin(angle)

                # Convert to GPS
                orbit_geo = GeoReference(
                    latitude=center_lat,
                    longitude=center_lon,
                    altitude=altitude_msl,
                )
                lat, lon, _ = orbit_geo.ned_to_gps(wp_n, wp_e, 0)

                # Fly to waypoint (short timeout since we're continuously moving)
                await self._mavlink.goto(lat, lon, altitude_msl, velocity)

                # Wait for part of the waypoint time
                await asyncio.sleep(duration_s / num_waypoints)

            logger.info("Orbit complete")
            return True

        except Exception as e:
            logger.error(f"Orbit failed: {e}")
            return False

    async def get_state(self) -> VehicleState | None:
        """Get current vehicle state."""
        if not self._mavlink or not self.is_connected:
            return None

        return self._mavlink.get_current_state()

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
