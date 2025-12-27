"""
MAVLink Interface

Manages MAVLink communication with ArduPilot/PX4 flight controllers.
Supports both SITL (UDP) and hardware (serial) connections.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from pymavlink import mavutil

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

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """MAVLink connection states."""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LOST = "lost"


@dataclass
class MAVLinkConfig:
    """MAVLink connection configuration."""
    
    connection_string: str = "udp:127.0.0.1:14550"
    source_system: int = 255
    source_component: int = 0
    timeout_ms: int = 1000
    heartbeat_interval_s: float = 1.0


class MAVLinkInterface:
    """
    Manages MAVLink connection and message handling.
    
    This class provides:
    - Connection management (connect/disconnect/reconnect)
    - Message parsing and aggregation
    - High-level command sending
    - State change callbacks
    
    Example:
        interface = MAVLinkInterface(config)
        await interface.connect()
        
        # Subscribe to state updates
        interface.on_state_update(my_callback)
        
        # Send commands
        await interface.arm()
        await interface.set_mode("GUIDED")
        await interface.goto(lat, lon, alt)
    """
    
    def __init__(self, config: Optional[MAVLinkConfig] = None):
        self.config = config or MAVLinkConfig()
        self._connection: Optional[mavutil.mavfile] = None
        self._state = ConnectionState.DISCONNECTED
        self._running = False
        
        # Callbacks
        self._state_callbacks: list[Callable[[VehicleState], None]] = []
        self._connection_callbacks: list[Callable[[ConnectionState], None]] = []
        
        # Cached telemetry for state aggregation
        self._last_position: Optional[Position] = None
        self._last_velocity: Optional[Velocity] = None
        self._last_attitude: Optional[Attitude] = None
        self._last_battery: Optional[BatteryState] = None
        self._last_gps: Optional[GPSState] = None
        self._last_heartbeat: Optional[datetime] = None
        self._armed: bool = False
        self._mode: FlightMode = FlightMode.UNKNOWN
        self._home_position: Optional[Position] = None
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if connected and receiving heartbeats."""
        return self._state == ConnectionState.CONNECTED
    
    async def connect(self) -> bool:
        """
        Establish MAVLink connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self._connection is not None:
            logger.warning("Already connected, disconnecting first")
            await self.disconnect()
        
        self._set_state(ConnectionState.CONNECTING)
        
        try:
            logger.info(f"Connecting to {self.config.connection_string}")
            
            self._connection = mavutil.mavlink_connection(
                self.config.connection_string,
                source_system=self.config.source_system,
                source_component=self.config.source_component,
            )
            
            # Wait for first heartbeat
            logger.info("Waiting for heartbeat...")
            self._connection.wait_heartbeat(timeout=self.config.timeout_ms / 1000)
            
            self._set_state(ConnectionState.CONNECTED)
            self._running = True
            
            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info(f"Connected to system {self._connection.target_system}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._set_state(ConnectionState.DISCONNECTED)
            return False
    
    async def disconnect(self) -> None:
        """Close MAVLink connection."""
        self._running = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._connection:
            self._connection.close()
            self._connection = None
        
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("Disconnected")
    
    def on_state_update(self, callback: Callable[[VehicleState], None]) -> None:
        """Register callback for vehicle state updates."""
        self._state_callbacks.append(callback)
    
    def on_connection_change(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register callback for connection state changes."""
        self._connection_callbacks.append(callback)
    
    async def arm(self) -> bool:
        """
        Arm the vehicle.
        
        Returns:
            True if arm command sent successfully
        """
        if not self._connection:
            return False
        
        self._connection.arducopter_arm()
        logger.info("Arm command sent")
        return True
    
    async def disarm(self) -> bool:
        """
        Disarm the vehicle.
        
        Returns:
            True if disarm command sent successfully
        """
        if not self._connection:
            return False
        
        self._connection.arducopter_disarm()
        logger.info("Disarm command sent")
        return True
    
    async def set_mode(self, mode: str) -> bool:
        """
        Set flight mode.
        
        Args:
            mode: Mode name (e.g., "GUIDED", "AUTO", "RTL")
            
        Returns:
            True if mode change command sent
        """
        if not self._connection:
            return False
        
        mode_id = self._connection.mode_mapping().get(mode)
        if mode_id is None:
            logger.error(f"Unknown mode: {mode}")
            return False
        
        self._connection.set_mode(mode_id)
        logger.info(f"Mode change to {mode} requested")
        return True
    
    async def takeoff(self, altitude: float) -> bool:
        """
        Command takeoff to specified altitude.
        
        Args:
            altitude: Target altitude in meters AGL
            
        Returns:
            True if takeoff command sent
        """
        if not self._connection:
            return False
        
        self._connection.mav.command_long_send(
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # confirmation
            0, 0, 0, 0,  # params 1-4 unused
            0, 0,  # lat, lon (use current)
            altitude,
        )
        logger.info(f"Takeoff to {altitude}m commanded")
        return True
    
    async def goto(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        ground_speed: Optional[float] = None,
    ) -> bool:
        """
        Command vehicle to fly to position.
        
        Args:
            latitude: Target latitude in degrees
            longitude: Target longitude in degrees
            altitude: Target altitude in meters MSL
            ground_speed: Optional ground speed in m/s
            
        Returns:
            True if goto command sent
        """
        if not self._connection:
            return False
        
        # Use SET_POSITION_TARGET_GLOBAL_INT for GUIDED mode
        type_mask = 0b0000111111111000  # Only position, ignore velocity/accel/yaw
        
        self._connection.mav.set_position_target_global_int_send(
            0,  # time_boot_ms
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            type_mask,
            int(latitude * 1e7),
            int(longitude * 1e7),
            altitude,
            0, 0, 0,  # velocity
            0, 0, 0,  # acceleration
            0, 0,  # yaw, yaw_rate
        )
        
        logger.info(f"Goto ({latitude:.6f}, {longitude:.6f}, {altitude:.1f}m) commanded")
        return True
    
    async def land(self) -> bool:
        """
        Command vehicle to land at current position.
        
        Returns:
            True if land command sent
        """
        return await self.set_mode("LAND")
    
    async def return_to_launch(self) -> bool:
        """
        Command vehicle to return to launch point.
        
        Returns:
            True if RTL command sent
        """
        return await self.set_mode("RTL")
    
    def get_current_state(self) -> Optional[VehicleState]:
        """
        Get the current aggregated vehicle state.
        
        Returns:
            VehicleState if sufficient data available, None otherwise
        """
        if not all([
            self._last_position,
            self._last_velocity,
            self._last_attitude,
            self._last_battery,
        ]):
            return None
        
        return VehicleState(
            timestamp=datetime.now(),
            position=self._last_position,
            velocity=self._last_velocity,
            attitude=self._last_attitude,
            battery=self._last_battery,
            mode=self._mode,
            armed=self._armed,
            in_air=self._armed and (self._last_position.altitude_agl > 0.5),
            gps=self._last_gps or GPSState(0, 0, 99.9, 99.9),
            health=VehicleHealth(True, True, True, True, True),  # Simplified
            home_position=self._home_position,
        )
    
    def _set_state(self, state: ConnectionState) -> None:
        """Update connection state and notify callbacks."""
        if state != self._state:
            self._state = state
            for callback in self._connection_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"Connection callback error: {e}")
    
    async def _receive_loop(self) -> None:
        """Background task to receive and process MAVLink messages."""
        while self._running and self._connection:
            try:
                # Non-blocking receive
                msg = self._connection.recv_match(blocking=False)
                if msg:
                    self._process_message(msg)
                else:
                    await asyncio.sleep(0.01)  # Prevent busy loop
                    
            except Exception as e:
                logger.error(f"Receive error: {e}")
                await asyncio.sleep(0.1)
    
    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeats and check connection."""
        while self._running and self._connection:
            try:
                # Send heartbeat
                self._connection.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0,
                )
                
                # Check for heartbeat timeout
                if self._last_heartbeat:
                    elapsed = (datetime.now() - self._last_heartbeat).total_seconds()
                    if elapsed > 5.0 and self._state == ConnectionState.CONNECTED:
                        logger.warning("Heartbeat timeout")
                        self._set_state(ConnectionState.LOST)
                
                await asyncio.sleep(self.config.heartbeat_interval_s)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1.0)
    
    def _process_message(self, msg) -> None:
        """Process a received MAVLink message."""
        msg_type = msg.get_type()
        
        if msg_type == "HEARTBEAT":
            self._process_heartbeat(msg)
        elif msg_type == "GLOBAL_POSITION_INT":
            self._process_global_position(msg)
        elif msg_type == "LOCAL_POSITION_NED":
            self._process_local_position(msg)
        elif msg_type == "ATTITUDE":
            self._process_attitude(msg)
        elif msg_type == "SYS_STATUS":
            self._process_sys_status(msg)
        elif msg_type == "GPS_RAW_INT":
            self._process_gps_raw(msg)
        elif msg_type == "HOME_POSITION":
            self._process_home_position(msg)
    
    def _process_heartbeat(self, msg) -> None:
        """Process HEARTBEAT message."""
        self._last_heartbeat = datetime.now()
        self._armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
        
        # Get mode name
        mode_mapping = self._connection.mode_mapping() if self._connection else {}
        mode_name = None
        for name, mode_id in mode_mapping.items():
            if mode_id == msg.custom_mode:
                mode_name = name
                break
        
        self._mode = FlightMode.from_string(mode_name or "UNKNOWN")
        
        if self._state == ConnectionState.LOST:
            self._set_state(ConnectionState.CONNECTED)
    
    def _process_global_position(self, msg) -> None:
        """Process GLOBAL_POSITION_INT message."""
        self._last_position = Position(
            latitude=msg.lat / 1e7,
            longitude=msg.lon / 1e7,
            altitude_msl=msg.alt / 1000,
            altitude_agl=msg.relative_alt / 1000,
        )
        
        self._last_velocity = Velocity(
            north=msg.vx / 100,
            east=msg.vy / 100,
            down=msg.vz / 100,
        )
        
        # Notify callbacks
        state = self.get_current_state()
        if state:
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
    
    def _process_local_position(self, msg) -> None:
        """Process LOCAL_POSITION_NED message."""
        # Could be used for more precise velocity
        pass
    
    def _process_attitude(self, msg) -> None:
        """Process ATTITUDE message."""
        self._last_attitude = Attitude(
            roll=msg.roll,
            pitch=msg.pitch,
            yaw=msg.yaw,
        )
    
    def _process_sys_status(self, msg) -> None:
        """Process SYS_STATUS message."""
        self._last_battery = BatteryState(
            voltage=msg.voltage_battery / 1000,
            current=msg.current_battery / 100 if msg.current_battery >= 0 else 0,
            remaining_percent=msg.battery_remaining if msg.battery_remaining >= 0 else 0,
        )
    
    def _process_gps_raw(self, msg) -> None:
        """Process GPS_RAW_INT message."""
        self._last_gps = GPSState(
            fix_type=msg.fix_type,
            satellites_visible=msg.satellites_visible,
            hdop=msg.eph / 100 if msg.eph != 65535 else 99.9,
            vdop=msg.epv / 100 if msg.epv != 65535 else 99.9,
        )
    
    def _process_home_position(self, msg) -> None:
        """Process HOME_POSITION message."""
        self._home_position = Position(
            latitude=msg.latitude / 1e7,
            longitude=msg.longitude / 1e7,
            altitude_msl=msg.altitude / 1000,
        )
