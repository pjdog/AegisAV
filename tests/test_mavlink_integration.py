"""
Integration tests for MAVLink interface and vehicle communication.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from tests.conftest import (
    TEST_MAVLINK_CONNECTION, TEST_MAVLINK_SYSTEM_ID, TEST_MAVLINK_COMPONENT_ID,
    TEST_HOME_POSITION, TEST_TIMEOUT_S, TEST_STEP_DELAY_S
)


class TestMAVLinkInterface:
    """Test MAVLink communication with mocked connection."""
    
    @pytest.mark.asyncio
    async def test_mavlink_connection_establishment(self, mock_mavlink_connection):
        """Test establishing MAVLink connection."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            # Import here to avoid import issues during pytest collection
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                
                # Verify connection was attempted
                mock_conn_func.assert_called_once_with(TEST_MAVLINK_CONNECTION, timeout=30)
                mock_mavlink_connection.wait_heartbeat.assert_called_once()
                
                # Verify connection state
                assert interface.is_connected
                
            except ImportError:
                pytest.skip("pymavlink not available for testing")
    
    @pytest.mark.asyncio
    async def test_telemetry_reception(self, mock_mavlink_connection):
        """Test receiving and processing telemetry data."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                
                # Mock message loop
                interface.mav_conn = mock_mavlink_connection
                
                # Test position message handling
                position_state = await interface._process_position_message()
                assert position_state is not None
                assert abs(position_state.latitude - TEST_HOME_POSITION['lat']) < 0.0001
                assert abs(position_state.longitude - TEST_HOME_POSITION['lon']) < 0.0001
                
            except ImportError:
                pytest.skip("pymavlink not available for testing")
    
    @pytest.mark.asyncio
    async def test_vehicle_state_aggregation(self, mock_mavlink_connection):
        """Test aggregation of multiple MAVLink messages into vehicle state."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                from autonomy.vehicle_state import VehicleState
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                interface.mav_conn = mock_mavlink_connection
                
                # Get aggregated vehicle state
                vehicle_state = await interface.get_vehicle_state()
                
                assert isinstance(vehicle_state, VehicleState)
                assert vehicle_state.timestamp is not None
                assert vehicle_state.position is not None
                assert vehicle_state.battery is not None
                assert vehicle_state.attitude is not None
                assert vehicle_state.mode is not None
                
                # Verify specific values
                assert vehicle_state.battery.remaining_percent == 80.0
                assert vehicle_state.position.latitude == TEST_HOME_POSITION['lat']
                
            except ImportError:
                pytest.skip("pymavlink not available for testing")
    
    @pytest.mark.asyncio
    async def test_command_sending(self, mock_mavlink_connection):
        """Test sending commands to vehicle via MAVLink."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                interface.mav_conn = mock_mavlink_connection
                
                # Test arm command
                await interface.arm()
                mock_mavlink_connection.mav.command_long_send.assert_called()
                
                # Test disarm command
                await interface.disarm()
                mock_mavlink_connection.mav.command_long_send.assert_called()
                
                # Test mode change
                await interface.set_mode("GUIDED")
                mock_mavlink_connection.mav.command_long_send.assert_called()
                
                # Test position command
                await self._send_position_command(interface)
                mock_mavlink_connection.mav.set_position_target_global_int_send.assert_called()
                
            except ImportError:
                pytest.skip("pymavlink not available for testing")
    
    async def _send_position_command(self, interface):
        """Helper method for position command testing."""
        await interface.send_position_command(
            latitude=TEST_HOME_POSITION['lat'],
            longitude=TEST_HOME_POSITION['lon'],
            altitude=TEST_HOME_POSITION['alt']
        )
    
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test handling of connection timeouts and reconnection."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn = MagicMock()
            mock_conn.wait_heartbeat = AsyncMock(side_effect=asyncio.TimeoutError("Connection timeout"))
            mock_conn_func.return_value = mock_conn
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                
                interface = MAVLinkInterface()
                
                # Should raise exception on timeout
                with pytest.raises(ConnectionError):
                    await interface.connect(TEST_MAVLINK_CONNECTION, timeout=1.0)
                
                assert not interface.is_connected
                
            except ImportError:
                pytest.skip("pymavlink not available for testing")
    
    @pytest.mark.asyncio
    async def test_heartbeat_monitoring(self, mock_mavlink_connection):
        """Test heartbeat monitoring and connection status tracking."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                interface.mav_conn = mock_mavlink_connection
                
                # Test heartbeat detection
                last_heartbeat = interface.last_heartbeat
                await interface._check_heartbeat()
                
                # Should update heartbeat timestamp
                assert interface.last_heartbeat >= last_heartbeat
                
            except ImportError:
                pytest.skip("pymavlink not available for testing")


class TestVehicleStateModels:
    """Test vehicle state models used in integration."""
    
    def test_vehicle_state_serialization(self):
        """Test vehicle state serialization for API communication."""
        try:
            from autonomy.vehicle_state import (
                VehicleState, Position, Velocity, Attitude, BatteryState,
                FlightMode, GPSState, VehicleHealth
            )
            
            vehicle_state = VehicleState(
                timestamp=datetime.now(),
                position=Position(
                    latitude=TEST_HOME_POSITION['lat'],
                    longitude=TEST_HOME_POSITION['lon'],
                    altitude_msl=TEST_HOME_POSITION['alt'],
                    altitude_agl=12.0
                ),
                velocity=Velocity(1.0, 2.0, 0.5),
                attitude=Attitude(0.1, 0.2, 3.14),
                battery=BatteryState(22.2, 6.5, 75.0),
                mode=FlightMode.GUIDED,
                armed=True,
                in_air=True,
                gps=GPSState(3, 10, 0.7, 0.8),
                health=VehicleHealth(True, True, True, True, True),
                home_position=Position(
                    latitude=TEST_HOME_POSITION['lat'],
                    longitude=TEST_HOME_POSITION['lon'],
                    altitude_msl=TEST_HOME_POSITION['alt'],
                    altitude_agl=0.0
                ),
            )
            
            # Test dict serialization
            state_dict = vehicle_state.to_dict()
            assert isinstance(state_dict, dict)
            assert 'position' in state_dict
            assert 'timestamp' in state_dict
            
            # Test JSON serialization
            import json
            json_str = json.dumps(state_dict, default=str)
            assert isinstance(json.loads(json_str), dict)
            
        except ImportError:
            pytest.skip("Vehicle state models not available for testing")
    
    def test_position_distance_calculations(self):
        """Test position distance calculation methods."""
        try:
            from autonomy.vehicle_state import Position
            
            pos1 = Position(47.397742, 8.545594, 488.0, 0.0)
            pos2 = Position(47.398500, 8.546500, 495.0, 12.0)
            
            # Test distance calculation
            distance = pos1.distance_to(pos2)
            assert distance > 0
            assert distance < 1000  # Should be reasonable for nearby coordinates
            
            # Test distance in 2D
            distance_2d = pos1.distance_2d_to(pos2)
            assert 0 < distance_2d < distance  # 2D distance should be less than 3D
            
        except ImportError:
            pytest.skip("Position model not available for testing")


class TestMissionPrimitivesIntegration:
    """Test integration of mission primitives with MAVLink interface."""
    
    @pytest.mark.asyncio
    async def test_takeoff_sequence(self, mock_mavlink_connection):
        """Test takeoff mission primitive."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                from autonomy.mission_primitives import TakeoffPrimitive
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                interface.mav_conn = mock_mavlink_connection
                
                # Test takeoff execution
                takeoff = TakeoffPrimitive(interface, altitude=10.0)
                success = await takeoff.execute()
                
                # Should send arm and takeoff commands
                assert mock_mavlink_connection.mav.command_long_send.called
                assert success is True
                
            except ImportError:
                pytest.skip("Mission primitives not available for testing")
    
    @pytest.mark.asyncio
    async def test_goto_navigation(self, mock_mavlink_connection):
        """Test goto navigation primitive."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                from autonomy.mission_primitives import GotoPrimitive
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                interface.mav_conn = mock_mavlink_connection
                
                # Test goto execution
                goto = GotoPrimitive(
                    interface,
                    latitude=TEST_HOME_POSITION['lat'],
                    longitude=TEST_HOME_POSITION['lon'],
                    altitude=TEST_HOME_POSITION['alt']
                )
                success = await goto.execute()
                
                # Should send position command
                assert mock_mavlink_connection.mav.set_position_target_global_int_send.called
                assert success is True
                
            except ImportError:
                pytest.skip("Mission primitives not available for testing")
    
    @pytest.mark.asyncio
    async def test_land_sequence(self, mock_mavlink_connection):
        """Test landing mission primitive."""
        with patch('pymavlink.mavutil.mavlink_connection') as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            
            try:
                from autonomy.mavlink_interface import MAVLinkInterface
                from autonomy.mission_primitives import LandPrimitive
                
                interface = MAVLinkInterface()
                await interface.connect(TEST_MAVLINK_CONNECTION)
                interface.mav_conn = mock_mavlink_connection
                
                # Test land execution
                land = LandPrimitive(interface)
                success = await land.execute()
                
                # Should send land command
                assert mock_mavlink_connection.mav.command_long_send.called
                assert success is True
                
            except ImportError:
                pytest.skip("Mission primitives not available for testing")