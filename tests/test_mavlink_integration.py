"""
Integration tests for MAVLink interface and vehicle communication.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import pymavlink  # noqa: F401

    from autonomy.mavlink_interface import (
        ConnectionState,
        MAVLinkConfig,
        MAVLinkInterface,
    )
    from autonomy.mission_primitives import (
        DockPlan,
        MissionPrimitives,
        OrbitPlan,
        PrimitiveConfig,
        PrimitiveResult,
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
    from tests.conftest import (
        TEST_HOME_POSITION,
        TEST_MAVLINK_COMPONENT_ID,
        TEST_MAVLINK_CONNECTION,
        TEST_MAVLINK_SYSTEM_ID,
    )
except ImportError:
    pytest.skip("pymavlink not installed", allow_module_level=True)


def _make_config() -> MAVLinkConfig:
    """Create a test MAVLink configuration payload."""
    return MAVLinkConfig(
        connection_string=TEST_MAVLINK_CONNECTION,
        source_system=TEST_MAVLINK_SYSTEM_ID,
        source_component=TEST_MAVLINK_COMPONENT_ID,
        timeout_ms=1000,
    )


def _mock_vehicle_state(altitude_agl: float, armed: bool) -> VehicleState:
    """Create a mock vehicle state for MAVLink tests."""
    return VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude_msl=TEST_HOME_POSITION["alt"],
            altitude_agl=altitude_agl,
        ),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(voltage=22.2, current=6.5, remaining_percent=75.0),
        mode=FlightMode.GUIDED,
        armed=armed,
        in_air=armed,
        gps=GPSState(fix_type=3, satellites_visible=10, hdop=0.7, vdop=0.8),
        health=VehicleHealth(
            sensors_healthy=True,
            gps_healthy=True,
            battery_healthy=True,
            motors_healthy=True,
            ekf_healthy=True,
        ),
        home_position=Position(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude_msl=TEST_HOME_POSITION["alt"],
            altitude_agl=0.0,
        ),
    )


class TestMAVLinkInterface:
    """Test MAVLink communication with mocked connection."""

    @pytest.mark.asyncio
    async def test_mavlink_connection_establishment(self, mock_mavlink_connection):
        """Test establishing MAVLink connection."""
        config = _make_config()
        with patch("pymavlink.mavutil.mavlink_connection") as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            interface = MAVLinkInterface(config)
            connected = await interface.connect()

            assert connected is True
            mock_conn_func.assert_called_once_with(
                config.connection_string,
                source_system=config.source_system,
                source_component=config.source_component,
            )
            mock_mavlink_connection.wait_heartbeat.assert_called_once_with(
                timeout=config.timeout_ms / 1000
            )
            assert interface.is_connected

            await interface.disconnect()

    @pytest.mark.asyncio
    async def test_telemetry_reception(self, mock_mavlink_connection):
        """Test receiving and processing telemetry data."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        interface._process_heartbeat(mock_mavlink_connection.messages["HEARTBEAT"])
        interface._process_global_position(mock_mavlink_connection.messages["GLOBAL_POSITION_INT"])
        interface._process_attitude(mock_mavlink_connection.messages["ATTITUDE"])
        interface._process_sys_status(mock_mavlink_connection.messages["SYS_STATUS"])
        interface._process_gps_raw(mock_mavlink_connection.messages["GPS_RAW_INT"])

        vehicle_state = interface.get_current_state()
        assert vehicle_state is not None
        assert abs(vehicle_state.position.latitude - TEST_HOME_POSITION["lat"]) < 0.0001
        assert abs(vehicle_state.position.longitude - TEST_HOME_POSITION["lon"]) < 0.0001

    @pytest.mark.asyncio
    async def test_command_sending(self, mock_mavlink_connection):
        """Test sending commands to vehicle via MAVLink."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        await interface.arm()
        mock_mavlink_connection.arducopter_arm.assert_called_once()

        await interface.disarm()
        mock_mavlink_connection.arducopter_disarm.assert_called_once()

        await interface.set_mode("GUIDED")
        mock_mavlink_connection.set_mode.assert_called_once_with(4)

        await interface.goto(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude=TEST_HOME_POSITION["alt"],
        )
        mock_mavlink_connection.mav.set_position_target_global_int_send.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_connection_timeout_handling(self):
        """Test handling of connection timeouts and reconnection."""
        with patch("pymavlink.mavutil.mavlink_connection") as mock_conn_func:
            mock_conn = MagicMock()
            mock_conn.wait_heartbeat.side_effect = TimeoutError("Connection timeout")
            mock_conn_func.return_value = mock_conn

            interface = MAVLinkInterface(_make_config())
            connected = await interface.connect()

            assert connected is False
            assert not interface.is_connected

    def test_heartbeat_updates_timestamp(self, mock_mavlink_connection):
        """Test heartbeat processing updates timestamps."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        assert interface._telemetry.status.last_heartbeat is None
        interface._process_heartbeat(mock_mavlink_connection.messages["HEARTBEAT"])
        assert interface._telemetry.status.last_heartbeat is not None


class TestVehicleStateModels:
    """Test vehicle state models used in integration."""

    def test_vehicle_state_serialization(self):
        """Test vehicle state serialization for API communication."""
        vehicle_state = _mock_vehicle_state(altitude_agl=12.0, armed=True)

        state_dict = vehicle_state.to_dict()
        assert isinstance(state_dict, dict)
        assert "position" in state_dict
        assert "timestamp" in state_dict

        json_str = json.dumps(state_dict, default=str)
        assert isinstance(json.loads(json_str), dict)

    def test_position_distance_calculations(self):
        """Test position distance calculation methods."""
        pos1 = Position(
            latitude=47.397742,
            longitude=8.545594,
            altitude_msl=488.0,
            altitude_agl=0.0,
        )
        pos2 = Position(
            latitude=47.398500,
            longitude=8.546500,
            altitude_msl=495.0,
            altitude_agl=12.0,
        )

        distance = pos1.distance_to(pos2)
        assert distance > 0
        assert distance < 1000

        pos2_horizontal = pos2.model_copy(update={"altitude_msl": pos1.altitude_msl})
        distance_2d = pos1.distance_to(pos2_horizontal)
        assert 0 < distance_2d < distance


class TestMissionPrimitivesIntegration:
    """Test integration of mission primitives with MAVLink interface."""

    @pytest.mark.asyncio
    async def test_takeoff_sequence(self, mock_mavlink_connection):
        """Test takeoff mission primitive."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        primitives = MissionPrimitives(interface)
        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                mock_state.return_value = _mock_vehicle_state(altitude_agl=10.0, armed=True)
                result = await primitives.arm_and_takeoff(altitude=10.0)

        assert result is not None
        assert mock_mavlink_connection.mav.command_long_send.called

    @pytest.mark.asyncio
    async def test_goto_navigation(self, mock_mavlink_connection):
        """Test goto navigation primitive."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        target = Position(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude_msl=TEST_HOME_POSITION["alt"],
        )
        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                mock_state.return_value = _mock_vehicle_state(altitude_agl=5.0, armed=True)
                result = await primitives.goto(target)

        assert result is not None
        assert mock_mavlink_connection.mav.set_position_target_global_int_send.called

    @pytest.mark.asyncio
    async def test_land_sequence(self, mock_mavlink_connection):
        """Test landing mission primitive."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"LAND": 9}

        primitives = MissionPrimitives(interface)
        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                mock_state.return_value = _mock_vehicle_state(altitude_agl=0.0, armed=False)
                result = await primitives.land()

        assert result is not None
        assert mock_mavlink_connection.set_mode.called


class TestMAVLinkInterfaceExtended:
    """Extended tests for MAVLink interface to increase coverage."""

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test disconnect when no connection exists."""
        interface = MAVLinkInterface(_make_config())
        # Should not raise even when not connected
        await interface.disconnect()
        assert interface.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_with_active_tasks(self, mock_mavlink_connection):
        """Test disconnect properly cancels background tasks."""
        config = _make_config()
        with patch("pymavlink.mavutil.mavlink_connection") as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            interface = MAVLinkInterface(config)
            await interface.connect()

            # Verify tasks were created
            assert interface._tasks.receive_task is not None
            assert interface._tasks.heartbeat_task is not None

            # Disconnect should cancel tasks
            await interface.disconnect()

            assert not interface.is_connected
            assert interface._connection is None

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_already_connected_warning(self, mock_mavlink_connection):
        """Test that connecting when already connected triggers disconnect first."""
        config = _make_config()
        with patch("pymavlink.mavutil.mavlink_connection") as mock_conn_func:
            mock_conn_func.return_value = mock_mavlink_connection
            interface = MAVLinkInterface(config)

            # First connection
            await interface.connect()
            assert interface.is_connected

            # Second connection should trigger disconnect first
            await interface.connect()
            assert interface.is_connected

            await interface.disconnect()

    @pytest.mark.asyncio
    async def test_arm_without_connection(self):
        """Test arm command returns False when not connected."""
        interface = MAVLinkInterface(_make_config())
        result = await interface.arm()
        assert result is False

    @pytest.mark.asyncio
    async def test_disarm_without_connection(self):
        """Test disarm command returns False when not connected."""
        interface = MAVLinkInterface(_make_config())
        result = await interface.disarm()
        assert result is False

    @pytest.mark.asyncio
    async def test_set_mode_without_connection(self):
        """Test set_mode returns False when not connected."""
        interface = MAVLinkInterface(_make_config())
        result = await interface.set_mode("GUIDED")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_set_mode_unknown_mode(self, mock_mavlink_connection):
        """Test set_mode returns False for unknown mode."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        result = await interface.set_mode("NONEXISTENT_MODE")
        assert result is False

    @pytest.mark.asyncio
    async def test_takeoff_without_connection(self):
        """Test takeoff command returns False when not connected."""
        interface = MAVLinkInterface(_make_config())
        result = await interface.takeoff(10.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_goto_without_connection(self):
        """Test goto command returns False when not connected."""
        interface = MAVLinkInterface(_make_config())
        result = await interface.goto(47.0, 8.0, 100.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_goto_with_ground_speed(self, mock_mavlink_connection):
        """Test goto command with optional ground speed parameter."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        result = await interface.goto(47.0, 8.0, 100.0, ground_speed=5.0)
        assert result is True
        mock_mavlink_connection.mav.set_position_target_global_int_send.assert_called()

    @pytest.mark.asyncio
    async def test_land_command(self, mock_mavlink_connection):
        """Test land command uses set_mode with LAND."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"LAND": 9}

        result = await interface.land()
        assert result is True
        mock_mavlink_connection.set_mode.assert_called_once_with(9)

    @pytest.mark.asyncio
    async def test_return_to_launch_command(self, mock_mavlink_connection):
        """Test RTL command uses set_mode with RTL."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"RTL": 6}

        result = await interface.return_to_launch()
        assert result is True
        mock_mavlink_connection.set_mode.assert_called_once_with(6)

    def test_get_current_state_missing_data(self):
        """Test get_current_state returns None when data is incomplete."""
        interface = MAVLinkInterface(_make_config())
        # No telemetry data set
        state = interface.get_current_state()
        assert state is None

    def test_get_current_state_partial_data(self, mock_mavlink_connection):
        """Test get_current_state returns None when only partial data."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        # Only process position, not all required data
        interface._process_global_position(mock_mavlink_connection.messages["GLOBAL_POSITION_INT"])

        state = interface.get_current_state()
        assert state is None  # Missing attitude and battery

    def test_get_current_state_complete_data(self, mock_mavlink_connection):
        """Test get_current_state returns state when all data available."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        # Process all required telemetry
        interface._process_global_position(mock_mavlink_connection.messages["GLOBAL_POSITION_INT"])
        interface._process_attitude(mock_mavlink_connection.messages["ATTITUDE"])
        interface._process_sys_status(mock_mavlink_connection.messages["SYS_STATUS"])

        state = interface.get_current_state()
        assert state is not None
        assert state.position is not None
        assert state.velocity is not None
        assert state.attitude is not None
        assert state.battery is not None

    def test_process_message_routing(self, mock_mavlink_connection):
        """Test _process_message routes messages to correct handlers."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        # Create message mocks with get_type method
        heartbeat_msg = MagicMock()
        heartbeat_msg.get_type.return_value = "HEARTBEAT"
        heartbeat_msg.base_mode = 217
        heartbeat_msg.custom_mode = 4

        position_msg = MagicMock()
        position_msg.get_type.return_value = "GLOBAL_POSITION_INT"
        position_msg.lat = int(47.0 * 1e7)
        position_msg.lon = int(8.0 * 1e7)
        position_msg.alt = 100000
        position_msg.relative_alt = 10000
        position_msg.vx = 0
        position_msg.vy = 0
        position_msg.vz = 0

        local_pos_msg = MagicMock()
        local_pos_msg.get_type.return_value = "LOCAL_POSITION_NED"

        attitude_msg = MagicMock()
        attitude_msg.get_type.return_value = "ATTITUDE"
        attitude_msg.roll = 0.1
        attitude_msg.pitch = 0.2
        attitude_msg.yaw = 1.5

        sys_status_msg = MagicMock()
        sys_status_msg.get_type.return_value = "SYS_STATUS"
        sys_status_msg.voltage_battery = 22800
        sys_status_msg.current_battery = 500
        sys_status_msg.battery_remaining = 80

        gps_msg = MagicMock()
        gps_msg.get_type.return_value = "GPS_RAW_INT"
        gps_msg.fix_type = 3
        gps_msg.satellites_visible = 10
        gps_msg.eph = 80
        gps_msg.epv = 100

        home_msg = MagicMock()
        home_msg.get_type.return_value = "HOME_POSITION"
        home_msg.latitude = int(47.0 * 1e7)
        home_msg.longitude = int(8.0 * 1e7)
        home_msg.altitude = 488000

        # Process all message types
        interface._process_message(heartbeat_msg)
        interface._process_message(position_msg)
        interface._process_message(local_pos_msg)
        interface._process_message(attitude_msg)
        interface._process_message(sys_status_msg)
        interface._process_message(gps_msg)
        interface._process_message(home_msg)

        # Verify data was processed
        assert interface._telemetry.core.position is not None
        assert interface._telemetry.core.attitude is not None
        assert interface._telemetry.core.battery is not None
        assert interface._telemetry.core.gps is not None
        assert interface._telemetry.status.home_position is not None

    def test_process_gps_raw_max_values(self, mock_mavlink_connection):
        """Test GPS processing with max DOP values (65535)."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        gps_msg = MagicMock()
        gps_msg.fix_type = 3
        gps_msg.satellites_visible = 10
        gps_msg.eph = 65535  # Invalid value
        gps_msg.epv = 65535  # Invalid value

        interface._process_gps_raw(gps_msg)

        assert interface._telemetry.core.gps is not None
        assert interface._telemetry.core.gps.hdop == 99.9
        assert interface._telemetry.core.gps.vdop == 99.9

    def test_process_sys_status_negative_current(self, mock_mavlink_connection):
        """Test sys_status processing with negative current (invalid)."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        sys_status_msg = MagicMock()
        sys_status_msg.voltage_battery = 22800
        sys_status_msg.current_battery = -1  # Invalid
        sys_status_msg.battery_remaining = -1  # Invalid

        interface._process_sys_status(sys_status_msg)

        assert interface._telemetry.core.battery is not None
        assert interface._telemetry.core.battery.current == 0
        assert interface._telemetry.core.battery.remaining_percent == 0

    def test_heartbeat_mode_mapping(self, mock_mavlink_connection):
        """Test heartbeat processing extracts correct mode."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4, "AUTO": 3}

        heartbeat_msg = MagicMock()
        heartbeat_msg.base_mode = 217  # Armed
        heartbeat_msg.custom_mode = 4  # GUIDED

        interface._process_heartbeat(heartbeat_msg)

        assert interface._telemetry.status.mode == FlightMode.GUIDED
        assert interface._telemetry.status.armed is True

    def test_heartbeat_unknown_mode(self, mock_mavlink_connection):
        """Test heartbeat processing with unknown mode."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        heartbeat_msg = MagicMock()
        heartbeat_msg.base_mode = 0  # Not armed
        heartbeat_msg.custom_mode = 999  # Unknown mode

        interface._process_heartbeat(heartbeat_msg)

        assert interface._telemetry.status.mode == FlightMode.UNKNOWN
        assert interface._telemetry.status.armed is False

    def test_heartbeat_restores_connection_from_lost(self, mock_mavlink_connection):
        """Test heartbeat restores connection state from LOST."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        # Set state to LOST
        interface._state = ConnectionState.LOST

        heartbeat_msg = MagicMock()
        heartbeat_msg.base_mode = 217
        heartbeat_msg.custom_mode = 4

        interface._process_heartbeat(heartbeat_msg)

        assert interface._state == ConnectionState.CONNECTED

    def test_on_state_update_callback(self, mock_mavlink_connection):
        """Test state update callbacks are invoked."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        callback_called = []

        def test_callback(state):
            callback_called.append(state)

        interface.on_state_update(test_callback)

        # Process all required telemetry to trigger callback
        interface._process_global_position(mock_mavlink_connection.messages["GLOBAL_POSITION_INT"])
        interface._process_attitude(mock_mavlink_connection.messages["ATTITUDE"])
        interface._process_sys_status(mock_mavlink_connection.messages["SYS_STATUS"])

        # Process another position to trigger callback
        interface._process_global_position(mock_mavlink_connection.messages["GLOBAL_POSITION_INT"])

        assert len(callback_called) > 0

    def test_on_connection_change_callback(self):
        """Test connection state change callbacks are invoked."""
        interface = MAVLinkInterface(_make_config())

        callback_states = []

        def test_callback(state):
            callback_states.append(state)

        interface.on_connection_change(test_callback)

        # Trigger state changes
        interface._set_state(ConnectionState.CONNECTING)
        interface._set_state(ConnectionState.CONNECTED)
        interface._set_state(ConnectionState.LOST)

        assert len(callback_states) == 3
        assert callback_states[0] == ConnectionState.CONNECTING
        assert callback_states[1] == ConnectionState.CONNECTED
        assert callback_states[2] == ConnectionState.LOST

    @pytest.mark.allow_error_logs
    def test_callback_exception_handling(self, mock_mavlink_connection):
        """Test that callback exceptions are caught and don't crash."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        def bad_callback(_state):
            raise RuntimeError("Callback error")

        interface.on_connection_change(bad_callback)

        # Should not raise even with bad callback
        interface._set_state(ConnectionState.CONNECTED)

    def test_set_state_no_change(self):
        """Test _set_state does not invoke callbacks when state unchanged."""
        interface = MAVLinkInterface(_make_config())

        callback_called = []

        def test_callback(state):
            callback_called.append(state)

        interface.on_connection_change(test_callback)

        # Set to same state twice
        interface._set_state(ConnectionState.DISCONNECTED)
        interface._set_state(ConnectionState.DISCONNECTED)

        # Initial state is DISCONNECTED, so no change
        assert len(callback_called) == 0

    def test_is_connected_property(self):
        """Test is_connected property returns correct value."""
        interface = MAVLinkInterface(_make_config())

        assert interface.is_connected is False

        interface._state = ConnectionState.CONNECTING
        assert interface.is_connected is False

        interface._state = ConnectionState.CONNECTED
        assert interface.is_connected is True

        interface._state = ConnectionState.LOST
        assert interface.is_connected is False

    def test_state_property(self):
        """Test state property returns current connection state."""
        interface = MAVLinkInterface(_make_config())

        assert interface.state == ConnectionState.DISCONNECTED

        interface._state = ConnectionState.CONNECTED
        assert interface.state == ConnectionState.CONNECTED


class TestMissionPrimitivesExtended:
    """Extended tests for MissionPrimitives to increase coverage."""

    @pytest.mark.asyncio
    async def test_abort_request_and_clear(self):
        """Test abort request and clear methods."""
        interface = MAVLinkInterface(_make_config())
        primitives = MissionPrimitives(interface)

        assert primitives._abort_requested is False

        primitives.request_abort()
        assert primitives._abort_requested is True

        primitives.clear_abort()
        assert primitives._abort_requested is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_arm_and_takeoff_set_mode_failure(self, mock_mavlink_connection):
        """Test arm_and_takeoff fails when set_mode fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {}  # Empty = mode not found

        primitives = MissionPrimitives(interface)
        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            result = await primitives.arm_and_takeoff(altitude=10.0)

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_arm_and_takeoff_arm_failure(self, mock_mavlink_connection):
        """Test arm_and_takeoff fails when arm fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}
        mock_mavlink_connection.arducopter_arm.side_effect = Exception("Arm failed")

        primitives = MissionPrimitives(interface)
        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            # Simulate arm returning False
            with patch.object(interface, "arm", return_value=False):
                result = await primitives.arm_and_takeoff(altitude=10.0)

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_arm_and_takeoff_takeoff_failure(self, mock_mavlink_connection):
        """Test arm_and_takeoff fails when takeoff command fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        primitives = MissionPrimitives(interface)
        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "takeoff", return_value=False):
                result = await primitives.arm_and_takeoff(altitude=10.0)

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    async def test_arm_and_takeoff_timeout(self, mock_mavlink_connection):
        """Test arm_and_takeoff times out waiting for altitude."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                # Return state at wrong altitude (never reaches target)
                mock_state.return_value = _mock_vehicle_state(altitude_agl=2.0, armed=True)
                result = await primitives.arm_and_takeoff(altitude=10.0, timeout=0.05)

        assert result == PrimitiveResult.TIMEOUT

    @pytest.mark.asyncio
    async def test_arm_and_takeoff_abort(self, mock_mavlink_connection):
        """Test arm_and_takeoff can be aborted during wait phase."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        call_count = [0]

        def mock_get_state():
            call_count[0] += 1
            # Request abort after first call (during wait loop)
            if call_count[0] >= 2:
                primitives.request_abort()
            return _mock_vehicle_state(altitude_agl=2.0, armed=True)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state", side_effect=mock_get_state):
                result = await primitives.arm_and_takeoff(altitude=10.0, timeout=10.0)

        assert result == PrimitiveResult.ABORTED

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_goto_failure(self):
        """Test goto fails when goto command fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = None  # No connection

        primitives = MissionPrimitives(interface)
        target = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            result = await primitives.goto(target)

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    async def test_goto_timeout(self, mock_mavlink_connection):
        """Test goto times out waiting for position."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        # Target far away
        target = Position(latitude=48.0, longitude=9.0, altitude_msl=100.0)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                mock_state.return_value = _mock_vehicle_state(altitude_agl=10.0, armed=True)
                result = await primitives.goto(target, timeout=0.05)

        assert result == PrimitiveResult.TIMEOUT

    @pytest.mark.asyncio
    async def test_goto_abort(self, mock_mavlink_connection):
        """Test goto can be aborted during wait phase."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)

        target = Position(latitude=48.0, longitude=9.0, altitude_msl=100.0)

        call_count = [0]

        def mock_get_state():
            call_count[0] += 1
            # Request abort after first call (during wait loop)
            if call_count[0] >= 2:
                primitives.request_abort()
            return _mock_vehicle_state(altitude_agl=10.0, armed=True)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state", side_effect=mock_get_state):
                result = await primitives.goto(target, timeout=10.0)

        assert result == PrimitiveResult.ABORTED

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_land_failure(self, mock_mavlink_connection):
        """Test land fails when land command fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {}  # LAND not found

        primitives = MissionPrimitives(interface)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            result = await primitives.land()

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    async def test_land_timeout(self, mock_mavlink_connection):
        """Test land times out waiting for disarm."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"LAND": 9}

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                # Vehicle stays armed
                mock_state.return_value = _mock_vehicle_state(altitude_agl=5.0, armed=True)
                result = await primitives.land(timeout=0.05)

        assert result == PrimitiveResult.TIMEOUT

    @pytest.mark.asyncio
    async def test_land_abort(self, mock_mavlink_connection):
        """Test land can be aborted during wait phase."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"LAND": 9}

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        call_count = [0]

        def mock_get_state():
            call_count[0] += 1
            # Request abort after first call (during wait loop)
            if call_count[0] >= 2:
                primitives.request_abort()
            return _mock_vehicle_state(altitude_agl=5.0, armed=True)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state", side_effect=mock_get_state):
                result = await primitives.land(timeout=10.0)

        assert result == PrimitiveResult.ABORTED

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_return_to_launch_failure(self, mock_mavlink_connection):
        """Test return_to_launch fails when RTL mode fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {}  # RTL not found

        primitives = MissionPrimitives(interface)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            result = await primitives.return_to_launch()

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    async def test_return_to_launch_success(self, mock_mavlink_connection):
        """Test successful return_to_launch."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"RTL": 6}

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                # Vehicle lands and disarms
                mock_state.return_value = _mock_vehicle_state(altitude_agl=0.0, armed=False)
                result = await primitives.return_to_launch()

        assert result == PrimitiveResult.SUCCESS

    @pytest.mark.asyncio
    async def test_orbit_success(self, mock_mavlink_connection):
        """Test successful orbit around a point."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.001
        primitives.config.position_tolerance_m = 100000  # Very large tolerance

        center = Position(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude_msl=TEST_HOME_POSITION["alt"],
        )
        plan = OrbitPlan(radius=50.0, altitude_agl=20.0, orbits=1, speed=5.0)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                mock_state.return_value = _mock_vehicle_state(altitude_agl=20.0, armed=True)
                result = await primitives.orbit(center, plan)

        assert result == PrimitiveResult.SUCCESS

    @pytest.mark.asyncio
    async def test_orbit_counter_clockwise(self, mock_mavlink_connection):
        """Test orbit in counter-clockwise direction."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.001
        primitives.config.position_tolerance_m = 100000

        center = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)
        plan = OrbitPlan(radius=50.0, altitude_agl=20.0, orbits=1, clockwise=False)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state") as mock_state:
                mock_state.return_value = _mock_vehicle_state(altitude_agl=20.0, armed=True)
                result = await primitives.orbit(center, plan)

        assert result == PrimitiveResult.SUCCESS

    @pytest.mark.asyncio
    async def test_orbit_abort(self, mock_mavlink_connection):
        """Test orbit can be aborted mid-execution."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.001
        primitives.config.position_tolerance_m = 100000

        center = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)
        plan = OrbitPlan(radius=50.0, altitude_agl=20.0, orbits=5)  # Many orbits

        call_count = [0]

        async def mock_goto_with_abort(_target, _speed=None, _timeout=None):
            call_count[0] += 1
            if call_count[0] >= 3:
                primitives.request_abort()
            await asyncio.sleep(0)
            return PrimitiveResult.SUCCESS

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(primitives, "goto", mock_goto_with_abort):
                result = await primitives.orbit(center, plan)

        assert result == PrimitiveResult.ABORTED

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_orbit_waypoint_failure(self, mock_mavlink_connection):
        """Test orbit fails when a waypoint fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)

        center = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)
        plan = OrbitPlan(radius=50.0, altitude_agl=20.0, orbits=1)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(primitives, "goto", return_value=PrimitiveResult.FAILED):
                result = await primitives.orbit(center, plan)

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    async def test_dock_success(self, mock_mavlink_connection):
        """Test successful dock operation."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"LAND": 9}

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.001
        primitives.config.position_tolerance_m = 100000

        dock_position = Position(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude_msl=TEST_HOME_POSITION["alt"],
        )

        async def mock_goto(_target, _speed=None, _timeout=None):
            await asyncio.sleep(0)
            return PrimitiveResult.SUCCESS

        async def mock_land(_timeout=None):
            await asyncio.sleep(0)
            return PrimitiveResult.SUCCESS

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(primitives, "goto", mock_goto):
                with patch.object(primitives, "land", mock_land):
                    result = await primitives.dock(dock_position)

        assert result == PrimitiveResult.SUCCESS

    @pytest.mark.asyncio
    async def test_dock_with_custom_plan(self, mock_mavlink_connection):
        """Test dock with custom DockPlan parameters."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        mock_mavlink_connection.mode_mapping.return_value = {"LAND": 9}

        primitives = MissionPrimitives(interface)

        dock_position = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)
        plan = DockPlan(approach_altitude=15.0, approach_speed=3.0, landing_speed=0.3)

        async def mock_goto(_target, _speed=None, _timeout=None):
            await asyncio.sleep(0)
            return PrimitiveResult.SUCCESS

        async def mock_land(_timeout=None):
            await asyncio.sleep(0)
            return PrimitiveResult.SUCCESS

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(primitives, "goto", mock_goto):
                with patch.object(primitives, "land", mock_land):
                    result = await primitives.dock(dock_position, plan)

        assert result == PrimitiveResult.SUCCESS

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_dock_approach_failure(self, mock_mavlink_connection):
        """Test dock fails when approach goto fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)

        dock_position = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(primitives, "goto", return_value=PrimitiveResult.FAILED):
                result = await primitives.dock(dock_position)

        assert result == PrimitiveResult.FAILED

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_dock_land_failure(self, mock_mavlink_connection):
        """Test dock fails when landing fails."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)

        dock_position = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)

        async def mock_goto(_target, _speed=None, _timeout=None):
            await asyncio.sleep(0)
            return PrimitiveResult.SUCCESS

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(primitives, "goto", mock_goto):
                with patch.object(primitives, "land", return_value=PrimitiveResult.FAILED):
                    result = await primitives.dock(dock_position)

        assert result == PrimitiveResult.FAILED

    def test_calculate_orbit_waypoint(self):
        """Test orbit waypoint calculation."""
        center = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)
        plan = OrbitPlan(radius=100.0, altitude_agl=50.0)

        # Test at angle 0
        waypoint = MissionPrimitives._calculate_orbit_waypoint(center, plan, 0.0)
        assert waypoint is not None
        assert waypoint.altitude_msl == 150.0  # center alt + altitude_agl

        # Test at angle pi/2 (90 degrees)
        import math

        waypoint2 = MissionPrimitives._calculate_orbit_waypoint(center, plan, math.pi / 2)
        assert waypoint2 is not None
        assert waypoint2.latitude != waypoint.latitude or waypoint2.longitude != waypoint.longitude

    @pytest.mark.asyncio
    async def test_wait_for_altitude_no_state(self, mock_mavlink_connection):
        """Test _wait_for_altitude handles missing state gracefully."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state", return_value=None):
                result = await primitives._wait_for_altitude(10.0, timeout=0.05)

        assert result == PrimitiveResult.TIMEOUT

    @pytest.mark.asyncio
    async def test_wait_for_position_no_state(self, mock_mavlink_connection):
        """Test _wait_for_position handles missing state gracefully."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        target = Position(latitude=47.0, longitude=8.0, altitude_msl=100.0)

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state", return_value=None):
                result = await primitives._wait_for_position(target, timeout=0.05)

        assert result == PrimitiveResult.TIMEOUT

    @pytest.mark.asyncio
    async def test_wait_for_disarm_no_state(self, mock_mavlink_connection):
        """Test _wait_for_disarm handles missing state gracefully."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        primitives = MissionPrimitives(interface)
        primitives.config.poll_interval_s = 0.01

        with patch("autonomy.mission_primitives.asyncio.sleep", new=AsyncMock()):
            with patch.object(interface, "get_current_state", return_value=None):
                result = await primitives._wait_for_disarm(timeout=0.05)

        assert result == PrimitiveResult.TIMEOUT

    def test_primitive_config_defaults(self):
        """Test PrimitiveConfig has correct defaults."""
        config = PrimitiveConfig()
        assert config.position_tolerance_m == 2.0
        assert config.altitude_tolerance_m == 1.0
        assert config.heading_tolerance_deg == 10.0
        assert config.default_timeout_s == 120.0
        assert config.poll_interval_s == 0.5

    def test_orbit_plan_defaults(self):
        """Test OrbitPlan has correct defaults."""
        plan = OrbitPlan(radius=50.0, altitude_agl=20.0)
        assert plan.orbits == 1
        assert plan.speed == 5.0
        assert plan.clockwise is True
        assert plan.timeout_s is None

    def test_dock_plan_defaults(self):
        """Test DockPlan has correct defaults."""
        plan = DockPlan()
        assert plan.approach_altitude == 10.0
        assert plan.approach_speed == 2.0
        assert plan.landing_speed == 0.5
        assert plan.timeout_s is None


class TestBackgroundLoops:
    """Test background loop functionality."""

    @pytest.mark.asyncio
    async def test_receive_loop_processes_messages(self, mock_mavlink_connection):
        """Test receive loop processes incoming messages."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = True

        # Create a message to receive
        heartbeat_msg = MagicMock()
        heartbeat_msg.get_type.return_value = "HEARTBEAT"
        heartbeat_msg.base_mode = 217
        heartbeat_msg.custom_mode = 4
        mock_mavlink_connection.mode_mapping.return_value = {"GUIDED": 4}

        # First call returns message, subsequent calls return None
        call_count = [0]

        def mock_recv(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return heartbeat_msg
            # Stop the loop
            interface._running = False
            return None

        mock_mavlink_connection.recv_match.side_effect = mock_recv

        await interface._receive_loop()

        # Verify message was processed
        assert call_count[0] >= 1

    @pytest.mark.asyncio
    async def test_receive_loop_handles_no_message(self, mock_mavlink_connection):
        """Test receive loop handles None messages by sleeping."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = True

        call_count = [0]

        def mock_recv(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                interface._running = False
            return None

        mock_mavlink_connection.recv_match.side_effect = mock_recv

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await interface._receive_loop()
            # Should have called sleep when no message
            mock_sleep.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_receive_loop_handles_exception(self, mock_mavlink_connection):
        """Test receive loop handles exceptions gracefully."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = True

        call_count = [0]

        def mock_recv(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Receive error")
            interface._running = False
            return None

        mock_mavlink_connection.recv_match.side_effect = mock_recv

        with patch("asyncio.sleep", new=AsyncMock()):
            await interface._receive_loop()

        # Loop should have continued after exception
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_heartbeat_loop_sends_heartbeat(self, mock_mavlink_connection):
        """Test heartbeat loop sends heartbeats."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = True
        interface.config.heartbeat_interval_s = 0.01

        call_count = [0]
        real_sleep = asyncio.sleep

        async def mock_sleep(_duration):
            call_count[0] += 1
            if call_count[0] >= 2:
                interface._running = False
            await real_sleep(0)

        with patch("asyncio.sleep", mock_sleep):
            await interface._heartbeat_loop()

        # Verify heartbeat was sent
        mock_mavlink_connection.mav.heartbeat_send.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_heartbeat_loop_detects_timeout(self, mock_mavlink_connection):
        """Test heartbeat loop detects connection timeout."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = True
        interface._state = ConnectionState.CONNECTED
        interface.config.heartbeat_interval_s = 0.01

        # Set last heartbeat to 10 seconds ago (exceeds 5s timeout)
        from datetime import timedelta

        interface._telemetry.status.last_heartbeat = datetime.now() - timedelta(seconds=10)

        call_count = [0]
        real_sleep = asyncio.sleep

        async def mock_sleep(_duration):
            call_count[0] += 1
            if call_count[0] >= 2:
                interface._running = False
            await real_sleep(0)

        with patch("asyncio.sleep", mock_sleep):
            await interface._heartbeat_loop()

        # Should have transitioned to LOST state
        assert interface._state == ConnectionState.LOST

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_heartbeat_loop_handles_exception(self, mock_mavlink_connection):
        """Test heartbeat loop handles exceptions gracefully."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = True
        interface.config.heartbeat_interval_s = 0.01

        call_count = [0]

        def mock_heartbeat_send(*_args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Heartbeat send error")
            interface._running = False

        mock_mavlink_connection.mav.heartbeat_send.side_effect = mock_heartbeat_send

        with patch("asyncio.sleep", new=AsyncMock()):
            await interface._heartbeat_loop()

        # Loop should have continued after exception
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_receive_loop_stops_when_not_running(self, mock_mavlink_connection):
        """Test receive loop stops when _running is False."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = False  # Not running

        await interface._receive_loop()
        # Should return immediately
        mock_mavlink_connection.recv_match.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_loop_stops_when_not_running(self, mock_mavlink_connection):
        """Test heartbeat loop stops when _running is False."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection
        interface._running = False  # Not running

        await interface._heartbeat_loop()
        # Should return immediately
        mock_mavlink_connection.mav.heartbeat_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_receive_loop_stops_when_no_connection(self):
        """Test receive loop stops when connection is None."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = None  # No connection
        interface._running = True

        await interface._receive_loop()
        # Should return immediately

    @pytest.mark.asyncio
    async def test_heartbeat_loop_stops_when_no_connection(self):
        """Test heartbeat loop stops when connection is None."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = None  # No connection
        interface._running = True

        await interface._heartbeat_loop()
        # Should return immediately

    @pytest.mark.allow_error_logs
    def test_state_callback_exception_in_global_position(self, mock_mavlink_connection):
        """Test state callback exceptions are caught in _process_global_position."""
        interface = MAVLinkInterface(_make_config())
        interface._connection = mock_mavlink_connection

        # Set up all required telemetry data first
        interface._process_attitude(mock_mavlink_connection.messages["ATTITUDE"])
        interface._process_sys_status(mock_mavlink_connection.messages["SYS_STATUS"])

        def bad_callback(_state):
            raise RuntimeError("Callback error")

        interface.on_state_update(bad_callback)

        # Now process global position which triggers state callback
        # Should not raise even with bad callback
        interface._process_global_position(mock_mavlink_connection.messages["GLOBAL_POSITION_INT"])
        interface._process_global_position(mock_mavlink_connection.messages["GLOBAL_POSITION_INT"])


class TestConnectionState:
    """Test ConnectionState enum."""

    def test_connection_state_values(self):
        """Test all ConnectionState values exist."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.LOST.value == "lost"


class TestPrimitiveResult:
    """Test PrimitiveResult enum."""

    def test_primitive_result_values(self):
        """Test all PrimitiveResult values exist."""
        assert PrimitiveResult.SUCCESS.value == "success"
        assert PrimitiveResult.TIMEOUT.value == "timeout"
        assert PrimitiveResult.ABORTED.value == "aborted"
        assert PrimitiveResult.FAILED.value == "failed"
        assert PrimitiveResult.IN_PROGRESS.value == "in_progress"
