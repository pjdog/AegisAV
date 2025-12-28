"""
Integration tests for MAVLink interface and vehicle communication.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import pymavlink  # noqa: F401

    from autonomy.mavlink_interface import MAVLinkConfig, MAVLinkInterface
    from autonomy.mission_primitives import MissionPrimitives
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
