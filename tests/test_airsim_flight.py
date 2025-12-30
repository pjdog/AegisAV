"""
Tests for AirSim flight control integration.

Tests coordinate conversion, action execution, and flight command translation.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simulation.airsim_action_executor import (
    AirSimActionExecutor,
    ExecutionResult,
    ExecutionStatus,
    FlightConfig,
)
from simulation.coordinate_utils import (
    GeoReference,
    haversine_distance,
    initial_bearing,
)


# =============================================================================
# Coordinate Utils Tests
# =============================================================================


class TestGeoReference:
    """Tests for GeoReference coordinate conversion."""

    @pytest.fixture
    def reference(self) -> GeoReference:
        """Create a reference point near Redmond, WA (AirSim default)."""
        return GeoReference(
            latitude=47.641468,
            longitude=-122.140165,
            altitude=0.0,
        )

    def test_gps_to_ned_at_origin(self, reference: GeoReference) -> None:
        """Origin should convert to (0, 0, 0)."""
        north, east, down = reference.gps_to_ned(
            reference.latitude,
            reference.longitude,
            reference.altitude,
        )
        assert abs(north) < 0.01
        assert abs(east) < 0.01
        assert abs(down) < 0.01

    def test_gps_to_ned_north(self, reference: GeoReference) -> None:
        """Moving north should increase north coordinate."""
        # Move ~111m north (approximately 0.001 degrees)
        north, east, down = reference.gps_to_ned(
            reference.latitude + 0.001,
            reference.longitude,
            reference.altitude,
        )
        assert north > 100  # Should be about 111m
        assert north < 120
        assert abs(east) < 1  # Should not move east
        assert abs(down) < 0.01

    def test_gps_to_ned_east(self, reference: GeoReference) -> None:
        """Moving east should increase east coordinate."""
        # Move east (approximately 0.001 degrees)
        north, east, down = reference.gps_to_ned(
            reference.latitude,
            reference.longitude + 0.001,
            reference.altitude,
        )
        assert abs(north) < 1  # Should not move north
        assert east > 50  # Should be positive (east)
        assert east < 100  # Less than north movement at this latitude

    def test_gps_to_ned_altitude(self, reference: GeoReference) -> None:
        """Higher altitude should give negative down (NED convention)."""
        north, east, down = reference.gps_to_ned(
            reference.latitude,
            reference.longitude,
            reference.altitude + 30.0,  # 30m higher
        )
        assert abs(north) < 0.01
        assert abs(east) < 0.01
        assert down == pytest.approx(-30.0, abs=0.01)

    def test_roundtrip_conversion(self, reference: GeoReference) -> None:
        """GPS -> NED -> GPS should return original coordinates."""
        original_lat = 47.642
        original_lon = -122.139
        original_alt = 30.0

        # Convert to NED
        north, east, down = reference.gps_to_ned(original_lat, original_lon, original_alt)

        # Convert back to GPS
        lat, lon, alt = reference.ned_to_gps(north, east, down)

        assert lat == pytest.approx(original_lat, abs=0.0001)
        assert lon == pytest.approx(original_lon, abs=0.0001)
        assert alt == pytest.approx(original_alt, abs=0.01)

    def test_distance_ned(self, reference: GeoReference) -> None:
        """Test NED distance calculation."""
        ned1 = (0.0, 0.0, 0.0)
        ned2 = (30.0, 40.0, 0.0)  # 3-4-5 triangle

        distance = reference.distance_ned(ned1, ned2)
        assert distance == pytest.approx(50.0, abs=0.01)

    def test_bearing_to(self, reference: GeoReference) -> None:
        """Test bearing calculation."""
        origin = (0.0, 0.0, 0.0)

        # Due north
        north_bearing = reference.bearing_to(origin, (100.0, 0.0, 0.0))
        assert north_bearing == pytest.approx(0.0, abs=0.1)

        # Due east
        east_bearing = reference.bearing_to(origin, (0.0, 100.0, 0.0))
        assert east_bearing == pytest.approx(90.0, abs=0.1)

        # Due south
        south_bearing = reference.bearing_to(origin, (-100.0, 0.0, 0.0))
        assert south_bearing == pytest.approx(180.0, abs=0.1)


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point(self) -> None:
        """Same point should have zero distance."""
        distance = haversine_distance(47.641468, -122.140165, 47.641468, -122.140165)
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self) -> None:
        """Test with known distance (roughly 1km)."""
        # Approximately 1km north
        distance = haversine_distance(
            47.641468, -122.140165,
            47.650468, -122.140165,
        )
        # Should be approximately 1km (9 * 111m per degree latitude)
        assert 900 < distance < 1100


class TestInitialBearing:
    """Tests for initial bearing calculation."""

    def test_due_north(self) -> None:
        """Bearing due north should be 0 degrees."""
        bearing = initial_bearing(47.641468, -122.140165, 47.651468, -122.140165)
        assert bearing == pytest.approx(0.0, abs=1.0)

    def test_due_east(self) -> None:
        """Bearing due east should be 90 degrees."""
        bearing = initial_bearing(47.641468, -122.140165, 47.641468, -122.130165)
        assert bearing == pytest.approx(90.0, abs=1.0)


# =============================================================================
# Mock AirSim Bridge
# =============================================================================


@dataclass
class MockPose:
    """Mock pose for testing."""

    @dataclass
    class Position:
        x: float = 0.0
        y: float = 0.0
        z: float = -30.0

    position: Position = field(default_factory=Position)


@dataclass
class MockState:
    """Mock synchronized state for testing."""

    pose: MockPose = field(default_factory=MockPose)


class MockRealtimeAirSimBridge:
    """Mock AirSim bridge for testing the action executor."""

    def __init__(self) -> None:
        self.connected = True
        self.armed = False
        self.is_flying = False

        # Track calls for assertions
        self.takeoff_calls: list[dict] = []
        self.land_calls: list[dict] = []
        self.move_to_position_calls: list[dict] = []
        self.orbit_calls: list[dict] = []
        self.hover_calls: list[dict] = []

        # Simulated position
        self.current_position = MockPose.Position()

    async def takeoff(self, altitude: float = 10.0, timeout: float = 30.0) -> bool:
        self.takeoff_calls.append({"altitude": altitude, "timeout": timeout})
        self.is_flying = True
        self.armed = True
        self.current_position.z = -altitude
        return True

    async def land(self, timeout: float = 30.0) -> bool:
        self.land_calls.append({"timeout": timeout})
        self.is_flying = False
        self.armed = False
        self.current_position.z = 0.0
        return True

    async def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 5.0,
        timeout: float = 60.0,
    ) -> bool:
        self.move_to_position_calls.append({
            "x": x, "y": y, "z": z,
            "velocity": velocity, "timeout": timeout,
        })
        self.current_position.x = x
        self.current_position.y = y
        self.current_position.z = z
        return True

    async def hover(self) -> bool:
        self.hover_calls.append({})
        return True

    async def orbit(
        self,
        center_x: float,
        center_y: float,
        center_z: float,
        radius: float = 20.0,
        velocity: float = 3.0,
        duration: float = 30.0,
    ) -> bool:
        self.orbit_calls.append({
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "radius": radius,
            "velocity": velocity,
            "duration": duration,
        })
        return True

    async def get_synchronized_state(self) -> MockState:
        state = MockState()
        state.pose.position = self.current_position
        return state


# =============================================================================
# Action Executor Tests
# =============================================================================


class TestAirSimActionExecutor:
    """Tests for the AirSim action executor."""

    @pytest.fixture
    def bridge(self) -> MockRealtimeAirSimBridge:
        return MockRealtimeAirSimBridge()

    @pytest.fixture
    def geo_ref(self) -> GeoReference:
        return GeoReference(
            latitude=47.641468,
            longitude=-122.140165,
            altitude=0.0,
        )

    @pytest.fixture
    def executor(
        self,
        bridge: MockRealtimeAirSimBridge,
        geo_ref: GeoReference,
    ) -> AirSimActionExecutor:
        return AirSimActionExecutor(
            bridge=bridge,  # type: ignore
            geo_ref=geo_ref,
            config=FlightConfig(),
            drone_id="TestDrone",
        )

    @pytest.mark.asyncio
    async def test_takeoff(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test takeoff action."""
        result = await executor.execute({
            "action": "takeoff",
            "parameters": {"altitude_agl": 25.0},
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "takeoff"
        assert len(bridge.takeoff_calls) == 1
        assert bridge.takeoff_calls[0]["altitude"] == 25.0
        assert executor.is_flying

    @pytest.mark.asyncio
    async def test_land(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test land action."""
        # First takeoff
        await executor.execute({"action": "takeoff"})

        # Then land
        result = await executor.execute({"action": "land"})

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "land"
        assert len(bridge.land_calls) == 1
        assert not executor.is_flying

    @pytest.mark.asyncio
    async def test_inspect_asset(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test inspect_asset action with GPS target."""
        result = await executor.execute({
            "action": "inspect_asset",
            "parameters": {
                "position": {
                    "latitude": 47.642,
                    "longitude": -122.139,
                    "altitude_agl": 30.0,
                },
                "orbit_radius_m": 15.0,
                "dwell_time_s": 20.0,
            },
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "inspect_asset"

        # Should have taken off first
        assert len(bridge.takeoff_calls) == 1

        # Should have moved to position
        assert len(bridge.move_to_position_calls) == 1
        move_call = bridge.move_to_position_calls[0]
        # North should be positive (target is north of reference)
        assert move_call["x"] > 0

        # Should have performed orbit
        assert len(bridge.orbit_calls) == 1
        orbit_call = bridge.orbit_calls[0]
        assert orbit_call["radius"] == 15.0
        assert orbit_call["duration"] == 20.0

    @pytest.mark.asyncio
    async def test_inspect_asset_with_target_asset(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test inspect_asset action with target_asset format."""
        result = await executor.execute({
            "action": "inspect_asset",
            "target_asset": {
                "asset_id": "solar_farm_a",
                "name": "Solar Farm Alpha",
                "latitude": 47.642,
                "longitude": -122.139,
                "inspection_altitude_agl": 25.0,
                "orbit_radius_m": 30.0,
                "dwell_time_s": 45.0,
            },
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.details["asset_id"] == "solar_farm_a"
        assert result.details["orbit_radius_m"] == 30.0
        assert result.details["dwell_time_s"] == 45.0

    @pytest.mark.asyncio
    async def test_return_low_battery(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test return_low_battery action."""
        # First takeoff
        await executor.execute({"action": "takeoff"})

        # Return
        result = await executor.execute({
            "action": "return_low_battery",
            "reason": "Battery at 15%",
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "return"
        assert "Battery" in result.details.get("reason", "")

        # Should have moved to home (0, 0, -altitude)
        assert len(bridge.move_to_position_calls) == 1
        move_call = bridge.move_to_position_calls[0]
        assert move_call["x"] == 0
        assert move_call["y"] == 0
        assert move_call["z"] < 0  # Should be negative (above ground)

        # Should have landed
        assert len(bridge.land_calls) == 1

    @pytest.mark.asyncio
    async def test_abort(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test abort action (emergency land)."""
        # First takeoff
        await executor.execute({"action": "takeoff"})

        # Abort
        result = await executor.execute({
            "action": "abort",
            "reason": "GPS signal lost",
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "abort"
        assert len(bridge.land_calls) == 1

    @pytest.mark.asyncio
    async def test_wait(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test wait action."""
        # Takeoff first
        await executor.execute({"action": "takeoff"})

        # Wait
        result = await executor.execute({
            "action": "wait",
            "parameters": {"duration_s": 0.1},  # Short for testing
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "wait"
        assert len(bridge.hover_calls) == 1

    @pytest.mark.asyncio
    async def test_goto(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test goto action."""
        result = await executor.execute({
            "action": "goto",
            "parameters": {
                "position": {
                    "latitude": 47.643,
                    "longitude": -122.138,
                    "altitude_agl": 40.0,
                },
                "speed_ms": 8.0,
            },
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "goto"

        # Should have moved to position with correct velocity
        move_call = bridge.move_to_position_calls[-1]
        assert move_call["velocity"] == 8.0

    @pytest.mark.asyncio
    async def test_unknown_action(
        self,
        executor: AirSimActionExecutor,
    ) -> None:
        """Test handling of unknown action type."""
        result = await executor.execute({
            "action": "do_a_barrel_roll",
        })

        assert result.status == ExecutionStatus.FAILED
        assert "Unknown action" in (result.error or "")

    @pytest.mark.asyncio
    async def test_inspect_without_position_fails(
        self,
        executor: AirSimActionExecutor,
    ) -> None:
        """Test that inspect without position fails gracefully."""
        result = await executor.execute({
            "action": "inspect_asset",
            "parameters": {},  # No position
        })

        assert result.status == ExecutionStatus.FAILED
        assert "No target position" in (result.error or "")

    @pytest.mark.asyncio
    async def test_action_enum_handling(
        self,
        executor: AirSimActionExecutor,
        bridge: MockRealtimeAirSimBridge,
    ) -> None:
        """Test that enum action types are handled correctly."""
        from enum import Enum

        class ActionType(Enum):
            TAKEOFF = "takeoff"

        result = await executor.execute({
            "action": ActionType.TAKEOFF,
        })

        assert result.status == ExecutionStatus.COMPLETED
        assert result.action == "takeoff"

    @pytest.mark.asyncio
    async def test_execution_callbacks(
        self,
        bridge: MockRealtimeAirSimBridge,
        geo_ref: GeoReference,
    ) -> None:
        """Test that execution callbacks are called."""
        start_callback = MagicMock()
        complete_callback = MagicMock()

        executor = AirSimActionExecutor(
            bridge=bridge,  # type: ignore
            geo_ref=geo_ref,
            on_execution_start=start_callback,
            on_execution_complete=complete_callback,
        )

        await executor.execute({"action": "takeoff"})

        start_callback.assert_called_once()
        complete_callback.assert_called_once()

        # Verify callback arguments
        start_args = start_callback.call_args[0]
        assert start_args[0] == "takeoff"

        complete_args = complete_callback.call_args[0]
        assert complete_args[0] == "takeoff"
        assert isinstance(complete_args[1], ExecutionResult)


class TestFlightConfig:
    """Tests for FlightConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FlightConfig()

        assert config.default_altitude_agl == 30.0
        assert config.default_velocity == 5.0
        assert config.max_velocity == 15.0
        assert config.inspection_orbit_radius == 20.0
        assert config.inspection_dwell_time == 30.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = FlightConfig(
            default_altitude_agl=50.0,
            default_velocity=10.0,
            inspection_orbit_radius=30.0,
        )

        assert config.default_altitude_agl == 50.0
        assert config.default_velocity == 10.0
        assert config.inspection_orbit_radius == 30.0


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="takeoff",
            drone_id="Drone1",
            duration_s=5.5,
            details={"altitude_agl": 30.0},
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "completed"
        assert result_dict["action"] == "takeoff"
        assert result_dict["drone_id"] == "Drone1"
        assert result_dict["duration_s"] == 5.5
        assert result_dict["details"]["altitude_agl"] == 30.0
        assert "timestamp" in result_dict
