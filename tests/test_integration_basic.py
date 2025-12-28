"""
Basic integration tests for AegisAV components.

These tests verify that core components work together properly
without requiring actual MAVLink connections.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.server.goal_selector import GoalSelector, GoalType
from agent.server.risk_evaluator import RiskEvaluator, RiskThresholds
from agent.server.world_model import WorldModel
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
from tests.conftest import TEST_BATTERY_CRITICAL, TEST_BATTERY_WARNING, TEST_HOME_POSITION


@pytest.mark.integration
class TestBasicIntegration:
    """Test basic integration between agent components."""

    @pytest.mark.asyncio
    async def test_world_model_goal_selector_integration(self):
        """Test that world model and goal selector work together."""
        # Create components
        world_model = WorldModel()
        goal_selector = GoalSelector()

        # Create test vehicle state
        vehicle_state = VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
                altitude_agl=12.0,
            ),
            velocity=Velocity(north=0.0, east=0.0, down=0.0),
            attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
            battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(fix_type=3, satellites_visible=8, hdop=0.8, vdop=0.8),
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
            ),
        )

        # Update world model
        world_model.set_dock(
            Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            )
        )
        world_model.update_vehicle(vehicle_state)

        # Get snapshot and select goal
        snapshot = world_model.get_snapshot()
        goal = await goal_selector.select_goal(snapshot)

        # Should select wait when no assets need inspection
        assert goal is not None
        assert goal.goal_type in [GoalType.WAIT, GoalType.INSPECT_ASSET]

    def test_risk_evaluator_with_world_model(self):
        """Test risk evaluator integration with world model."""
        world_model = WorldModel()
        thresholds = RiskThresholds(
            battery_warning_percent=TEST_BATTERY_WARNING,
            battery_critical_percent=TEST_BATTERY_CRITICAL,
            wind_warning_ms=8.0,
            wind_abort_ms=12.0,
        )
        risk_evaluator = RiskEvaluator(thresholds)

        # Create low battery vehicle state
        vehicle_state = VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            ),
            velocity=Velocity(north=0.0, east=0.0, down=0.0),
            attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
            battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=25.0),  # Low battery
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            health=VehicleHealth(
                sensors_healthy=True,
                gps_healthy=True,
                battery_healthy=True,
                motors_healthy=True,
                ekf_healthy=True,
            ),
        )

        # Update world model
        world_model.set_dock(
            Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            )
        )
        world_model.update_vehicle(vehicle_state)
        snapshot = world_model.get_snapshot()

        # Evaluate risk
        risk = risk_evaluator.evaluate(snapshot)

        # Should detect battery risk
        assert risk is not None
        assert risk.overall_score > 0.2  # Weighted risk
        assert any("battery" in warning.lower() for warning in risk.warnings)

    @pytest.mark.asyncio
    async def test_decision_pipeline_integration(self):
        """Test full decision pipeline: world -> risk -> goal."""
        world_model = WorldModel()
        goal_selector = GoalSelector(battery_return_threshold=30.0)

        thresholds = RiskThresholds(
            battery_warning_percent=TEST_BATTERY_WARNING,
            battery_critical_percent=TEST_BATTERY_CRITICAL,
            wind_warning_ms=8.0,
            wind_abort_ms=12.0,
        )
        risk_evaluator = RiskEvaluator(thresholds)

        # Create vehicle with critical battery
        vehicle_state = VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            ),
            velocity=Velocity(north=5.0, east=0.0, down=-1.0),
            attitude=Attitude(roll=0.1, pitch=0.2, yaw=1.57),
            battery=BatteryState(
                voltage=20.0, current=10.0, remaining_percent=18.0
            ),  # Critical battery
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(fix_type=3, satellites_visible=10, hdop=0.6, vdop=0.8),
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
            ),
        )

        # Run decision pipeline
        world_model.set_dock(
            Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            )
        )
        world_model.update_vehicle(vehicle_state)
        snapshot = world_model.get_snapshot()

        # Risk assessment
        risk = risk_evaluator.evaluate(snapshot)
        assert risk.overall_score > 0.2  # High risk due to battery (weighted)

        # Goal selection (should prioritize battery)
        goal = await goal_selector.select_goal(snapshot)
        assert goal.goal_type in [GoalType.RETURN_LOW_BATTERY, GoalType.ABORT]

        # Verify reasoning includes battery
        assert "battery" in goal.reason.lower()  # Changed from reasoning to reason


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncIntegration:
    """Test asynchronous integration components."""

    async def test_mock_mavlink_interface(self):
        """Test with mocked MAVLink interface."""
        # Create mock MAVLink connection
        mock_conn = MagicMock()
        mock_conn.wait_heartbeat = AsyncMock(return_value=True)
        mock_conn.target_system = 1
        mock_conn.target_component = 1

        # Mock messages
        mock_conn.messages = {
            "HEARTBEAT": MagicMock(type=2, base_mode=217, custom_mode=4),
            "GLOBAL_POSITION_INT": MagicMock(
                lat=int(TEST_HOME_POSITION["lat"] * 1e7),
                lon=int(TEST_HOME_POSITION["lon"] * 1e7),
                alt=int(TEST_HOME_POSITION["alt"] * 1000),
                relative_alt=int(12 * 1000),
                vx=0,
                vy=0,
                vz=0,
            ),
            "SYS_STATUS": MagicMock(
                voltage_battery=22800, current_battery=500, battery_remaining=80
            ),
        }

        # Test that we can extract state from mock
        assert mock_conn.messages["GLOBAL_POSITION_INT"].lat / 1e7 == TEST_HOME_POSITION["lat"]
        assert mock_conn.messages["SYS_STATUS"].battery_remaining == 80

        # Verify mock is properly configured
        assert mock_conn.target_system == 1
        assert asyncio.iscoroutinefunction(mock_conn.wait_heartbeat)

    async def test_state_updates(self):
        """Test state update flow with timing."""
        world_model = WorldModel()

        # Create initial state
        initial_time = datetime.now()
        initial_state = VehicleState(
            timestamp=initial_time,
            position=Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            ),
            velocity=Velocity(north=0.0, east=0.0, down=0.0),
            attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
            battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=False,
        )

        # Update world model
        world_model.set_dock(
            Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            )
        )
        world_model.update_vehicle(initial_state)
        snapshot1 = world_model.get_snapshot()

        # Create updated state (1 second later, armed)
        updated_time = initial_time + timedelta(seconds=1)
        updated_state = VehicleState(
            timestamp=updated_time,
            position=Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"] + 5.0,  # Climbed
            ),
            velocity=Velocity(north=0.0, east=0.0, down=-5.0),  # Climbing
            attitude=Attitude(roll=0.0, pitch=0.1, yaw=0.0),  # Slight pitch up
            battery=BatteryState(
                voltage=22.7, current=8.0, remaining_percent=79.5
            ),  # Slightly lower
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,  # Now in air
        )

        # Update and verify changes
        world_model.update_vehicle(updated_state)
        snapshot2 = world_model.get_snapshot()

        # Verify state progression
        assert snapshot2.vehicle.timestamp > snapshot1.vehicle.timestamp
        assert snapshot2.vehicle.in_air
        assert not snapshot1.vehicle.in_air
        assert snapshot2.vehicle.position.altitude_msl > snapshot1.vehicle.position.altitude_msl
        assert (
            snapshot2.vehicle.battery.remaining_percent
            < snapshot1.vehicle.battery.remaining_percent
        )


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration scenarios."""

    @pytest.mark.asyncio
    async def test_missing_gps_handling(self):
        """Test decision making with missing GPS."""
        world_model = WorldModel()
        goal_selector = GoalSelector()

        # Create vehicle without GPS fix
        vehicle_state = VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            ),
            velocity=Velocity(north=0.0, east=0.0, down=0.0),
            attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
            battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(fix_type=0, satellites_visible=0, hdop=99.9, vdop=99.9),  # No GPS fix
            health=VehicleHealth(
                sensors_healthy=True,
                gps_healthy=False,
                battery_healthy=True,
                motors_healthy=True,
                ekf_healthy=True,
                error_messages=["gps loss"],
            ),  # GPS unhealthy
        )

        world_model.set_dock(
            Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            )
        )
        world_model.update_vehicle(vehicle_state)
        snapshot = world_model.get_snapshot()

        # Should abort due to no GPS
        goal = await goal_selector.select_goal(snapshot)
        assert goal.goal_type == GoalType.ABORT
        assert "gps" in goal.reason.lower()  # Changed from reasoning to reason

    @pytest.mark.asyncio
    async def test_unhealthy_vehicle_handling(self):
        """Test decision making with unhealthy vehicle."""
        world_model = WorldModel()
        goal_selector = GoalSelector()

        # Create vehicle with multiple health issues
        vehicle_state = VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            ),
            velocity=Velocity(north=0.0, east=0.0, down=0.0),
            attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
            battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(fix_type=3, satellites_visible=8, hdop=0.8, vdop=0.8),
            health=VehicleHealth(
                sensors_healthy=False,
                gps_healthy=True,
                battery_healthy=False,
                motors_healthy=False,
                ekf_healthy=True,
            ),  # Multiple failures
        )

        world_model.set_dock(
            Position(
                latitude=TEST_HOME_POSITION["lat"],
                longitude=TEST_HOME_POSITION["lon"],
                altitude_msl=TEST_HOME_POSITION["alt"],
            )
        )
        world_model.update_vehicle(vehicle_state)
        snapshot = world_model.get_snapshot()

        # Should abort due to health issues
        goal = await goal_selector.select_goal(snapshot)
        assert goal.goal_type == GoalType.ABORT
        assert any(
            keyword in goal.reason.lower() for keyword in ["health", "sensors", "battery", "motors"]
        )
