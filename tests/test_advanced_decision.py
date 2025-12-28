from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.server.advanced_decision import AdvancedDecisionEngine, MissionDecision
from agent.server.goals import GoalType
from agent.server.world_model import (
    DockState,
    DockStatus,
    EnvironmentState,
    MissionState,
    WorldSnapshot,
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


@pytest.fixture
def mock_world_snapshot():
    """Create a baseline world snapshot for advanced decision tests."""
    return WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=VehicleState(
            timestamp=datetime.now(),
            position=Position(latitude=45.0, longitude=-75.0, altitude_msl=100.0),
            velocity=Velocity(north=0.0, east=0.0, down=0.0),
            attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
            battery=BatteryState(voltage=24.0, current=1.0, remaining_percent=80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            health=VehicleHealth(
                sensors_healthy=True,
                gps_healthy=True,
                battery_healthy=True,
                motors_healthy=True,
                ekf_healthy=True,
            ),
            gps=GPSState(fix_type=3, satellites_visible=10, hdop=1.0),
        ),
        assets=[],
        anomalies=[],
        dock=DockState(
            position=Position(latitude=45.0, longitude=-75.0, altitude_msl=0.0),
            status=DockStatus.AVAILABLE,
        ),
        environment=EnvironmentState(timestamp=datetime.now()),
        mission=MissionState(mission_id="test", mission_name="Test Mission"),
    )


@pytest.mark.asyncio
async def test_advanced_decision_mock(mock_world_snapshot):
    """Ensure advanced decisions use the configured model output."""
    # Set up engine with a test model that returns a specific decision
    engine = AdvancedDecisionEngine()

    # Define what the mock model should return
    mock_decision = MissionDecision(
        goal_type=GoalType.INSPECT_ASSET,
        priority=30,
        reason="Routine inspection of scheduled assets.",
        target_asset_id=None,
    )

    engine.modules.agent.run = AsyncMock(return_value=SimpleNamespace(output=mock_decision))
    goal, context = await engine.make_advanced_decision(mock_world_snapshot)

    assert goal.goal_type == GoalType.INSPECT_ASSET
    assert goal.priority == 30
    assert "Routine inspection" in goal.reason
    assert context.cognitive_level is not None


@pytest.mark.asyncio
async def test_advanced_decision_low_battery_mock(mock_world_snapshot):
    """Ensure low-battery decisions map to return goals."""
    # Update battery to low using model_copy since it's frozen
    new_battery = mock_world_snapshot.vehicle.battery.model_copy(update={"remaining_percent": 10.0})
    mock_world_snapshot.vehicle.battery = new_battery

    engine = AdvancedDecisionEngine()

    mock_decision = MissionDecision(
        goal_type=GoalType.RETURN_LOW_BATTERY,
        priority=0,
        reason="Battery critical, returning to dock immediately.",
        target_asset_id=None,
    )

    engine.modules.agent.run = AsyncMock(return_value=SimpleNamespace(output=mock_decision))
    goal, _context = await engine.make_advanced_decision(mock_world_snapshot)

    assert goal.goal_type == GoalType.RETURN_LOW_BATTERY
    assert "Battery critical" in goal.reason


@pytest.mark.asyncio
async def test_advanced_decision_fallback(mock_world_snapshot):
    """Verify fallback behavior when the LLM decision fails."""
    # Test fallback if agent fails
    engine = AdvancedDecisionEngine()

    # We break the agent by giving it a model that raises an error
    engine.modules.agent.run = AsyncMock(side_effect=RuntimeError("Agent injection failure"))
    # This should trigger the try-except in make_advanced_decision
    # and fall back to _make_reactive_decision
    goal, _context = await engine.make_advanced_decision(mock_world_snapshot)

    # In mock_world_snapshot, battery is 80%, so it should return WAIT
    assert goal.goal_type == GoalType.WAIT
    assert "No immediate action" in goal.reason
