"""
Unit tests for multi-agent critic system.

Tests all three critics and the orchestrator with various scenarios.
"""

from datetime import datetime

import pytest

from agent.api_models import ActionType
from agent.server.critics import (
    AuthorityModel,
    CriticOrchestrator,
    EfficiencyCritic,
    GoalAlignmentCritic,
    SafetyCritic,
)
from agent.server.decision import Decision
from agent.server.models.critic_models import CriticVerdict
from agent.server.risk_evaluator import RiskAssessment, RiskFactor, RiskLevel
from agent.server.world_model import (
    Asset,
    AssetType,
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
    VehicleState,
    Velocity,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_position():
    """Create a mock position."""
    return Position(latitude=37.7749, longitude=-122.4194, altitude_msl=50.0, altitude_agl=50.0)


@pytest.fixture
def mock_battery_good():
    """Create a mock battery with good charge."""
    return BatteryState(voltage=16.8, current=5.0, remaining_percent=75.0, time_remaining_s=1800)


@pytest.fixture
def mock_battery_low():
    """Create a mock battery with low charge."""
    return BatteryState(voltage=14.4, current=5.0, remaining_percent=15.0, time_remaining_s=180)


@pytest.fixture
def mock_gps_good():
    """Create a mock GPS with good signal."""
    return GPSState(satellites_visible=12, hdop=0.8, fix_type=3)


@pytest.fixture
def mock_gps_poor():
    """Create a mock GPS with poor signal."""
    return GPSState(satellites_visible=4, hdop=3.5, fix_type=2)


@pytest.fixture
def mock_environment_calm():
    """Create calm weather environment."""
    return EnvironmentState(
        timestamp=datetime.now(),
        wind_speed_ms=3.0,
        wind_direction_deg=180.0,
        visibility_m=5000.0,
        temperature_c=20.0,
    )


@pytest.fixture
def mock_environment_windy():
    """Create windy environment."""
    return EnvironmentState(
        timestamp=datetime.now(),
        wind_speed_ms=13.0,
        wind_direction_deg=270.0,
        visibility_m=2000.0,
        temperature_c=18.0,
    )


@pytest.fixture
def mock_vehicle_good(mock_position, mock_battery_good, mock_gps_good):
    """Create mock vehicle in good state."""
    return VehicleState(
        timestamp=datetime.now(),
        position=mock_position,
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=mock_battery_good,
        gps=mock_gps_good,
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
    )


@pytest.fixture
def mock_vehicle_low_battery(mock_position, mock_battery_low, mock_gps_good):
    """Create mock vehicle with low battery."""
    return VehicleState(
        timestamp=datetime.now(),
        position=mock_position,
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=mock_battery_low,
        gps=mock_gps_good,
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
    )


@pytest.fixture
def mock_world_good(mock_vehicle_good, mock_environment_calm):
    """Create mock world in good state."""
    return WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=mock_vehicle_good,
        environment=mock_environment_calm,
        assets=[
            Asset(
                asset_id="asset_1",
                name="Bridge Asset",
                position=Position(
                    latitude=37.7750, longitude=-122.4195, altitude_msl=0, altitude_agl=0
                ),
                asset_type=AssetType.BUILDING,
                priority=8,
            ),
            Asset(
                asset_id="asset_2",
                name="Tower Asset",
                position=Position(
                    latitude=37.7760, longitude=-122.4200, altitude_msl=0, altitude_agl=0
                ),
                asset_type=AssetType.BUILDING,
                priority=5,
            ),
        ],
        mission=MissionState(
            mission_id="test_mission",
            mission_name="Test Mission",
            assets_total=2,
            assets_inspected=0,
        ),
        anomalies=[],
        dock=DockState(
            position=Position(
                latitude=37.7749, longitude=-122.4194, altitude_msl=0, altitude_agl=0
            ),
            status=DockStatus.AVAILABLE,
        ),
    )


@pytest.fixture
def mock_risk_low():
    """Create low-risk assessment."""
    return RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery",
                value=0.1,
                threshold=0.5,
                critical=0.8,
                description="Battery healthy",
            )
        },
    )


@pytest.fixture
def mock_risk_high():
    """Create high-risk assessment."""
    return RiskAssessment(
        overall_level=RiskLevel.HIGH,
        overall_score=0.8,
        factors={
            "battery": RiskFactor(
                name="battery",
                value=0.9,
                threshold=0.5,
                critical=0.8,
                description="Battery critical",
            )
        },
        abort_recommended=True,
        abort_reason="Battery critical",
    )


# ============================================================================
# SafetyCritic Tests
# ============================================================================


@pytest.mark.asyncio
async def test_safety_critic_approves_good_conditions(mock_world_good, mock_risk_low):
    """Test that SafetyCritic approves decisions in good conditions."""
    critic = SafetyCritic()

    decision = Decision(
        action=ActionType.INSPECT,
        parameters={"asset_id": "asset_1"},
        confidence=0.9,
        reasoning="Test inspection",
    )

    response = await critic.evaluate_fast(decision, mock_world_good, mock_risk_low)

    assert response.verdict in [CriticVerdict.APPROVE, CriticVerdict.APPROVE_WITH_CONCERNS]
    assert response.risk_score < 0.5


@pytest.mark.asyncio
async def test_safety_critic_rejects_low_battery(
    mock_vehicle_low_battery, mock_environment_calm, mock_risk_high
):
    """Test that SafetyCritic rejects decisions with low battery."""
    critic = SafetyCritic()

    world = WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=mock_vehicle_low_battery,
        environment=mock_environment_calm,
        assets=[],
        anomalies=[],
        mission=MissionState(
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(
                latitude=37.7749, longitude=-122.4194, altitude_msl=0, altitude_agl=0
            ),
            status=DockStatus.AVAILABLE,
        ),
    )

    decision = Decision(
        action=ActionType.INSPECT, parameters={"asset_id": "asset_1"}, confidence=0.9
    )

    response = await critic.evaluate_fast(decision, world, mock_risk_high)

    assert response.verdict in [CriticVerdict.REJECT, CriticVerdict.ESCALATE]
    assert len(response.concerns) > 0
    assert any("battery" in c.lower() for c in response.concerns)


@pytest.mark.asyncio
async def test_safety_critic_rejects_high_wind(
    mock_world_good, mock_environment_windy, mock_risk_high
):
    """Test that SafetyCritic rejects decisions in high wind."""
    critic = SafetyCritic()

    world = WorldSnapshot(
        timestamp=mock_world_good.timestamp,
        vehicle=mock_world_good.vehicle,
        environment=mock_environment_windy,
        assets=mock_world_good.assets,
        anomalies=[],
        mission=mock_world_good.mission,
        dock=mock_world_good.dock,
    )

    decision = Decision(action=ActionType.ORBIT, parameters={"asset_id": "asset_1"}, confidence=0.9)

    response = await critic.evaluate_fast(decision, world, mock_risk_high)

    assert response.verdict in [CriticVerdict.REJECT, CriticVerdict.ESCALATE]
    assert any("wind" in c.lower() for c in response.concerns)


# ============================================================================
# EfficiencyCritic Tests
# ============================================================================


@pytest.mark.asyncio
async def test_efficiency_critic_approves_efficient_decision(mock_world_good, mock_risk_low):
    """Test that EfficiencyCritic approves efficient decisions."""
    critic = EfficiencyCritic()

    decision = Decision(
        action=ActionType.INSPECT, parameters={"asset_id": "asset_1"}, confidence=0.9
    )

    response = await critic.evaluate_fast(decision, mock_world_good, mock_risk_low)

    assert response.verdict in [CriticVerdict.APPROVE, CriticVerdict.APPROVE_WITH_CONCERNS]


@pytest.mark.asyncio
async def test_efficiency_critic_flags_inefficient_wait(mock_world_good, mock_risk_low):
    """Test that EfficiencyCritic flags inefficient WAIT decisions."""
    critic = EfficiencyCritic()

    decision = Decision(
        action=ActionType.WAIT,
        parameters={"duration_s": 300},  # 5 minutes
        confidence=0.9,
    )

    response = await critic.evaluate_fast(decision, mock_world_good, mock_risk_low)

    # Should have concerns about waiting with high battery
    assert response.verdict == CriticVerdict.APPROVE_WITH_CONCERNS
    assert len(response.concerns) > 0


# ============================================================================
# GoalAlignmentCritic Tests
# ============================================================================


@pytest.mark.asyncio
async def test_goal_alignment_approves_aligned_decision(mock_world_good, mock_risk_low):
    """Test that GoalAlignmentCritic approves aligned decisions."""
    critic = GoalAlignmentCritic()

    decision = Decision(
        action=ActionType.INSPECT,
        parameters={"asset_id": "asset_1"},  # High priority asset
        confidence=0.9,
    )

    response = await critic.evaluate_fast(decision, mock_world_good, mock_risk_low)

    assert response.verdict in [CriticVerdict.APPROVE, CriticVerdict.APPROVE_WITH_CONCERNS]


@pytest.mark.asyncio
async def test_goal_alignment_flags_mission_complete_return(mock_world_good, mock_risk_low):
    """Test that GoalAlignmentCritic flags early return when mission incomplete."""
    critic = GoalAlignmentCritic()

    decision = Decision(action=ActionType.RETURN, confidence=0.9)

    response = await critic.evaluate_fast(decision, mock_world_good, mock_risk_low)

    # Should flag concern about returning with uninspected assets
    assert len(response.concerns) > 0 or response.verdict == CriticVerdict.APPROVE_WITH_CONCERNS


# ============================================================================
# CriticOrchestrator Tests
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_advisory_mode_always_approves(mock_world_good, mock_risk_low):
    """Test that orchestrator in advisory mode always approves."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ADVISORY)

    decision = Decision(
        action=ActionType.INSPECT, parameters={"asset_id": "asset_1"}, confidence=0.9
    )

    approved, escalation = await orchestrator.validate_decision(
        decision, mock_world_good, mock_risk_low
    )

    assert approved is True
    assert escalation is None


@pytest.mark.asyncio
async def test_orchestrator_escalation_mode_low_risk(mock_world_good, mock_risk_low):
    """Test orchestrator escalation mode with low risk."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

    decision = Decision(
        action=ActionType.INSPECT, parameters={"asset_id": "asset_1"}, confidence=0.9
    )

    approved, _escalation = await orchestrator.validate_decision(
        decision, mock_world_good, mock_risk_low
    )

    # Low risk should approve
    assert approved is True


@pytest.mark.asyncio
async def test_orchestrator_escalation_mode_high_risk(
    mock_vehicle_low_battery, mock_environment_calm, mock_risk_high
):
    """Test orchestrator escalation mode with high risk."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION, enable_llm=False)

    world = WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=mock_vehicle_low_battery,
        environment=mock_environment_calm,
        assets=[],
        anomalies=[],
        mission=MissionState(
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(
                latitude=37.7749, longitude=-122.4194, altitude_msl=0, altitude_agl=0
            ),
            status=DockStatus.AVAILABLE,
        ),
    )

    decision = Decision(
        action=ActionType.INSPECT, parameters={"asset_id": "asset_1"}, confidence=0.9
    )

    _approved, escalation = await orchestrator.validate_decision(decision, world, mock_risk_high)

    # High risk should trigger escalation
    assert escalation is not None
    # May be approved or blocked depending on critic responses


@pytest.mark.asyncio
async def test_orchestrator_gets_stats():
    """Test that orchestrator returns statistics."""
    orchestrator = CriticOrchestrator()

    stats = orchestrator.get_stats()

    assert "authority_model" in stats
    assert "total_critics" in stats
    assert stats["total_critics"] == 3
    assert "critic_stats" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
