"""
Integration tests for decision pipeline with critics.

Tests the full flow from state ingestion → decision → critic validation → outcome tracking.
"""

import asyncio
from datetime import datetime

import pytest

from agent.api_models import ActionType
from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.decision import Decision
from agent.server.models.critic_models import EscalationLevel
from agent.server.models.outcome_models import DecisionFeedback, ExecutionStatus
from agent.server.monitoring import OutcomeTracker
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
def good_vehicle_state():
    """Create a vehicle in good condition."""
    return VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=37.7749, longitude=-122.4194, altitude_msl=50.0, altitude_agl=50.0
        ),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(
            voltage=16.8, current=5.0, remaining_percent=75.0, time_remaining_s=1800
        ),
        gps=GPSState(satellites_visible=12, hdop=0.8, fix_type=3),
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
    )


@pytest.fixture
def low_battery_vehicle_state():
    """Create a vehicle with critically low battery."""
    return VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=37.7749, longitude=-122.4194, altitude_msl=50.0, altitude_agl=50.0
        ),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(
            voltage=14.0, current=5.0, remaining_percent=12.0, time_remaining_s=120
        ),
        gps=GPSState(satellites_visible=12, hdop=0.8, fix_type=3),
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
    )


@pytest.fixture
def good_world_snapshot(good_vehicle_state):
    """Create a world snapshot in good conditions."""
    return WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=good_vehicle_state,
        environment=EnvironmentState(
            timestamp=datetime.now(),
            wind_speed_ms=3.0,
            wind_direction_deg=180.0,
            visibility_m=5000.0,
            temperature_c=20.0,
        ),
        assets=[
            Asset(
                asset_id="asset_1",
                name="Test Asset",
                position=Position(
                    latitude=37.7750, longitude=-122.4195, altitude_msl=0, altitude_agl=0
                ),
                asset_type=AssetType.BUILDING,
                priority=1,  # High priority
            )
        ],
        anomalies=[],
        mission=MissionState(
            mission_id="test_mission",
            mission_name="Integration Test Mission",
            assets_total=1,
            assets_inspected=0,
        ),
        dock=DockState(
            position=Position(
                latitude=37.7749, longitude=-122.4194, altitude_msl=0, altitude_agl=0
            ),
            status=DockStatus.AVAILABLE,
        ),
    )


@pytest.fixture
def low_risk():
    """Create a low-risk assessment."""
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
def high_risk():
    """Create a high-risk assessment."""
    return RiskAssessment(
        overall_level=RiskLevel.HIGH,
        overall_score=0.85,
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
        abort_reason="Battery critically low",
    )


# ============================================================================
# Normal Decision Flow Tests
# ============================================================================


@pytest.mark.asyncio
async def test_normal_decision_approved_by_critics(good_world_snapshot, low_risk):
    """Test that a normal decision flows through critics and gets approved."""
    # Setup
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)
    outcome_tracker = OutcomeTracker(log_dir="logs/test_outcomes")

    # Create a normal inspection decision
    decision = Decision(
        action=ActionType.INSPECT,
        parameters={"asset_id": "asset_1"},
        confidence=0.9,
        reasoning="Inspecting high-priority asset",
    )

    # Validate with critics
    approved, escalation = await orchestrator.validate_decision(
        decision, good_world_snapshot, low_risk
    )

    # Assertions
    assert approved is True, "Decision should be approved in good conditions"

    # Escalation might be None (advisory mode) or have approved=True
    if escalation:
        assert escalation.approved is True
        assert escalation.escalation_level in [EscalationLevel.ADVISORY, EscalationLevel.NONE]

    # Create outcome tracking
    outcome = outcome_tracker.create_outcome(decision)
    assert outcome is not None
    assert outcome.decision_id == decision.decision_id
    assert outcome.execution_status == ExecutionStatus.PENDING


@pytest.mark.asyncio
@pytest.mark.allow_error_logs
async def test_decision_blocked_by_low_battery(low_battery_vehicle_state, high_risk):
    """Test that critics block unsafe decision with low battery."""
    # Setup
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

    world = WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=low_battery_vehicle_state,
        environment=EnvironmentState(timestamp=datetime.now()),
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

    # Create risky decision (INSPECT with low battery)
    decision = Decision(
        action=ActionType.INSPECT,
        parameters={"asset_id": "asset_1"},
        confidence=0.9,
    )

    # Validate with critics
    approved, escalation = await orchestrator.validate_decision(decision, world, high_risk)

    # Assertions
    # With high risk (0.85), should trigger escalation
    assert escalation is not None, "Should have escalation decision"

    # May be blocked or escalated depending on critic consensus
    if not approved:
        assert escalation.approved is False
        assert len(escalation.reason) > 0
        assert any(
            "battery" in concern.lower()
            for resp in escalation.critic_responses
            for concern in resp.concerns
        )


@pytest.mark.asyncio
async def test_hierarchical_review_triggered(good_world_snapshot):
    """Test that very high risk triggers hierarchical review."""
    # Setup
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION, enable_llm=False)

    # Critical risk (0.9+)
    critical_risk = RiskAssessment(
        overall_level=RiskLevel.CRITICAL,
        overall_score=0.95,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.95, threshold=0.5, critical=0.8, description="Critical"
            ),
            "weather": RiskFactor(
                name="weather", value=0.9, threshold=0.5, critical=0.8, description="Severe weather"
            ),
        },
        abort_recommended=True,
        abort_reason="Multiple critical factors",
    )

    decision = Decision(
        action=ActionType.ORBIT,
        parameters={"asset_id": "asset_1"},
        confidence=0.8,
    )

    # Validate - should trigger hierarchical review
    _approved, escalation = await orchestrator.validate_decision(
        decision, good_world_snapshot, critical_risk
    )

    assert escalation is not None
    assert escalation.escalation_level in [EscalationLevel.HIERARCHICAL, EscalationLevel.BLOCKING]


# ============================================================================
# Feedback Loop Tests
# ============================================================================


@pytest.mark.asyncio
async def test_outcome_tracking_and_feedback():
    """Test complete feedback loop: decision → execution → feedback → outcome update."""
    # Setup
    outcome_tracker = OutcomeTracker(log_dir="logs/test_outcomes")

    # Create decision
    decision = Decision(
        action=ActionType.INSPECT,
        parameters={
            "asset_id": "asset_1",
            "predicted_battery_consumption": 5.0,
            "estimated_duration_s": 120,
        },
        confidence=0.9,
    )

    # Track outcome
    outcome = outcome_tracker.create_outcome(decision)
    assert outcome.execution_status == ExecutionStatus.PENDING
    assert outcome.predicted_battery_consumed == 5.0
    assert outcome.predicted_duration_s == 120

    # Simulate client feedback
    feedback = DecisionFeedback(
        decision_id=decision.decision_id,
        status=ExecutionStatus.SUCCESS,
        battery_consumed=4.8,
        duration_s=115,
        mission_objective_achieved=True,
        asset_inspected="asset_1",
    )

    # Process feedback
    updated_outcome = await outcome_tracker.process_feedback(feedback)

    # Verify outcome updated
    assert updated_outcome is not None
    assert updated_outcome.execution_status == ExecutionStatus.SUCCESS
    assert updated_outcome.actual_battery_consumed == 4.8
    assert updated_outcome.actual_duration_s == 115
    assert updated_outcome.mission_objective_achieved is True

    # Check prediction errors calculated
    assert updated_outcome.prediction_error_battery == abs(5.0 - 4.8)
    assert updated_outcome.prediction_error_duration == abs(120 - 115)

    # Verify stats updated
    stats = outcome_tracker.get_stats_dict()
    assert stats["total_tracked"] == 1
    assert stats["successful"] == 1
    assert stats["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_feedback_for_unknown_decision():
    """Test that feedback for unknown decision is handled gracefully."""
    outcome_tracker = OutcomeTracker(log_dir="logs/test_outcomes")

    feedback = DecisionFeedback(
        decision_id="unknown_decision_123",
        status=ExecutionStatus.SUCCESS,
        battery_consumed=3.0,
    )

    result = await outcome_tracker.process_feedback(feedback)

    # Should return None for unknown decision
    assert result is None


# ============================================================================
# Escalation Model Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.allow_error_logs
async def test_advisory_mode_always_approves():
    """Test that ADVISORY mode never blocks decisions."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ADVISORY)

    # Even with high risk, should approve
    high_risk = RiskAssessment(
        overall_level=RiskLevel.HIGH,
        overall_score=0.8,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.8, threshold=0.5, critical=0.8, description="High"
            )
        },
    )

    world = WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=VehicleState(
            timestamp=datetime.now(),
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=50, altitude_agl=50),
            velocity=Velocity(north=0, east=0, down=0),
            attitude=Attitude(roll=0, pitch=0, yaw=0),
            battery=BatteryState(voltage=15.0, current=5.0, remaining_percent=25.0),
            gps=GPSState(fix_type=3, satellites_visible=10, hdop=1.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
        ),
        environment=EnvironmentState(timestamp=datetime.now()),
        assets=[],
        anomalies=[],
        mission=MissionState(
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.8)

    approved, escalation = await orchestrator.validate_decision(decision, world, high_risk)

    # Advisory mode always approves
    assert approved is True
    # May log concerns but doesn't block
    if escalation:
        assert escalation.approved is True


# ============================================================================
# Concurrent Decision Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_decision_validation():
    """Test that multiple decisions can be validated concurrently."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

    # Create multiple decisions
    decisions = [
        Decision(action=ActionType.INSPECT, parameters={"asset_id": f"asset_{i}"}, confidence=0.9)
        for i in range(5)
    ]

    world = WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=VehicleState(
            timestamp=datetime.now(),
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=50, altitude_agl=50),
            velocity=Velocity(north=0, east=0, down=0),
            attitude=Attitude(roll=0, pitch=0, yaw=0),
            battery=BatteryState(voltage=16.8, current=5.0, remaining_percent=75.0),
            gps=GPSState(fix_type=3, satellites_visible=12, hdop=0.8),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
        ),
        environment=EnvironmentState(timestamp=datetime.now()),
        assets=[],
        anomalies=[],
        mission=MissionState(
            mission_id="test", mission_name="Test", assets_total=5, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    low_risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    # Validate all decisions concurrently
    tasks = [orchestrator.validate_decision(d, world, low_risk) for d in decisions]
    results = await asyncio.gather(*tasks)

    # All should be approved
    assert len(results) == 5
    for approved, _escalation in results:
        assert approved is True


# ============================================================================
# Statistics and Monitoring Tests
# ============================================================================


@pytest.mark.asyncio
async def test_outcome_statistics_tracking():
    """Test that outcome statistics are correctly tracked."""
    outcome_tracker = OutcomeTracker(log_dir="logs/test_outcomes")

    # Create multiple outcomes
    decisions = [
        Decision(action=ActionType.INSPECT, parameters={}, confidence=0.9) for _ in range(10)
    ]

    outcomes = [outcome_tracker.create_outcome(d) for d in decisions]

    # Simulate different execution results
    for i, outcome in enumerate(outcomes):
        if i < 7:
            # 7 successes
            feedback = DecisionFeedback(
                decision_id=outcome.decision_id,
                status=ExecutionStatus.SUCCESS,
                mission_objective_achieved=True,
            )
        else:
            # 3 failures
            feedback = DecisionFeedback(
                decision_id=outcome.decision_id,
                status=ExecutionStatus.FAILED,
                mission_objective_achieved=False,
                errors=["Simulated failure"],
            )

        await outcome_tracker.process_feedback(feedback)

    # Check statistics
    stats = outcome_tracker.get_stats_dict()
    assert stats["total_tracked"] == 10
    assert stats["successful"] == 7
    assert stats["failed"] == 3
    assert stats["success_rate"] == 0.7


def test_orchestrator_statistics():
    """Test that critic orchestrator provides statistics."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

    stats = orchestrator.get_stats()

    assert "authority_model" in stats
    assert "total_critics" in stats
    assert stats["total_critics"] == 3
    assert "critic_stats" in stats
    assert len(stats["critic_stats"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
