"""
Edge case and performance tests for the agent server.

Tests error handling, invalid inputs, timeouts, and performance characteristics.
"""

import asyncio
import time
from datetime import datetime

import pytest

from agent.api_models import ActionType
from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.critics.safety_critic import SafetyCritic
from agent.server.decision import Decision
from agent.server.models.critic_models import CriticVerdict
from agent.server.models.outcome_models import DecisionFeedback, ExecutionStatus
from agent.server.monitoring import OutcomeTracker
from agent.server.risk_evaluator import RiskAssessment, RiskFactor, RiskLevel
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
    VehicleState,
    Velocity,
)

# ============================================================================
# Invalid Input Tests
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_decision_parameters():
    """Test that critics handle decisions with invalid parameters gracefully."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

    # Create decision with missing required parameters
    decision = Decision(
        action=ActionType.INSPECT,
        parameters={},  # Missing asset_id
        confidence=0.9,
    )

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
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    # Should not crash, should handle gracefully
    try:
        _approved, _escalation = await orchestrator.validate_decision(decision, world, risk)
        # As long as it doesn't crash, test passes
        assert True
    except Exception as e:
        pytest.fail(f"Critics should handle invalid parameters gracefully: {e}")


@pytest.mark.asyncio
async def test_none_values_in_world_snapshot():
    """Test handling of None/missing values in world snapshot."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

    # Create vehicle with minimal data (some fields might be None)
    vehicle = VehicleState(
        timestamp=datetime.now(),
        position=Position(latitude=37.0, longitude=-122.0, altitude_msl=50, altitude_agl=50),
        velocity=Velocity(north=0, east=0, down=0),
        attitude=Attitude(roll=0, pitch=0, yaw=0),
        battery=BatteryState(voltage=16.8, current=5.0, remaining_percent=75.0),
        gps=None,  # No GPS data
        mode=FlightMode.GUIDED,
        armed=True,
    )

    world = WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=vehicle,
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

    risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    decision = Decision(action=ActionType.WAIT, parameters={"duration_s": 10}, confidence=0.9)

    # Should handle gracefully
    try:
        _approved, _escalation = await orchestrator.validate_decision(decision, world, risk)
        assert True
    except Exception as e:
        pytest.fail(f"Should handle None values gracefully: {e}")


@pytest.mark.asyncio
async def test_feedback_with_invalid_decision_id():
    """Test that invalid decision IDs in feedback are handled correctly."""
    tracker = OutcomeTracker(log_dir="logs/test_outcomes")

    # Feedback for non-existent decision
    feedback = DecisionFeedback(
        decision_id="nonexistent_123",
        status=ExecutionStatus.SUCCESS,
        battery_consumed=5.0,
    )

    result = await tracker.process_feedback(feedback)

    # Should return None for unknown decision, not crash
    assert result is None


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_critic_evaluation_performance():
    """Test that critic evaluation completes within acceptable time (<200ms)."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION, enable_llm=False)

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
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    decision = Decision(action=ActionType.INSPECT, parameters={"asset_id": "test"}, confidence=0.9)

    # Measure execution time
    start_time = time.perf_counter()
    _approved, _escalation = await orchestrator.validate_decision(decision, world, risk)
    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000

    # Should complete in under 200ms for classical evaluation
    assert elapsed_ms < 200, f"Critic evaluation took {elapsed_ms:.2f}ms (target: <200ms)"


@pytest.mark.asyncio
async def test_concurrent_decision_throughput():
    """Test system can handle multiple concurrent decisions efficiently."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION, enable_llm=False)

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
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    # Create 20 decisions
    num_decisions = 20
    decisions = [
        Decision(action=ActionType.INSPECT, parameters={"asset_id": f"asset_{i}"}, confidence=0.9)
        for i in range(num_decisions)
    ]

    # Process all concurrently
    start_time = time.perf_counter()
    tasks = [orchestrator.validate_decision(d, world, risk) for d in decisions]
    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_decisions

    assert len(results) == num_decisions
    assert avg_time_ms < 100, f"Average time per decision: {avg_time_ms:.2f}ms (target: <100ms)"


@pytest.mark.asyncio
async def test_outcome_tracker_performance():
    """Test that outcome tracking scales well with many outcomes."""
    tracker = OutcomeTracker(log_dir="logs/test_outcomes")

    # Create 100 outcomes
    num_outcomes = 100
    decisions = [
        Decision(action=ActionType.INSPECT, parameters={}, confidence=0.9)
        for _ in range(num_outcomes)
    ]

    start_time = time.perf_counter()

    # Create all outcomes
    for decision in decisions:
        tracker.create_outcome(decision)

    # Simulate feedback for all
    for decision in decisions:
        feedback = DecisionFeedback(
            decision_id=decision.decision_id,
            status=ExecutionStatus.SUCCESS,
            battery_consumed=5.0,
        )
        await tracker.process_feedback(feedback)

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    # Should handle 100 outcomes quickly
    assert total_time_ms < 1000, f"Outcome tracking took {total_time_ms:.2f}ms (target: <1000ms)"


# ============================================================================
# Boundary Condition Tests
# ============================================================================


@pytest.mark.asyncio
async def test_extreme_risk_values():
    """Test handling of extreme risk values (0.0, 1.0)."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

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
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    # Test with zero risk
    zero_risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.0,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.0, threshold=0.5, critical=0.8, description="Perfect"
            )
        },
    )

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=1.0)

    approved, escalation = await orchestrator.validate_decision(decision, world, zero_risk)
    assert approved is True  # Zero risk should approve

    # Test with maximum risk
    max_risk = RiskAssessment(
        overall_level=RiskLevel.CRITICAL,
        overall_score=1.0,
        factors={
            "battery": RiskFactor(
                name="battery", value=1.0, threshold=0.5, critical=0.8, description="Critical"
            ),
        },
        abort_recommended=True,
    )

    approved, escalation = await orchestrator.validate_decision(decision, world, max_risk)
    assert escalation is not None  # Max risk should trigger escalation


@pytest.mark.asyncio
async def test_empty_assets_list():
    """Test handling of missions with no assets."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)

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
        assets=[],  # Empty assets
        anomalies=[],
        mission=MissionState(
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    decision = Decision(action=ActionType.RETURN, parameters={}, confidence=0.9)

    # Should handle gracefully
    approved, _escalation = await orchestrator.validate_decision(decision, world, risk)
    assert approved is True  # Returning with no assets should be fine


# ============================================================================
# Memory and Resource Tests
# ============================================================================


@pytest.mark.asyncio
async def test_outcome_tracker_pending_cleanup():
    """Test that completed outcomes are removed from pending dict (memory management)."""
    tracker = OutcomeTracker(log_dir="logs/test_outcomes")

    # Create outcomes
    decisions = [
        Decision(action=ActionType.INSPECT, parameters={}, confidence=0.9) for _ in range(10)
    ]

    for decision in decisions:
        tracker.create_outcome(decision)

    # All should be pending
    assert tracker.get_pending_count() == 10

    # Complete half of them
    for i in range(5):
        feedback = DecisionFeedback(
            decision_id=decisions[i].decision_id,
            status=ExecutionStatus.SUCCESS,
        )
        await tracker.process_feedback(feedback)

    # Only 5 should remain pending
    assert tracker.get_pending_count() == 5


# ============================================================================
# Error Recovery Tests
# ============================================================================


@pytest.mark.asyncio
async def test_critic_graceful_degradation():
    """Test that if one critic fails, system continues with remaining critics."""
    # We'll test that the orchestrator handles this internally
    # For now, just verify critics don't crash with unusual inputs
    safety_critic = SafetyCritic()

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
            mission_id="test", mission_name="Test", assets_total=0, assets_inspected=0
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.9)

    # Should not crash
    try:
        response = await safety_critic.evaluate_fast(decision, world, risk)
        assert response.verdict in [CriticVerdict.APPROVE, CriticVerdict.APPROVE_WITH_CONCERNS]
    except Exception as e:
        pytest.fail(f"Critic should handle gracefully: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
