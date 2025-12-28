"""
Comprehensive tests for LLM integration in critic system.

Tests LLM evaluation, hybrid logic, explanation agent, cost tracking, and fallback.
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Set fake API key for testing (must be before imports)
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing-only"

from agent.api_models import ActionType
from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.critics.safety_critic import SafetyCritic
from agent.server.decision import Decision
from agent.server.models.critic_models import CriticVerdict
from agent.server.monitoring.explanation_agent import ExplanationAgent
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
# Fixtures
# ============================================================================


@pytest.fixture
def good_world_snapshot():
    """Create a world snapshot in good conditions."""
    return WorldSnapshot(
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
            mission_id="test", mission_name="Test", assets_total=5, assets_inspected=2
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )


@pytest.fixture
def borderline_risk():
    """Create borderline risk (0.5-0.7 range where LLM might be useful)."""
    return RiskAssessment(
        overall_level=RiskLevel.MODERATE,
        overall_score=0.55,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.5, threshold=0.5, critical=0.8, description="Borderline"
            ),
            "weather": RiskFactor(
                name="weather", value=0.6, threshold=0.5, critical=0.8, description="Moderate wind"
            ),
        },
    )


# ============================================================================
# Hybrid Evaluation Logic Tests
# ============================================================================


def test_should_use_llm_high_risk():
    """Test that LLM is triggered for high-risk scenarios."""
    critic = SafetyCritic(llm_model="gpt-4o-mini")
    critic.config.use_llm = True
    critic.config.llm_threshold = 0.5

    high_risk = RiskAssessment(
        overall_level=RiskLevel.HIGH,
        overall_score=0.7,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.7, threshold=0.5, critical=0.8, description="High"
            )
        },
    )

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.8)

    # Should use LLM for high risk
    should_use = critic._should_use_llm(decision, high_risk)
    assert should_use is True


def test_should_use_llm_low_confidence():
    """Test that LLM is triggered for low-confidence decisions."""
    critic = SafetyCritic(llm_model="gpt-4o-mini")
    critic.config.use_llm = True

    low_risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.2,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.2, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    low_confidence_decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.5)

    # Should use LLM for low confidence even with low risk
    should_use = critic._should_use_llm(low_confidence_decision, low_risk)
    assert should_use is True


def test_should_not_use_llm_routine_decision():
    """Test that LLM is NOT used for routine, low-risk decisions."""
    critic = SafetyCritic(llm_model="gpt-4o-mini")
    critic.config.use_llm = True
    critic.config.llm_threshold = 0.5

    low_risk = RiskAssessment(
        overall_level=RiskLevel.LOW,
        overall_score=0.1,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.1, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    routine_decision = Decision(action=ActionType.WAIT, parameters={}, confidence=0.9)

    # Should NOT use LLM for routine decision
    should_use = critic._should_use_llm(routine_decision, low_risk)
    assert should_use is False


# ============================================================================
# LLM Evaluation Tests (Mocked)
# ============================================================================


@pytest.mark.asyncio
async def test_llm_evaluation_approve(good_world_snapshot, borderline_risk):
    """Test LLM evaluation returns APPROVE verdict."""
    critic = SafetyCritic(llm_model="gpt-4o-mini")

    decision = Decision(action=ActionType.INSPECT, parameters={"asset_id": "test"}, confidence=0.8)

    # Mock LLM response
    mock_result = Mock()
    mock_result.data = (
        "After analyzing the situation, I APPROVE this decision. Battery is sufficient at 75%."
    )

    # Mock the Agent class to avoid real API calls (patch where it's imported from)
    with (
        patch("agent.server.critics.safety_critic.Agent") as MockAgent,
        patch("agent.server.critics.safety_critic.OpenAIModel"),
    ):
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        response = await critic.evaluate_llm(decision, good_world_snapshot, borderline_risk)

        assert response.verdict == CriticVerdict.APPROVE
        assert response.used_llm is True
        assert "LLM Safety Analysis" in response.reasoning


@pytest.mark.asyncio
async def test_llm_evaluation_reject(good_world_snapshot):
    """Test LLM evaluation returns REJECT verdict."""
    critic = SafetyCritic(llm_model="gpt-4o-mini")

    high_risk = RiskAssessment(
        overall_level=RiskLevel.HIGH,
        overall_score=0.8,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.85, threshold=0.5, critical=0.8, description="Critical"
            )
        },
    )

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.7)

    # Mock LLM response with REJECT
    mock_result = Mock()
    mock_result.data = (
        "I must REJECT this decision due to critically low battery and high risk score."
    )

    with (
        patch("agent.server.critics.safety_critic.Agent") as MockAgent,
        patch("agent.server.critics.safety_critic.OpenAIModel"),
    ):
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        response = await critic.evaluate_llm(decision, good_world_snapshot, high_risk)

        assert response.verdict == CriticVerdict.REJECT
        assert response.used_llm is True


@pytest.mark.asyncio
async def test_llm_evaluation_with_concerns(good_world_snapshot, borderline_risk):
    """Test LLM identifies specific concerns."""
    critic = SafetyCritic(llm_model="gpt-4o-mini")

    decision = Decision(action=ActionType.ORBIT, parameters={}, confidence=0.75)

    # Mock LLM response mentioning battery and wind
    mock_result = Mock()
    mock_result.data = """APPROVE with CONCERNS.
    Battery is low and wind conditions are high.
    Monitor battery closely during orbit."""

    with (
        patch("agent.server.critics.safety_critic.Agent") as MockAgent,
        patch("agent.server.critics.safety_critic.OpenAIModel"),
    ):
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        response = await critic.evaluate_llm(decision, good_world_snapshot, borderline_risk)

        assert response.verdict == CriticVerdict.APPROVE_WITH_CONCERNS
        assert response.used_llm is True
        assert len(response.concerns) > 0


# ============================================================================
# LLM Fallback Tests
# ============================================================================


@pytest.mark.asyncio
async def test_llm_fallback_on_error(good_world_snapshot, borderline_risk):
    """Test that system falls back to classical evaluation if LLM fails."""
    critic = SafetyCritic(llm_model="gpt-4o-mini")

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.8)

    # Mock LLM to raise exception
    with (
        patch("agent.server.critics.safety_critic.Agent") as MockAgent,
        patch("agent.server.critics.safety_critic.OpenAIModel"),
    ):
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("API timeout"))
        MockAgent.return_value = mock_agent

        response = await critic.evaluate_llm(decision, good_world_snapshot, borderline_risk)

        # Should fallback to fast evaluation
        assert response.used_llm is False or response.used_llm is None
        # Should still return a valid response
        assert response.verdict in [CriticVerdict.APPROVE, CriticVerdict.APPROVE_WITH_CONCERNS]


# ============================================================================
# Explanation Agent Tests
# ============================================================================


@pytest.mark.asyncio
async def test_explanation_agent_generates_audit_trail(good_world_snapshot, borderline_risk):
    """Test that explanation agent generates comprehensive audit trails."""
    agent = ExplanationAgent(llm_model="gpt-4o-mini")

    decision = Decision(
        action=ActionType.INSPECT,
        parameters={"asset_id": "test"},
        confidence=0.85,
        reasoning="Inspecting high-priority asset",
    )

    audit_trail = await agent.generate_audit_trail(decision, good_world_snapshot, borderline_risk)

    # Verify audit trail structure
    assert audit_trail.decision_id == decision.decision_id
    assert audit_trail.approved is True
    assert audit_trail.approver == "decision_maker"
    assert "inspect" in audit_trail.summary.lower()
    assert len(audit_trail.reasoning_steps) >= 3
    assert len(audit_trail.factor_contributions) > 0
    assert len(audit_trail.counterfactuals) > 0


@pytest.mark.asyncio
async def test_explanation_agent_identifies_factors(good_world_snapshot, borderline_risk):
    """Test that explanation agent identifies key decision factors."""
    agent = ExplanationAgent(llm_model="gpt-4o-mini")

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.8)

    audit_trail = await agent.generate_audit_trail(decision, good_world_snapshot, borderline_risk)

    # Should identify battery, risk, and mission progress factors
    factor_names = [f.factor_name for f in audit_trail.factor_contributions]
    assert "battery_level" in factor_names
    assert "overall_risk" in factor_names
    assert "mission_progress" in factor_names


@pytest.mark.asyncio
async def test_explanation_agent_generates_counterfactuals():
    """Test that explanation agent generates counterfactual scenarios."""
    agent = ExplanationAgent(llm_model="gpt-4o-mini")

    # Create world with low battery to trigger counterfactuals
    low_battery_world = WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=VehicleState(
            timestamp=datetime.now(),
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=50, altitude_agl=50),
            velocity=Velocity(north=0, east=0, down=0),
            attitude=Attitude(roll=0, pitch=0, yaw=0),
            battery=BatteryState(voltage=14.5, current=5.0, remaining_percent=25.0),
            gps=GPSState(fix_type=3, satellites_visible=12, hdop=0.8),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
        ),
        environment=EnvironmentState(timestamp=datetime.now()),
        assets=[],
        anomalies=[],
        mission=MissionState(
            mission_id="test", mission_name="Test", assets_total=5, assets_inspected=2
        ),
        dock=DockState(
            position=Position(latitude=37.0, longitude=-122.0, altitude_msl=0, altitude_agl=0),
            status=DockStatus.AVAILABLE,
        ),
    )

    high_risk = RiskAssessment(
        overall_level=RiskLevel.HIGH,
        overall_score=0.75,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.8, threshold=0.5, critical=0.8, description="Low"
            )
        },
    )

    decision = Decision(action=ActionType.RETURN, parameters={}, confidence=0.9)

    audit_trail = await agent.generate_audit_trail(decision, low_battery_world, high_risk)

    # Should generate counterfactuals for low battery and high risk scenarios
    assert len(audit_trail.counterfactuals) > 0
    counterfactual_names = [c.scenario_name for c in audit_trail.counterfactuals]
    assert any("battery" in desc.lower() for desc in counterfactual_names)


@pytest.mark.asyncio
async def test_explanation_agent_llm_explanation(good_world_snapshot, borderline_risk):
    """Test that explanation agent generates natural language explanations."""
    agent = ExplanationAgent(llm_model="gpt-4o-mini")

    decision = Decision(
        action=ActionType.INSPECT,
        parameters={"asset_id": "test"},
        confidence=0.85,
        reasoning="Inspecting asset based on priority",
    )

    # Mock LLM response
    mock_result = Mock()
    mock_result.data = "The drone decided to inspect the asset because battery is sufficient and mission progress is on track."

    with (
        patch("agent.server.monitoring.explanation_agent.Agent") as MockAgent,
        patch("agent.server.monitoring.explanation_agent.OpenAIModel"),
    ):
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        explanation = await agent.explain_decision_llm(
            decision, good_world_snapshot, borderline_risk
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "drone" in explanation.lower() or "asset" in explanation.lower()


@pytest.mark.asyncio
async def test_explanation_agent_fallback_on_llm_failure(good_world_snapshot, borderline_risk):
    """Test explanation agent falls back gracefully if LLM fails."""
    agent = ExplanationAgent(llm_model="gpt-4o-mini")

    decision = Decision(
        action=ActionType.INSPECT,
        parameters={},
        confidence=0.8,
        reasoning="Original reasoning from decision maker",
    )

    # Mock LLM to fail
    with (
        patch("agent.server.monitoring.explanation_agent.Agent") as MockAgent,
        patch("agent.server.monitoring.explanation_agent.OpenAIModel"),
    ):
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM unavailable"))
        MockAgent.return_value = mock_agent

        explanation = await agent.explain_decision_llm(
            decision, good_world_snapshot, borderline_risk
        )

        # Should fallback to original reasoning
        assert explanation == decision.reasoning


# ============================================================================
# Orchestrator LLM Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_uses_llm_for_hierarchical_review(good_world_snapshot):
    """Test that orchestrator triggers LLM for hierarchical review in high-risk scenarios."""
    orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION, enable_llm=True)

    critical_risk = RiskAssessment(
        overall_level=RiskLevel.CRITICAL,
        overall_score=0.9,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.95, threshold=0.5, critical=0.8, description="Critical"
            ),
        },
        abort_recommended=True,
    )

    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.7)

    # Mock LLM for all critics
    mock_result = Mock()
    mock_result.data = "REJECT - Critical battery level makes this decision unsafe."

    with (
        patch("agent.server.critics.safety_critic.Agent") as SafetyAgent,
        patch("agent.server.critics.safety_critic.OpenAIModel"),
        patch("agent.server.critics.efficiency_critic.Agent") as EfficiencyAgent,
        patch("agent.server.critics.efficiency_critic.OpenAIModel"),
        patch("agent.server.critics.goal_alignment_critic.Agent") as GoalAgent,
        patch("agent.server.critics.goal_alignment_critic.OpenAIModel"),
    ):
        SafetyAgent.return_value.run = AsyncMock(return_value=mock_result)
        EfficiencyAgent.return_value.run = AsyncMock(return_value=mock_result)
        GoalAgent.return_value.run = AsyncMock(return_value=mock_result)

        _approved, escalation = await orchestrator.validate_decision(
            decision, good_world_snapshot, critical_risk
        )

        # For critical risk (0.9), should trigger hierarchical review
        assert escalation is not None
        # Should use LLM for at least some critics in hierarchical mode
        # (exact behavior depends on force_llm parameter in orchestrator)


# ============================================================================
# Cost and Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_llm_calls_tracked_for_cost_monitoring():
    """Test that LLM calls are tracked for cost monitoring."""
    # This is a placeholder for future cost tracking implementation
    # In production, track:
    # - Number of LLM calls
    # - Tokens used (input + output)
    # - Estimated cost
    # - Call latency

    # For now, just verify we can count calls
    call_count = 0

    async def mock_llm_call(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)
        mock_result = Mock()
        mock_result.data = "APPROVE"
        return mock_result

    critic = SafetyCritic(llm_model="gpt-4o-mini")
    decision = Decision(action=ActionType.INSPECT, parameters={}, confidence=0.8)
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
        overall_level=RiskLevel.MODERATE,
        overall_score=0.5,
        factors={
            "battery": RiskFactor(
                name="battery", value=0.5, threshold=0.5, critical=0.8, description="OK"
            )
        },
    )

    with (
        patch("agent.server.critics.safety_critic.Agent") as MockAgent,
        patch("agent.server.critics.safety_critic.OpenAIModel"),
    ):
        mock_agent = Mock()
        mock_agent.run = mock_llm_call
        MockAgent.return_value = mock_agent

        await critic.evaluate_llm(decision, world, risk)

        # Verify LLM was called
        assert call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
