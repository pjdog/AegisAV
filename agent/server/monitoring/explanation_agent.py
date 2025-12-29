"""Explanation Agent.

Generates detailed audit trails and explanations for decisions using LLM.
Provides counterfactual analysis and factor contributions.
"""

import logging
import time
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from agent.server.decision import Decision
from agent.server.models.audit_models import (
    AuditTrail,
    CounterfactualScenario,
    FactorContribution,
    ReasoningStep,
)
from agent.server.models.critic_models import EscalationDecision
from agent.server.monitoring.cost_tracker import CallDetails, estimate_tokens, get_cost_tracker
from agent.server.risk_evaluator import RiskAssessment
from agent.server.world_model import WorldSnapshot

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """Generates detailed explanations and audit trails for decisions.

    Responsibilities:
    - Create comprehensive audit trails
    - Generate counterfactual scenarios ("what if" analysis)
    - Explain factor contributions
    - Provide natural language explanations
    """

    def __init__(self, llm_model: str = "gpt-4o-mini") -> None:
        """Initialize explanation agent.

        Args:
            llm_model: LLM model to use for explanations
        """
        self.llm_model = llm_model
        self.logger = logger

    async def generate_audit_trail(
        self,
        decision: Decision,
        world: WorldSnapshot,
        risk: RiskAssessment,
        escalation: EscalationDecision | None = None,
    ) -> AuditTrail:
        """Generate a comprehensive audit trail for a decision.

        Args:
            decision: The decision that was made
            world: World state at time of decision
            risk: Risk assessment
            escalation: Escalation decision (if any)

        Returns:
            Complete AuditTrail with reasoning steps and explanations
        """
        # Build reasoning steps
        reasoning_steps = [
            ReasoningStep(
                step_number=1,
                description="World State Analysis",
                inputs={
                    "battery_percent": world.vehicle.battery.remaining_percent,
                    "mission_progress": world.mission.progress_percent,
                    "wind_speed_ms": world.environment.wind_speed_ms,
                },
                outputs={"state_summary": "Analyzed current operational state"},
                confidence=0.95,
            ),
            ReasoningStep(
                step_number=2,
                description="Risk Evaluation",
                inputs={
                    "risk_level": risk.overall_level.value,
                    "risk_score": risk.overall_score,
                },
                outputs={"risk_assessment": f"Risk level: {risk.overall_level.value}"},
                confidence=0.90,
            ),
            ReasoningStep(
                step_number=3,
                description="Decision Selection",
                inputs={
                    "action": decision.action.value,
                    "parameters": decision.parameters,
                },
                outputs={"decision": f"Selected action: {decision.action.value}"},
                confidence=decision.confidence,
            ),
        ]

        # Add critic review step if escalation occurred
        if escalation:
            reasoning_steps.append(
                ReasoningStep(
                    step_number=4,
                    description="Multi-Agent Critic Review",
                    inputs={
                        "escalation_level": escalation.escalation_level.value,
                        "num_critics": len(escalation.critic_responses),
                    },
                    outputs={
                        "approved": escalation.approved,
                        "reason": escalation.reason,
                    },
                    confidence=escalation.consensus_score,
                )
            )

        # Identify key factors
        factor_contributions = self._identify_factors(decision, world, risk)

        # Generate counterfactuals
        counterfactuals = await self._generate_counterfactuals(decision, world, risk)

        return AuditTrail(
            decision_id=decision.decision_id,
            timestamp=datetime.now(),
            reasoning_steps=reasoning_steps,
            factor_contributions=factor_contributions,
            counterfactuals=counterfactuals,
            approved=True,  # If we're generating an audit trail, the decision was approved
            approval_timestamp=datetime.now(),
            approver="decision_maker",
            summary=(
                f"{decision.action.value} decision with {len(counterfactuals)} "
                "counterfactuals analyzed"
            ),
        )

    def _identify_factors(
        self, _decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> list[FactorContribution]:
        """Identify key factors that influenced the decision.

        Returns:
            List of FactorContribution objects
        """
        factors = []

        # Battery factor
        battery_pct = world.vehicle.battery.remaining_percent
        if battery_pct < 30:
            weight = 0.9
        elif battery_pct < 50:
            weight = 0.6
        else:
            weight = 0.3

        # Normalize battery to 0-1 range (0% = 0.0, 100% = 1.0)
        normalized_battery = battery_pct / 100.0
        factors.append(
            FactorContribution(
                factor_name="battery_level",
                value=battery_pct,
                weight=weight,
                contribution=weight * normalized_battery,
                unit="percent",
            )
        )

        # Risk factor (already 0-1)
        risk_weight = risk.overall_score
        factors.append(
            FactorContribution(
                factor_name="overall_risk",
                value=risk.overall_score,
                weight=risk_weight,
                contribution=risk_weight * risk.overall_score,
                unit="score",
            )
        )

        # Mission progress factor
        progress_pct = world.mission.progress_percent
        mission_weight = 0.4 + (progress_pct / 100) * 0.4
        normalized_progress = progress_pct / 100.0
        factors.append(
            FactorContribution(
                factor_name="mission_progress",
                value=progress_pct,
                weight=mission_weight,
                contribution=mission_weight * normalized_progress,
                unit="percent",
            )
        )

        return factors

    async def _generate_counterfactuals(
        self, _decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> list[CounterfactualScenario]:
        """Generate counterfactual "what if" scenarios.

        Returns:
            List of CounterfactualScenario objects
        """
        counterfactuals = []

        # Counterfactual: What if battery was higher?
        if world.vehicle.battery.remaining_percent < 50:
            counterfactuals.append(
                CounterfactualScenario(
                    scenario_name="What if battery was at 80%?",
                    changed_factors={"battery_percent": 80.0},
                    predicted_outcome=(
                        "Mission could continue with more aggressive inspection strategy"
                    ),
                    confidence=0.75,
                    would_change_decision=True,
                )
            )

        # Counterfactual: What if risk was lower?
        if risk.overall_score > 0.5:
            counterfactuals.append(
                CounterfactualScenario(
                    scenario_name="What if risk score was 0.2?",
                    changed_factors={"risk_score": 0.2},
                    predicted_outcome="Decision would likely proceed with higher confidence",
                    confidence=0.80,
                    would_change_decision=False,
                )
            )

        # Counterfactual: What if mission was complete?
        if world.mission.progress_percent < 100:
            counterfactuals.append(
                CounterfactualScenario(
                    scenario_name="What if all assets were inspected?",
                    changed_factors={"mission_progress": 100.0},
                    predicted_outcome=(
                        "Decision would prioritize return to dock over continuing inspection"
                    ),
                    confidence=0.90,
                    would_change_decision=True,
                )
            )

        return counterfactuals

    async def explain_decision_llm(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> str:
        """Generate natural language explanation using LLM.

        Args:
            decision: Decision to explain
            world: World state
            risk: Risk assessment

        Returns:
            Natural language explanation string
        """
        model = OpenAIModel(self.llm_model)
        agent = Agent(
            model,
            system_prompt="""You are an explainability AI for autonomous drone operations.

Your role is to provide clear, concise explanations of decisions made by the drone agent.

Explain:
- What decision was made and why
- Key factors that influenced the decision
- Trade-offs and alternatives considered
- Potential risks and mitigations

Use simple language suitable for operators who may not be AI experts.
Focus on actionable insights and safety-critical information.""",
        )

        start_time = time.time()
        cost_tracker = get_cost_tracker()

        try:
            prompt_text = f"""Explain this drone decision in 2-3 sentences:

Decision: {decision.action.value}
Parameters: {decision.parameters}
Reasoning: {decision.reasoning}

Context:
- Battery: {world.vehicle.battery.remaining_percent:.1f}%
- Mission Progress: {world.mission.progress_percent:.0f}%
- Risk Level: {risk.overall_level.value} (score: {risk.overall_score:.2f})
- Wind: {world.environment.wind_speed_ms:.1f} m/s

Provide a clear, concise explanation suitable for a human operator."""

            result = await agent.run(prompt_text)
            llm_response = result.data

            # Track cost
            latency_ms = (time.time() - start_time) * 1000
            prompt_tokens = estimate_tokens(prompt_text)
            completion_tokens = estimate_tokens(llm_response)

            cost_tracker.record_call(
                CallDetails(
                    model=self.llm_model,
                    context="explanation_agent",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    success=True,
                )
            )

            return llm_response

        except Exception as e:
            # Track failed call
            latency_ms = (time.time() - start_time) * 1000
            cost_tracker.record_call(
                CallDetails(
                    model=self.llm_model,
                    context="explanation_agent",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error_message=str(e),
                )
            )

            self.logger.error(f"LLM explanation failed: {e}")
            return decision.reasoning  # Fallback to original reasoning
