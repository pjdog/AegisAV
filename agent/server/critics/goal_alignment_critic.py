"""
Goal Alignment Critic

Validates that decisions align with mission objectives, asset priorities,
and strategic goals.
"""

import logging

from agent.server.critics.base import BaseCritic
from agent.server.decision import Decision
from agent.server.models.critic_models import (
    CriticConfig,
    CriticResponse,
    CriticType,
    CriticVerdict,
)
from agent.server.risk_evaluator import RiskAssessment
from agent.server.world_model import WorldSnapshot

logger = logging.getLogger(__name__)


class GoalAlignmentCriticConfig(CriticConfig):
    """Goal alignment specific configuration."""

    # Priority thresholds
    max_priority_deviation: int = 2  # How far can we deviate from highest priority
    min_asset_priority_for_inspection: int = 3  # Min priority to justify inspection

    # Mission consistency
    allow_out_of_order_inspection: bool = True  # Allow inspecting lower priority first
    require_anomaly_followup: bool = True  # Must re-inspect anomalies

    # Strategic alignment
    max_detour_for_low_priority: float = 200.0  # Max meters detour for low priority assets


class GoalAlignmentCritic(BaseCritic):
    """
    Goal alignment critic validates mission consistency.

    Checks:
    - Decision aligns with current mission objectives
    - Asset priorities are respected
    - Anomalies are followed up appropriately
    - Strategic consistency (not contradicting recent decisions)
    - Logical flow of actions
    """

    def __init__(
        self, config: GoalAlignmentCriticConfig | None = None, llm_model: str | None = None
    ):
        """Initialize goal alignment critic with configuration."""
        self.alignment_config = config or GoalAlignmentCriticConfig()
        super().__init__(config=self.alignment_config, llm_model=llm_model)

    def _get_critic_type(self) -> CriticType:
        """Return GOAL_ALIGNMENT critic type."""
        return CriticType.GOAL_ALIGNMENT

    async def evaluate_fast(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> CriticResponse:
        """
        Fast classical goal alignment evaluation.

        Performs rule-based checks on:
        - Mission objective alignment
        - Asset priority respect
        - Anomaly follow-up
        - Strategic consistency

        Returns:
            CriticResponse with verdict and alignment analysis
        """
        concerns: list[str] = []
        alternatives: list[str] = []
        max_risk_score = 0.0

        # 1. Mission Objective Alignment
        objective_concern, objective_alternatives, objective_risk = self._check_mission_alignment(
            decision, world
        )
        if objective_concern:
            concerns.extend(objective_concern)
            alternatives.extend(objective_alternatives)
        max_risk_score = max(max_risk_score, objective_risk)

        # 2. Asset Priority Check
        priority_concern, priority_alternatives, priority_risk = self._check_asset_priorities(
            decision, world
        )
        if priority_concern:
            concerns.extend(priority_concern)
            alternatives.extend(priority_alternatives)
        max_risk_score = max(max_risk_score, priority_risk)

        # 3. Anomaly Follow-up Check
        anomaly_concern, anomaly_alternatives, anomaly_risk = self._check_anomaly_followup(
            decision, world
        )
        if anomaly_concern:
            concerns.extend(anomaly_concern)
            alternatives.extend(anomaly_alternatives)
        max_risk_score = max(max_risk_score, anomaly_risk)

        # 4. Strategic Consistency Check
        strategy_concern, strategy_alternatives, strategy_risk = self._check_strategic_consistency(
            decision, world, risk
        )
        if strategy_concern:
            concerns.extend(strategy_concern)
            alternatives.extend(strategy_alternatives)
        max_risk_score = max(max_risk_score, strategy_risk)

        # Determine verdict
        verdict, reasoning, confidence = self._determine_verdict(concerns, max_risk_score, decision)

        return CriticResponse(
            critic_type=self.critic_type,
            verdict=verdict,
            confidence=confidence,
            concerns=concerns,
            alternatives=alternatives,
            reasoning=reasoning,
            risk_score=max_risk_score,
        )

    def _check_mission_alignment(
        self, decision: Decision, world: WorldSnapshot
    ) -> tuple[list[str], list[str], float]:
        """
        Check if decision aligns with overall mission objectives.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        # Check if mission is complete but we're not returning
        if world.mission.assets_inspected >= world.mission.assets_total:
            if decision.action.value not in {"RETURN", "DOCK", "RECHARGE"}:
                concerns.append(
                    f"Mission complete ({world.mission.assets_inspected}/{world.mission.assets_total} assets) "
                    f"but decision is {decision.action.value}"
                )
                alternatives.append("Return to dock - mission complete")
                risk_score = 0.5

        # Check if we're returning when mission is incomplete and battery is sufficient
        if decision.action.value == "return":
            remaining = world.mission.assets_total - world.mission.assets_inspected
            if remaining > 0 and world.vehicle.battery.remaining_percent > 50.0:
                # Check if there are pending assets (high priority = low number)
                pending_assets = world.get_pending_assets()
                if pending_assets:
                    # High priority = priority <= 3
                    high_priority = [a for a in pending_assets if a.priority <= 3]
                    if high_priority:
                        concerns.append(
                            f"Returning with {len(high_priority)} high-priority assets uninspected "
                            f"and {world.vehicle.battery.remaining_percent:.1f}% battery"
                        )
                        alternatives.append("Inspect high-priority assets before return")
                        risk_score = 0.6
                    elif len(pending_assets) > 0:
                        # Even if no high-priority, flag any pending assets with good battery
                        concerns.append(
                            f"Returning with {len(pending_assets)} assets uninspected "
                            f"and {world.vehicle.battery.remaining_percent:.1f}% battery"
                        )
                        alternatives.append("Inspect remaining assets before return")
                        risk_score = 0.4

        return concerns, alternatives, risk_score

    def _check_asset_priorities(
        self, decision: Decision, world: WorldSnapshot
    ) -> tuple[list[str], list[str], float]:
        """
        Check if asset priorities are being respected.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        # Only applicable to inspection decisions
        if decision.action.value not in {"INSPECT", "GOTO"}:
            return concerns, alternatives, risk_score

        # Get target asset
        target_asset_id = decision.parameters.get("asset_id")
        if not target_asset_id:
            return concerns, alternatives, risk_score

        target_asset = next((a for a in world.assets if a.asset_id == target_asset_id), None)
        if not target_asset:
            concerns.append(f"Target asset {target_asset_id} not found in mission")
            alternatives.append("Select valid asset from mission list")
            risk_score = 0.7
            return concerns, alternatives, risk_score

        # Check if inspecting very low priority asset
        if target_asset.priority < self.alignment_config.min_asset_priority_for_inspection:
            concerns.append(f"Inspecting low-priority asset (priority {target_asset.priority}/10)")
            alternatives.append("Focus on higher-priority assets")
            risk_score = 0.3

        # Check if higher priority assets are pending
        pending_assets = world.get_pending_assets()
        if pending_assets:
            highest_priority = max(a.priority for a in pending_assets)
            priority_gap = highest_priority - target_asset.priority

            if priority_gap > self.alignment_config.max_priority_deviation:
                concerns.append(
                    f"Inspecting priority {target_asset.priority} asset when "
                    f"priority {highest_priority} assets are pending"
                )
                alternatives.append("Inspect highest priority assets first")
                alternatives.append("Reorder inspection sequence by priority")
                risk_score = max(risk_score, 0.5)

        return concerns, alternatives, risk_score

    def _check_anomaly_followup(
        self, decision: Decision, world: WorldSnapshot
    ) -> tuple[list[str], list[str], float]:
        """
        Check if anomalies are being followed up appropriately.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        if not self.alignment_config.require_anomaly_followup:
            return concerns, alternatives, risk_score

        # Check if there are unresolved anomalies
        anomaly_assets = world.get_anomaly_assets()
        unresolved = [a for a in anomaly_assets if not a.anomaly_resolved]

        if unresolved and decision.action.value in {"INSPECT", "GOTO"}:
            target_asset_id = decision.parameters.get("asset_id")

            # If inspecting asset without anomaly while anomalies exist
            if target_asset_id and target_asset_id not in [a.asset_id for a in unresolved]:
                # Check severity - only concerned about high-severity anomalies
                high_severity = [
                    a for a in unresolved if a.anomaly_severity and a.anomaly_severity >= 7
                ]
                if high_severity:
                    concerns.append(
                        f"{len(high_severity)} high-severity anomalies unresolved, "
                        f"but inspecting asset without anomaly"
                    )
                    alternatives.append("Prioritize anomaly re-inspection")
                    alternatives.append("Follow up on detected anomalies")
                    risk_score = 0.4

        return concerns, alternatives, risk_score

    def _check_strategic_consistency(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> tuple[list[str], list[str], float]:
        """
        Check for strategic consistency and logical flow.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        # Check for ABORT without clear reason
        if decision.is_abort:
            # Abort should have high risk or clear justification
            if risk.overall_score < 0.6 and not decision.reasoning:
                concerns.append("Abort decision with moderate risk and no clear reasoning")
                alternatives.append("Continue mission if risk is acceptable")
                risk_score = 0.5

        # Check for WAIT without clear purpose
        if decision.action.value == "WAIT":
            wait_duration = decision.parameters.get("duration_s", 0)
            # Long wait without reason is concerning
            if wait_duration > 120 and len(decision.reasoning) < 20:
                concerns.append(f"Long wait ({wait_duration:.0f}s) without clear justification")
                alternatives.append("Continue mission if no blocking conditions")
                risk_score = 0.3

        # Check for contradiction: DOCK when not at dock
        if decision.action.value == "DOCK":
            distance_to_dock = world.distance_to_dock()
            if distance_to_dock > 50:  # More than 50m from dock
                concerns.append(f"DOCK command issued at {distance_to_dock:.0f}m from dock")
                alternatives.append("Navigate to dock vicinity first (GOTO)")
                risk_score = 0.6

        return concerns, alternatives, risk_score

    async def evaluate_llm(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> CriticResponse:
        """
        LLM-based goal alignment evaluation for complex strategic scenarios.

        Uses language model to provide nuanced strategic analysis when:
        - Multiple competing mission objectives exist
        - Priority trade-offs are complex
        - Anomaly handling requires judgment
        - Strategic consistency is unclear

        Returns:
            CriticResponse with detailed LLM reasoning
        """
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel

        # Prepare context for LLM
        pending_assets = world.get_pending_assets()
        anomaly_assets = world.get_anomaly_assets()

        context = {
            "decision": {
                "action": decision.action.value,
                "parameters": decision.parameters,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            },
            "mission": {
                "assets_total": world.mission.assets_total,
                "assets_inspected": world.mission.assets_inspected,
                "progress_percent": world.mission.progress_percent,
                "pending_count": len(pending_assets),
                "anomaly_count": len(anomaly_assets),
            },
            "pending_assets": [
                {
                    "asset_id": a.asset_id,
                    "priority": a.priority,
                    "has_anomaly": a.has_anomaly,
                    "anomaly_severity": a.anomaly_severity if a.has_anomaly else None,
                }
                for a in pending_assets[:5]  # Top 5 pending assets
            ],
            "anomalies": [
                {
                    "asset_id": a.asset_id,
                    "severity": a.anomaly_severity,
                    "resolved": a.anomaly_resolved,
                }
                for a in anomaly_assets
                if a.anomaly_severity and a.anomaly_severity >= 5
            ],
            "vehicle": {
                "battery_percent": world.vehicle.battery.remaining_percent,
            },
            "risk": {
                "overall_level": risk.overall_level.value,
                "overall_score": risk.overall_score,
            },
        }

        # Create LLM agent
        model = OpenAIModel(self.llm_model)
        agent = Agent(
            model,
            system_prompt="""You are a Goal Alignment Critic for autonomous drone operations.

Your role is to evaluate whether decisions align with mission objectives including:
- Mission completion and progress toward goals
- Asset priority respect (priority 1-10, higher = more important)
- Anomaly detection follow-up and resolution
- Strategic consistency and logical flow of actions

Analyze the provided decision context and provide:
1. Goal alignment verdict: APPROVE, APPROVE_WITH_CONCERNS, REJECT, or ESCALATE
2. Specific alignment concerns (if any)
3. Alternative actions that better serve mission objectives (if decision is misaligned)
4. Detailed reasoning explaining your strategic assessment

Focus on mission success and optimal prioritization.
Consider long-term strategic implications, not just immediate actions.
Identify goal conflicts that rule-based systems might miss.""",
        )

        # Get LLM evaluation
        try:
            result = await agent.run(
                f"""Evaluate this drone decision for goal alignment:

Decision: {context["decision"]["action"]}
Parameters: {context["decision"]["parameters"]}
Agent Reasoning: {context["decision"]["reasoning"]}

Mission Status:
- Progress: {context["mission"]["progress_percent"]:.0f}% ({context["mission"]["assets_inspected"]}/{context["mission"]["assets_total"]} assets)
- Pending Assets: {context["mission"]["pending_count"]}
- Active Anomalies: {context["mission"]["anomaly_count"]}
- Battery: {context["vehicle"]["battery_percent"]:.1f}%

Pending Assets (Priority Ordered):
{chr(10).join([f"  - {a['asset_id']}: Priority {a['priority']}/10" + (f", Anomaly Severity {a['anomaly_severity']}" if a["has_anomaly"] else "") for a in context["pending_assets"]])}

Unresolved Anomalies:
{chr(10).join([f"  - {a['asset_id']}: Severity {a['severity']}/10" for a in context["anomalies"]]) if context["anomalies"] else "  None"}

Risk Assessment: {context["risk"]["overall_level"]} (score: {context["risk"]["overall_score"]:.2f})

Evaluate the goal alignment of this decision. Consider:
1. Does this decision advance mission objectives effectively?
2. Are asset priorities being respected?
3. Should anomalies be prioritized?
4. Is there a better strategic alternative?

Provide your goal alignment verdict and reasoning."""
            )

            llm_response = result.data

            # Parse LLM response and map to verdict
            response_lower = llm_response.lower()

            if "reject" in response_lower:
                verdict = CriticVerdict.REJECT
                confidence = 0.80
            elif "escalate" in response_lower:
                verdict = CriticVerdict.ESCALATE
                confidence = 0.75
            elif "concerns" in response_lower or "concern" in response_lower:
                verdict = CriticVerdict.APPROVE_WITH_CONCERNS
                confidence = 0.70
            else:
                verdict = CriticVerdict.APPROVE
                confidence = 0.75

            # Extract concerns from response
            concerns = []
            if "priority" in response_lower or "priorities" in response_lower:
                concerns.append("LLM flagged asset priority concerns")
            if "anomaly" in response_lower or "anomalies" in response_lower:
                concerns.append("LLM flagged anomaly follow-up concerns")
            if "mission" in response_lower and (
                "incomplete" in response_lower or "objective" in response_lower
            ):
                concerns.append("LLM flagged mission objective concerns")
            if "strategic" in response_lower or "inconsistent" in response_lower:
                concerns.append("LLM flagged strategic consistency concerns")

            return CriticResponse(
                critic_type=self.critic_type,
                verdict=verdict,
                confidence=confidence,
                concerns=concerns,
                alternatives=[],
                reasoning=f"LLM Goal Alignment Analysis: {llm_response[:500]}",
                risk_score=risk.overall_score,
                used_llm=True,
            )

        except Exception as e:
            # Fallback to fast evaluation if LLM fails
            self.logger.error(f"LLM evaluation failed: {e}, falling back to classical")
            return await self.evaluate_fast(decision, world, risk)

    def _determine_verdict(
        self, concerns: list[str], max_risk_score: float, _decision: Decision
    ) -> tuple[CriticVerdict, str, float]:
        """
        Determine final verdict based on goal alignment concerns.

        Returns:
            (verdict, reasoning, confidence)
        """
        # Goal alignment issues are usually advisory unless severe

        if max_risk_score >= 0.7 or len(concerns) >= 3:
            verdict = CriticVerdict.ESCALATE
            reasoning = (
                f"Significant goal alignment issues ({len(concerns)} concerns). "
                f"Risk score: {max_risk_score:.2f}. Decision may not serve mission objectives."
            )
            confidence = 0.75

        elif max_risk_score >= 0.5 or len(concerns) >= 2:
            verdict = CriticVerdict.APPROVE_WITH_CONCERNS
            reasoning = (
                f"Goal alignment concerns noted ({len(concerns)} issues). "
                f"Risk score: {max_risk_score:.2f}. Decision proceeds but may not be optimal."
            )
            confidence = 0.70

        elif len(concerns) > 0:
            verdict = CriticVerdict.APPROVE_WITH_CONCERNS
            reasoning = (
                f"Minor goal alignment concerns ({len(concerns)} issue). "
                f"Risk score: {max_risk_score:.2f}. Decision aligns with mission objectives."
            )
            confidence = 0.80

        else:
            verdict = CriticVerdict.APPROVE
            reasoning = (
                f"Decision well-aligned with mission objectives. Risk score: {max_risk_score:.2f}. "
                f"Priorities respected and strategy consistent."
            )
            confidence = 0.90

        return verdict, reasoning, confidence
