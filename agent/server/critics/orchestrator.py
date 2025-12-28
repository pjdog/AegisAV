"""
Critic Orchestrator

Coordinates multiple critic agents and implements the escalation authority model.
Manages parallel critic execution and consensus-based decision validation.
"""

import asyncio
import logging
from enum import Enum

from agent.server.critics.efficiency_critic import EfficiencyCritic
from agent.server.critics.goal_alignment_critic import GoalAlignmentCritic
from agent.server.critics.safety_critic import SafetyCritic
from agent.server.decision import Decision
from agent.server.models.critic_models import (
    CriticResponse,
    CriticVerdict,
    EscalationDecision,
    EscalationLevel,
)
from agent.server.risk_evaluator import RiskAssessment
from agent.server.world_model import WorldSnapshot

logger = logging.getLogger(__name__)


class AuthorityModel(Enum):
    """Authority models for critic orchestration."""

    ADVISORY = "advisory"  # Critics advise but don't block
    BLOCKING = "blocking"  # Critics can veto decisions
    ESCALATION = "escalation"  # Escalation based on risk level
    HIERARCHICAL = "hierarchical"  # Always use hierarchical review


class CriticOrchestrator:
    """
    Orchestrates multiple critics and applies authority model.

    The orchestrator:
    1. Runs all critics in parallel for efficiency
    2. Aggregates their responses
    3. Applies the authority model to determine if decision proceeds
    4. Generates escalation decisions when needed
    """

    def __init__(
        self,
        authority_model: str | AuthorityModel = AuthorityModel.ESCALATION,
        enable_llm: bool = True,
    ):
        """
        Initialize critic orchestrator.

        Args:
            authority_model: Which authority model to use
            enable_llm: Whether to enable LLM evaluation for critics
        """
        self.authority_model = (
            AuthorityModel(authority_model) if isinstance(authority_model, str) else authority_model
        )

        # Initialize all critics
        self.critics = [
            SafetyCritic(llm_model="openai:gpt-4o-mini"),
            EfficiencyCritic(llm_model="openai:gpt-4o-mini"),
            GoalAlignmentCritic(llm_model="openai:gpt-4o-mini"),
        ]

        # Configure LLM usage
        for critic in self.critics:
            critic.config.use_llm = enable_llm

        logger.info(
            f"Critic Orchestrator initialized: authority={self.authority_model.value}, "
            f"critics={len(self.critics)}, llm_enabled={enable_llm}"
        )

    async def validate_decision(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> tuple[bool, EscalationDecision | None]:
        """
        Validate decision using all critics.

        Args:
            decision: The decision to validate
            world: Current world state
            risk: Risk assessment

        Returns:
            (approved, escalation_decision)
            - approved: True if decision can proceed, False if blocked
            - escalation_decision: Details if decision was escalated/blocked
        """
        logger.info(
            f"Validating decision: {decision.action.value} "
            f"(confidence: {decision.confidence:.2f}, risk: {risk.overall_score:.2f})"
        )

        # Run all critics in parallel
        critic_responses = await self._run_critics(decision, world, risk)

        # Apply authority model
        if self.authority_model == AuthorityModel.ADVISORY:
            return await self._apply_advisory_model(decision, risk, critic_responses)
        if self.authority_model == AuthorityModel.BLOCKING:
            return await self._apply_blocking_model(decision, risk, critic_responses)
        if self.authority_model == AuthorityModel.ESCALATION:
            return await self._apply_escalation_model(decision, world, risk, critic_responses)
        if self.authority_model == AuthorityModel.HIERARCHICAL:
            return await self._apply_hierarchical_model(decision, world, risk, critic_responses)

        logger.error("Unknown authority model: %s", self.authority_model)
        return True, None

    async def _run_critics(
        self,
        decision: Decision,
        world: WorldSnapshot,
        risk: RiskAssessment,
        force_llm: bool = False,
    ) -> list[CriticResponse]:
        """
        Run all critics in parallel.

        Args:
            decision: Decision to evaluate
            world: World state
            risk: Risk assessment
            force_llm: Force LLM evaluation for all critics

        Returns:
            List of CriticResponse from all critics
        """
        # Execute all critics concurrently
        tasks = [
            critic.evaluate(decision, world, risk, force_llm=force_llm) for critic in self.critics
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(
                    f"Critic {self.critics[i].critic_type.value} failed: {response}",
                    exc_info=response,
                )
            else:
                valid_responses.append(response)

        return valid_responses

    async def _apply_advisory_model(
        self, _decision: Decision, _risk: RiskAssessment, responses: list[CriticResponse]
    ) -> tuple[bool, EscalationDecision | None]:
        """
        Advisory mode: Always approve, just log concerns.

        Returns:
            (True, None) - Always approved
        """
        # Count concerns for logging
        total_concerns = sum(len(r.concerns) for r in responses)
        rejections = [r for r in responses if r.verdict == CriticVerdict.REJECT]

        if total_concerns > 0:
            logger.warning(
                f"Advisory mode: {total_concerns} concerns from {len(responses)} critics "
                f"({len(rejections)} rejections), but approving decision"
            )

        return True, None

    async def _apply_blocking_model(
        self, _decision: Decision, _risk: RiskAssessment, responses: list[CriticResponse]
    ) -> tuple[bool, EscalationDecision | None]:
        """
        Blocking mode: Reject if ANY critic rejects.

        Returns:
            (approved, escalation_decision)
        """
        rejections = [r for r in responses if r.verdict == CriticVerdict.REJECT]

        if rejections:
            # Decision blocked
            consensus_score = self._calculate_consensus(responses)
            escalation = EscalationDecision(
                escalation_level=EscalationLevel.BLOCKING,
                reason=f"{len(rejections)} critic(s) rejected decision",
                recommended_action="Abort or select alternative",
                critic_responses=responses,
                consensus_score=consensus_score,
                approved=False,
                requires_human_review=len(rejections) >= 2,  # Multiple rejections need review
            )

            logger.warning(
                f"Decision BLOCKED: {len(rejections)} rejections, consensus: {consensus_score:.2f}"
            )

            return False, escalation

        return True, None

    async def _apply_escalation_model(
        self,
        decision: Decision,
        world: WorldSnapshot,
        risk: RiskAssessment,
        responses: list[CriticResponse],
    ) -> tuple[bool, EscalationDecision | None]:
        """
        Escalation mode: Advisory → Blocking → Hierarchical based on risk.

        Rules:
        - risk >= 0.7: HIERARCHICAL (re-evaluate with LLM, full review)
        - risk >= 0.4: BLOCKING (can reject decisions)
        - risk < 0.4: ADVISORY (log warnings only)

        Returns:
            (approved, escalation_decision)
        """
        # Count rejections and escalations
        rejections = [r for r in responses if r.verdict == CriticVerdict.REJECT]
        escalations = [r for r in responses if r.verdict == CriticVerdict.ESCALATE]

        # HIERARCHICAL REVIEW: High risk or explicit escalations
        if risk.overall_score >= 0.7 or len(escalations) > 0:
            logger.info(
                f"HIERARCHICAL review triggered: risk={risk.overall_score:.2f}, "
                f"escalations={len(escalations)}"
            )
            return await self._hierarchical_review(decision, world, risk, responses)

        # BLOCKING MODE: Moderate risk
        if risk.overall_score >= 0.4:
            if rejections:
                consensus_score = self._calculate_consensus(responses)
                escalation = EscalationDecision(
                    escalation_level=EscalationLevel.BLOCKING,
                    reason=f"{len(rejections)} rejections in moderate risk scenario",
                    recommended_action="Abort or select safer alternative",
                    critic_responses=responses,
                    consensus_score=consensus_score,
                    approved=False,
                )

                logger.warning(f"Decision BLOCKED (moderate risk): {len(rejections)} rejections")
                return False, escalation

            # Approved but log concerns
            total_concerns = sum(len(r.concerns) for r in responses)
            if total_concerns > 0:
                logger.info(f"Decision APPROVED with {total_concerns} concerns (moderate risk)")
            return True, None

        # ADVISORY MODE: Low risk
        total_concerns = sum(len(r.concerns) for r in responses)
        if total_concerns > 0:
            logger.info("Advisory mode: %s concerns logged, decision approved", total_concerns)
        return True, None

    async def _apply_hierarchical_model(
        self,
        decision: Decision,
        world: WorldSnapshot,
        risk: RiskAssessment,
        responses: list[CriticResponse],
    ) -> tuple[bool, EscalationDecision | None]:
        """
        Hierarchical mode: Always use full LLM review.

        Returns:
            (approved, escalation_decision)
        """
        logger.info("Hierarchical model: Forcing LLM review for all critics")
        return await self._hierarchical_review(decision, world, risk, responses)

    async def _hierarchical_review(
        self,
        decision: Decision,
        world: WorldSnapshot,
        risk: RiskAssessment,
        _initial_responses: list[CriticResponse],
    ) -> tuple[bool, EscalationDecision | None]:
        """
        Perform hierarchical review with full LLM evaluation.

        Re-evaluates with all critics using LLM for deeper analysis.

        Returns:
            (approved, escalation_decision)
        """
        logger.info("Initiating hierarchical review with LLM evaluation")

        # Re-run all critics with LLM forced on
        detailed_responses = await self._run_critics(decision, world, risk, force_llm=True)

        # Analyze consensus
        rejections = [r for r in detailed_responses if r.verdict == CriticVerdict.REJECT]
        concerns_with = [
            r for r in detailed_responses if r.verdict == CriticVerdict.APPROVE_WITH_CONCERNS
        ]
        consensus_score = self._calculate_consensus(detailed_responses)

        # Generate recommendation based on detailed analysis
        if len(rejections) >= 2:
            # Multiple critics reject: Strong recommendation to abort
            recommended_action = "Abort mission - multiple safety/alignment issues"
            approved = False
            requires_review = True
        elif len(rejections) == 1:
            # Single rejection: Evaluate severity
            rejection = rejections[0]
            if rejection.risk_score >= 0.8:
                recommended_action = "Abort or wait for conditions to improve"
                approved = False
                requires_review = True
            else:
                recommended_action = "Consider alternative or proceed with extreme caution"
                approved = False
                requires_review = False
        else:
            # No rejections but many concerns
            if len(concerns_with) >= 2:
                recommended_action = "Proceed with heightened monitoring"
                approved = True
                requires_review = False
            else:
                recommended_action = "Approve - concerns manageable"
                approved = True
                requires_review = False

        escalation = EscalationDecision(
            escalation_level=EscalationLevel.HIERARCHICAL,
            reason=(
                f"Hierarchical review: {len(rejections)} rejections, {len(concerns_with)} concerns"
            ),
            recommended_action=recommended_action,
            critic_responses=detailed_responses,
            consensus_score=consensus_score,
            approved=approved,
            requires_human_review=requires_review,
        )

        logger.info(
            f"Hierarchical review complete: approved={approved}, "
            f"consensus={consensus_score:.2f}, requires_review={requires_review}"
        )

        return approved, escalation

    def _calculate_consensus(self, responses: list[CriticResponse]) -> float:
        """
        Calculate consensus score among critics.

        Consensus is based on:
        - Agreement on verdict
        - Similarity in risk scores
        - Overlapping concerns

        Returns:
            Consensus score (0.0 - 1.0), higher = more agreement
        """
        if not responses:
            return 0.0

        # Count verdicts
        verdict_counts = {}
        for response in responses:
            verdict = response.verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Most common verdict
        max_count = max(verdict_counts.values())
        verdict_consensus = max_count / len(responses)

        # Risk score similarity (using standard deviation)
        risk_scores = [r.risk_score for r in responses]
        if len(risk_scores) > 1:
            mean_risk = sum(risk_scores) / len(risk_scores)
            variance = sum((r - mean_risk) ** 2 for r in risk_scores) / len(risk_scores)
            std_dev = variance**0.5
            # Convert to similarity score (lower std_dev = higher consensus)
            risk_consensus = max(0.0, 1.0 - (std_dev * 2))  # Scale std_dev
        else:
            risk_consensus = 1.0

        # Weighted average
        consensus = (verdict_consensus * 0.7) + (risk_consensus * 0.3)

        return consensus

    def get_stats(self) -> dict:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary with critic stats and orchestrator metrics
        """
        return {
            "authority_model": self.authority_model.value,
            "total_critics": len(self.critics),
            "critic_stats": [critic.get_stats() for critic in self.critics],
        }
