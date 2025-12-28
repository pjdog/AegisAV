"""
Base Critic Architecture

Provides abstract base class for all critic agents with hybrid classical/LLM evaluation.
"""

import logging
import time
from abc import ABC, abstractmethod

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


class BaseCritic(ABC):
    """
    Base class for all critic agents.

    Provides hybrid evaluation strategy:
    - Fast classical algorithms for routine decisions (< 50ms)
    - LLM-based reasoning for complex/ambiguous cases (< 2s)

    Subclasses must implement:
    - _get_critic_type(): Return the critic's type
    - evaluate_fast(): Classical algorithm evaluation
    - evaluate_llm(): LLM-based evaluation (optional, has default)
    """

    def __init__(self, config: CriticConfig | None = None, llm_model: str | None = None):
        """
        Initialize critic.

        Args:
            config: Critic configuration (enables, thresholds, etc.)
            llm_model: LLM model to use (default: "openai:gpt-4o-mini")
        """
        self.config = config or CriticConfig()
        self.llm_model = llm_model or "openai:gpt-4o-mini"
        self.critic_type = self._get_critic_type()
        self.evaluations_performed = 0
        self.llm_evaluations = 0
        self.logger = logger  # Add instance logger for subclasses

        logger.info(
            f"Initialized {self.critic_type.value} critic "
            f"(LLM: {self.config.use_llm}, model: {self.llm_model})"
        )

    @abstractmethod
    def _get_critic_type(self) -> CriticType:
        """
        Return the critic type.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    async def evaluate_fast(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> CriticResponse:
        """
        Fast classical algorithm evaluation.

        This method should use deterministic, rule-based logic to quickly
        assess the decision. Target latency: < 50ms.

        Args:
            decision: The decision to evaluate
            world: Current world state
            risk: Risk assessment for the decision

        Returns:
            CriticResponse with verdict and reasoning

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def evaluate_llm(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> CriticResponse:
        """
        LLM-based evaluation for complex cases.

        Default implementation returns APPROVE with note that LLM is not implemented.
        Subclasses should override this to provide LLM-based reasoning.

        Target latency: < 2s.

        Args:
            decision: The decision to evaluate
            world: Current world state
            risk: Risk assessment for the decision

        Returns:
            CriticResponse with verdict and reasoning
        """
        logger.warning(
            f"{self.critic_type.value} critic: LLM evaluation not implemented, "
            "falling back to fast evaluation"
        )
        return await self.evaluate_fast(decision, world, risk)

    async def evaluate(
        self,
        decision: Decision,
        world: WorldSnapshot,
        risk: RiskAssessment,
        force_llm: bool = False,
    ) -> CriticResponse:
        """
        Main evaluation entry point.

        Chooses between fast (classical) or LLM-based evaluation based on:
        - Configuration (use_llm enabled/disabled)
        - Complexity heuristics (risk score, decision confidence, action type)
        - force_llm parameter (for hierarchical review)

        Args:
            decision: The decision to evaluate
            world: Current world state
            risk: Risk assessment for the decision
            force_llm: Force LLM evaluation regardless of heuristics

        Returns:
            CriticResponse with verdict and reasoning
        """
        start_time = time.time()
        self.evaluations_performed += 1

        try:
            # Determine evaluation method
            use_llm = force_llm or (self.config.use_llm and self._should_use_llm(decision, risk))

            # Perform evaluation
            if use_llm:
                self.llm_evaluations += 1
                logger.debug(f"{self.critic_type.value}: Using LLM evaluation (force={force_llm})")
                response = await self.evaluate_llm(decision, world, risk)
                response.used_llm = True
            else:
                logger.debug(f"{self.critic_type.value}: Using fast evaluation")
                response = await self.evaluate_fast(decision, world, risk)
                response.used_llm = False

            # Record processing time
            response.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"{self.critic_type.value} verdict: {response.verdict.value} "
                f"(confidence: {response.confidence:.2f}, "
                f"time: {response.processing_time_ms:.1f}ms, "
                f"llm: {response.used_llm})"
            )

            return response

        except Exception as e:
            logger.error(f"{self.critic_type.value} evaluation failed: {e}", exc_info=True)
            # Return safe fallback: approve with low confidence and concern
            processing_time = (time.time() - start_time) * 1000
            return CriticResponse(
                critic_type=self.critic_type,
                verdict=CriticVerdict.APPROVE_WITH_CONCERNS,
                confidence=0.3,
                concerns=[f"Evaluation failed: {e!s}"],
                alternatives=[],
                reasoning=(
                    "Critic evaluation encountered error, defaulting to approval with concerns"
                ),
                risk_score=0.5,
                processing_time_ms=processing_time,
                used_llm=False,
            )

    def _should_use_llm(self, decision: Decision, risk: RiskAssessment) -> bool:
        """
        Determine if LLM evaluation is needed.

        Heuristics:
        - High risk (> llm_threshold): Use LLM for safety-critical analysis
        - Low confidence decisions (< 0.7): Use LLM for additional scrutiny
        - Complex actions (ABORT, INSPECT): Use LLM for nuanced reasoning

        Args:
            decision: The decision being evaluated
            risk: Current risk assessment

        Returns:
            True if LLM should be used, False for classical evaluation
        """
        # Check risk score
        if risk.overall_score > self.config.llm_threshold:
            logger.debug(
                f"{self.critic_type.value}: LLM triggered by high risk "
                f"({risk.overall_score:.2f} > {self.config.llm_threshold})"
            )
            return True

        # Check decision confidence
        if decision.confidence < 0.7:
            logger.debug(
                f"{self.critic_type.value}: LLM triggered by low confidence "
                f"({decision.confidence:.2f})"
            )
            return True

        # Check for complex action types
        complex_actions = {"ABORT", "INSPECT", "DOCK"}
        if decision.action.value in complex_actions:
            logger.debug(
                f"{self.critic_type.value}: LLM triggered by complex action "
                f"({decision.action.value})"
            )
            return True

        return False

    def get_stats(self) -> dict:
        """
        Get critic statistics.

        Returns:
            Dictionary with evaluation counts and LLM usage
        """
        return {
            "critic_type": self.critic_type.value,
            "total_evaluations": self.evaluations_performed,
            "llm_evaluations": self.llm_evaluations,
            "llm_usage_rate": (
                self.llm_evaluations / self.evaluations_performed
                if self.evaluations_performed > 0
                else 0.0
            ),
        }
