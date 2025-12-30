"""Safety Critic.

Validates decisions against safety constraints using battery, GPS, weather,
and vehicle health checks.
"""

import logging
import time
from typing import Any

try:
    from pydantic_ai import Agent
except ImportError:  # pragma: no cover - optional dependency
    Agent = None

from agent.server.critics.base import BaseCritic
from agent.server.decision import Decision
from agent.server.models.critic_models import (
    CriticConfig,
    CriticResponse,
    CriticType,
    CriticVerdict,
)
from agent.server.monitoring.cost_tracker import CallDetails, estimate_tokens, get_cost_tracker
from agent.server.risk_evaluator import RiskAssessment
from agent.server.world_model import WorldSnapshot

logger = logging.getLogger(__name__)


class SafetyCriticConfig(CriticConfig):
    """Safety-specific configuration thresholds."""

    # Battery thresholds
    min_battery_percent: float = 20.0
    min_battery_margin: float = 10.0  # Extra margin for return

    # GPS thresholds
    min_gps_satellites: int = 6
    max_gps_hdop: float = 2.0

    # Weather thresholds
    max_wind_ms: float = 12.0
    max_wind_warning_ms: float = 8.0

    # Distance thresholds
    max_safe_distance_m: float = 5000.0


class SafetyCritic(BaseCritic):
    """Safety critic validates decisions against safety constraints.

    Checks:
    - Battery sufficiency for mission + return
    - GPS quality (satellite count, HDOP)
    - Weather safety (wind speed, visibility)
    - Vehicle health status
    - Distance from dock/safe zones
    """

    def __init__(
        self, config: SafetyCriticConfig | None = None, llm_model: str | None = None
    ) -> None:
        """Initialize safety critic with configuration."""
        self.safety_config = config or SafetyCriticConfig()
        super().__init__(config=self.safety_config, llm_model=llm_model)

    def _get_critic_type(self) -> CriticType:
        """Return SAFETY critic type."""
        return CriticType.SAFETY

    async def evaluate_fast(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> CriticResponse:
        """Fast classical safety evaluation.

        Performs rule-based checks on:
        - Battery level and margin
        - GPS quality
        - Weather conditions
        - Vehicle health

        Returns:
            CriticResponse with verdict (APPROVE, APPROVE_WITH_CONCERNS, REJECT, ESCALATE)
        """
        concerns: list[str] = []
        alternatives: list[str] = []
        max_risk_score = 0.0

        for check, args in (
            (self._check_battery, (decision, world, risk)),
            (self._check_gps, (world, risk)),
            (self._check_weather, (decision, world, risk)),
            (self._check_vehicle_health, (world, risk)),
        ):
            check_concerns, check_alternatives, check_risk = check(*args)
            if check_concerns:
                concerns.extend(check_concerns)
                alternatives.extend(check_alternatives)
            max_risk_score = max(max_risk_score, check_risk)

        # Determine verdict based on concerns and risk level
        verdict, reasoning, confidence = self._determine_verdict(
            concerns, max_risk_score, decision, risk
        )

        return CriticResponse(
            critic_type=self.critic_type,
            verdict=verdict,
            confidence=confidence,
            concerns=concerns,
            alternatives=alternatives,
            reasoning=reasoning,
            risk_score=max_risk_score,
        )

    def _check_battery(
        self, decision: Decision, world: WorldSnapshot, _risk: RiskAssessment
    ) -> tuple[list[str], list[str], float]:
        """Check battery safety.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        battery_percent = world.vehicle.battery.remaining_percent

        # Critical: Battery below minimum threshold
        if battery_percent < self.safety_config.min_battery_percent:
            concerns.append(
                f"Battery critically low at {battery_percent:.1f}% "
                f"(min: {self.safety_config.min_battery_percent}%)"
            )
            alternatives.append("Return to dock immediately")
            alternatives.append("Land at current position")
            risk_score = 0.9

        # Warning: Battery below margin threshold
        elif battery_percent < (
            self.safety_config.min_battery_percent + self.safety_config.min_battery_margin
        ):
            concerns.append(f"Battery at {battery_percent:.1f}% approaching minimum threshold")
            alternatives.append("Consider returning to dock soon")
            risk_score = 0.6

        # Check if decision involves movement with low battery
        if decision.is_movement and battery_percent < 25.0:
            # Estimate if there's enough battery for this action + return
            distance_to_dock = world.distance_to_dock()
            if distance_to_dock > 500 and battery_percent < 30.0:
                concerns.append(
                    f"Insufficient battery ({battery_percent:.1f}%) for movement "
                    f"at {distance_to_dock:.0f}m from dock"
                )
                alternatives.append("Reduce distance from dock before continuing mission")
                risk_score = max(risk_score, 0.7)

        return concerns, alternatives, risk_score

    def _check_gps(
        self, world: WorldSnapshot, _risk: RiskAssessment
    ) -> tuple[list[str], list[str], float]:
        """Check GPS quality.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        gps = world.vehicle.gps

        # Check satellite count
        if gps.satellites_visible < self.safety_config.min_gps_satellites:
            concerns.append(
                f"GPS satellite count low ({gps.satellites_visible} satellites, "
                f"min: {self.safety_config.min_gps_satellites})"
            )
            alternatives.append("Wait for better GPS fix")
            alternatives.append("Abort mission if GPS doesn't improve")
            risk_score = 0.8

        # Check HDOP (horizontal dilution of precision)
        if gps.hdop > self.safety_config.max_gps_hdop:
            concerns.append(
                f"GPS accuracy degraded (HDOP: {gps.hdop:.2f}, "
                f"max: {self.safety_config.max_gps_hdop})"
            )
            alternatives.append("Wait for improved GPS accuracy")
            risk_score = max(risk_score, 0.7)

        return concerns, alternatives, risk_score

    def _check_weather(
        self, decision: Decision, world: WorldSnapshot, _risk: RiskAssessment
    ) -> tuple[list[str], list[str], float]:
        """Check weather safety.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        env = world.environment
        wind_speed = env.wind_speed_ms

        # Critical: Wind above abort threshold
        if wind_speed >= self.safety_config.max_wind_ms:
            concerns.append(
                f"Wind speed critically high ({wind_speed:.1f} m/s, "
                f"max: {self.safety_config.max_wind_ms} m/s)"
            )
            alternatives.append("Land immediately")
            alternatives.append("Return to dock if wind decreases")
            risk_score = 0.95

        # Warning: Wind above warning threshold
        elif wind_speed >= self.safety_config.max_wind_warning_ms:
            concerns.append(
                f"Wind speed elevated ({wind_speed:.1f} m/s, "
                f"warning: {self.safety_config.max_wind_warning_ms} m/s)"
            )
            alternatives.append("Monitor wind conditions closely")
            alternatives.append("Consider reducing mission scope")
            risk_score = 0.6

        # Check visibility if available
        if env.visibility_m is not None and env.visibility_m < 100:
            concerns.append(f"Visibility very poor ({env.visibility_m:.0f}m)")
            alternatives.append("Wait for improved visibility")
            risk_score = max(risk_score, 0.7)

        # Check if decision is complex action in marginal weather
        if decision.action.value in {"INSPECT", "ORBIT"} and wind_speed > 6.0:
            concerns.append(
                f"Complex maneuver ({decision.action.value}) in elevated wind "
                f"({wind_speed:.1f} m/s)"
            )
            alternatives.append("Postpone inspection until wind decreases")
            risk_score = max(risk_score, 0.5)

        return concerns, alternatives, risk_score

    def _check_vehicle_health(
        self, world: WorldSnapshot, _risk: RiskAssessment
    ) -> tuple[list[str], list[str], float]:
        """Check vehicle health status.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        vehicle = world.vehicle

        # Check if vehicle is healthy
        if vehicle.health and not vehicle.health.is_healthy:
            concerns.append("Vehicle reporting unhealthy status")
            alternatives.append("Run diagnostics")
            alternatives.append("Return to dock for maintenance")
            risk_score = 0.85

        # Check if vehicle is armed (for movement decisions)
        # (This is more of a logic check than safety, but included here)

        return concerns, alternatives, risk_score

    async def evaluate_llm(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> CriticResponse:
        """LLM-based safety evaluation for complex scenarios.

        Uses language model to provide nuanced safety analysis when:
        - Multiple safety factors conflict
        - Edge cases not captured by rules
        - Risk assessment is borderline

        Returns:
            CriticResponse with detailed LLM reasoning
        """
        if Agent is None:
            logger.warning("pydantic_ai unavailable, falling back to fast evaluation")
            return await self.evaluate_fast(decision, world, risk)

        context = self._build_llm_context(decision, world, risk)
        agent = self._create_llm_agent()
        cost_context = self._start_cost_tracking()

        try:
            prompt_text = self._build_llm_prompt(context)
            result = await agent.run(prompt_text)
            llm_response = result.data

            self._record_llm_success(cost_context, prompt_text, llm_response)
            verdict, confidence = self._parse_llm_verdict(llm_response)
            concerns = self._extract_llm_concerns(llm_response)

            return CriticResponse(
                critic_type=self.critic_type,
                verdict=verdict,
                confidence=confidence,
                concerns=concerns,
                alternatives=[],
                reasoning=f"LLM Safety Analysis: {llm_response[:500]}",
                risk_score=risk.overall_score,
                used_llm=True,
            )

        except Exception as e:
            self._record_llm_failure(cost_context, e)
            self.logger.error(f"LLM evaluation failed: {e}, falling back to classical")
            return await self.evaluate_fast(decision, world, risk)

    def _create_llm_agent(self) -> Agent:
        """Create the LLM agent for safety evaluation."""
        return Agent(
            self.llm_model,
            system_prompt="""You are a Safety Critic for autonomous drone operations.

Your role is to evaluate decisions for safety including:
- Battery safety margin and return capability
- Weather safety (wind, visibility)
- GPS quality and navigation reliability
- Vehicle health and system status

Analyze the provided decision context and provide:
1. Safety verdict: APPROVE, APPROVE_WITH_CONCERNS, REJECT, or ESCALATE
2. Specific safety concerns (if any)
3. Alternative safer actions (if decision is unsafe)
4. Detailed reasoning explaining your safety assessment

Prioritize safety above mission objectives.""",
        )

    @staticmethod
    def _start_cost_tracking() -> tuple[float, Any]:
        """Return a tracking context for LLM cost reporting."""
        return time.time(), get_cost_tracker()

    def _record_llm_success(
        self,
        cost_context: tuple[float, Any],
        prompt_text: str,
        llm_response: str,
    ) -> None:
        """Record a successful LLM call in the cost tracker."""
        start_time, cost_tracker = cost_context
        latency_ms = (time.time() - start_time) * 1000
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(llm_response)

        cost_tracker.record_call(
            CallDetails(
                model=self.llm_model,
                context="safety_critic",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=True,
            )
        )

    def _record_llm_failure(self, cost_context: tuple[float, Any], error: Exception) -> None:
        """Record a failed LLM call in the cost tracker."""
        start_time, cost_tracker = cost_context
        latency_ms = (time.time() - start_time) * 1000
        cost_tracker.record_call(
            CallDetails(
                model=self.llm_model,
                context="safety_critic",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error_message=str(error),
            )
        )

    @staticmethod
    def _build_llm_context(
        decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> dict[str, Any]:
        """Build the LLM context payload for safety evaluation."""
        environment = world.environment
        wind_speed = 0.0
        visibility_m = 0.0
        if environment is not None:
            try:
                wind_speed = environment.wind_speed_ms
            except AttributeError:
                wind_speed = 0.0
            try:
                visibility_m = environment.visibility_m
            except AttributeError:
                visibility_m = 0.0
        return {
            "decision": {
                "action": decision.action.value,
                "parameters": decision.parameters,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            },
            "vehicle": {
                "battery_percent": world.vehicle.battery.remaining_percent,
                "gps_satellites": world.vehicle.gps.satellites_visible if world.vehicle.gps else 0,
                "gps_hdop": world.vehicle.gps.hdop if world.vehicle.gps else 99.9,
                "mode": world.vehicle.mode.value,
                "armed": world.vehicle.armed,
                "in_air": world.vehicle.in_air,
            },
            "environment": {
                "wind_speed": wind_speed,
                "visibility_m": visibility_m,
            },
            "mission": {
                "assets_total": world.mission.assets_total,
                "assets_inspected": world.mission.assets_inspected,
                "progress_percent": world.mission.progress_percent,
            },
            "risk": {
                "overall_level": risk.overall_level.value,
                "overall_score": risk.overall_score,
                "abort_recommended": risk.abort_recommended,
            },
            "vision": {
                "active_anomalies": len(world.anomalies),
                "anomaly_details": [
                    {
                        "asset_id": a.asset_id,
                        "severity": a.severity,
                        "description": a.description[:100] if a.description else "",
                    }
                    for a in world.anomalies[:5]  # Limit to 5 most recent
                ],
            },
        }

    @staticmethod
    def _build_llm_prompt(context: dict[str, Any]) -> str:
        """Build the LLM prompt from safety context."""
        status_line = (
            f"- Mode: {context['vehicle']['mode']}, Armed: {context['vehicle']['armed']}, "
            f"In Air: {context['vehicle']['in_air']}"
        )
        mission_line = (
            f"Mission Progress: {context['mission']['progress_percent']:.0f}% "
            f"({context['mission']['assets_inspected']}/"
            f"{context['mission']['assets_total']} assets)"
        )
        return f"""Evaluate this drone decision for safety:

Decision: {context["decision"]["action"]}
Parameters: {context["decision"]["parameters"]}
Agent Reasoning: {context["decision"]["reasoning"]}

Vehicle State:
- Battery: {context["vehicle"]["battery_percent"]:.1f}%
- GPS: {context["vehicle"]["gps_satellites"]} sats, HDOP {context["vehicle"]["gps_hdop"]:.2f}
{status_line}

Environment:
- Wind: {context["environment"]["wind_speed"]:.1f} m/s
- Visibility: {context["environment"]["visibility_m"]:.0f} m

{mission_line}

Risk Assessment: {context["risk"]["overall_level"]} (score: {context["risk"]["overall_score"]:.2f})
Abort Recommended: {context["risk"]["abort_recommended"]}

Provide your safety verdict and reasoning."""

    @staticmethod
    def _parse_llm_verdict(response: str) -> tuple[CriticVerdict, float]:
        """Parse LLM response into a verdict and confidence."""
        response_lower = response.lower()
        if "reject" in response_lower:
            return CriticVerdict.REJECT, 0.85
        if "escalate" in response_lower:
            return CriticVerdict.ESCALATE, 0.80
        if "concerns" in response_lower or "concern" in response_lower:
            return CriticVerdict.APPROVE_WITH_CONCERNS, 0.75
        return CriticVerdict.APPROVE, 0.80

    @staticmethod
    def _extract_llm_concerns(response: str) -> list[str]:
        """Extract structured concerns from LLM response text."""
        response_lower = response.lower()
        concerns: list[str] = []
        if "battery" in response_lower and (
            "low" in response_lower or "insufficient" in response_lower
        ):
            concerns.append("LLM flagged battery safety concerns")
        if "gps" in response_lower and ("poor" in response_lower or "weak" in response_lower):
            concerns.append("LLM flagged GPS quality concerns")
        if "wind" in response_lower and ("high" in response_lower or "unsafe" in response_lower):
            concerns.append("LLM flagged weather safety concerns")
        if "health" in response_lower and ("issue" in response_lower or "fault" in response_lower):
            concerns.append("LLM flagged vehicle health concerns")
        return concerns

    def _determine_verdict(
        self,
        concerns: list[str],
        max_risk_score: float,
        decision: Decision,
        _risk: RiskAssessment,
    ) -> tuple[CriticVerdict, str, float]:
        """Determine final verdict based on concerns and risk.

        Returns:
            (verdict, reasoning, confidence)
        """
        # Critical risk or many concerns: REJECT
        if max_risk_score >= 0.85 or len(concerns) >= 3:
            verdict = CriticVerdict.REJECT
            reasoning = (
                f"Safety check failed: {len(concerns)} critical concerns identified. "
                f"Risk score: {max_risk_score:.2f}. Decision cannot proceed safely."
            )
            confidence = 0.95

        # High risk or abort decision with concerns: ESCALATE
        elif max_risk_score >= 0.7 or (decision.is_abort and len(concerns) > 0):
            verdict = CriticVerdict.ESCALATE
            reasoning = (
                f"Safety concerns require review: {len(concerns)} issues identified. "
                f"Risk score: {max_risk_score:.2f}. Escalating for hierarchical review."
            )
            confidence = 0.80

        # Moderate risk or some concerns: APPROVE_WITH_CONCERNS
        elif max_risk_score >= 0.4 or len(concerns) > 0:
            verdict = CriticVerdict.APPROVE_WITH_CONCERNS
            reasoning = (
                f"Decision approved with {len(concerns)} safety concerns noted. "
                f"Risk score: {max_risk_score:.2f}. Monitor conditions closely."
            )
            confidence = 0.75

        # Low risk and no concerns: APPROVE
        else:
            verdict = CriticVerdict.APPROVE
            reasoning = (
                f"All safety checks passed. Risk score: {max_risk_score:.2f}. "
                f"Decision is safe to proceed."
            )
            confidence = 0.90

        return verdict, reasoning, confidence
