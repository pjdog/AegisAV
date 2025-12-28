"""
Efficiency Critic

Evaluates resource usage and goal efficiency for decisions.
Focuses on battery consumption, path optimization, and mission progress.
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


class EfficiencyCriticConfig(CriticConfig):
    """Efficiency-specific configuration thresholds."""

    # Battery efficiency
    max_battery_waste_percent: float = 5.0  # Max acceptable battery waste
    min_battery_efficiency: float = 0.7  # Min efficiency ratio (useful / total)

    # Mission progress
    min_mission_progress_rate: float = 0.1  # Min assets inspected per hour
    max_time_per_asset_s: float = 600.0  # 10 minutes max per asset

    # Path efficiency
    max_path_deviation_percent: float = 30.0  # Max acceptable path deviation
    min_direct_flight_ratio: float = 0.6  # Min ratio of direct distance


class EfficiencyCritic(BaseCritic):
    """
    Efficiency critic evaluates resource usage and goal efficiency.

    Checks:
    - Battery consumption vs. mission value
    - Path efficiency (direct vs. planned distance)
    - Mission progress rate
    - Time utilization
    - Resource allocation
    """

    def __init__(self, config: EfficiencyCriticConfig | None = None, llm_model: str | None = None):
        """Initialize efficiency critic with configuration."""
        self.efficiency_config = config or EfficiencyCriticConfig()
        super().__init__(config=self.efficiency_config, llm_model=llm_model)

    def _get_critic_type(self) -> CriticType:
        """Return EFFICIENCY critic type."""
        return CriticType.EFFICIENCY

    async def evaluate_fast(
        self, decision: Decision, world: WorldSnapshot, risk: RiskAssessment
    ) -> CriticResponse:
        """
        Fast classical efficiency evaluation.

        Performs rule-based checks on:
        - Battery consumption efficiency
        - Mission progress rate
        - Path optimization
        - Resource allocation

        Returns:
            CriticResponse with verdict and efficiency analysis
        """
        concerns: list[str] = []
        alternatives: list[str] = []
        max_risk_score = 0.0

        # 1. Battery Efficiency Check
        battery_concern, battery_alternatives, battery_risk = self._check_battery_efficiency(
            decision, world
        )
        if battery_concern:
            concerns.extend(battery_concern)
            alternatives.extend(battery_alternatives)
        max_risk_score = max(max_risk_score, battery_risk)

        # 2. Mission Progress Check
        progress_concern, progress_alternatives, progress_risk = self._check_mission_progress(
            decision, world
        )
        if progress_concern:
            concerns.extend(progress_concern)
            alternatives.extend(progress_alternatives)
        max_risk_score = max(max_risk_score, progress_risk)

        # 3. Path Efficiency Check
        path_concern, path_alternatives, path_risk = self._check_path_efficiency(decision, world)
        if path_concern:
            concerns.extend(path_concern)
            alternatives.extend(path_alternatives)
        max_risk_score = max(max_risk_score, path_risk)

        # 4. Resource Allocation Check
        resource_concern, resource_alternatives, resource_risk = self._check_resource_allocation(
            decision, world
        )
        if resource_concern:
            concerns.extend(resource_concern)
            alternatives.extend(resource_alternatives)
        max_risk_score = max(max_risk_score, resource_risk)

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

    def _check_battery_efficiency(
        self, decision: Decision, world: WorldSnapshot
    ) -> tuple[list[str], list[str], float]:
        """
        Check battery usage efficiency.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        battery_percent = world.vehicle.battery.remaining_percent

        # Check if decision is WAIT with high battery (waste)
        if decision.action.value == "WAIT" and battery_percent > 70.0:
            wait_duration = decision.parameters.get("duration_s", 0)
            if wait_duration > 60:
                concerns.append(
                    f"Inefficient WAIT with {battery_percent:.1f}% battery remaining "
                    f"({wait_duration:.0f}s wait time)"
                )
                alternatives.append("Continue mission instead of waiting")
                alternatives.append("Use wait time to inspect nearby assets")
                risk_score = 0.4

        # Check for low-value actions with limited battery
        if battery_percent < 40.0 and decision.action.value == "INSPECT":
            # Inspecting with low battery might be inefficient if asset is low priority
            asset_id = decision.parameters.get("asset_id")
            if asset_id:
                asset = next((a for a in world.assets if a.asset_id == asset_id), None)
                if asset and asset.priority < 5:  # Low priority
                    concerns.append(
                        f"Low priority inspection ({asset.priority}/10) with "
                        f"limited battery ({battery_percent:.1f}%)"
                    )
                    alternatives.append("Prioritize high-value assets before return")
                    risk_score = max(risk_score, 0.5)

        return concerns, alternatives, risk_score

    def _check_mission_progress(
        self, decision: Decision, world: WorldSnapshot
    ) -> tuple[list[str], list[str], float]:
        """
        Check mission progress efficiency.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        # Calculate mission progress
        if world.mission.assets_total > 0:
            progress_rate = world.mission.assets_inspected / world.mission.assets_total

            # Check if we're making reasonable progress
            if progress_rate < 0.3 and world.vehicle.battery.remaining_percent < 50.0:
                concerns.append(
                    f"Low mission progress ({progress_rate * 100:.0f}% complete) "
                    f"with {world.vehicle.battery.remaining_percent:.1f}% battery"
                )
                alternatives.append("Focus on completing high-priority inspections")
                alternatives.append("Optimize inspection route")
                risk_score = 0.4

        # Check for RETURN decision with mission incomplete
        if decision.action.value == "RETURN" and world.mission.assets_total > 0:
            remaining_assets = world.mission.assets_total - world.mission.assets_inspected
            if remaining_assets > 0 and world.vehicle.battery.remaining_percent > 40.0:
                concerns.append(
                    f"Returning with {remaining_assets} assets uninspected "
                    f"and {world.vehicle.battery.remaining_percent:.1f}% battery"
                )
                alternatives.append("Inspect remaining high-priority assets")
                alternatives.append("Optimize route to cover more assets before return")
                risk_score = max(risk_score, 0.5)

        return concerns, alternatives, risk_score

    def _check_path_efficiency(
        self, decision: Decision, world: WorldSnapshot
    ) -> tuple[list[str], list[str], float]:
        """
        Check path efficiency for movement decisions.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        # Only applicable to movement decisions
        if not decision.is_movement:
            return concerns, alternatives, risk_score

        # Check if moving to far asset when closer ones need inspection
        if decision.action.value == "GOTO":
            target_pos = decision.target_position
            if target_pos:
                # Find pending assets and check if there are closer ones
                pending_assets = world.get_pending_assets()
                if len(pending_assets) > 1:
                    # Calculate distance to target
                    from autonomy.vehicle_state import calculate_distance

                    target_distance = calculate_distance(world.vehicle.position, target_pos)

                    # Check if there's a closer asset
                    closest_distance = min(
                        calculate_distance(world.vehicle.position, asset.position)
                        for asset in pending_assets
                    )

                    if target_distance > closest_distance * 1.5:  # 50% longer path
                        concerns.append(
                            f"Flying to asset at {target_distance:.0f}m when closer "
                            f"assets exist at {closest_distance:.0f}m"
                        )
                        alternatives.append("Inspect closer assets first")
                        alternatives.append("Optimize inspection sequence")
                        risk_score = 0.3

        return concerns, alternatives, risk_score

    def _check_resource_allocation(
        self, decision: Decision, world: WorldSnapshot
    ) -> tuple[list[str], list[str], float]:
        """
        Check resource allocation efficiency.

        Returns:
            (concerns, alternatives, risk_score)
        """
        concerns = []
        alternatives = []
        risk_score = 0.0

        battery_percent = world.vehicle.battery.remaining_percent

        # Check for ORBIT/INSPECT with very low battery
        if decision.action.value in {"ORBIT", "INSPECT"}:
            orbit_duration = decision.parameters.get("dwell_time_s", 30.0)

            # Rough estimate: 1% battery per 30s of operation
            estimated_consumption = (orbit_duration / 30.0) * 1.0

            if estimated_consumption > battery_percent * 0.5:  # Using >50% remaining battery
                concerns.append(
                    f"Action will consume ~{estimated_consumption:.1f}% battery "
                    f"({battery_percent:.1f}% available)"
                )
                alternatives.append("Reduce orbit duration")
                alternatives.append("Skip detailed inspection, return to dock")
                risk_score = 0.5

        # Check for DOCK decision when not needed (battery still high)
        if decision.action.value == "DOCK" and battery_percent > 60.0:
            pending_assets = world.get_pending_assets()
            if len(pending_assets) > 0:
                concerns.append(
                    f"Docking with {battery_percent:.1f}% battery and "
                    f"{len(pending_assets)} assets pending"
                )
                alternatives.append("Continue mission before docking")
                alternatives.append("Inspect remaining assets")
                risk_score = 0.4

        # Check for WAIT decision when battery is high and assets are pending
        if decision.action.value == "wait" and battery_percent > 60.0:
            pending_assets = world.get_pending_assets()
            if len(pending_assets) > 0:
                wait_duration = decision.parameters.get("duration_s", 0)
                concerns.append(
                    f"Waiting {wait_duration}s with {battery_percent:.1f}% battery and "
                    f"{len(pending_assets)} assets pending"
                )
                alternatives.append("Inspect pending assets instead of waiting")
                alternatives.append("Continue mission progress")
                risk_score = max(risk_score, 0.3)

        return concerns, alternatives, risk_score

    def _determine_verdict(
        self, concerns: list[str], max_risk_score: float, decision: Decision
    ) -> tuple[CriticVerdict, str, float]:
        """
        Determine final verdict based on efficiency concerns.

        Returns:
            (verdict, reasoning, confidence)
        """
        # Efficiency is advisory - we rarely REJECT on efficiency alone
        # Only escalate for very inefficient decisions

        if max_risk_score >= 0.7 or len(concerns) >= 3:
            verdict = CriticVerdict.ESCALATE
            reasoning = (
                f"Significant efficiency concerns ({len(concerns)} issues). "
                f"Risk score: {max_risk_score:.2f}. Recommend reviewing alternatives."
            )
            confidence = 0.75

        elif max_risk_score >= 0.5 or len(concerns) >= 2:
            verdict = CriticVerdict.APPROVE_WITH_CONCERNS
            reasoning = (
                f"Decision approved with {len(concerns)} efficiency concerns. "
                f"Risk score: {max_risk_score:.2f}. Consider optimizations."
            )
            confidence = 0.70

        elif len(concerns) > 0:
            verdict = CriticVerdict.APPROVE_WITH_CONCERNS
            reasoning = (
                f"Minor efficiency concerns noted ({len(concerns)} issues). "
                f"Risk score: {max_risk_score:.2f}. Proceed but monitor resource usage."
            )
            confidence = 0.80

        else:
            verdict = CriticVerdict.APPROVE
            reasoning = (
                f"Decision is resource-efficient. Risk score: {max_risk_score:.2f}. "
                f"Good resource allocation and mission progress."
            )
            confidence = 0.85

        return verdict, reasoning, confidence
