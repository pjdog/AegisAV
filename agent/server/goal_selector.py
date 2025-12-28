"""
Enhanced Goal Selector with Advanced Agentic Capabilities

Selects the next goal based on world state, mission objectives,
operational constraints, and learning from experience. This implements
hierarchical decision making with predictive capabilities and adaptive learning.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta

from agent.server.advanced_decision import AdvancedDecisionEngine, create_advanced_decision_engine
from agent.server.goals import Goal, GoalSelectorConfig, GoalType
from agent.server.world_model import WorldSnapshot

logger = logging.getLogger(__name__)


class GoalSelector:
    """
    Selects the next goal based on world state.

    The goal selector evaluates the current situation and chooses
    the most appropriate goal. It implements a priority-based
    decision tree:

    1. ABORT if critical risk conditions
    2. RETURN if battery critical or weather unsafe
    3. INSPECT_ANOMALY if anomalies need attention
    4. INSPECT_ASSET if assets need inspection
    5. RETURN if mission complete
    6. WAIT if nothing to do

    The selector is stateless - all decisions are based on the
    provided WorldSnapshot.

    Example:
        selector = GoalSelector(config)
        goal = selector.select_goal(world_snapshot)

        if goal.is_abort:
            # Handle abort
        elif goal.is_return:
            # Return to dock
        else:
            # Process inspection or other goal
    """

    def __init__(self, config: GoalSelectorConfig | None = None):
        self.config = config or GoalSelectorConfig()
        self.anomaly_revisit_interval = timedelta(
            minutes=self.config.anomaly_revisit_interval_minutes
        )
        self.normal_cadence = timedelta(minutes=self.config.normal_cadence_minutes)
        self.use_advanced_engine = self.config.use_advanced_engine

        # Initialize advanced components
        self.advanced_engine: AdvancedDecisionEngine | None = None
        self.performance_history = deque(maxlen=100)
        self.adaptive_thresholds = {
            "battery_return": self.config.battery_return_threshold,
            "anomaly_revisit": self.config.anomaly_revisit_interval_minutes,
            "weather_limit": 12.0,  # m/s
        }

    async def initialize(self) -> None:
        """Initialize advanced decision engine if enabled.

        Returns:
            None
        """
        if self.use_advanced_engine:
            self.advanced_engine = await create_advanced_decision_engine()
            logger.info("Advanced decision engine initialized")

    async def select_goal(self, world: WorldSnapshot) -> Goal:
        """
        Select the next goal based on current world state.

        Args:
            world (WorldSnapshot): Current world snapshot.

        Returns:
            Goal: Selected goal with priority and context.
        """
        goal: Goal | None = None

        # If advanced engine is enabled and initialized, use it
        if self.use_advanced_engine and self.advanced_engine:
            try:
                goal, _context = await self.advanced_engine.make_advanced_decision(world)
                logger.info("Advanced goal selected: %s - %s", goal.goal_type, goal.reason)
            except Exception as e:
                logger.error("Advanced decision failed, falling back to rules: %s", e)

        if goal is None:
            checks = (
                (self._check_abort_conditions, logger.warning, "ABORT condition: %s"),
                (self._check_battery, logger.info, "Battery goal: %s"),
                (self._check_weather, logger.info, "Weather goal: %s"),
                (self._check_anomalies, logger.info, "Anomaly goal: %s"),
                (self._check_inspections, logger.info, "Inspection goal: %s"),
            )
            for check, log_func, message in checks:
                candidate = check(world)
                if candidate:
                    log_func(message, candidate.reason)
                    goal = candidate
                    break

        # Priority 6: Check if mission complete
        if goal is None and world.mission.is_active:
            if world.mission.assets_inspected >= world.mission.assets_total:
                goal = Goal(
                    goal_type=GoalType.RETURN_MISSION_COMPLETE,
                    priority=50,
                    reason="All assets inspected, mission complete",
                )

        # Default: Wait
        if goal is None:
            goal = Goal(
                goal_type=GoalType.WAIT,
                priority=100,
                reason="No actionable goals, holding position",
            )

        return goal

    def _check_abort_conditions(self, world: WorldSnapshot) -> Goal | None:
        """Check for conditions that require immediate abort."""

        # Critical battery
        if world.vehicle.battery.remaining_percent < 15:
            return Goal(
                goal_type=GoalType.ABORT,
                priority=0,
                reason=f"Critical battery: {world.vehicle.battery.remaining_percent:.1f}%",
            )

        # Vehicle unhealthy
        if world.vehicle.health and not world.vehicle.health.is_healthy:
            unhealthy = []
            if not world.vehicle.health.sensors_healthy:
                unhealthy.append("sensors")
            if not world.vehicle.health.gps_healthy:
                unhealthy.append("gps")
            if not world.vehicle.health.battery_healthy:
                unhealthy.append("battery")
            if not world.vehicle.health.motors_healthy:
                unhealthy.append("motors")
            if not world.vehicle.health.ekf_healthy:
                unhealthy.append("ekf")

            error_list = unhealthy + world.vehicle.health.error_messages
            errors = ", ".join(error_list) or "unknown"
            return Goal(
                goal_type=GoalType.ABORT,
                priority=0,
                reason=f"Vehicle health critical: {errors}",
            )

        # GPS lost
        if world.vehicle.gps and not world.vehicle.gps.has_fix:
            return Goal(
                goal_type=GoalType.ABORT,
                priority=0,
                reason="GPS fix lost",
            )

        return None

    async def orchestrate(self, enabled: bool) -> None:
        """Toggle the advanced engine at runtime and initialize if needed."""
        self.use_advanced_engine = enabled
        if enabled and not self.advanced_engine:
            await self.initialize()
        logger.info(f"Advanced engine orchestrated: {enabled}")

    def _check_battery(self, world: WorldSnapshot) -> Goal | None:
        """Check if battery level requires return."""

        battery_percent = world.vehicle.battery.remaining_percent

        # Critical - must return immediately
        if battery_percent < self.config.battery_critical_threshold:
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                priority=5,
                reason=f"Battery critical: {battery_percent:.1f}%",
                confidence=1.0,
            )

        # Low - should return unless very close to completing task
        if battery_percent < self.config.battery_return_threshold:
            # Could add logic to check if we can complete current task
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                priority=10,
                reason=f"Battery low: {battery_percent:.1f}%",
                confidence=0.9,
            )

        return None

    def _check_weather(self, world: WorldSnapshot) -> Goal | None:
        """Check if weather requires return."""

        if not world.environment.is_flyable:
            reasons = []
            if world.environment.wind_speed_ms >= 12:
                reasons.append(f"wind {world.environment.wind_speed_ms:.1f}m/s")
            if world.environment.visibility_m < 1000:
                reasons.append(f"visibility {world.environment.visibility_m:.0f}m")
            if world.environment.precipitation not in ("none", "light_rain"):
                reasons.append(world.environment.precipitation)

            return Goal(
                goal_type=GoalType.RETURN_WEATHER,
                priority=15,
                reason=f"Weather unflyable: {', '.join(reasons)}",
            )

        return None

    def _check_anomalies(self, world: WorldSnapshot) -> Goal | None:
        """Check for anomalies needing re-inspection."""

        anomaly_assets = world.get_anomaly_assets()

        for asset in anomaly_assets:
            # Check if enough time has passed for re-inspection
            if asset.last_inspection:
                time_since = datetime.now() - asset.last_inspection
                if time_since < self.anomaly_revisit_interval:
                    continue

            return Goal(
                goal_type=GoalType.INSPECT_ANOMALY,
                priority=20,
                target_asset=asset,
                reason=f"Re-inspect anomaly at {asset.name}",
                confidence=0.95,
            )

        return None

    def _check_inspections(self, world: WorldSnapshot) -> Goal | None:
        """Check for assets needing routine inspection."""

        pending_assets = world.get_pending_assets()

        if pending_assets:
            # Take highest priority (lowest number) asset
            asset = pending_assets[0]

            return Goal(
                goal_type=GoalType.INSPECT_ASSET,
                priority=30 + asset.priority,
                target_asset=asset,
                reason=f"Scheduled inspection of {asset.name}",
                confidence=0.9,
            )

        return None
