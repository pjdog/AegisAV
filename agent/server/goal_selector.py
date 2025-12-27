"""
Goal Selector

Chooses the next goal based on current world state, mission objectives,
and operational constraints. This is a core component of the agentic
decision-making system.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from agent.server.world_model import Asset, WorldSnapshot

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Types of goals the agent can pursue."""
    
    # Inspection goals
    INSPECT_ASSET = "inspect_asset"       # Inspect a specific asset
    INSPECT_ANOMALY = "inspect_anomaly"   # Re-inspect asset with anomaly
    
    # Return goals
    RETURN_LOW_BATTERY = "return_low_battery"
    RETURN_MISSION_COMPLETE = "return_complete"
    RETURN_WEATHER = "return_weather"
    
    # Control goals
    WAIT = "wait"
    ABORT = "abort"
    
    # Dock goals
    RECHARGE = "recharge"
    
    # No goal
    NONE = "none"


@dataclass
class Goal:
    """
    A goal selected by the agent.
    
    Goals represent high-level objectives that the planner will
    translate into specific actions.
    """
    
    goal_type: GoalType
    priority: int  # Lower = higher priority
    
    # Target (for inspection goals)
    target_asset: Optional[Asset] = None
    
    # Context
    reason: str = ""
    confidence: float = 1.0
    
    # Constraints
    deadline: Optional[datetime] = None
    
    @property
    def is_abort(self) -> bool:
        return self.goal_type == GoalType.ABORT
    
    @property
    def is_return(self) -> bool:
        return self.goal_type in {
            GoalType.RETURN_LOW_BATTERY,
            GoalType.RETURN_MISSION_COMPLETE,
            GoalType.RETURN_WEATHER,
        }


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
    
    def __init__(
        self,
        battery_return_threshold: float = 30.0,
        battery_critical_threshold: float = 20.0,
        anomaly_revisit_interval_minutes: float = 10.0,
        normal_cadence_minutes: float = 30.0,
    ):
        self.battery_return_threshold = battery_return_threshold
        self.battery_critical_threshold = battery_critical_threshold
        self.anomaly_revisit_interval = timedelta(minutes=anomaly_revisit_interval_minutes)
        self.normal_cadence = timedelta(minutes=normal_cadence_minutes)
    
    def select_goal(self, world: WorldSnapshot) -> Goal:
        """
        Select the next goal based on current world state.
        
        Args:
            world: Current world snapshot
            
        Returns:
            Selected goal with priority and context
        """
        # Priority 1: Check for critical conditions requiring abort
        abort_goal = self._check_abort_conditions(world)
        if abort_goal:
            logger.warning(f"ABORT condition: {abort_goal.reason}")
            return abort_goal
        
        # Priority 2: Check battery level
        battery_goal = self._check_battery(world)
        if battery_goal:
            logger.info(f"Battery goal: {battery_goal.reason}")
            return battery_goal
        
        # Priority 3: Check weather conditions
        weather_goal = self._check_weather(world)
        if weather_goal:
            logger.info(f"Weather goal: {weather_goal.reason}")
            return weather_goal
        
        # Priority 4: Check for anomalies needing attention
        anomaly_goal = self._check_anomalies(world)
        if anomaly_goal:
            logger.info(f"Anomaly goal: {anomaly_goal.reason}")
            return anomaly_goal
        
        # Priority 5: Check for assets needing inspection
        inspection_goal = self._check_inspections(world)
        if inspection_goal:
            logger.info(f"Inspection goal: {inspection_goal.reason}")
            return inspection_goal
        
        # Priority 6: Check if mission complete
        if world.mission.is_active and world.mission.assets_inspected >= world.mission.assets_total:
            return Goal(
                goal_type=GoalType.RETURN_MISSION_COMPLETE,
                priority=50,
                reason="All assets inspected, mission complete",
            )
        
        # Default: Wait
        return Goal(
            goal_type=GoalType.WAIT,
            priority=100,
            reason="No actionable goals, holding position",
        )
    
    def _check_abort_conditions(self, world: WorldSnapshot) -> Optional[Goal]:
        """Check for conditions that require immediate abort."""
        
        # Critical battery
        if world.vehicle.battery.remaining_percent < 15:
            return Goal(
                goal_type=GoalType.ABORT,
                priority=0,
                reason=f"Critical battery: {world.vehicle.battery.remaining_percent:.1f}%",
            )
        
        # Vehicle unhealthy
        if not world.vehicle.health.is_healthy:
            errors = ", ".join(world.vehicle.health.error_messages) or "unknown"
            return Goal(
                goal_type=GoalType.ABORT,
                priority=0,
                reason=f"Vehicle health critical: {errors}",
            )
        
        # GPS lost
        if not world.vehicle.gps.has_fix:
            return Goal(
                goal_type=GoalType.ABORT,
                priority=0,
                reason="GPS fix lost",
            )
        
        return None
    
    def _check_battery(self, world: WorldSnapshot) -> Optional[Goal]:
        """Check if battery level requires return."""
        
        battery_percent = world.vehicle.battery.remaining_percent
        
        # Critical - must return immediately
        if battery_percent < self.battery_critical_threshold:
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                priority=5,
                reason=f"Battery critical: {battery_percent:.1f}%",
                confidence=1.0,
            )
        
        # Low - should return unless very close to completing task
        if battery_percent < self.battery_return_threshold:
            # Could add logic to check if we can complete current task
            return Goal(
                goal_type=GoalType.RETURN_LOW_BATTERY,
                priority=10,
                reason=f"Battery low: {battery_percent:.1f}%",
                confidence=0.9,
            )
        
        return None
    
    def _check_weather(self, world: WorldSnapshot) -> Optional[Goal]:
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
    
    def _check_anomalies(self, world: WorldSnapshot) -> Optional[Goal]:
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
    
    def _check_inspections(self, world: WorldSnapshot) -> Optional[Goal]:
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
