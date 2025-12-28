"""
Goal Models

Defines goal types and goal payloads used by the planning pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict

from agent.server.world_model import Asset


class GoalType(Enum):
    """Types of goals the agent can pursue."""

    # Inspection goals
    INSPECT_ASSET = "inspect_asset"  # Inspect a specific asset
    INSPECT_ANOMALY = "inspect_anomaly"  # Re-inspect asset with anomaly

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


class Goal(BaseModel):
    """A goal selected by the agent.

    Goals represent high-level objectives that the planner will
    translate into specific actions.

    Attributes:
        goal_type (GoalType): Goal type enum.
        priority (int): Priority value (lower is higher).
        target_asset (Asset | None): Target asset (:class:`agent.server.world_model.Asset`).
        reason (str): Human-readable rationale.
        confidence (float): Confidence score.
        deadline (datetime | None): Optional deadline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    goal_type: GoalType
    priority: int
    target_asset: Asset | None = None
    reason: str = ""
    confidence: float = 1.0
    deadline: datetime | None = None

    @property
    def is_abort(self) -> bool:
        """Check if this is an abort goal."""
        return self.goal_type == GoalType.ABORT

    @property
    def is_return(self) -> bool:
        """Check if this is a return goal."""
        return self.goal_type in (
            GoalType.RETURN_LOW_BATTERY,
            GoalType.RETURN_MISSION_COMPLETE,
            GoalType.RETURN_WEATHER,
        )


@dataclass(frozen=True)
class GoalSelectorConfig:
    """Configuration values for goal selection.

    Attributes:
        battery_return_threshold (float): Battery percent to trigger return.
        battery_critical_threshold (float): Battery percent to trigger abort.
        anomaly_revisit_interval_minutes (float): Minutes between anomaly revisits.
        normal_cadence_minutes (float): Minutes between normal inspections.
        use_advanced_engine (bool): Whether to enable the advanced decision engine.
    """

    battery_return_threshold: float = 30.0
    battery_critical_threshold: float = 20.0
    anomaly_revisit_interval_minutes: float = 10.0
    normal_cadence_minutes: float = 30.0
    use_advanced_engine: bool = True
