"""
Decision Data Models

Defines the Decision type which is the primary output of the agent server.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.api_models import ActionType
from autonomy.vehicle_state import Position


class Decision(BaseModel):
    """
    The primary output of the agent's decision-making process.

    Each decision includes:
    - The action to take
    - Parameters for the action
    - Confidence level
    - Human-readable reasoning (for explainability)
    - Risk assessment at time of decision
    """

    action: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""

    # Context
    timestamp: datetime = Field(default_factory=datetime.now)
    risk_factors: dict[str, float] = Field(default_factory=dict)

    # Tracking
    decision_id: str = Field(default="")
    supersedes: str | None = None  # ID of decision this replaces

    def model_post_init(self, __context: Any) -> None:
        """Generate decision ID if not provided."""
        if not self.decision_id:
            self.decision_id = f"dec_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

    @property
    def is_movement(self) -> bool:
        """Check if this decision involves vehicle movement."""
        return self.action in {
            ActionType.GOTO,
            ActionType.TAKEOFF,
            ActionType.LAND,
            ActionType.RTL,
            ActionType.DOCK,
            ActionType.ORBIT,
        }

    @property
    def is_abort(self) -> bool:
        """Check if this is an abort decision."""
        return self.action == ActionType.ABORT

    @property
    def target_position(self) -> Position | None:
        """Extract target position from parameters if present."""
        pos_data = self.parameters.get("position")
        if pos_data:
            return Position(
                latitude=pos_data["latitude"],
                longitude=pos_data["longitude"],
                altitude_msl=pos_data.get("altitude_msl", 0),
                altitude_agl=pos_data.get("altitude_agl", 0),
            )
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        """Create Decision from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def wait(cls, reason: str, duration_s: float = 0) -> "Decision":
        """Create a WAIT decision."""
        return cls(
            action=ActionType.WAIT,
            parameters={"duration_s": duration_s},
            reasoning=reason,
        )

    @classmethod
    def abort(cls, reason: str, risk_factors: dict | None = None) -> "Decision":
        """Create an ABORT decision."""
        return cls(
            action=ActionType.ABORT,
            reasoning=reason,
            confidence=1.0,
            risk_factors=risk_factors or {},
        )

    @classmethod
    def goto(
        cls,
        position: Position,
        reason: str,
        speed: float | None = None,
        confidence: float = 1.0,
    ) -> "Decision":
        """Create a GOTO decision."""
        params = {
            "position": {
                "latitude": position.latitude,
                "longitude": position.longitude,
                "altitude_msl": position.altitude_msl,
            }
        }
        if speed:
            params["speed_ms"] = speed

        return cls(
            action=ActionType.GOTO,
            parameters=params,
            reasoning=reason,
            confidence=confidence,
        )

    @classmethod
    def inspect(
        cls,
        asset_id: str,
        position: Position,
        reason: str,
        orbit_radius: float = 20.0,
        dwell_time_s: float = 30.0,
    ) -> "Decision":
        """Create an INSPECT decision."""
        return cls(
            action=ActionType.INSPECT,
            parameters={
                "asset_id": asset_id,
                "position": {
                    "latitude": position.latitude,
                    "longitude": position.longitude,
                    "altitude_msl": position.altitude_msl,
                },
                "orbit_radius_m": orbit_radius,
                "dwell_time_s": dwell_time_s,
            },
            reasoning=reason,
        )

    @classmethod
    def return_to_dock(cls, reason: str, confidence: float = 1.0) -> "Decision":
        """Create a DOCK decision to return to dock."""
        return cls(
            action=ActionType.DOCK,
            reasoning=reason,
            confidence=confidence,
        )
