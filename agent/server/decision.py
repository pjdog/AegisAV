"""
Decision Data Models

Defines the Decision type which is the primary output of the agent server.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from autonomy.vehicle_state import Position


class ActionType(Enum):
    """Types of actions the agent can command."""
    
    # Movement actions
    GOTO = "goto"                    # Fly to a position
    TAKEOFF = "takeoff"              # Take off to altitude
    LAND = "land"                    # Land at current position
    RTL = "rtl"                      # Return to launch
    
    # Mission actions
    INSPECT = "inspect"              # Perform asset inspection
    ORBIT = "orbit"                  # Orbit around a point
    
    # Dock actions
    DOCK = "dock"                    # Return to dock and land
    RECHARGE = "recharge"            # Recharge at dock
    UNDOCK = "undock"                # Take off from dock
    
    # Control actions
    WAIT = "wait"                    # Hold position, wait for condition
    ABORT = "abort"                  # Emergency abort mission
    
    # No action
    NONE = "none"                    # No action required


@dataclass
class Decision:
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
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""
    
    # Context
    timestamp: datetime = field(default_factory=datetime.now)
    risk_factors: dict[str, float] = field(default_factory=dict)
    
    # Tracking
    decision_id: str = ""
    supersedes: Optional[str] = None  # ID of decision this replaces
    
    def __post_init__(self) -> None:
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
    def target_position(self) -> Optional[Position]:
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
        return {
            "decision_id": self.decision_id,
            "action": self.action.value,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "risk_factors": self.risk_factors,
            "supersedes": self.supersedes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        """Create Decision from dictionary."""
        return cls(
            action=ActionType(data["action"]),
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            risk_factors=data.get("risk_factors", {}),
            decision_id=data.get("decision_id", ""),
            supersedes=data.get("supersedes"),
        )
    
    @classmethod
    def wait(cls, reason: str, duration_s: float = 0) -> "Decision":
        """Create a WAIT decision."""
        return cls(
            action=ActionType.WAIT,
            parameters={"duration_s": duration_s},
            reasoning=reason,
        )
    
    @classmethod
    def abort(cls, reason: str, risk_factors: Optional[dict] = None) -> "Decision":
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
        speed: Optional[float] = None,
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
