"""
Event Broadcasting System

Handles real-time event broadcasting via WebSocket for dashboard updates.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class EventType(str, Enum):
    """Types of events that can be broadcast."""

    SERVER_DECISION = "server_decision"
    CLIENT_EXECUTION = "client_execution"
    RISK_UPDATE = "risk_update"
    GOAL_SELECTED = "goal_selected"
    CRITIC_VALIDATION = "critic_validation"
    VISION_DETECTION = "vision_detection"
    ANOMALY_CREATED = "anomaly_created"


class EventSeverity(str, Enum):
    """Event severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Event(BaseModel):
    """Event to be broadcast to connected clients."""

    event_type: EventType
    timestamp: datetime
    data: dict[str, Any]
    severity: EventSeverity = EventSeverity.INFO

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}
