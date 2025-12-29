"""Tests for event broadcasting models."""

from datetime import datetime

from agent.server.events import Event, EventSeverity, EventType


def test_event_defaults_to_info_severity() -> None:
    """Events default to INFO severity."""
    timestamp = datetime(2025, 1, 1, 12, 0, 0)
    event = Event(event_type=EventType.SERVER_DECISION, timestamp=timestamp, data={"id": "dec_1"})

    assert event.severity == EventSeverity.INFO
    assert event.event_type == EventType.SERVER_DECISION
    assert event.timestamp == timestamp
    assert event.data["id"] == "dec_1"


def test_event_severity_enum_values() -> None:
    """Event severity enums map to expected values."""
    assert EventSeverity.INFO.value == "info"
    assert EventSeverity.WARNING.value == "warning"
    assert EventSeverity.ERROR.value == "error"
    assert EventSeverity.CRITICAL.value == "critical"
