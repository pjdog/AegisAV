"""
Tests for the events module.
"""

from datetime import datetime

from agent.server.events import Event, EventSeverity, EventType


class TestEventType:
    """Test EventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types exist."""
        expected = [
            "SERVER_DECISION",
            "CLIENT_EXECUTION",
            "RISK_UPDATE",
            "GOAL_SELECTED",
            "CRITIC_VALIDATION",
            "VISION_DETECTION",
            "ANOMALY_CREATED",
        ]
        for event_type in expected:
            assert hasattr(EventType, event_type)

    def test_event_type_values(self):
        """Test event type string values."""
        assert EventType.SERVER_DECISION.value == "server_decision"
        assert EventType.CLIENT_EXECUTION.value == "client_execution"
        assert EventType.VISION_DETECTION.value == "vision_detection"


class TestEventSeverity:
    """Test EventSeverity enum."""

    def test_all_severities_exist(self):
        """Test that all severity levels exist."""
        expected = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        for severity in expected:
            assert hasattr(EventSeverity, severity)

    def test_severity_values(self):
        """Test severity string values."""
        assert EventSeverity.INFO.value == "info"
        assert EventSeverity.WARNING.value == "warning"
        assert EventSeverity.ERROR.value == "error"
        assert EventSeverity.CRITICAL.value == "critical"


class TestEvent:
    """Test Event model."""

    def test_create_event_minimal(self):
        """Test creating event with minimal data."""
        event = Event(
            event_type=EventType.SERVER_DECISION,
            timestamp=datetime.now(),
            data={"action": "INSPECT"},
        )
        assert event.event_type == EventType.SERVER_DECISION
        assert event.severity == EventSeverity.INFO  # default
        assert event.data["action"] == "INSPECT"

    def test_create_event_with_severity(self):
        """Test creating event with custom severity."""
        event = Event(
            event_type=EventType.RISK_UPDATE,
            timestamp=datetime.now(),
            data={"risk_level": "HIGH"},
            severity=EventSeverity.WARNING,
        )
        assert event.severity == EventSeverity.WARNING

    def test_event_json_serialization(self):
        """Test that events can be serialized to JSON."""
        timestamp = datetime(2024, 1, 15, 12, 0, 0)
        event = Event(
            event_type=EventType.VISION_DETECTION,
            timestamp=timestamp,
            data={"detections": [{"class": "crack", "confidence": 0.95}]},
            severity=EventSeverity.INFO,
        )

        json_data = event.model_dump_json()
        assert "vision_detection" in json_data
        assert "2024-01-15" in json_data
        assert "crack" in json_data

    def test_event_dict_conversion(self):
        """Test converting event to dict."""
        event = Event(
            event_type=EventType.GOAL_SELECTED,
            timestamp=datetime.now(),
            data={"goal": "INSPECT_ASSET", "asset_id": "asset_001"},
        )

        data = event.model_dump()
        assert data["event_type"] == EventType.GOAL_SELECTED
        assert data["data"]["goal"] == "INSPECT_ASSET"
        assert data["data"]["asset_id"] == "asset_001"

    def test_server_decision_event(self):
        """Test creating a server decision event."""
        event = Event(
            event_type=EventType.SERVER_DECISION,
            timestamp=datetime.now(),
            data={
                "decision_id": "dec_001",
                "action": "INSPECT",
                "confidence": 0.85,
                "goal": "INSPECT_ASSET",
            },
        )
        assert event.data["confidence"] == 0.85
        assert event.data["action"] == "INSPECT"

    def test_client_execution_event(self):
        """Test creating a client execution event."""
        event = Event(
            event_type=EventType.CLIENT_EXECUTION,
            timestamp=datetime.now(),
            data={
                "decision_id": "dec_001",
                "status": "COMPLETED",
                "duration_ms": 5000,
            },
        )
        assert event.data["status"] == "COMPLETED"
        assert event.data["duration_ms"] == 5000

    def test_critic_validation_event(self):
        """Test creating a critic validation event."""
        event = Event(
            event_type=EventType.CRITIC_VALIDATION,
            timestamp=datetime.now(),
            data={
                "critic": "SafetyCritic",
                "approved": True,
                "confidence": 0.92,
                "reasoning": "Battery level sufficient for mission",
            },
            severity=EventSeverity.INFO,
        )
        assert event.data["approved"] is True
        assert event.data["critic"] == "SafetyCritic"

    def test_anomaly_created_event(self):
        """Test creating an anomaly event."""
        event = Event(
            event_type=EventType.ANOMALY_CREATED,
            timestamp=datetime.now(),
            data={
                "anomaly_id": "anom_001",
                "asset_id": "asset_001",
                "severity": 0.85,
                "description": "Potential crack detected on solar panel",
            },
            severity=EventSeverity.WARNING,
        )
        assert event.data["anomaly_id"] == "anom_001"
        assert event.severity == EventSeverity.WARNING

    def test_critical_event(self):
        """Test creating a critical event."""
        event = Event(
            event_type=EventType.RISK_UPDATE,
            timestamp=datetime.now(),
            data={
                "risk_level": "ABORT",
                "battery_percent": 10,
                "reason": "Battery critically low",
            },
            severity=EventSeverity.CRITICAL,
        )
        assert event.severity == EventSeverity.CRITICAL
        assert event.data["risk_level"] == "ABORT"
