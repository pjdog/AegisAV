"""
Comprehensive tests for Decision data models.
"""

from datetime import datetime

import pytest

from agent.api_models import ActionType
from agent.server.decision import Decision
from autonomy.vehicle_state import Position


class TestDecisionBasic:
    """Test basic Decision functionality."""

    def test_decision_with_defaults(self):
        """Test creating a decision with default values."""
        decision = Decision(action=ActionType.WAIT)
        assert decision.action == ActionType.WAIT
        assert decision.parameters == {}
        assert decision.confidence == 1.0
        assert decision.reasoning == ""
        assert decision.risk_factors == {}
        assert decision.supersedes is None
        assert decision.decision_id.startswith("dec_")

    def test_decision_with_all_fields(self):
        """Test creating a decision with all fields."""
        timestamp = datetime(2023, 6, 15, 12, 0, 0)
        decision = Decision(
            action=ActionType.GOTO,
            parameters={"position": {"latitude": 37.0, "longitude": -122.0, "altitude_msl": 100.0}},
            confidence=0.95,
            reasoning="Moving to inspection target",
            timestamp=timestamp,
            risk_factors={"battery": 0.2, "wind": 0.1},
            decision_id="dec_custom_001",
            supersedes="dec_previous_001",
        )
        assert decision.action == ActionType.GOTO
        assert decision.confidence == 0.95
        assert decision.reasoning == "Moving to inspection target"
        assert decision.timestamp == timestamp
        assert decision.risk_factors == {"battery": 0.2, "wind": 0.1}
        assert decision.decision_id == "dec_custom_001"
        assert decision.supersedes == "dec_previous_001"

    def test_decision_id_auto_generated(self):
        """Test that decision_id is auto-generated when not provided."""
        decision = Decision(action=ActionType.WAIT)
        assert decision.decision_id.startswith("dec_")
        assert len(decision.decision_id) > 4

    def test_decision_id_format(self):
        """Test decision_id format includes timestamp."""
        decision = Decision(action=ActionType.WAIT)
        # Format should be dec_YYYYMMDD_HHMMSS_microseconds
        parts = decision.decision_id.split("_")
        assert parts[0] == "dec"
        assert len(parts) == 4  # dec, date, time, microseconds

    def test_decision_custom_id_preserved(self):
        """Test that custom decision_id is preserved."""
        decision = Decision(action=ActionType.WAIT, decision_id="my_custom_id")
        assert decision.decision_id == "my_custom_id"


class TestDecisionProperties:
    """Test Decision property methods."""

    def test_is_movement_goto(self):
        """Test is_movement for GOTO action."""
        decision = Decision(action=ActionType.GOTO)
        assert decision.is_movement is True

    def test_is_movement_takeoff(self):
        """Test is_movement for TAKEOFF action."""
        decision = Decision(action=ActionType.TAKEOFF)
        assert decision.is_movement is True

    def test_is_movement_land(self):
        """Test is_movement for LAND action."""
        decision = Decision(action=ActionType.LAND)
        assert decision.is_movement is True

    def test_is_movement_rtl(self):
        """Test is_movement for RTL action."""
        decision = Decision(action=ActionType.RTL)
        assert decision.is_movement is True

    def test_is_movement_dock(self):
        """Test is_movement for DOCK action."""
        decision = Decision(action=ActionType.DOCK)
        assert decision.is_movement is True

    def test_is_movement_orbit(self):
        """Test is_movement for ORBIT action."""
        decision = Decision(action=ActionType.ORBIT)
        assert decision.is_movement is True

    def test_is_not_movement_wait(self):
        """Test is_movement for WAIT action."""
        decision = Decision(action=ActionType.WAIT)
        assert decision.is_movement is False

    def test_is_not_movement_inspect(self):
        """Test is_movement for INSPECT action."""
        decision = Decision(action=ActionType.INSPECT)
        assert decision.is_movement is False

    def test_is_not_movement_abort(self):
        """Test is_movement for ABORT action."""
        decision = Decision(action=ActionType.ABORT)
        assert decision.is_movement is False

    def test_is_not_movement_none(self):
        """Test is_movement for NONE action."""
        decision = Decision(action=ActionType.NONE)
        assert decision.is_movement is False

    def test_is_abort_true(self):
        """Test is_abort for ABORT action."""
        decision = Decision(action=ActionType.ABORT)
        assert decision.is_abort is True

    def test_is_abort_false(self):
        """Test is_abort for non-ABORT actions."""
        for action_type in [ActionType.WAIT, ActionType.GOTO, ActionType.LAND, ActionType.INSPECT]:
            decision = Decision(action=action_type)
            assert decision.is_abort is False

    def test_target_position_present(self):
        """Test target_position when position is in parameters."""
        decision = Decision(
            action=ActionType.GOTO,
            parameters={
                "position": {
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "altitude_msl": 150.0,
                    "altitude_agl": 50.0,
                }
            },
        )
        pos = decision.target_position
        assert pos is not None
        assert pos.latitude == 37.7749
        assert pos.longitude == -122.4194
        assert pos.altitude_msl == 150.0
        assert pos.altitude_agl == 50.0

    def test_target_position_minimal(self):
        """Test target_position with minimal position data."""
        decision = Decision(
            action=ActionType.GOTO,
            parameters={
                "position": {
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                }
            },
        )
        pos = decision.target_position
        assert pos is not None
        assert pos.latitude == 37.7749
        assert pos.longitude == -122.4194
        assert pos.altitude_msl == 0
        assert pos.altitude_agl == 0

    def test_target_position_absent(self):
        """Test target_position when no position in parameters."""
        decision = Decision(action=ActionType.WAIT)
        assert decision.target_position is None

    def test_target_position_empty_parameters(self):
        """Test target_position with empty parameters."""
        decision = Decision(action=ActionType.GOTO, parameters={})
        assert decision.target_position is None


class TestDecisionSerialization:
    """Test Decision serialization methods."""

    def test_to_dict(self):
        """Test to_dict method."""
        timestamp = datetime(2023, 6, 15, 12, 0, 0)
        decision = Decision(
            action=ActionType.GOTO,
            parameters={"speed_ms": 5.0},
            confidence=0.9,
            reasoning="Test reason",
            timestamp=timestamp,
            risk_factors={"battery": 0.1},
            decision_id="dec_test_001",
        )
        result = decision.to_dict()

        assert result["action"] == "goto"
        assert result["parameters"] == {"speed_ms": 5.0}
        assert result["confidence"] == 0.9
        assert result["reasoning"] == "Test reason"
        assert result["risk_factors"] == {"battery": 0.1}
        assert result["decision_id"] == "dec_test_001"
        assert result["supersedes"] is None

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "action": "goto",
            "parameters": {"speed_ms": 5.0},
            "confidence": 0.9,
            "reasoning": "Test reason",
            "timestamp": "2023-06-15T12:00:00",
            "risk_factors": {"battery": 0.1},
            "decision_id": "dec_test_001",
            "supersedes": None,
        }
        decision = Decision.from_dict(data)

        assert decision.action == ActionType.GOTO
        assert decision.parameters == {"speed_ms": 5.0}
        assert decision.confidence == 0.9
        assert decision.reasoning == "Test reason"
        assert decision.decision_id == "dec_test_001"

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are reversible."""
        original = Decision(
            action=ActionType.INSPECT,
            parameters={"asset_id": "asset_001", "dwell_time_s": 30},
            confidence=0.85,
            reasoning="Inspecting asset",
            risk_factors={"wind": 0.2},
            decision_id="dec_round_001",
        )

        serialized = original.to_dict()
        restored = Decision.from_dict(serialized)

        assert restored.action == original.action
        assert restored.parameters == original.parameters
        assert restored.confidence == original.confidence
        assert restored.reasoning == original.reasoning
        assert restored.decision_id == original.decision_id


class TestDecisionFactoryMethods:
    """Test Decision factory class methods."""

    def test_wait_with_reason(self):
        """Test creating a WAIT decision with reason."""
        decision = Decision.wait(reason="Waiting for GPS fix")
        assert decision.action == ActionType.WAIT
        assert decision.reasoning == "Waiting for GPS fix"
        assert decision.parameters == {"duration_s": 0}

    def test_wait_with_duration(self):
        """Test creating a WAIT decision with duration."""
        decision = Decision.wait(reason="Cooldown period", duration_s=30.0)
        assert decision.action == ActionType.WAIT
        assert decision.parameters["duration_s"] == 30.0

    def test_abort_basic(self):
        """Test creating an ABORT decision."""
        decision = Decision.abort(reason="Battery critical")
        assert decision.action == ActionType.ABORT
        assert decision.reasoning == "Battery critical"
        assert decision.confidence == 1.0
        assert decision.risk_factors == {}

    def test_abort_with_risk_factors(self):
        """Test creating an ABORT decision with risk factors."""
        risk_factors = {"battery": 0.95, "wind": 0.8}
        decision = Decision.abort(reason="Multiple critical risks", risk_factors=risk_factors)
        assert decision.action == ActionType.ABORT
        assert decision.risk_factors == risk_factors

    def test_goto_basic(self):
        """Test creating a GOTO decision."""
        position = Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=100.0,
        )
        decision = Decision.goto(position=position, reason="Moving to target")
        assert decision.action == ActionType.GOTO
        assert decision.reasoning == "Moving to target"
        assert decision.confidence == 1.0
        assert decision.parameters["position"]["latitude"] == 37.7749
        assert decision.parameters["position"]["longitude"] == -122.4194
        assert decision.parameters["position"]["altitude_msl"] == 100.0
        assert "speed_ms" not in decision.parameters

    def test_goto_with_speed(self):
        """Test creating a GOTO decision with speed."""
        position = Position(latitude=37.0, longitude=-122.0, altitude_msl=50.0)
        decision = Decision.goto(position=position, reason="Fast transit", speed=10.0)
        assert decision.parameters["speed_ms"] == 10.0

    def test_goto_with_confidence(self):
        """Test creating a GOTO decision with custom confidence."""
        position = Position(latitude=37.0, longitude=-122.0, altitude_msl=50.0)
        decision = Decision.goto(position=position, reason="Uncertain path", confidence=0.7)
        assert decision.confidence == 0.7

    def test_inspect_basic(self):
        """Test creating an INSPECT decision."""
        position = Position(latitude=37.0, longitude=-122.0, altitude_msl=50.0)
        decision = Decision.inspect(
            asset_id="asset_001",
            position=position,
            reason="Routine inspection",
        )
        assert decision.action == ActionType.INSPECT
        assert decision.reasoning == "Routine inspection"
        assert decision.parameters["asset_id"] == "asset_001"
        assert decision.parameters["position"]["latitude"] == 37.0
        assert decision.parameters["orbit_radius_m"] == 20.0  # default
        assert decision.parameters["dwell_time_s"] == 30.0  # default

    def test_inspect_with_custom_parameters(self):
        """Test creating an INSPECT decision with custom parameters."""
        position = Position(latitude=37.0, longitude=-122.0, altitude_msl=50.0)
        inspection = {"orbit_radius_m": 25.0, "dwell_time_s": 45.0}
        decision = Decision.inspect(
            asset_id="asset_002",
            position=position,
            reason="Detailed inspection",
            inspection=inspection,
        )
        assert decision.parameters["orbit_radius_m"] == 25.0
        assert decision.parameters["dwell_time_s"] == 45.0

    def test_return_to_dock_basic(self):
        """Test creating a DOCK decision."""
        decision = Decision.return_to_dock(reason="Low battery")
        assert decision.action == ActionType.DOCK
        assert decision.reasoning == "Low battery"
        assert decision.confidence == 1.0

    def test_return_to_dock_with_confidence(self):
        """Test creating a DOCK decision with custom confidence."""
        decision = Decision.return_to_dock(reason="Mission complete", confidence=0.9)
        assert decision.confidence == 0.9


class TestDecisionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_reasoning(self):
        """Test decision with empty reasoning."""
        decision = Decision(action=ActionType.WAIT, reasoning="")
        assert decision.reasoning == ""

    def test_zero_confidence(self):
        """Test decision with zero confidence."""
        decision = Decision(action=ActionType.WAIT, confidence=0.0)
        assert decision.confidence == 0.0

    def test_complex_parameters(self):
        """Test decision with complex nested parameters."""
        params = {
            "waypoints": [
                {"lat": 37.0, "lon": -122.0},
                {"lat": 37.1, "lon": -122.1},
            ],
            "options": {
                "avoid_zones": ["zone_a", "zone_b"],
                "max_altitude": 200,
            },
        }
        decision = Decision(action=ActionType.GOTO, parameters=params)
        assert decision.parameters["waypoints"][0]["lat"] == 37.0
        assert decision.parameters["options"]["max_altitude"] == 200

    def test_multiple_risk_factors(self):
        """Test decision with multiple risk factors."""
        risk_factors = {
            "battery": 0.5,
            "wind": 0.3,
            "gps": 0.1,
            "health": 0.0,
            "distance": 0.2,
        }
        decision = Decision(action=ActionType.INSPECT, risk_factors=risk_factors)
        assert len(decision.risk_factors) == 5
        assert decision.risk_factors["battery"] == 0.5

    def test_supersedes_chain(self):
        """Test decision that supersedes another."""
        original = Decision(action=ActionType.GOTO, decision_id="dec_001")
        replacement = Decision(
            action=ActionType.ABORT,
            decision_id="dec_002",
            supersedes="dec_001",
        )
        assert replacement.supersedes == original.decision_id

    def test_all_movement_actions(self):
        """Test that all expected movement actions are recognized."""
        movement_actions = [
            ActionType.GOTO,
            ActionType.TAKEOFF,
            ActionType.LAND,
            ActionType.RTL,
            ActionType.DOCK,
            ActionType.ORBIT,
        ]
        for action in movement_actions:
            decision = Decision(action=action)
            assert decision.is_movement is True, f"{action} should be a movement action"

    def test_all_non_movement_actions(self):
        """Test that non-movement actions are recognized."""
        non_movement_actions = [
            ActionType.WAIT,
            ActionType.ABORT,
            ActionType.INSPECT,
            ActionType.RECHARGE,
            ActionType.UNDOCK,
            ActionType.RETURN,
            ActionType.NONE,
        ]
        for action in non_movement_actions:
            decision = Decision(action=action)
            assert decision.is_movement is False, f"{action} should not be a movement action"


class TestDecisionTimestamp:
    """Test Decision timestamp handling."""

    def test_default_timestamp_is_now(self):
        """Test that default timestamp is approximately now."""
        before = datetime.now()
        decision = Decision(action=ActionType.WAIT)
        after = datetime.now()

        assert before <= decision.timestamp <= after

    def test_custom_timestamp(self):
        """Test setting a custom timestamp."""
        custom_time = datetime(2023, 1, 15, 10, 30, 0)
        decision = Decision(action=ActionType.WAIT, timestamp=custom_time)
        assert decision.timestamp == custom_time

    def test_timestamp_in_decision_id(self):
        """Test that decision_id contains timestamp information."""
        timestamp = datetime(2023, 6, 15, 12, 30, 45)
        decision = Decision(action=ActionType.WAIT, timestamp=timestamp)

        # The auto-generated ID should contain the timestamp
        assert "20230615" in decision.decision_id
        assert "123045" in decision.decision_id


class TestDecisionActionTypes:
    """Test all action types are supported."""

    @pytest.mark.parametrize(
        "action_type",
        [
            ActionType.GOTO,
            ActionType.TAKEOFF,
            ActionType.LAND,
            ActionType.RTL,
            ActionType.INSPECT,
            ActionType.ORBIT,
            ActionType.DOCK,
            ActionType.RECHARGE,
            ActionType.UNDOCK,
            ActionType.WAIT,
            ActionType.ABORT,
            ActionType.RETURN,
            ActionType.NONE,
        ],
    )
    def test_all_action_types_create_valid_decision(self, action_type):
        """Test that all action types can create valid decisions."""
        decision = Decision(action=action_type)
        assert decision.action == action_type
        assert decision.decision_id.startswith("dec_")

    @pytest.mark.parametrize(
        "action_type,expected_is_movement",
        [
            (ActionType.GOTO, True),
            (ActionType.TAKEOFF, True),
            (ActionType.LAND, True),
            (ActionType.RTL, True),
            (ActionType.DOCK, True),
            (ActionType.ORBIT, True),
            (ActionType.INSPECT, False),
            (ActionType.RECHARGE, False),
            (ActionType.UNDOCK, False),
            (ActionType.WAIT, False),
            (ActionType.ABORT, False),
            (ActionType.RETURN, False),
            (ActionType.NONE, False),
        ],
    )
    def test_is_movement_for_all_types(self, action_type, expected_is_movement):
        """Test is_movement property for all action types."""
        decision = Decision(action=action_type)
        assert decision.is_movement == expected_is_movement
