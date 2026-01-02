"""Tests for map update safety gates."""

from mapping.safety_gates import MapUpdateGate, SafetyGateConfig, SafetyGateResult


def test_map_update_gate_rejects_low_quality() -> None:
    gate = MapUpdateGate(SafetyGateConfig(min_map_confidence=0.7))
    new_map = {
        "metadata": {
            "map_quality_score": 0.2,
            "slam_confidence": 1.0,
            "splat_quality": 1.0,
        },
        "obstacles": [],
    }

    result = gate.check_update(new_map)
    assert result.result == SafetyGateResult.REJECT
    assert "quality" in result.reason


def test_map_update_gate_warns_on_low_splat_quality() -> None:
    gate = MapUpdateGate(SafetyGateConfig(min_splat_quality=0.6))
    new_map = {
        "metadata": {
            "map_quality_score": 0.9,
            "slam_confidence": 0.9,
            "splat_quality": 0.2,
        },
        "obstacles": [],
    }

    result = gate.check_update(new_map)
    assert result.result == SafetyGateResult.WARN
    assert "splat" in result.gate_name
