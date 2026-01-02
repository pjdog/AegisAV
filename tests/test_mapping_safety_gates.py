"""Tests for mapping safety gates and output validation."""

from datetime import datetime, timedelta

from mapping.decision_context import MapContext
from mapping.safety_gates import (
    MapUpdateGate,
    PlannerSafetyGate,
    SafetyGateConfig,
    SafetyGateResult,
    validate_map_output,
)


def test_map_update_gate_rejects_low_quality() -> None:
    gate = MapUpdateGate(SafetyGateConfig(min_map_confidence=0.8))
    nav_map = {
        "generated_at": datetime.now().isoformat(),
        "obstacles": [],
        "metadata": {
            "map_quality_score": 0.2,
            "slam_confidence": 1.0,
            "splat_quality": 1.0,
        },
    }

    result = gate.check_update(nav_map, None)

    assert result.result == SafetyGateResult.REJECT


def test_planner_gate_freezes_on_stale_map() -> None:
    config = SafetyGateConfig(max_map_age_s=1.0, freeze_on_stale=True)
    gate = PlannerSafetyGate(config)
    stale_time = (datetime.now() - timedelta(seconds=5)).isoformat()

    nav_map = {
        "generated_at": stale_time,
        "obstacles": [],
        "metadata": {"map_quality_score": 1.0},
    }
    map_context = MapContext.from_navigation_map(
        nav_map,
        stale_threshold_s=config.max_map_age_s,
        min_quality_score=config.min_map_confidence,
    )

    result = gate.check_planning_allowed(map_context)

    assert result.result == SafetyGateResult.FREEZE


def test_validate_map_output_flags_missing_fields() -> None:
    nav_map = {
        "generated_at": datetime.now().isoformat(),
        "obstacles": [{"height_m": 5.0}],
    }

    valid, errors = validate_map_output(nav_map, require_obstacles=True, min_obstacles=1)

    assert valid is False
    assert any("radius_m" in err for err in errors)
