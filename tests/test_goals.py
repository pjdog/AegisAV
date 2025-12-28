"""
Tests for goal models and configuration defaults.
"""

from agent.server.goals import Goal, GoalSelectorConfig, GoalType


def test_goal_properties():
    """Verify Goal helper properties for abort/return."""
    goal_abort = Goal(goal_type=GoalType.ABORT, priority=1, reason="Emergency")
    goal_return = Goal(goal_type=GoalType.RETURN_LOW_BATTERY, priority=5, reason="Battery")
    goal_wait = Goal(goal_type=GoalType.WAIT, priority=10, reason="Idle")

    assert goal_abort.is_abort is True
    assert goal_abort.is_return is False

    assert goal_return.is_abort is False
    assert goal_return.is_return is True

    assert goal_wait.is_abort is False
    assert goal_wait.is_return is False


def test_goal_defaults():
    """Verify Goal default fields and optional values."""
    goal = Goal(goal_type=GoalType.INSPECT_ASSET, priority=10, reason="Inspect")

    assert goal.confidence == 1.0
    assert goal.deadline is None
    assert goal.target_asset is None


def test_goal_selector_config_defaults():
    """Verify GoalSelectorConfig default thresholds."""
    config = GoalSelectorConfig()

    assert config.battery_return_threshold == 30.0
    assert config.battery_critical_threshold == 20.0
    assert config.anomaly_revisit_interval_minutes == 10.0
    assert config.normal_cadence_minutes == 30.0
    assert config.use_advanced_engine is True


def test_goal_selector_config_customization():
    """Verify GoalSelectorConfig supports overrides."""
    config = GoalSelectorConfig(
        battery_return_threshold=25.0,
        battery_critical_threshold=15.0,
        anomaly_revisit_interval_minutes=5.0,
        normal_cadence_minutes=20.0,
        use_advanced_engine=False,
    )

    assert config.battery_return_threshold == 25.0
    assert config.battery_critical_threshold == 15.0
    assert config.anomaly_revisit_interval_minutes == 5.0
    assert config.normal_cadence_minutes == 20.0
    assert config.use_advanced_engine is False
