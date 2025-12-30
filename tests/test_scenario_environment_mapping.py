"""Tests for scenario environment -> AirSim mapping helpers."""

from agent.server import main as main_module
from agent.server.scenarios import EnvironmentConditions


def test_map_scenario_environment_rain() -> None:
    env = EnvironmentConditions(
        precipitation="rain",
        visibility_m=10000.0,
        is_daylight=True,
        wind_speed_ms=5.0,
        wind_direction_deg=90.0,
    )

    mapped = main_module._map_scenario_environment(env)

    assert mapped["rain"] == 0.7
    assert mapped["snow"] == 0.0
    assert mapped["fog"] == 0.0
    assert mapped["dust"] == 0.0
    assert mapped["hour"] == 12
    assert mapped["wind_speed_ms"] == 5.0
    assert mapped["wind_direction_deg"] == 90.0


def test_map_scenario_environment_visibility_drives_fog() -> None:
    env = EnvironmentConditions(
        precipitation="none",
        visibility_m=2000.0,
        is_daylight=False,
        wind_speed_ms=0.0,
        wind_direction_deg=0.0,
    )

    mapped = main_module._map_scenario_environment(env)

    assert mapped["fog"] == 0.8
    assert mapped["hour"] == 22


def test_env_changed_thresholds() -> None:
    baseline = {
        "rain": 0.3,
        "snow": 0.0,
        "fog": 0.2,
        "dust": 0.0,
        "hour": 12,
        "is_daylight": True,
        "wind_speed_ms": 4.0,
        "wind_direction_deg": 90.0,
    }

    small_change = dict(baseline, rain=0.32, wind_speed_ms=4.1)
    assert main_module._env_changed(baseline, small_change) is False

    larger_change = dict(baseline, rain=0.4, wind_speed_ms=4.5)
    assert main_module._env_changed(baseline, larger_change) is True

    day_change = dict(baseline, is_daylight=False)
    assert main_module._env_changed(baseline, day_change) is True
