"""Tests for AirSim environment helpers and status endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from agent.server import main as main_module


class DummyBridge:
    """Minimal AirSim bridge stub for environment application tests."""

    def __init__(self) -> None:
        self.connected = True
        self.set_weather = AsyncMock()
        self.set_time_of_day = AsyncMock()
        self.set_wind = AsyncMock()
        self.set_vehicle_pose = AsyncMock(return_value=True)


class DummyTask:
    """Minimal asyncio task stub."""

    def done(self) -> bool:
        return False


def _dummy_manager(enabled: bool = True) -> SimpleNamespace:
    config = SimpleNamespace(
        simulation=SimpleNamespace(
            airsim_enabled=enabled,
            airsim_host="10.0.0.5",
            airsim_vehicle_name="DroneX",
        ),
    )
    return SimpleNamespace(config=config)


@pytest.mark.asyncio
async def test_apply_airsim_environment_calls_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = DummyBridge()
    monkeypatch.setattr(main_module.server_state, "airsim_bridge", bridge)
    monkeypatch.setattr(main_module.unreal_manager, "active_connections", 0)

    mapped = {
        "rain": 0.25,
        "snow": 0.0,
        "fog": 0.1,
        "dust": 0.0,
        "hour": 18,
        "wind_speed_ms": 4.5,
        "wind_direction_deg": 270.0,
    }

    await main_module._apply_airsim_environment(mapped)

    bridge.set_weather.assert_awaited_once_with(rain=0.25, snow=0.0, fog=0.1, dust=0.0)
    bridge.set_time_of_day.assert_awaited_once_with(hour=18, is_enabled=True, celestial_clock_speed=1.0)
    bridge.set_wind.assert_awaited_once_with(speed_ms=4.5, direction_deg=270.0)


@pytest.mark.asyncio
async def test_get_airsim_status_reports_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "get_config_manager", lambda: _dummy_manager())
    monkeypatch.setattr(main_module, "_airsim_launch_supported", lambda: True)
    monkeypatch.setattr(main_module.server_state, "airsim_bridge", None)
    monkeypatch.setattr(main_module.server_state, "airsim_connect_task", DummyTask())
    monkeypatch.setattr(main_module.server_state, "airsim_last_error", "Still connecting")

    status = await main_module.get_airsim_status()

    assert status["enabled"] is True
    assert status["host"] == "10.0.0.5"
    assert status["vehicle_name"] == "DroneX"
    assert status["connecting"] is True
    assert status["bridge_connected"] is False
    assert status["launch_supported"] is True
    assert status["last_error"] == "Still connecting"


@pytest.mark.asyncio
async def test_start_airsim_schedules_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "get_config_manager", lambda: _dummy_manager())
    monkeypatch.setattr(main_module, "_airsim_bridge_connected", lambda: False)
    monkeypatch.setattr(
        main_module,
        "_launch_airsim_process",
        lambda: (True, True, "AirSim launch initiated."),
    )
    schedule_calls: dict[str, int] = {"count": 0}

    def _fake_schedule() -> bool:
        schedule_calls["count"] += 1
        return True

    monkeypatch.setattr(main_module, "_schedule_airsim_connect", _fake_schedule)

    response = await main_module.start_airsim()

    assert response["launch_supported"] is True
    assert response["launch_started"] is True
    assert response["connecting"] is True
    assert schedule_calls["count"] == 1


@pytest.mark.asyncio
async def test_start_airsim_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "get_config_manager", lambda: _dummy_manager(enabled=False))
    with pytest.raises(HTTPException) as exc_info:
        await main_module.start_airsim()

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_sync_airsim_scene_sets_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = DummyBridge()
    monkeypatch.setattr(main_module.server_state, "airsim_bridge", bridge)
    monkeypatch.setattr(main_module, "get_config_manager", lambda: _dummy_manager())

    scenario = SimpleNamespace(
        scenario_id="sync_001",
        name="Sync Test",
        drones=[SimpleNamespace(latitude=1.0, longitude=2.0, altitude_agl=12.0)],
        environment=None,
    )

    result = await main_module._sync_airsim_scene(scenario)

    assert result["synced"] is True
    bridge.set_vehicle_pose.assert_awaited_once_with(0.0, 0.0, -12.0)


@pytest.mark.asyncio
async def test_sync_airsim_scene_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = DummyBridge()
    bridge.connected = False
    monkeypatch.setattr(main_module.server_state, "airsim_bridge", bridge)
    monkeypatch.setattr(main_module, "get_config_manager", lambda: _dummy_manager())

    scenario = SimpleNamespace(
        scenario_id="sync_002",
        name="No Connect",
        drones=[SimpleNamespace(latitude=1.0, longitude=2.0, altitude_agl=0.0)],
        environment=None,
    )

    result = await main_module._sync_airsim_scene(scenario)

    assert result["synced"] is False
    assert result["reason"] == "airsim_not_connected"
