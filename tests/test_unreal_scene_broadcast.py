"""Tests for scenario-to-Unreal broadcast helpers."""

from unittest.mock import AsyncMock

import pytest

from agent.server import main as main_module
from agent.server.scenarios import get_scenario
from agent.server.unreal_stream import UnrealMessageType
from agent.server.world_model import DockStatus, WorldModel
from autonomy.vehicle_state import Position


@pytest.mark.asyncio
async def test_broadcast_scenario_scene_sends_dock_assets_and_defects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = get_scenario("normal_ops_001")
    assert scenario is not None

    world_model = WorldModel()
    dock_pos = Position(latitude=37.0, longitude=-122.0, altitude_msl=0.0, altitude_agl=0.0)
    world_model.set_dock(dock_pos, DockStatus.AVAILABLE)

    monkeypatch.setattr(main_module.server_state, "world_model", world_model)
    monkeypatch.setattr(main_module.server_state, "airsim_env_last", None)
    monkeypatch.setattr(main_module.unreal_manager, "connections", {"client": object()})

    broadcast = AsyncMock()
    monkeypatch.setattr(main_module.unreal_manager, "broadcast", broadcast)

    await main_module.broadcast_scenario_scene(scenario, include_defects=True)

    payloads = [call.args[0] for call in broadcast.await_args_list]
    types = [payload.get("type") for payload in payloads]

    assert UnrealMessageType.CLEAR_ASSETS.value in types
    assert UnrealMessageType.CLEAR_ANOMALY_MARKERS.value in types
    assert UnrealMessageType.DOCK_UPDATE.value in types
    assert types.count(UnrealMessageType.SPAWN_ASSET.value) == len(scenario.assets)
    assert UnrealMessageType.CLEAR_DEFECTS.value in types
    assert types.count(UnrealMessageType.SPAWN_DEFECT.value) == len(scenario.defects)

    dock_payload = next(p for p in payloads if p.get("type") == UnrealMessageType.DOCK_UPDATE.value)
    assert dock_payload["beacon_active"] is True


@pytest.mark.asyncio
async def test_broadcast_scenario_scene_skips_without_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = get_scenario("normal_ops_001")
    assert scenario is not None

    monkeypatch.setattr(main_module.unreal_manager, "connections", {})
    broadcast = AsyncMock()
    monkeypatch.setattr(main_module.unreal_manager, "broadcast", broadcast)

    await main_module.broadcast_scenario_scene(scenario, include_defects=True)

    broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_dock_state_honors_beacon_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module.unreal_manager, "connections", {"client": object()})
    broadcast = AsyncMock()
    monkeypatch.setattr(main_module.unreal_manager, "broadcast", broadcast)

    await main_module.broadcast_dock_state(
        dock_id="dock_main",
        status="available",
        latitude=1.0,
        longitude=2.0,
        altitude_m=3.0,
        beacon_active=False,
    )

    payload = broadcast.await_args.args[0]
    assert payload["type"] == UnrealMessageType.DOCK_UPDATE.value
    assert payload["beacon_active"] is False
