"""Scenario data integrity tests."""

from agent.server.scenarios import DOCK_LATITUDE, DOCK_LONGITUDE, get_all_scenarios
from agent.server.world_model import AssetType


def test_scenario_ids_are_unique() -> None:
    scenarios = get_all_scenarios()
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))


def test_scenario_assets_and_defects_are_consistent() -> None:
    scenarios = get_all_scenarios()
    valid_asset_types = {t.value for t in AssetType}

    for scenario in scenarios:
        asset_ids = [a.asset_id for a in scenario.assets]
        assert len(asset_ids) == len(set(asset_ids)), scenario.scenario_id

        for asset in scenario.assets:
            assert asset.asset_type in valid_asset_types, scenario.scenario_id
            assert asset.inspection_altitude_agl > 0, scenario.scenario_id
            assert asset.orbit_radius_m > 0, scenario.scenario_id
            assert asset.dwell_time_s > 0, scenario.scenario_id
            assert 0.0 <= asset.anomaly_severity <= 1.0, scenario.scenario_id

        for defect in scenario.defects:
            assert defect.asset_id in asset_ids, scenario.scenario_id
            assert 0.0 <= defect.severity <= 1.0, scenario.scenario_id


def test_scenario_locations_are_close_to_dock() -> None:
    scenarios = get_all_scenarios()
    max_delta = 0.01  # ~1.1km buffer from dock for demo visibility

    for scenario in scenarios:
        for asset in scenario.assets:
            assert abs(asset.latitude - DOCK_LATITUDE) <= max_delta, scenario.scenario_id
            assert abs(asset.longitude - DOCK_LONGITUDE) <= max_delta, scenario.scenario_id


def test_scenario_drone_ids_are_unique() -> None:
    scenarios = get_all_scenarios()
    for scenario in scenarios:
        drone_ids = [d.drone_id for d in scenario.drones]
        assert len(drone_ids) == len(set(drone_ids)), scenario.scenario_id
