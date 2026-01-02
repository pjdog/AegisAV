"""Tests for navigation map and avoidance helpers."""

from types import SimpleNamespace

from agent.server.navigation_map import build_navigation_map
from agent.server.vision.vision_service import VisionService
from simulation.airsim_action_executor import AirSimActionExecutor
from simulation.coordinate_utils import GeoReference


class DummyBridge:
    """Minimal bridge stub."""


def test_build_navigation_map_filters_assets() -> None:
    assets = [
        {"asset_id": "house_1", "asset_type": "building", "latitude": 1.0, "longitude": 2.0},
        {"asset_id": "panel_1", "asset_type": "solar_panel", "latitude": 3.0, "longitude": 4.0},
    ]

    nav_map = build_navigation_map(assets, "scenario_1")

    assert nav_map["scenario_id"] == "scenario_1"
    assert len(nav_map["obstacles"]) == 2
    assert {obs["asset_id"] for obs in nav_map["obstacles"]} == {"house_1", "panel_1"}


def test_vision_service_seeds_navigation_map() -> None:
    assets = [
        SimpleNamespace(asset_id="house_2", asset_type="building", latitude=5.0, longitude=6.0),
    ]
    vision = VisionService(world_model=SimpleNamespace())
    nav_map = vision.seed_navigation_map(assets, "scenario_2")

    assert nav_map["source"] == "vision_seed"
    assert len(nav_map["obstacles"]) == 1


def test_avoidance_adjusts_target() -> None:
    geo_ref = GeoReference(latitude=37.0, longitude=-122.0, altitude=0.0)
    executor = AirSimActionExecutor(DummyBridge(), geo_ref)
    executor.set_avoid_zones([
        {
            "asset_id": "house_3",
            "asset_type": "building",
            "latitude": 37.0001,
            "longitude": -122.0001,
            "radius_m": 25.0,
            "height_m": 10.0,
        }
    ])

    lat, lon, alt, avoidance = executor._apply_avoidance(37.0001, -122.0001, 5.0, "test")

    assert avoidance is not None
    assert alt >= 10.0
    assert (lat, lon) != (37.0001, -122.0001)


def test_navigation_map_avoidance_uses_ned_buffers() -> None:
    geo_ref = GeoReference(latitude=37.0, longitude=-122.0, altitude=0.0)
    executor = AirSimActionExecutor(DummyBridge(), geo_ref)
    nav_map = {
        "metadata": {"resolution_m": 6.0},
        "obstacles": [
            {
                "obstacle_id": "obs_ned",
                "x_ned": 0.0,
                "y_ned": 0.0,
                "radius_m": 5.0,
                "height_m": 4.0,
            }
        ],
    }
    executor.set_navigation_map(nav_map)

    target_lat, target_lon, _ = geo_ref.ned_to_gps(0.0, 0.0, geo_ref.altitude + 2.0)
    lat, lon, alt, avoidance = executor._apply_avoidance(target_lat, target_lon, 2.0, "ned_test")

    assert avoidance is not None
    assert alt >= 16.0
    assert (lat, lon) != (target_lat, target_lon)
