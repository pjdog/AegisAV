"""Tests for navigation map API responses."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI

from agent.server.api_navigation import register_navigation_routes
from agent.server.state import server_state


@pytest.mark.asyncio
async def test_map_status_no_map(make_async_client) -> None:
    app = FastAPI()
    register_navigation_routes(app)

    async with make_async_client(app) as client:
        resp = await client.get("/api/navigation/map/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["map_available"] is False
    assert data["obstacle_count"] == 0


@pytest.mark.asyncio
async def test_map_status_with_map(make_async_client) -> None:
    app = FastAPI()
    register_navigation_routes(app)

    previous_map = server_state.navigation_map
    try:
        server_state.navigation_map = {
            "scenario_id": "scenario_test",
            "generated_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "source": "slam",
            "obstacles": [{"obstacle_id": "obs_1"}],
            "metadata": {"version": 2, "map_quality_score": 0.6},
        }

        async with make_async_client(app) as client:
            resp = await client.get("/api/navigation/map/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["map_available"] is True
        assert data["obstacle_count"] == 1
        assert data["map_version"] == 2
    finally:
        server_state.navigation_map = previous_map


@pytest.mark.asyncio
async def test_map_health_report(make_async_client) -> None:
    app = FastAPI()
    register_navigation_routes(app)

    previous_map = server_state.navigation_map
    previous_last_valid = getattr(server_state, "last_valid_navigation_map", None)
    try:
        server_state.navigation_map = {
            "scenario_id": "scenario_health",
            "generated_at": (datetime.now() - timedelta(seconds=5)).isoformat(),
            "last_updated": datetime.now().isoformat(),
            "source": "slam",
            "obstacles": [{"obstacle_id": "obs_1"}],
            "metadata": {
                "version": 1,
                "map_quality_score": 0.8,
                "slam_confidence": 0.9,
                "splat_quality": 0.9,
            },
        }
        server_state.last_valid_navigation_map = server_state.navigation_map

        async with make_async_client(app) as client:
            resp = await client.get("/api/navigation/map/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["map_available"] is True
        assert data["map_valid"] is True
        assert data["gate_status"] in {"pass", "warn"}
    finally:
        server_state.navigation_map = previous_map
        server_state.last_valid_navigation_map = previous_last_valid


@pytest.mark.asyncio
async def test_splat_proxy_endpoint(make_async_client) -> None:
    app = FastAPI()
    register_navigation_routes(app)

    repo_root = Path(__file__).resolve().parents[1]
    repo_splat_dir = repo_root / "data" / "splats"
    target_scene = repo_splat_dir / "scene_testproxy" / "v1"
    target_scene.mkdir(parents=True, exist_ok=True)
    (target_scene / "planning_proxy.json").write_text(
        '{"metadata": {"map_id": "testproxy"}, "obstacles": []}'
    )

    try:
        async with make_async_client(app) as client:
            resp = await client.get("/api/navigation/splat/proxy/testproxy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["proxy"]["metadata"]["map_id"] == "testproxy"
    finally:
        for path in target_scene.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
        for path in sorted((repo_splat_dir / "scene_testproxy").rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        if (repo_splat_dir / "scene_testproxy").exists():
            (repo_splat_dir / "scene_testproxy").rmdir()


@pytest.mark.asyncio
async def test_map_preview_no_map(make_async_client) -> None:
    """Test map preview endpoint with no map."""
    app = FastAPI()
    register_navigation_routes(app)

    previous_map = server_state.navigation_map
    try:
        server_state.navigation_map = None

        async with make_async_client(app) as client:
            resp = await client.get("/api/navigation/map/preview")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("preview") is None or data.get("available") is False
    finally:
        server_state.navigation_map = previous_map


@pytest.mark.asyncio
async def test_map_preview_with_obstacles(make_async_client) -> None:
    """Test map preview endpoint with obstacles."""
    app = FastAPI()
    register_navigation_routes(app)

    previous_map = server_state.navigation_map
    try:
        server_state.navigation_map = {
            "scenario_id": "preview_test",
            "generated_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "source": "slam",
            "obstacles": [
                {
                    "obstacle_id": "obs_1",
                    "x": 10.0,
                    "y": 20.0,
                    "radius_m": 5.0,
                    "height_m": 10.0,
                    "type": "building",
                },
                {
                    "obstacle_id": "obs_2",
                    "x": -5.0,
                    "y": 15.0,
                    "radius_m": 3.0,
                    "height_m": 5.0,
                    "type": "tree",
                },
            ],
            "metadata": {
                "version": 1,
                "bounds_min_x": -20.0,
                "bounds_max_x": 30.0,
                "bounds_min_y": -10.0,
                "bounds_max_y": 40.0,
                "resolution_m": 2.0,
            },
        }

        async with make_async_client(app) as client:
            resp = await client.get("/api/navigation/map/preview")
        assert resp.status_code == 200
        data = resp.json()

        preview = data.get("preview")
        if preview:
            assert preview.get("obstacle_count") == 2
            assert "bounds" in preview
    finally:
        server_state.navigation_map = previous_map


@pytest.mark.asyncio
async def test_splat_scenes_empty(make_async_client) -> None:
    """Test splat scenes endpoint with no scenes."""
    app = FastAPI()
    register_navigation_routes(app)

    async with make_async_client(app) as client:
        resp = await client.get("/api/navigation/splat/scenes")
    assert resp.status_code == 200
    data = resp.json()
    assert "scenes" in data
    assert isinstance(data["scenes"], list)


@pytest.mark.asyncio
async def test_splat_proxy_not_found(make_async_client) -> None:
    """Test splat proxy endpoint with non-existent run ID."""
    app = FastAPI()
    register_navigation_routes(app)

    async with make_async_client(app) as client:
        resp = await client.get("/api/navigation/splat/proxy/nonexistent_run")
    # Should return 404 or empty response
    assert resp.status_code in {200, 404}
    if resp.status_code == 200:
        data = resp.json()
        assert data.get("proxy") is None or data.get("available") is False
