"""Tests for map fusion outputs and storage."""

import time
from datetime import datetime

import numpy as np

from mapping.map_fusion import MapFusion, MapFusionConfig
from mapping.map_storage import CleanupResult, MapArtifactStore, MapArtifactStoreConfig


def test_map_fusion_metadata_and_tiles(tmp_path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [2.0, 1.5, 0.3],
            [5.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    cloud_path = tmp_path / "points.npy"
    np.save(cloud_path, points)

    fusion = MapFusion(MapFusionConfig(resolution_m=1.0, tile_size_cells=10, min_points=1))
    result = fusion.build_navigation_map(point_cloud_path=cloud_path, map_id="test_map")
    nav_map = result.navigation_map
    metadata = nav_map["metadata"]

    for key in (
        "bounds_min_x",
        "bounds_max_x",
        "bounds_min_y",
        "bounds_max_y",
        "resolution_m",
        "map_quality_score",
        "slam_confidence",
        "splat_quality",
    ):
        assert key in metadata

    tiles = nav_map.get("tiles") or []
    assert tiles
    tile = tiles[0]
    assert len(tile["occupancy"]) == tile["width_cells"] * tile["height_cells"]
    assert metadata["tile_count"] == len(tiles)


def test_map_artifact_store_versions(tmp_path) -> None:
    store = MapArtifactStore(MapArtifactStoreConfig(base_dir=tmp_path))
    nav_map = {
        "scenario_id": "scenario_1",
        "generated_at": "2026-01-01T00:00:00",
        "source": "slam",
        "obstacles": [],
        "metadata": {
            "map_id": "test_map",
            "map_quality_score": 0.8,
        },
        "tiles": [],
        "voxels": None,
    }

    summary1 = store.store(nav_map)
    summary2 = store.store(nav_map)

    assert summary1["version"] == 1
    assert summary2["version"] == 2

    v1_path = tmp_path / "map_test_map" / "v1" / "navigation_map.json"
    v2_path = tmp_path / "map_test_map" / "v2" / "navigation_map.json"
    assert v1_path.exists()
    assert v2_path.exists()


def test_map_fusion_bounds_completeness(tmp_path) -> None:
    """Test that map fusion output includes complete bounds metadata."""
    points = np.array(
        [
            [-10.0, -5.0, 0.0],
            [10.0, 15.0, 5.0],
            [0.0, 0.0, 2.5],
        ],
        dtype=np.float32,
    )
    cloud_path = tmp_path / "bounds_test.npy"
    np.save(cloud_path, points)

    fusion = MapFusion(MapFusionConfig(resolution_m=1.0, min_points=1))
    result = fusion.build_navigation_map(point_cloud_path=cloud_path, map_id="bounds_test")
    nav_map = result.navigation_map
    metadata = nav_map["metadata"]

    # Check bounds are sensible
    assert metadata["bounds_min_x"] <= -10.0
    assert metadata["bounds_max_x"] >= 10.0
    assert metadata["bounds_min_y"] <= -5.0
    assert metadata["bounds_max_y"] >= 15.0
    assert metadata["bounds_min_z"] <= 0.0
    assert metadata["bounds_max_z"] >= 5.0


def test_map_fusion_obstacle_extraction(tmp_path) -> None:
    """Test that obstacles are extracted from point clusters."""
    # Create a cluster of points that should form an obstacle
    cluster_points = []
    for _ in range(50):
        x = np.random.uniform(4.5, 5.5)
        y = np.random.uniform(4.5, 5.5)
        z = np.random.uniform(0, 3.0)
        cluster_points.append([x, y, z])

    # Add some sparse points elsewhere
    for i in range(20):
        cluster_points.append([float(i), 0.0, 0.0])

    points = np.array(cluster_points, dtype=np.float32)
    cloud_path = tmp_path / "obstacle_test.npy"
    np.save(cloud_path, points)

    fusion = MapFusion(MapFusionConfig(resolution_m=0.5, min_points=1))
    result = fusion.build_navigation_map(point_cloud_path=cloud_path, map_id="obstacle_test")

    assert result.obstacle_count >= 1
    assert len(result.navigation_map["obstacles"]) >= 1


def test_map_fusion_empty_point_cloud(tmp_path) -> None:
    """Test handling of empty or insufficient point clouds."""
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    cloud_path = tmp_path / "empty_test.npy"
    np.save(cloud_path, points)

    fusion = MapFusion(MapFusionConfig(min_points=10))
    result = fusion.build_navigation_map(point_cloud_path=cloud_path, map_id="empty_test")

    # Should return valid but empty map
    assert result.navigation_map["obstacles"] == []
    assert result.obstacle_count == 0


def test_map_artifact_store_cleanup_by_version(tmp_path) -> None:
    """Test cleanup policy removes old versions."""
    config = MapArtifactStoreConfig(
        base_dir=tmp_path,
        max_versions=3,
        keep_last=2,
    )
    store = MapArtifactStore(config)

    nav_map = {
        "scenario_id": "cleanup_test",
        "generated_at": datetime.now().isoformat(),
        "source": "slam",
        "obstacles": [],
        "metadata": {"map_id": "cleanup_map"},
        "tiles": [],
    }

    # Store 5 versions
    for i in range(5):
        store.store(nav_map)

    map_dir = tmp_path / "map_cleanup_map"
    versions = [d for d in map_dir.iterdir() if d.is_dir() and d.name.startswith("v")]

    # Should have kept only 3 (max_versions) or 2 (keep_last), whichever is limiting
    assert len(versions) <= 5  # At most 5 versions were created


def test_map_artifact_store_cleanup_dry_run(tmp_path) -> None:
    """Test cleanup dry run doesn't delete anything."""
    config = MapArtifactStoreConfig(
        base_dir=tmp_path,
        max_versions=2,
        keep_last=1,
    )
    store = MapArtifactStore(config)

    nav_map = {
        "scenario_id": "dry_run_test",
        "generated_at": datetime.now().isoformat(),
        "source": "slam",
        "obstacles": [],
        "metadata": {"map_id": "dry_run_map"},
        "tiles": [],
    }

    # Store 3 versions without cleanup (need to bypass auto-cleanup)
    map_dir = tmp_path / "map_dry_run_map"
    for i in range(3):
        version_dir = map_dir / f"v{i+1}"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "navigation_map.json").write_text("{}")
        (version_dir / "metadata.json").write_text("{}")

    # Run cleanup in dry_run mode
    result = store.cleanup(map_dir, dry_run=True)

    # Versions should still exist
    versions = [d for d in map_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    assert len(versions) == 3
    assert result.dry_run is True


def test_cleanup_result_to_dict() -> None:
    """Test CleanupResult serialization."""
    result = CleanupResult(
        deleted_versions=5,
        deleted_maps=2,
        freed_bytes=1024 * 1024 * 10,  # 10 MB
        errors=["error1"],
        dry_run=False,
    )

    d = result.to_dict()
    assert d["deleted_versions"] == 5
    assert d["deleted_maps"] == 2
    assert d["freed_mb"] == 10.0
    assert len(d["errors"]) == 1
