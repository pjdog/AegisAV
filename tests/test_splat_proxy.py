"""Tests for splat proxy output and storage."""

import json

from mapping.splat_storage import SplatMetadata, SplatScene, SplatStorage


def test_splat_metadata_serialization() -> None:
    """Test SplatMetadata can round-trip through JSON."""
    metadata = SplatMetadata(
        run_id="test_run",
        scene_id="test_scene",
        version=1,
        keyframe_count=100,
        total_iterations=30000,
        final_loss=0.01,
        bounds_min=[-10.0, -10.0, 0.0],
        bounds_max=[10.0, 10.0, 5.0],
        gaussian_count=50000,
        compressed_size_mb=25.5,
        psnr=28.5,
        ssim=0.92,
        lpips=0.15,
    )

    d = metadata.to_dict()

    # Check schema completeness
    assert "run_id" in d
    assert "scene_id" in d
    assert "version" in d
    assert "training" in d
    assert "bounds" in d
    assert "gaussians" in d
    assert "quality" in d
    assert "files" in d

    # Verify nested structure
    assert d["training"]["keyframe_count"] == 100
    assert d["bounds"]["min"] == [-10.0, -10.0, 0.0]
    assert d["bounds"]["max"] == [10.0, 10.0, 5.0]
    assert d["gaussians"]["count"] == 50000
    assert d["quality"]["psnr"] == 28.5
    assert d["quality"]["ssim"] == 0.92

    # Round-trip through from_dict
    restored = SplatMetadata.from_dict(d)
    assert restored.run_id == metadata.run_id
    assert restored.gaussian_count == metadata.gaussian_count
    assert restored.psnr == metadata.psnr
    assert restored.bounds_min == metadata.bounds_min


def test_splat_storage_store_and_retrieve(tmp_path) -> None:
    """Test storing and retrieving splat scenes."""
    storage = SplatStorage(tmp_path)

    # Create a mock model file
    model_path = tmp_path / "source_model.ply"
    model_path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0")

    # Create a mock preview file
    preview_path = tmp_path / "source_preview.ply"
    preview_path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0")

    metadata = SplatMetadata(
        run_id="test_run",
        scene_id="test_run",
        gaussian_count=1000,
        psnr=25.0,
        ssim=0.85,
    )

    # Store scene
    scene = storage.store_scene(
        run_id="test_run",
        model_path=model_path,
        preview_path=preview_path,
        metadata=metadata,
    )

    assert scene.metadata.version == 1
    assert scene.model_path.exists()
    assert scene.preview_path.exists()

    # Retrieve scene
    retrieved = storage.get_scene("test_run")
    assert retrieved is not None
    assert retrieved.metadata.run_id == "test_run"
    assert retrieved.metadata.gaussian_count == 1000


def test_splat_storage_versioning(tmp_path) -> None:
    """Test that multiple versions are stored correctly."""
    storage = SplatStorage(tmp_path)

    model_path = tmp_path / "model.ply"
    model_path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0")

    # Store multiple versions
    for i in range(3):
        metadata = SplatMetadata(
            run_id="versioned_run",
            scene_id="versioned_run",
            gaussian_count=1000 * (i + 1),
        )
        storage.store_scene(
            run_id="versioned_run",
            model_path=model_path,
            metadata=metadata,
        )

    # Check versions
    versions = storage.list_versions("versioned_run")
    assert len(versions) == 3
    assert versions == [1, 2, 3]

    # Latest should be v3
    latest = storage.get_scene("versioned_run")
    assert latest is not None
    assert latest.metadata.version == 3
    assert latest.metadata.gaussian_count == 3000

    # Can retrieve specific version
    v1 = storage.get_scene("versioned_run", version=1)
    assert v1 is not None
    assert v1.metadata.version == 1


def test_splat_storage_list_scenes(tmp_path) -> None:
    """Test listing all scenes."""
    storage = SplatStorage(tmp_path)

    model_path = tmp_path / "model.ply"
    model_path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0")

    # Store multiple scenes
    for run_id in ["scene_a", "scene_b", "scene_c"]:
        storage.store_scene(
            run_id=run_id,
            model_path=model_path,
            metadata=SplatMetadata(run_id=run_id, scene_id=run_id),
        )

    scenes = storage.list_scenes()
    assert len(scenes) == 3
    run_ids = {s.metadata.run_id for s in scenes}
    assert run_ids == {"scene_a", "scene_b", "scene_c"}


def test_splat_storage_delete_scene(tmp_path) -> None:
    """Test deleting scenes."""
    storage = SplatStorage(tmp_path)

    model_path = tmp_path / "model.ply"
    model_path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0")

    # Store scene
    storage.store_scene(
        run_id="delete_test",
        model_path=model_path,
        metadata=SplatMetadata(run_id="delete_test", scene_id="delete_test"),
    )

    assert storage.get_scene("delete_test") is not None

    # Delete scene
    result = storage.delete_scene("delete_test")
    assert result is True
    assert storage.get_scene("delete_test") is None


def test_splat_storage_stats(tmp_path) -> None:
    """Test storage statistics."""
    storage = SplatStorage(tmp_path)

    model_path = tmp_path / "model.ply"
    model_path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0")

    # Store some scenes
    for i in range(2):
        storage.store_scene(
            run_id=f"stats_test_{i}",
            model_path=model_path,
            metadata=SplatMetadata(run_id=f"stats_test_{i}", scene_id=f"stats_test_{i}"),
        )

    stats = storage.get_storage_stats()
    assert stats["scene_count"] == 2
    assert stats["total_versions"] == 2
    assert stats["total_size_mb"] >= 0


def test_splat_metadata_obstacle_count_sanity() -> None:
    """Test that gaussian count is reasonable for proxy generation."""
    # Typical scene should have thousands of gaussians
    metadata = SplatMetadata(
        run_id="proxy_test",
        scene_id="proxy_test",
        gaussian_count=50000,
    )

    # Gaussian count should be positive
    assert metadata.gaussian_count > 0

    # Check serialization includes count
    d = metadata.to_dict()
    assert d["gaussians"]["count"] == 50000


def test_splat_scene_paths(tmp_path) -> None:
    """Test SplatScene path properties."""
    metadata = SplatMetadata(
        run_id="path_test",
        scene_id="path_test",
        version=2,
        model_file="custom_model.ply",
        preview_file="custom_preview.ply",
    )

    scene = SplatScene(metadata=metadata, base_path=tmp_path)

    assert scene.scene_dir == tmp_path / "scene_path_test"
    assert scene.version_dir == tmp_path / "scene_path_test" / "v2"
    assert scene.model_path == tmp_path / "scene_path_test" / "v2" / "custom_model.ply"
    assert scene.preview_path == tmp_path / "scene_path_test" / "v2" / "custom_preview.ply"
