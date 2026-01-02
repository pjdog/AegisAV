"""Tests for mapping manifest generation and replay."""

from mapping.capture_replay import CaptureReplay
from mapping.manifest import DatasetManifest, ManifestBuilder, SensorCalibration, validate_manifest


def test_manifest_roundtrip(tmp_path) -> None:
    sequence_dir = tmp_path / "sequence_roundtrip"
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True)

    (frames_dir / "0001.png").write_bytes(b"fake")
    (frames_dir / "0001_depth.npy").write_bytes(b"fake_depth")

    calibration = SensorCalibration(
        fx=120.0,
        fy=121.0,
        cx=64.0,
        cy=48.0,
        width=128,
        height=96,
    )
    builder = ManifestBuilder(
        dataset_id="dataset_1",
        sequence_id="sequence_1",
        output_dir=sequence_dir,
        calibration=calibration,
    )
    builder.set_origin(47.0, -122.0, 10.0)
    builder.add_frame(
        timestamp_s=1.23,
        image_path="frames/0001.png",
        depth_path="frames/0001_depth.npy",
        position=[1.0, 2.0, 3.0],
        orientation=[10.0, 20.0, 30.0],
        is_keyframe=True,
        blur_score=0.1,
        feature_count=120,
    )
    manifest_path = builder.save()

    loaded = DatasetManifest.load(manifest_path)
    assert loaded.dataset_id == "dataset_1"
    assert loaded.sequence_id == "sequence_1"
    assert loaded.frame_count == 1
    assert loaded.calibration.fx == 120.0
    assert loaded.frames[0].image_path == "frames/0001.png"
    assert loaded.frames[0].depth_path == "frames/0001_depth.npy"


def test_capture_replay_uses_manifest(tmp_path) -> None:
    sequence_dir = tmp_path / "sequence_replay"
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True)

    img_path = frames_dir / "frame_a.png"
    depth_path = frames_dir / "frame_a_depth.npy"
    img_path.write_bytes(b"image")
    depth_path.write_bytes(b"depth")

    calibration = SensorCalibration(
        fx=300.0,
        fy=301.0,
        cx=160.0,
        cy=120.0,
        width=320,
        height=240,
    )
    builder = ManifestBuilder(
        dataset_id="dataset_2",
        sequence_id="sequence_2",
        output_dir=sequence_dir,
        calibration=calibration,
    )
    builder.add_frame(
        timestamp_s=2.5,
        image_path="frames/frame_a.png",
        depth_path="frames/frame_a_depth.npy",
        position=[4.0, 5.0, 6.0],
        orientation=[1.0, 2.0, 3.0],
    )
    builder.save()

    replay = CaptureReplay.from_directory(sequence_dir)
    assert replay.sequence.frame_count == 1

    frame = replay.sequence.frames[0]
    assert frame.image_path == img_path
    assert frame.depth_path == depth_path
    assert frame.fx == 300.0
    assert frame.fy == 301.0
    assert frame.width == 320
    assert frame.height == 240
    assert frame.pose.x == 4.0
    assert frame.pose.y == 5.0
    assert frame.pose.z == 6.0


def test_validate_manifest_detects_issues(tmp_path) -> None:
    sequence_dir = tmp_path / "sequence_bad"
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True)

    builder = ManifestBuilder(
        dataset_id="dataset_bad",
        sequence_id="sequence_bad",
        output_dir=sequence_dir,
        calibration=SensorCalibration(fx=0.0, fy=0.0, width=0, height=0),
    )
    builder.add_frame(
        timestamp_s=2.0,
        image_path="frames/missing.png",
        depth_path="frames/missing_depth.npy",
    )
    builder.add_frame(
        timestamp_s=1.0,
        image_path="frames/missing2.png",
    )
    manifest = builder.get_manifest()

    issues = validate_manifest(manifest, sequence_dir)
    assert "invalid_intrinsics" in issues
    assert "invalid_resolution" in issues
    assert "non_monotonic_timestamps" in issues
    assert any(issue.startswith("missing_image:") for issue in issues)
