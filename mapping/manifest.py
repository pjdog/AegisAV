"""Dataset manifest generation for SLAM and splatting pipelines.

Phase 1 Worker B: Dataset manifest generation (sequence index, sensor metadata).

The manifest provides a standardized index of capture sequences for training
SLAM and Gaussian splatting models. It includes sensor calibration, sequence
metadata, and frame indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SensorCalibration:
    """Camera/sensor calibration data."""

    # Camera intrinsics
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 1280
    height: int = 720

    # Distortion coefficients (k1, k2, p1, p2, k3)
    distortion: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # Camera-to-body transform (4x4 matrix, row-major)
    T_body_camera: list[float] = field(
        default_factory=lambda: [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    )

    # IMU-to-body transform (if different from identity)
    T_body_imu: list[float] | None = None

    # Depth camera calibration (if available)
    depth_scale: float = 1.0  # meters per unit
    depth_min_m: float = 0.1
    depth_max_m: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intrinsics": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "width": self.width,
                "height": self.height,
            },
            "distortion": self.distortion,
            "T_body_camera": self.T_body_camera,
            "T_body_imu": self.T_body_imu,
            "depth": {
                "scale": self.depth_scale,
                "min_m": self.depth_min_m,
                "max_m": self.depth_max_m,
            },
        }


@dataclass
class ManifestEntry:
    """A single frame entry in the manifest."""

    frame_index: int
    timestamp_s: float
    image_path: str
    depth_path: str | None = None
    pose_path: str | None = None

    # Pose (if inline rather than in separate file)
    position: list[float] | None = None  # [x, y, z] in NED
    orientation: list[float] | None = None  # [roll, pitch, yaw] in degrees

    # Keyframe info
    is_keyframe: bool = False
    keyframe_index: int | None = None

    # Quality metrics
    blur_score: float = 0.0
    feature_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "image_path": self.image_path,
        }

        if self.depth_path:
            result["depth_path"] = self.depth_path
        if self.pose_path:
            result["pose_path"] = self.pose_path
        if self.position:
            result["position"] = self.position
        if self.orientation:
            result["orientation"] = self.orientation
        if self.is_keyframe:
            result["is_keyframe"] = True
            result["keyframe_index"] = self.keyframe_index

        result["quality"] = {
            "blur_score": self.blur_score,
            "feature_count": self.feature_count,
        }

        return result


@dataclass
class DatasetManifest:
    """Complete dataset manifest for SLAM/splatting training.

    The manifest serves as the central index for a capture dataset,
    containing all metadata needed to train SLAM and Gaussian splatting
    models.
    """

    # Identification
    dataset_id: str
    sequence_id: str
    version: int = 1

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    capture_start: str | None = None
    capture_end: str | None = None

    # Paths (relative to manifest location)
    base_path: str = "."
    frames_dir: str = "frames"
    keyframes_dir: str = "keyframes"
    poses_dir: str = "poses"

    # Sensor configuration
    sensor_type: str = "airsim"  # airsim, realsense, custom
    calibration: SensorCalibration = field(default_factory=SensorCalibration)

    # Coordinate frame
    coordinate_frame: str = "NED"  # NED, ENU, camera
    origin_latitude: float | None = None
    origin_longitude: float | None = None
    origin_altitude_m: float | None = None

    # Sequence statistics
    frame_count: int = 0
    keyframe_count: int = 0
    duration_s: float = 0.0
    avg_fps: float = 0.0

    # Frame entries
    frames: list[ManifestEntry] = field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "manifest_version": "1.0",
            "dataset_id": self.dataset_id,
            "sequence_id": self.sequence_id,
            "version": self.version,
            "created_at": self.created_at,
            "capture_start": self.capture_start,
            "capture_end": self.capture_end,
            "paths": {
                "base": self.base_path,
                "frames": self.frames_dir,
                "keyframes": self.keyframes_dir,
                "poses": self.poses_dir,
            },
            "sensor": {
                "type": self.sensor_type,
                "calibration": self.calibration.to_dict(),
            },
            "coordinate_frame": {
                "type": self.coordinate_frame,
                "origin_latitude": self.origin_latitude,
                "origin_longitude": self.origin_longitude,
                "origin_altitude_m": self.origin_altitude_m,
            },
            "statistics": {
                "frame_count": self.frame_count,
                "keyframe_count": self.keyframe_count,
                "duration_s": self.duration_s,
                "avg_fps": self.avg_fps,
            },
            "frames": [f.to_dict() for f in self.frames],
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            path: Output path for manifest.json
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(
            "manifest_saved",
            path=str(path),
            frame_count=self.frame_count,
            keyframe_count=self.keyframe_count,
        )

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        """Load manifest from JSON file.

        Args:
            path: Path to manifest.json

        Returns:
            DatasetManifest instance
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        # Parse calibration
        sensor_data = data.get("sensor", {})
        cal_data = sensor_data.get("calibration", {})
        intrinsics = cal_data.get("intrinsics", {})
        depth_data = cal_data.get("depth", {})

        calibration = SensorCalibration(
            fx=intrinsics.get("fx", 0.0),
            fy=intrinsics.get("fy", 0.0),
            cx=intrinsics.get("cx", 0.0),
            cy=intrinsics.get("cy", 0.0),
            width=intrinsics.get("width", 1280),
            height=intrinsics.get("height", 720),
            distortion=cal_data.get("distortion", [0.0] * 5),
            T_body_camera=cal_data.get(
                "T_body_camera", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            ),
            T_body_imu=cal_data.get("T_body_imu"),
            depth_scale=depth_data.get("scale", 1.0),
            depth_min_m=depth_data.get("min_m", 0.1),
            depth_max_m=depth_data.get("max_m", 100.0),
        )

        # Parse frames
        frames = []
        for f_data in data.get("frames", []):
            quality = f_data.get("quality", {})
            frames.append(
                ManifestEntry(
                    frame_index=f_data.get("frame_index", 0),
                    timestamp_s=f_data.get("timestamp_s", 0.0),
                    image_path=f_data.get("image_path", ""),
                    depth_path=f_data.get("depth_path"),
                    pose_path=f_data.get("pose_path"),
                    position=f_data.get("position"),
                    orientation=f_data.get("orientation"),
                    is_keyframe=f_data.get("is_keyframe", False),
                    keyframe_index=f_data.get("keyframe_index"),
                    blur_score=quality.get("blur_score", 0.0),
                    feature_count=quality.get("feature_count", 0),
                )
            )

        # Parse coordinate frame
        coord_data = data.get("coordinate_frame", {})
        paths_data = data.get("paths", {})
        stats_data = data.get("statistics", {})

        manifest = cls(
            dataset_id=data.get("dataset_id", "unknown"),
            sequence_id=data.get("sequence_id", "unknown"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now().isoformat()),
            capture_start=data.get("capture_start"),
            capture_end=data.get("capture_end"),
            base_path=paths_data.get("base", "."),
            frames_dir=paths_data.get("frames", "frames"),
            keyframes_dir=paths_data.get("keyframes", "keyframes"),
            poses_dir=paths_data.get("poses", "poses"),
            sensor_type=sensor_data.get("type", "airsim"),
            calibration=calibration,
            coordinate_frame=coord_data.get("type", "NED"),
            origin_latitude=coord_data.get("origin_latitude"),
            origin_longitude=coord_data.get("origin_longitude"),
            origin_altitude_m=coord_data.get("origin_altitude_m"),
            frame_count=stats_data.get("frame_count", len(frames)),
            keyframe_count=stats_data.get("keyframe_count", 0),
            duration_s=stats_data.get("duration_s", 0.0),
            avg_fps=stats_data.get("avg_fps", 0.0),
            frames=frames,
            metadata=data.get("metadata", {}),
        )

        logger.info(
            "manifest_loaded",
            path=str(path),
            frame_count=manifest.frame_count,
            keyframe_count=manifest.keyframe_count,
        )

        return manifest

    def add_frame(self, entry: ManifestEntry) -> None:
        """Add a frame entry to the manifest."""
        self.frames.append(entry)
        self.frame_count = len(self.frames)

        if entry.is_keyframe:
            self.keyframe_count = sum(1 for f in self.frames if f.is_keyframe)

    def get_keyframes(self) -> list[ManifestEntry]:
        """Get all keyframe entries."""
        return [f for f in self.frames if f.is_keyframe]

    def compute_statistics(self) -> None:
        """Recompute statistics from frame data."""
        self.frame_count = len(self.frames)
        self.keyframe_count = sum(1 for f in self.frames if f.is_keyframe)

        if self.frames:
            start_time = self.frames[0].timestamp_s
            end_time = self.frames[-1].timestamp_s
            self.duration_s = end_time - start_time

            if self.duration_s > 0:
                self.avg_fps = self.frame_count / self.duration_s
            else:
                self.avg_fps = 0.0


class ManifestBuilder:
    """Builder for creating manifests during capture.

    Usage:
        builder = ManifestBuilder(
            dataset_id="drone_survey_001",
            sequence_id="flight_01",
            output_dir=Path("data/captures/survey_001"),
        )

        for frame in capture_stream:
            entry = builder.add_frame(
                timestamp_s=frame.timestamp,
                image_path="frames/000001.png",
                position=[frame.x, frame.y, frame.z],
                orientation=[frame.roll, frame.pitch, frame.yaw],
                is_keyframe=keyframe_selector.check_keyframe(frame).is_keyframe,
            )

        builder.finalize()
        builder.save()
    """

    def __init__(
        self,
        dataset_id: str,
        sequence_id: str,
        output_dir: Path,
        sensor_type: str = "airsim",
        calibration: SensorCalibration | None = None,
    ) -> None:
        """Initialize the manifest builder.

        Args:
            dataset_id: Unique dataset identifier.
            sequence_id: Unique sequence identifier.
            output_dir: Directory where manifest will be saved.
            sensor_type: Type of sensor used.
            calibration: Sensor calibration data.
        """
        self.output_dir = Path(output_dir)
        self._keyframe_index = 0

        self.manifest = DatasetManifest(
            dataset_id=dataset_id,
            sequence_id=sequence_id,
            sensor_type=sensor_type,
            calibration=calibration or SensorCalibration(),
            base_path=".",
        )

    def set_origin(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
    ) -> None:
        """Set the geographic origin for the coordinate frame."""
        self.manifest.origin_latitude = latitude
        self.manifest.origin_longitude = longitude
        self.manifest.origin_altitude_m = altitude_m

    def set_calibration(self, calibration: SensorCalibration) -> None:
        """Update sensor calibration."""
        self.manifest.calibration = calibration

    def add_frame(
        self,
        timestamp_s: float,
        image_path: str,
        depth_path: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        is_keyframe: bool = False,
        blur_score: float = 0.0,
        feature_count: int = 0,
    ) -> ManifestEntry:
        """Add a frame to the manifest.

        Args:
            timestamp_s: Frame timestamp in seconds.
            image_path: Relative path to image file.
            depth_path: Relative path to depth file (optional).
            position: [x, y, z] position in NED frame.
            orientation: [roll, pitch, yaw] in degrees.
            is_keyframe: Whether this frame is a keyframe.
            blur_score: Motion blur score (0-1).
            feature_count: Number of detected features.

        Returns:
            The created ManifestEntry.
        """
        keyframe_index = None
        if is_keyframe:
            keyframe_index = self._keyframe_index
            self._keyframe_index += 1

        entry = ManifestEntry(
            frame_index=len(self.manifest.frames),
            timestamp_s=timestamp_s,
            image_path=image_path,
            depth_path=depth_path,
            position=position,
            orientation=orientation,
            is_keyframe=is_keyframe,
            keyframe_index=keyframe_index,
            blur_score=blur_score,
            feature_count=feature_count,
        )

        self.manifest.add_frame(entry)

        # Update capture time range
        if self.manifest.capture_start is None:
            self.manifest.capture_start = datetime.now().isoformat()
        self.manifest.capture_end = datetime.now().isoformat()

        return entry

    def finalize(self) -> None:
        """Finalize the manifest by computing final statistics."""
        self.manifest.compute_statistics()

    def save(self, filename: str = "manifest.json") -> Path:
        """Save the manifest to disk.

        Args:
            filename: Manifest filename.

        Returns:
            Path to saved manifest.
        """
        self.finalize()
        output_path = self.output_dir / filename
        self.manifest.save(output_path)
        return output_path

    def get_manifest(self) -> DatasetManifest:
        """Get the current manifest."""
        return self.manifest


def validate_manifest(manifest: DatasetManifest, base_dir: Path | None = None) -> list[str]:
    """Validate manifest integrity checks."""
    issues: list[str] = []
    if manifest.calibration.fx <= 0 or manifest.calibration.fy <= 0:
        issues.append("invalid_intrinsics")
    if manifest.calibration.width <= 0 or manifest.calibration.height <= 0:
        issues.append("invalid_resolution")

    timestamps = [frame.timestamp_s for frame in manifest.frames]
    if any(t <= 0 for t in timestamps):
        issues.append("non_positive_timestamp")
    if timestamps != sorted(timestamps):
        issues.append("non_monotonic_timestamps")

    if base_dir:
        base_dir = Path(base_dir)
        for frame in manifest.frames:
            if frame.image_path:
                img_path = base_dir / frame.image_path
                if not img_path.exists():
                    issues.append(f"missing_image:{frame.image_path}")
            if frame.depth_path:
                depth_path = base_dir / frame.depth_path
                if not depth_path.exists():
                    issues.append(f"missing_depth:{frame.depth_path}")

    return issues
