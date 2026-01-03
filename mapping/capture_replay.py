"""Capture replay tool for SLAM and splatting pipelines.

Phase 1 Worker B: Replay tool to feed recorded captures to SLAM/splat pipelines.

This module provides tools to:
1. Load recorded capture bundles from disk
2. Stream frames at real-time or accelerated rates
3. Feed data to SLAM/splat processing pipelines
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from mapping.keyframe_selector import FramePose
from mapping.manifest import DatasetManifest, validate_manifest

logger = structlog.get_logger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for capture replay.

    Attributes:
        time_scale: Playback speed multiplier (1.0 = realtime, 2.0 = 2x speed).
        loop: Whether to loop the sequence.
        start_frame: Frame index to start from.
        end_frame: Frame index to end at (None = end of sequence).
        skip_frames: Number of frames to skip between outputs.
        include_images: Whether to load image data.
    """

    time_scale: float = 1.0
    loop: bool = False
    start_frame: int = 0
    end_frame: int | None = None
    skip_frames: int = 0
    include_images: bool = True


@dataclass
class CaptureFrame:
    """A single captured frame with pose and optional image data."""

    frame_index: int
    timestamp_s: float
    pose: FramePose
    frame_id: str = ""
    image_path: Path | None = None
    depth_path: Path | None = None
    camera_pose: dict[str, Any] | None = None
    image_data: bytes | None = None
    depth_data: bytes | None = None

    # Camera intrinsics
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0

    # Distortion coefficients [k1, k2, p1, p2, k3]
    distortion: list[float] | None = None

    # Depth calibration
    depth_scale: float = 1.0  # Scale factor to convert raw depth to meters
    depth_min_m: float = 0.1  # Minimum valid depth in meters
    depth_max_m: float = 100.0  # Maximum valid depth in meters

    # IMU data if available
    imu_data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_index": self.frame_index,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "pose": {
                "x": self.pose.x,
                "y": self.pose.y,
                "z": self.pose.z,
                "roll_deg": self.pose.roll_deg,
                "pitch_deg": self.pose.pitch_deg,
                "yaw_deg": self.pose.yaw_deg,
                "vx": self.pose.vx,
                "vy": self.pose.vy,
                "vz": self.pose.vz,
            },
            "image_path": str(self.image_path) if self.image_path else None,
            "depth_path": str(self.depth_path) if self.depth_path else None,
            "camera_pose": self.camera_pose,
            "intrinsics": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "width": self.width,
                "height": self.height,
            },
            "imu_data": self.imu_data,
        }


@dataclass
class CaptureSequence:
    """A sequence of captured frames."""

    sequence_id: str
    base_path: Path
    frames: list[CaptureFrame] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Total duration of the sequence in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp_s - self.frames[0].timestamp_s

    @property
    def frame_count(self) -> int:
        """Number of frames in the sequence."""
        return len(self.frames)


class CaptureReplay:
    """Replays captured frame sequences for SLAM/splat processing.

    Supports both synchronous and asynchronous iteration, with
    configurable playback speed.

    Usage:
        # Load and replay at 2x speed
        replay = CaptureReplay.from_directory(capture_path)
        replay.config.time_scale = 2.0

        async for frame in replay.stream_async():
            process_frame(frame)
    """

    def __init__(
        self,
        sequence: CaptureSequence,
        config: ReplayConfig | None = None,
    ) -> None:
        """Initialize the replay tool.

        Args:
            sequence: The capture sequence to replay.
            config: Replay configuration.
        """
        self.sequence = sequence
        self.config = config or ReplayConfig()
        self._current_index = 0
        self._is_playing = False
        self._on_frame_callback: Callable[[CaptureFrame], None] | None = None

    @classmethod
    def from_directory(
        cls,
        path: Path,
        config: ReplayConfig | None = None,
    ) -> CaptureReplay:
        """Load a capture sequence from a directory.

        Expected directory structure:
            sequence_YYYYMMDD/
                manifest.json
                frames/
                    000000.png
                    000000.json  (pose, IMU, intrinsics)
                    000001.png
                    000001.json
                    ...

        Args:
            path: Path to the sequence directory.
            config: Replay configuration.

        Returns:
            CaptureReplay instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Capture directory not found: {path}")

        # Load manifest if it exists
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = DatasetManifest.load(manifest_path)
            issues = validate_manifest(manifest, path)
            if issues:
                logger.warning("manifest_validation_issues", issues=issues)
            frames = cls._frames_from_manifest(path, manifest)
            if frames:
                sequence = CaptureSequence(
                    sequence_id=manifest.sequence_id,
                    base_path=path,
                    frames=frames,
                    metadata=manifest.to_dict(),
                )
                logger.info(
                    "sequence_loaded",
                    sequence_id=sequence.sequence_id,
                    frame_count=len(frames),
                    duration_s=sequence.duration_s,
                )
                return cls(sequence, config)

        metadata: dict[str, Any] = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                metadata = json.load(f)

        sequence_id = metadata.get("sequence_id", path.name)

        # Find frames directory
        frames_dir = path / "frames"
        if not frames_dir.exists():
            frames_dir = path  # Frames might be directly in the directory

        # Load frames
        frames: list[CaptureFrame] = []
        json_files = sorted(frames_dir.glob("*.json"))

        for json_path in json_files:
            if json_path.name == "manifest.json":
                continue

            try:
                frame = cls._load_frame(json_path, frames_dir)
                if frame:
                    frames.append(frame)
            except Exception as e:
                logger.warning("frame_load_error", path=str(json_path), error=str(e))

        # Sort by timestamp
        frames.sort(key=lambda f: f.timestamp_s)

        # Assign frame indices
        for i, frame in enumerate(frames):
            frame.frame_index = i

        sequence = CaptureSequence(
            sequence_id=sequence_id,
            base_path=path,
            frames=frames,
            metadata=metadata,
        )

        logger.info(
            "sequence_loaded",
            sequence_id=sequence_id,
            frame_count=len(frames),
            duration_s=sequence.duration_s,
        )

        return cls(sequence, config)

    @staticmethod
    def _frames_from_manifest(path: Path, manifest: DatasetManifest) -> list[CaptureFrame]:
        calibration = manifest.calibration
        frames: list[CaptureFrame] = []
        base_path = Path(path)

        for entry in manifest.frames:
            image_path = base_path / entry.image_path if entry.image_path else None
            depth_path = base_path / entry.depth_path if entry.depth_path else None

            position = entry.position
            orientation = entry.orientation

            if (position is None or orientation is None) and entry.pose_path:
                pose_path = base_path / entry.pose_path
                if pose_path.exists():
                    try:
                        with open(pose_path) as f:
                            pose_payload = json.load(f)
                        pose_data = pose_payload.get("pose", pose_payload)
                        pos_data = pose_data.get("position") or pose_payload.get("position")
                        ori_data = pose_data.get("orientation") or pose_payload.get("orientation")
                        if pos_data:
                            if isinstance(pos_data, list):
                                position = [float(v) for v in pos_data[:3]]
                            elif isinstance(pos_data, dict):
                                position = [
                                    float(pos_data.get("x", 0.0)),
                                    float(pos_data.get("y", 0.0)),
                                    float(pos_data.get("z", 0.0)),
                                ]
                        if ori_data:
                            if isinstance(ori_data, list):
                                if len(ori_data) == 4:
                                    orientation = list(
                                        CaptureReplay._quat_to_euler_deg(
                                            float(ori_data[0]),
                                            float(ori_data[1]),
                                            float(ori_data[2]),
                                            float(ori_data[3]),
                                        )
                                    )
                                else:
                                    orientation = [float(v) for v in ori_data[:3]]
                            elif isinstance(ori_data, dict):
                                orientation = list(
                                    CaptureReplay._quat_to_euler_deg(
                                        float(ori_data.get("w", 1.0)),
                                        float(ori_data.get("x", 0.0)),
                                        float(ori_data.get("y", 0.0)),
                                        float(ori_data.get("z", 0.0)),
                                    )
                                )
                    except Exception as exc:
                        logger.warning(
                            "manifest_pose_load_failed", path=str(pose_path), error=str(exc)
                        )

            if position is None:
                position = [0.0, 0.0, 0.0]
            if orientation is None:
                orientation = [0.0, 0.0, 0.0]

            pose = FramePose(
                timestamp_s=entry.timestamp_s,
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                roll_deg=float(orientation[0]),
                pitch_deg=float(orientation[1]),
                yaw_deg=float(orientation[2]),
                feature_count=entry.feature_count,
                blur_score=entry.blur_score,
            )

            frame_id = (
                Path(entry.image_path).stem if entry.image_path else f"frame_{entry.frame_index}"
            )
            frames.append(
                CaptureFrame(
                    frame_index=entry.frame_index,
                    frame_id=frame_id,
                    timestamp_s=entry.timestamp_s,
                    pose=pose,
                    image_path=image_path,
                    depth_path=depth_path,
                    fx=calibration.fx,
                    fy=calibration.fy,
                    cx=calibration.cx,
                    cy=calibration.cy,
                    width=calibration.width,
                    height=calibration.height,
                    distortion=calibration.distortion,
                    depth_scale=calibration.depth_scale,
                    depth_min_m=calibration.depth_min_m,
                    depth_max_m=calibration.depth_max_m,
                )
            )

        frames.sort(key=lambda f: f.timestamp_s)
        for i, frame in enumerate(frames):
            frame.frame_index = i
        return frames

    @staticmethod
    def _quat_to_euler_deg(
        qw: float, qx: float, qy: float, qz: float
    ) -> tuple[float, float, float]:
        """Convert quaternion (w, x, y, z) to Euler angles in degrees."""
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    @staticmethod
    def _load_frame(json_path: Path, frames_dir: Path) -> CaptureFrame | None:
        """Load a single frame from its JSON metadata file."""
        with open(json_path) as f:
            data = json.load(f)

        # Extract timestamp from filename or data
        timestamp = data.get("timestamp_s") or 0.0
        if not timestamp:
            timestamp_ns = data.get("timestamp_ns")
            if timestamp_ns:
                timestamp = float(timestamp_ns) / 1e9
        if not timestamp:
            timestamp_str = data.get("timestamp")
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    timestamp = dt.timestamp()
                except ValueError:
                    timestamp = 0.0
        if not timestamp:
            # Try to parse from filename (e.g., "000000.json" or "1234567890.json")
            stem = json_path.stem
            try:
                timestamp = float(stem) / 1000.0  # Assume milliseconds
            except ValueError:
                timestamp = 0.0

        # Load pose
        pose_data = data.get("pose", {})
        telemetry = data.get("telemetry") or {}
        telemetry_pose = telemetry.get("pose") if isinstance(telemetry, dict) else None
        if telemetry_pose and not pose_data:
            position = telemetry_pose.get("position", {})
            orientation = telemetry_pose.get("orientation", {})
            roll_deg, pitch_deg, yaw_deg = CaptureReplay._quat_to_euler_deg(
                orientation.get("w", 1.0),
                orientation.get("x", 0.0),
                orientation.get("y", 0.0),
                orientation.get("z", 0.0),
            )
            velocity = telemetry_pose.get("linear_velocity", {})
            pose_data = {
                "x": position.get("x", 0.0),
                "y": position.get("y", 0.0),
                "z": position.get("z", 0.0),
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
                "vx": velocity.get("x", 0.0),
                "vy": velocity.get("y", 0.0),
                "vz": velocity.get("z", 0.0),
            }

        pose = FramePose(
            timestamp_s=timestamp,
            x=pose_data.get("x", 0.0),
            y=pose_data.get("y", 0.0),
            z=pose_data.get("z", 0.0),
            roll_deg=pose_data.get("roll_deg", 0.0),
            pitch_deg=pose_data.get("pitch_deg", 0.0),
            yaw_deg=pose_data.get("yaw_deg", 0.0),
            vx=pose_data.get("vx", 0.0),
            vy=pose_data.get("vy", 0.0),
            vz=pose_data.get("vz", 0.0),
            feature_count=data.get("feature_count", 0),
            blur_score=data.get("blur_score", 0.0),
        )

        # Find corresponding image
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = frames_dir / f"{json_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        # Find depth image if available
        depth_path = None
        for suffix in ["_depth.png", "_depth.exr", "_depth.npy"]:
            candidate = frames_dir / f"{json_path.stem}{suffix}"
            if candidate.exists():
                depth_path = candidate
                break

        # Camera intrinsics
        camera = data.get("camera", {}) if isinstance(data.get("camera"), dict) else {}
        intrinsics = data.get("intrinsics", {}) or camera.get("intrinsics", {})
        resolution = camera.get("resolution", [])

        camera_pose = camera.get("pose")

        return CaptureFrame(
            frame_index=0,  # Will be set after sorting
            frame_id=data.get("frame_id", json_path.stem),
            timestamp_s=timestamp,
            pose=pose,
            image_path=image_path,
            depth_path=depth_path,
            camera_pose=camera_pose,
            fx=intrinsics.get("fx", 0.0),
            fy=intrinsics.get("fy", 0.0),
            cx=intrinsics.get("cx", 0.0),
            cy=intrinsics.get("cy", 0.0),
            width=intrinsics.get("width", 0) or (resolution[0] if len(resolution) > 1 else 0),
            height=intrinsics.get("height", 0) or (resolution[1] if len(resolution) > 1 else 0),
            imu_data=data.get("imu") or telemetry.get("imu"),
        )

    def set_on_frame(self, callback: Callable[[CaptureFrame], None]) -> None:
        """Set callback for each frame during playback."""
        self._on_frame_callback = callback

    def _get_frame_range(self) -> tuple[int, int]:
        """Get the frame range to replay."""
        start = max(0, self.config.start_frame)
        end = self.config.end_frame
        if end is None or end > len(self.sequence.frames):
            end = len(self.sequence.frames)
        return start, end

    def _load_frame_data(self, frame: CaptureFrame) -> CaptureFrame:
        """Load image data for a frame if configured."""
        if not self.config.include_images:
            return frame

        if frame.image_path and frame.image_path.exists():
            frame.image_data = frame.image_path.read_bytes()

        if frame.depth_path and frame.depth_path.exists():
            frame.depth_data = frame.depth_path.read_bytes()

        return frame

    def iterate(self) -> Iterator[CaptureFrame]:
        """Synchronous iterator over frames (no timing delays)."""
        start, end = self._get_frame_range()
        skip = self.config.skip_frames + 1

        while True:
            for i in range(start, end, skip):
                frame = self._load_frame_data(self.sequence.frames[i])

                if self._on_frame_callback:
                    self._on_frame_callback(frame)

                yield frame

            if not self.config.loop:
                break

    async def stream_async(self) -> AsyncIterator[CaptureFrame]:
        """Async iterator that respects timing based on time_scale."""
        start, end = self._get_frame_range()
        skip = self.config.skip_frames + 1
        self._is_playing = True

        try:
            while self._is_playing:
                playback_start = time.perf_counter()
                sequence_start_time = self.sequence.frames[start].timestamp_s

                for i in range(start, end, skip):
                    if not self._is_playing:
                        break

                    frame = self._load_frame_data(self.sequence.frames[i])

                    # Calculate when this frame should be played
                    frame_offset = frame.timestamp_s - sequence_start_time
                    target_time = playback_start + (frame_offset / self.config.time_scale)

                    # Wait until it's time to emit this frame
                    now = time.perf_counter()
                    if target_time > now:
                        await asyncio.sleep(target_time - now)

                    if self._on_frame_callback:
                        self._on_frame_callback(frame)

                    yield frame

                if not self.config.loop:
                    break

        finally:
            self._is_playing = False

    def stop(self) -> None:
        """Stop async playback."""
        self._is_playing = False

    def get_stats(self) -> dict[str, Any]:
        """Get replay statistics."""
        return {
            "sequence_id": self.sequence.sequence_id,
            "frame_count": self.sequence.frame_count,
            "duration_s": self.sequence.duration_s,
            "config": {
                "time_scale": self.config.time_scale,
                "loop": self.config.loop,
                "start_frame": self.config.start_frame,
                "end_frame": self.config.end_frame,
                "skip_frames": self.config.skip_frames,
                "include_images": self.config.include_images,
            },
            "is_playing": self._is_playing,
        }


@dataclass
class CaptureIntegrityResult:
    """Results from capture integrity validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    frame_count: int
    frames_with_timestamps: int
    frames_with_intrinsics: int
    frames_with_depth: int
    frames_with_images: int
    timestamp_gaps: list[tuple[int, float]]  # (frame_index, gap_seconds)
    intrinsics_consistent: bool
    avg_frame_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "frame_count": self.frame_count,
            "frames_with_timestamps": self.frames_with_timestamps,
            "frames_with_intrinsics": self.frames_with_intrinsics,
            "frames_with_depth": self.frames_with_depth,
            "frames_with_images": self.frames_with_images,
            "timestamp_gaps": [
                {"frame_index": idx, "gap_s": gap} for idx, gap in self.timestamp_gaps
            ],
            "intrinsics_consistent": self.intrinsics_consistent,
            "avg_frame_rate": self.avg_frame_rate,
        }


class CaptureIntegrityChecker:
    """Validates capture data integrity for SLAM pipeline compatibility.

    Checks that captures have:
    - Monotonically increasing timestamps
    - No excessive timestamp gaps
    - Valid camera intrinsics
    - Available depth data (if required)
    - Consistent resolution
    """

    def __init__(
        self,
        max_timestamp_gap_s: float = 1.0,
        require_depth: bool = False,
        require_intrinsics: bool = True,
        min_frames: int = 10,
    ) -> None:
        """Initialize the integrity checker.

        Args:
            max_timestamp_gap_s: Maximum allowed gap between consecutive frames.
            require_depth: Whether depth data is required for all frames.
            require_intrinsics: Whether camera intrinsics are required.
            min_frames: Minimum number of valid frames required.
        """
        self.max_timestamp_gap_s = max_timestamp_gap_s
        self.require_depth = require_depth
        self.require_intrinsics = require_intrinsics
        self.min_frames = min_frames

    def check(self, sequence: CaptureSequence) -> CaptureIntegrityResult:
        """Validate capture sequence integrity.

        Args:
            sequence: The capture sequence to validate.

        Returns:
            CaptureIntegrityResult with validation details.
        """
        errors: list[str] = []
        warnings: list[str] = []
        timestamp_gaps: list[tuple[int, float]] = []

        frames = sequence.frames
        frame_count = len(frames)

        if frame_count == 0:
            return CaptureIntegrityResult(
                valid=False,
                errors=["No frames in sequence"],
                warnings=[],
                frame_count=0,
                frames_with_timestamps=0,
                frames_with_intrinsics=0,
                frames_with_depth=0,
                frames_with_images=0,
                timestamp_gaps=[],
                intrinsics_consistent=False,
                avg_frame_rate=0.0,
            )

        # Count frames with valid data
        frames_with_timestamps = 0
        frames_with_intrinsics = 0
        frames_with_depth = 0
        frames_with_images = 0

        # Track intrinsics for consistency check
        first_intrinsics: tuple[float, float, float, float, int, int] | None = None
        intrinsics_consistent = True

        prev_timestamp: float | None = None

        for i, frame in enumerate(frames):
            # Check timestamp
            if frame.timestamp_s > 0:
                frames_with_timestamps += 1

                # Check for timestamp gaps
                if prev_timestamp is not None:
                    gap = frame.timestamp_s - prev_timestamp
                    if gap < 0:
                        errors.append(
                            f"Non-monotonic timestamp at frame {i}: "
                            f"{prev_timestamp:.3f} -> {frame.timestamp_s:.3f}"
                        )
                    elif gap > self.max_timestamp_gap_s:
                        timestamp_gaps.append((i, gap))
                        warnings.append(f"Large timestamp gap at frame {i}: {gap:.2f}s")

                prev_timestamp = frame.timestamp_s

            # Check intrinsics
            if frame.fx > 0 and frame.fy > 0 and frame.cx > 0 and frame.cy > 0:
                frames_with_intrinsics += 1
                current = (frame.fx, frame.fy, frame.cx, frame.cy, frame.width, frame.height)

                if first_intrinsics is None:
                    first_intrinsics = current
                elif current != first_intrinsics:
                    intrinsics_consistent = False

            # Check depth availability
            if frame.depth_path and frame.depth_path.exists():
                frames_with_depth += 1

            # Check image availability
            if frame.image_path and frame.image_path.exists():
                frames_with_images += 1

        # Validate requirements
        if frame_count < self.min_frames:
            errors.append(f"Insufficient frames: {frame_count} < {self.min_frames}")

        if frames_with_timestamps < frame_count * 0.9:
            errors.append(
                f"Too few frames with valid timestamps: {frames_with_timestamps}/{frame_count}"
            )

        if self.require_intrinsics and frames_with_intrinsics == 0:
            errors.append("No frames have valid camera intrinsics")
        elif self.require_intrinsics and not intrinsics_consistent:
            warnings.append("Camera intrinsics vary across frames")

        if self.require_depth and frames_with_depth < frame_count * 0.8:
            errors.append(f"Insufficient depth data: {frames_with_depth}/{frame_count}")

        if frames_with_images < frame_count * 0.9:
            errors.append(f"Too many missing images: {frames_with_images}/{frame_count}")

        # Calculate average frame rate
        avg_frame_rate = 0.0
        if sequence.duration_s > 0 and frame_count > 1:
            avg_frame_rate = (frame_count - 1) / sequence.duration_s

        valid = len(errors) == 0

        result = CaptureIntegrityResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            frame_count=frame_count,
            frames_with_timestamps=frames_with_timestamps,
            frames_with_intrinsics=frames_with_intrinsics,
            frames_with_depth=frames_with_depth,
            frames_with_images=frames_with_images,
            timestamp_gaps=timestamp_gaps,
            intrinsics_consistent=intrinsics_consistent,
            avg_frame_rate=avg_frame_rate,
        )

        logger.info(
            "capture_integrity_check",
            sequence_id=sequence.sequence_id,
            valid=valid,
            errors=len(errors),
            warnings=len(warnings),
            frame_count=frame_count,
        )

        return result

    def check_replay(self, replay: CaptureReplay) -> CaptureIntegrityResult:
        """Validate a CaptureReplay instance.

        Args:
            replay: The CaptureReplay instance to validate.

        Returns:
            CaptureIntegrityResult with validation details.
        """
        return self.check(replay.sequence)
