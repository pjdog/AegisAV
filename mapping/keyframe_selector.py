"""Keyframe selection for SLAM and Gaussian splatting pipelines.

Phase 1 Worker B: Implement keyframe selection rules.

Keyframe selection determines which frames from a capture sequence should be
used for SLAM tracking and Gaussian splatting training. Good keyframe selection
balances coverage (enough frames for reconstruction) with efficiency (not too
many redundant frames).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KeyframeSelectionConfig:
    """Configuration for keyframe selection rules.

    Attributes:
        min_time_interval_s: Minimum time between keyframes (seconds).
        max_time_interval_s: Maximum time without a keyframe (seconds).
        min_translation_m: Minimum translation to trigger keyframe (meters).
        min_rotation_deg: Minimum rotation to trigger keyframe (degrees).
        velocity_threshold_ms: Below this velocity, use time-based selection.
        blur_threshold: Maximum motion blur score (0-1) for a valid keyframe.
        min_feature_count: Minimum feature count for valid keyframe.
        overlap_target: Target overlap between consecutive keyframes (0-1).
    """

    min_time_interval_s: float = 0.1
    max_time_interval_s: float = 2.0
    min_translation_m: float = 0.5
    min_rotation_deg: float = 5.0
    velocity_threshold_ms: float = 1.0
    blur_threshold: float = 0.7
    min_feature_count: int = 50
    overlap_target: float = 0.6


@dataclass
class FramePose:
    """Pose information for a captured frame.

    Uses NED (North-East-Down) coordinate frame.
    """

    timestamp_s: float
    x: float  # North (meters)
    y: float  # East (meters)
    z: float  # Down (meters)
    roll_deg: float  # Roll angle (degrees)
    pitch_deg: float  # Pitch angle (degrees)
    yaw_deg: float  # Yaw angle (degrees)

    # Optional velocity info
    vx: float = 0.0  # North velocity (m/s)
    vy: float = 0.0  # East velocity (m/s)
    vz: float = 0.0  # Down velocity (m/s)

    # Optional quality metrics
    feature_count: int = 0
    blur_score: float = 0.0

    def speed(self) -> float:
        """Calculate total speed in m/s."""
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def translation_from(self, other: FramePose) -> float:
        """Calculate 3D translation distance from another pose."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def rotation_from(self, other: FramePose) -> float:
        """Calculate maximum rotation difference from another pose (degrees)."""
        # Normalize yaw difference to [-180, 180]
        dyaw = self.yaw_deg - other.yaw_deg
        while dyaw > 180:
            dyaw -= 360
        while dyaw < -180:
            dyaw += 360

        droll = abs(self.roll_deg - other.roll_deg)
        dpitch = abs(self.pitch_deg - other.pitch_deg)

        return max(abs(dyaw), droll, dpitch)


@dataclass
class KeyframeResult:
    """Result of keyframe selection decision."""

    is_keyframe: bool
    reason: str
    frame_index: int
    timestamp_s: float
    translation_since_last: float = 0.0
    rotation_since_last: float = 0.0
    time_since_last: float = 0.0


class KeyframeSelector:
    """Selects keyframes from a stream of captured frames.

    Implements multiple selection criteria:
    1. Time-based: Ensures keyframes at regular intervals
    2. Motion-based: Triggers on significant translation or rotation
    3. Quality-based: Filters out blurry or feature-poor frames

    Usage:
        selector = KeyframeSelector(config)

        for frame_idx, (image, pose) in enumerate(capture_stream):
            result = selector.check_keyframe(frame_idx, pose)
            if result.is_keyframe:
                save_keyframe(frame_idx, image, pose)
    """

    def __init__(self, config: KeyframeSelectionConfig | None = None) -> None:
        """Initialize the keyframe selector.

        Args:
            config: Selection configuration. Uses defaults if None.
        """
        self.config = config or KeyframeSelectionConfig()
        self._last_keyframe_pose: FramePose | None = None
        self._last_keyframe_time: float = 0.0
        self._last_keyframe_index: int = -1
        self._keyframe_count: int = 0
        self._frame_count: int = 0

    def reset(self) -> None:
        """Reset selector state for a new sequence."""
        self._last_keyframe_pose = None
        self._last_keyframe_time = 0.0
        self._last_keyframe_index = -1
        self._keyframe_count = 0
        self._frame_count = 0

    def check_keyframe(self, frame_index: int, pose: FramePose) -> KeyframeResult:
        """Check if a frame should be selected as a keyframe.

        Args:
            frame_index: Index of the frame in the sequence.
            pose: Pose information for the frame.

        Returns:
            KeyframeResult indicating if frame is a keyframe and why.
        """
        self._frame_count += 1

        # First frame is always a keyframe
        if self._last_keyframe_pose is None:
            return self._accept_keyframe(frame_index, pose, "first_frame")

        # Calculate metrics
        time_since_last = pose.timestamp_s - self._last_keyframe_time
        translation = pose.translation_from(self._last_keyframe_pose)
        rotation = pose.rotation_from(self._last_keyframe_pose)
        speed = pose.speed()

        # Quality checks (reject if too blurry or few features)
        if pose.blur_score > self.config.blur_threshold:
            return self._reject_keyframe(
                frame_index, pose, "blur_too_high",
                translation, rotation, time_since_last
            )

        if pose.feature_count > 0 and pose.feature_count < self.config.min_feature_count:
            return self._reject_keyframe(
                frame_index, pose, "insufficient_features",
                translation, rotation, time_since_last
            )

        # Minimum time interval check
        if time_since_last < self.config.min_time_interval_s:
            return self._reject_keyframe(
                frame_index, pose, "too_soon",
                translation, rotation, time_since_last
            )

        # Maximum time interval - force keyframe
        if time_since_last >= self.config.max_time_interval_s:
            return self._accept_keyframe(frame_index, pose, "max_time_exceeded")

        # Velocity-based selection: if moving slowly, use time intervals
        if speed < self.config.velocity_threshold_ms:
            # Low velocity - use longer time intervals
            if time_since_last >= self.config.max_time_interval_s * 0.5:
                return self._accept_keyframe(frame_index, pose, "time_interval_slow")
            return self._reject_keyframe(
                frame_index, pose, "waiting_time_slow",
                translation, rotation, time_since_last
            )

        # Motion-based selection
        if translation >= self.config.min_translation_m:
            return self._accept_keyframe(frame_index, pose, "translation_threshold")

        if rotation >= self.config.min_rotation_deg:
            return self._accept_keyframe(frame_index, pose, "rotation_threshold")

        # Not enough motion yet
        return self._reject_keyframe(
            frame_index, pose, "insufficient_motion",
            translation, rotation, time_since_last
        )

    def _accept_keyframe(
        self,
        frame_index: int,
        pose: FramePose,
        reason: str,
    ) -> KeyframeResult:
        """Accept a frame as a keyframe."""
        translation = 0.0
        rotation = 0.0
        time_since = 0.0

        if self._last_keyframe_pose is not None:
            translation = pose.translation_from(self._last_keyframe_pose)
            rotation = pose.rotation_from(self._last_keyframe_pose)
            time_since = pose.timestamp_s - self._last_keyframe_time

        self._last_keyframe_pose = pose
        self._last_keyframe_time = pose.timestamp_s
        self._last_keyframe_index = frame_index
        self._keyframe_count += 1

        logger.debug(
            "keyframe_selected",
            frame_index=frame_index,
            reason=reason,
            translation=translation,
            rotation=rotation,
            keyframe_count=self._keyframe_count,
        )

        return KeyframeResult(
            is_keyframe=True,
            reason=reason,
            frame_index=frame_index,
            timestamp_s=pose.timestamp_s,
            translation_since_last=translation,
            rotation_since_last=rotation,
            time_since_last=time_since,
        )

    def _reject_keyframe(
        self,
        frame_index: int,
        pose: FramePose,
        reason: str,
        translation: float,
        rotation: float,
        time_since: float,
    ) -> KeyframeResult:
        """Reject a frame as a keyframe."""
        return KeyframeResult(
            is_keyframe=False,
            reason=reason,
            frame_index=frame_index,
            timestamp_s=pose.timestamp_s,
            translation_since_last=translation,
            rotation_since_last=rotation,
            time_since_last=time_since,
        )

    @property
    def keyframe_count(self) -> int:
        """Number of keyframes selected so far."""
        return self._keyframe_count

    @property
    def frame_count(self) -> int:
        """Total number of frames processed."""
        return self._frame_count

    @property
    def keyframe_ratio(self) -> float:
        """Ratio of keyframes to total frames."""
        if self._frame_count == 0:
            return 0.0
        return self._keyframe_count / self._frame_count

    def get_stats(self) -> dict[str, Any]:
        """Get selection statistics."""
        return {
            "frame_count": self._frame_count,
            "keyframe_count": self._keyframe_count,
            "keyframe_ratio": self.keyframe_ratio,
            "last_keyframe_index": self._last_keyframe_index,
            "config": {
                "min_time_interval_s": self.config.min_time_interval_s,
                "max_time_interval_s": self.config.max_time_interval_s,
                "min_translation_m": self.config.min_translation_m,
                "min_rotation_deg": self.config.min_rotation_deg,
                "velocity_threshold_ms": self.config.velocity_threshold_ms,
            },
        }
