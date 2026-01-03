"""Visual Odometry.

Estimates camera/vehicle motion from tracked features using essential matrix decomposition.
Provides relative pose estimates for GPS-denied navigation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from pydantic import BaseModel

from vision.localization.tracking import TrackingResult

logger = logging.getLogger(__name__)


class VOState(Enum):
    """Visual odometry state."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"


class VOResult(BaseModel):
    """Result of visual odometry computation."""

    timestamp: float
    state: str  # VOState value

    # Relative motion (camera frame)
    delta_x: float = 0.0  # Forward motion (meters)
    delta_y: float = 0.0  # Right motion (meters)
    delta_z: float = 0.0  # Down motion (meters)

    # Rotation (radians)
    delta_roll: float = 0.0
    delta_pitch: float = 0.0
    delta_yaw: float = 0.0

    # Quality metrics
    num_inliers: int = 0
    confidence: float = 0.0
    computation_time_ms: float = 0.0

    # Scale information
    scale_source: str = "unknown"  # altitude, imu, fixed
    scale_factor: float = 1.0

    @property
    def is_valid(self) -> bool:
        """Check if result is valid for use."""
        return self.state == VOState.TRACKING.value and self.confidence > 0.3


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)

    # Optional distortion coefficients
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @property
    def matrix(self) -> np.ndarray:
        """Get 3x3 camera matrix."""
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        """Get distortion coefficients."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    @classmethod
    def from_fov(cls, fov_deg: float, width: int, height: int) -> CameraIntrinsics:
        """Create intrinsics from field of view.

        Args:
            fov_deg: Horizontal field of view in degrees
            width: Image width
            height: Image height

        Returns:
            CameraIntrinsics
        """
        fov_rad = np.deg2rad(fov_deg)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2
        return cls(fx=fx, fy=fy, cx=cx, cy=cy)


@dataclass
class VOConfig:
    """Configuration for visual odometry."""

    # Camera intrinsics (will be auto-detected if None)
    camera: CameraIntrinsics | None = None

    # Default camera FOV (used if intrinsics not provided)
    default_fov_deg: float = 90.0

    # Essential matrix estimation
    essential_method: int = cv2.RANSAC
    essential_threshold: float = 1.0  # pixels
    essential_confidence: float = 0.999

    # Minimum inliers for valid pose
    min_inliers: int = 15

    # Scale recovery
    use_altitude_scale: bool = True
    fixed_scale: float = 1.0  # meters per unit if no altitude

    # Keyframe selection
    min_parallax_deg: float = 1.0
    min_translation_m: float = 0.1

    # Pose filtering
    max_rotation_deg: float = 30.0  # Max rotation per frame
    max_translation_m: float = 5.0  # Max translation per frame


class VisualOdometry:
    """Estimates vehicle motion from visual features.

    Uses essential matrix decomposition to recover relative pose
    between consecutive frames. Scale is recovered from altitude
    measurements or IMU integration.

    Example:
        vo = VisualOdometry(config)
        vo.set_camera_intrinsics(intrinsics)

        # Process tracked features
        result = vo.process(tracking_result, altitude=30.0)

        if result.is_valid:
            print(f"Motion: ({result.delta_x:.2f}, {result.delta_y:.2f}, {result.delta_z:.2f})")
    """

    def __init__(self, config: VOConfig | None = None) -> None:
        """Initialize visual odometry.

        Args:
            config: VO configuration
        """
        self._config = config or VOConfig()
        self._camera = config.camera if config else None
        self._state = VOState.UNINITIALIZED

        # Accumulated pose
        self._position = np.zeros(3)  # x, y, z in camera frame
        self._rotation = np.eye(3)  # Rotation matrix

        # Previous frame data
        self._prev_keyframe_pts: np.ndarray | None = None
        self._prev_altitude: float | None = None

        # Frame counter
        self._frame_count = 0

        logger.info("VisualOdometry initialized")

    @property
    def state(self) -> VOState:
        """Get current VO state."""
        return self._state

    def set_camera_intrinsics(self, intrinsics: CameraIntrinsics) -> None:
        """Set camera intrinsic parameters.

        Args:
            intrinsics: Camera intrinsics
        """
        self._camera = intrinsics
        logger.info(f"Camera intrinsics set: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    def process(
        self,
        tracking: TrackingResult,
        altitude: float | None = None,
        imu_delta: np.ndarray | None = None,
    ) -> VOResult:
        """Process tracked features to estimate motion.

        Args:
            tracking: Tracking result with matched features
            altitude: Current altitude AGL (for scale recovery)
            imu_delta: Optional IMU delta for scale (dx, dy, dz)

        Returns:
            VOResult with motion estimate
        """
        start_time = time.perf_counter()
        self._frame_count += 1

        # Check if we have camera intrinsics
        if self._camera is None:
            # Try to infer from image size if available
            if tracking.matches and hasattr(tracking, "width"):
                self._camera = CameraIntrinsics.from_fov(
                    self._config.default_fov_deg,
                    tracking.width if hasattr(tracking, "width") else 1280,
                    tracking.height if hasattr(tracking, "height") else 720,
                )
            else:
                # Use default
                self._camera = CameraIntrinsics.from_fov(self._config.default_fov_deg, 1280, 720)

        # Check minimum matches
        if tracking.num_matches < self._config.min_inliers:
            self._state = VOState.LOST if self._state == VOState.TRACKING else self._state
            return VOResult(
                timestamp=tracking.timestamp,
                state=self._state.value,
                confidence=0.0,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Get matched points
        prev_pts, curr_pts = tracking.get_matched_points()

        # Initialize if needed
        if self._state == VOState.UNINITIALIZED:
            self._state = VOState.INITIALIZING
            self._prev_keyframe_pts = curr_pts
            self._prev_altitude = altitude
            return VOResult(
                timestamp=tracking.timestamp,
                state=self._state.value,
                confidence=0.5,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            prev_pts,
            curr_pts,
            self._camera.matrix,
            method=self._config.essential_method,
            prob=self._config.essential_confidence,
            threshold=self._config.essential_threshold,
        )

        if E is None or mask is None:
            self._state = VOState.LOST
            return VOResult(
                timestamp=tracking.timestamp,
                state=self._state.value,
                confidence=0.0,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        num_inliers = int(mask.sum())
        if num_inliers < self._config.min_inliers:
            self._state = VOState.LOST
            return VOResult(
                timestamp=tracking.timestamp,
                state=self._state.value,
                num_inliers=num_inliers,
                confidence=num_inliers / max(len(mask), 1),
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Recover pose from essential matrix
        _, R, t, pose_mask = cv2.recoverPose(E, prev_pts, curr_pts, self._camera.matrix, mask=mask)

        # Translation is unit vector, need to recover scale
        t = t.flatten()
        scale, scale_source = self._recover_scale(t, altitude, imu_delta)

        # Apply scale to translation
        translation = t * scale

        # Validate motion
        if not self._validate_motion(R, translation):
            return VOResult(
                timestamp=tracking.timestamp,
                state=VOState.LOST.value,
                num_inliers=num_inliers,
                confidence=0.2,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Extract rotation angles
        roll, pitch, yaw = self._rotation_to_euler(R)

        # Update state
        self._state = VOState.TRACKING
        self._prev_keyframe_pts = curr_pts
        self._prev_altitude = altitude

        # Update accumulated pose
        self._rotation = R @ self._rotation
        self._position += self._rotation @ translation

        # Calculate confidence
        confidence = min(num_inliers / self._config.min_inliers, 1.0) * 0.8
        if scale_source == "altitude":
            confidence += 0.2

        computation_time = (time.perf_counter() - start_time) * 1000

        return VOResult(
            timestamp=tracking.timestamp,
            state=self._state.value,
            delta_x=float(translation[0]),  # Forward
            delta_y=float(translation[1]),  # Right
            delta_z=float(translation[2]),  # Down
            delta_roll=roll,
            delta_pitch=pitch,
            delta_yaw=yaw,
            num_inliers=num_inliers,
            confidence=confidence,
            computation_time_ms=computation_time,
            scale_source=scale_source,
            scale_factor=scale,
        )

    def _recover_scale(
        self,
        translation: np.ndarray,
        altitude: float | None,
        imu_delta: np.ndarray | None,
    ) -> tuple[float, str]:
        """Recover absolute scale for translation.

        Args:
            translation: Unit translation vector
            altitude: Current altitude AGL
            imu_delta: Optional IMU delta

        Returns:
            Tuple of (scale, source)
        """
        # Priority 1: Altitude difference
        if (
            self._config.use_altitude_scale
            and altitude is not None
            and self._prev_altitude is not None
        ):
            alt_diff = abs(altitude - self._prev_altitude)
            if alt_diff > 0.1 and abs(translation[2]) > 0.01:
                scale = alt_diff / abs(translation[2])
                return scale, "altitude"

        # Priority 2: IMU integration
        if imu_delta is not None:
            imu_magnitude = np.linalg.norm(imu_delta)
            if imu_magnitude > 0.1:
                scale = imu_magnitude / np.linalg.norm(translation)
                return scale, "imu"

        # Priority 3: Fixed scale
        return self._config.fixed_scale, "fixed"

    def _validate_motion(self, R: np.ndarray, t: np.ndarray) -> bool:
        """Validate estimated motion is reasonable.

        Args:
            R: Rotation matrix
            t: Translation vector

        Returns:
            True if motion is valid
        """
        # Check translation magnitude
        trans_mag = np.linalg.norm(t)
        if trans_mag > self._config.max_translation_m:
            logger.warning(
                f"Translation {trans_mag:.2f}m exceeds max {self._config.max_translation_m}m"
            )
            return False

        # Check rotation magnitude
        roll, pitch, yaw = self._rotation_to_euler(R)
        max_rot = max(abs(roll), abs(pitch), abs(yaw))
        max_rot_deg = np.rad2deg(max_rot)
        if max_rot_deg > self._config.max_rotation_deg:
            logger.warning(
                f"Rotation {max_rot_deg:.1f}° exceeds max {self._config.max_rotation_deg}°"
            )
            return False

        return True

    def _rotation_to_euler(self, R: np.ndarray) -> tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (roll, pitch, yaw).

        Args:
            R: 3x3 rotation matrix

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return roll, pitch, yaw

    def get_accumulated_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get accumulated pose since initialization.

        Returns:
            Tuple of (position, rotation_matrix)
        """
        return self._position.copy(), self._rotation.copy()

    def reset(self) -> None:
        """Reset visual odometry state."""
        self._state = VOState.UNINITIALIZED
        self._position = np.zeros(3)
        self._rotation = np.eye(3)
        self._prev_keyframe_pts = None
        self._prev_altitude = None
        self._frame_count = 0
        logger.info("VisualOdometry reset")
