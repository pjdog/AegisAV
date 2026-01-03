"""Localization Pipeline.

Orchestrates the visual localization system by combining feature extraction,
tracking, and visual odometry into a unified pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel

from autonomy.state_estimator import StateEstimator, VisualOdometryInput
from vision.localization.features import FrameFeatures, ORBConfig, ORBFeatureExtractor
from vision.localization.tracking import FeatureTracker, TrackerConfig, TrackingResult
from vision.localization.visual_odometry import (
    CameraIntrinsics,
    VisualOdometry,
    VOConfig,
    VOResult,
    VOState,
)

logger = logging.getLogger(__name__)


class LocalizationResult(BaseModel):
    """Result from the localization pipeline."""

    timestamp: datetime
    frame_id: int

    # Feature extraction
    num_features: int = 0
    extraction_time_ms: float = 0.0

    # Tracking
    num_tracked: int = 0
    tracking_confidence: float = 0.0
    tracking_time_ms: float = 0.0

    # Visual odometry
    vo_state: str = "uninitialized"
    delta_north: float = 0.0
    delta_east: float = 0.0
    delta_down: float = 0.0
    vo_confidence: float = 0.0
    vo_time_ms: float = 0.0

    # Total pipeline metrics
    total_time_ms: float = 0.0
    is_valid: bool = False

    # Position estimate (accumulated)
    position_north: float = 0.0
    position_east: float = 0.0
    position_down: float = 0.0


@dataclass
class LocalizationConfig:
    """Configuration for the localization pipeline."""

    # Component configs
    orb_config: ORBConfig = field(default_factory=ORBConfig)
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    vo_config: VOConfig = field(default_factory=VOConfig)

    # Camera intrinsics (auto-detected if None)
    camera_intrinsics: CameraIntrinsics | None = None

    # Performance targets
    target_fps: float = 20.0
    max_latency_ms: float = 50.0

    # State estimator integration
    feed_state_estimator: bool = True

    # Coordinate transform (camera frame to NED)
    # VOResult outputs: delta_x=forward, delta_y=right, delta_z=down
    # NED frame: north, east, down
    # For camera pointing north (yaw=0): forward=north, right=east, down=down
    # This matrix transforms [forward, right, down] -> [north, east, down]
    camera_to_ned_rotation: list[list[float]] = field(
        default_factory=lambda: [
            [1, 0, 0],  # NED North = Camera Forward (delta_x)
            [0, 1, 0],  # NED East = Camera Right (delta_y)
            [0, 0, 1],  # NED Down = Camera Down (delta_z)
        ]
    )


class LocalizationPipeline:
    """Orchestrates visual localization components.

    Combines feature extraction, tracking, and visual odometry into a
    unified pipeline that produces position estimates for GPS-denied
    navigation.

    The pipeline:
    1. Extracts ORB features from input image
    2. Tracks features from previous frame
    3. Estimates motion using visual odometry
    4. Optionally feeds results to state estimator

    Example:
        pipeline = LocalizationPipeline(config)
        pipeline.set_camera_intrinsics(intrinsics)

        # Process frames
        for image in camera_stream:
            result = pipeline.process(image, altitude=30.0)
            if result.is_valid:
                print(f"Position: ({result.position_north:.2f}, {result.position_east:.2f})")
    """

    def __init__(
        self,
        config: LocalizationConfig | None = None,
        state_estimator: StateEstimator | None = None,
    ) -> None:
        """Initialize the localization pipeline.

        Args:
            config: Pipeline configuration
            state_estimator: Optional state estimator for fusion
        """
        self._config = config or LocalizationConfig()

        # Create components
        self._extractor = ORBFeatureExtractor(self._config.orb_config)
        self._tracker = FeatureTracker(self._config.tracker_config)
        self._vo = VisualOdometry(self._config.vo_config)

        # State estimator for fusion
        self._state_estimator = state_estimator

        # Set camera intrinsics if provided
        if self._config.camera_intrinsics:
            self._vo.set_camera_intrinsics(self._config.camera_intrinsics)

        # Pipeline state
        self._prev_image: np.ndarray | None = None
        self._prev_features: FrameFeatures | None = None
        self._frame_count = 0

        # Accumulated position (in NED)
        self._position_ned = np.zeros(3)

        # Camera to NED rotation
        self._camera_to_ned = np.array(self._config.camera_to_ned_rotation)

        # Callbacks
        self._on_result_callbacks: list[Callable[[LocalizationResult], None]] = []

        logger.info("LocalizationPipeline initialized")

    def set_camera_intrinsics(self, intrinsics: CameraIntrinsics) -> None:
        """Set camera intrinsic parameters.

        Args:
            intrinsics: Camera intrinsics
        """
        self._vo.set_camera_intrinsics(intrinsics)

    def set_state_estimator(self, estimator: StateEstimator) -> None:
        """Set state estimator for fusion.

        Args:
            estimator: State estimator instance
        """
        self._state_estimator = estimator

    def process(
        self,
        image: np.ndarray,
        altitude: float | None = None,
        timestamp: float | None = None,
    ) -> LocalizationResult:
        """Process a frame through the localization pipeline.

        Args:
            image: Input image (BGR or grayscale)
            altitude: Optional altitude AGL for scale recovery
            timestamp: Optional timestamp

        Returns:
            LocalizationResult with position and quality metrics
        """
        start_time = time.perf_counter()
        self._frame_count += 1
        timestamp = timestamp or time.time()

        # Step 1: Extract features
        features = self._extractor.extract(image, timestamp=timestamp)

        # Step 2: Track features
        tracking: TrackingResult | None = None
        if self._prev_features is not None and self._prev_image is not None:
            tracking = self._tracker.track(
                self._prev_features,
                features,
                self._prev_image,
                image,
            )

        # Step 3: Visual odometry
        vo_result: VOResult | None = None
        if tracking is not None:
            vo_result = self._vo.process(tracking, altitude=altitude)

        # Step 4: Transform to NED and accumulate
        delta_ned = np.zeros(3)
        if vo_result is not None and vo_result.is_valid:
            # Camera frame motion
            camera_delta = np.array([
                vo_result.delta_x,
                vo_result.delta_y,
                vo_result.delta_z,
            ])

            # Transform to NED
            delta_ned = self._camera_to_ned @ camera_delta

            # Accumulate position
            self._position_ned += delta_ned

            # Feed to state estimator if configured
            if self._config.feed_state_estimator and self._state_estimator:
                vo_input = VisualOdometryInput(
                    timestamp=datetime.fromtimestamp(timestamp),
                    delta_north=float(delta_ned[0]),
                    delta_east=float(delta_ned[1]),
                    delta_down=float(delta_ned[2]),
                    confidence=vo_result.confidence,
                    features_tracked=vo_result.num_inliers,
                    is_valid=vo_result.is_valid,
                )
                self._state_estimator.update_visual(vo_input)

        # Update state for next frame
        self._prev_image = image.copy()
        self._prev_features = features

        # Build result
        total_time = (time.perf_counter() - start_time) * 1000

        result = LocalizationResult(
            timestamp=datetime.fromtimestamp(timestamp),
            frame_id=self._frame_count,
            num_features=features.num_features,
            extraction_time_ms=features.extraction_time_ms,
            num_tracked=tracking.num_tracked if tracking else 0,
            tracking_confidence=tracking.tracking_confidence if tracking else 0.0,
            tracking_time_ms=tracking.tracking_time_ms if tracking else 0.0,
            vo_state=vo_result.state if vo_result else VOState.UNINITIALIZED.value,
            delta_north=float(delta_ned[0]),
            delta_east=float(delta_ned[1]),
            delta_down=float(delta_ned[2]),
            vo_confidence=vo_result.confidence if vo_result else 0.0,
            vo_time_ms=vo_result.computation_time_ms if vo_result else 0.0,
            total_time_ms=total_time,
            is_valid=vo_result.is_valid if vo_result else False,
            position_north=float(self._position_ned[0]),
            position_east=float(self._position_ned[1]),
            position_down=float(self._position_ned[2]),
        )

        # Notify callbacks
        for callback in self._on_result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Localization callback error: {e}")

        # Log performance warning
        if total_time > self._config.max_latency_ms:
            logger.warning(
                f"Localization pipeline slow: {total_time:.1f}ms > {self._config.max_latency_ms}ms"
            )

        return result

    def process_async(
        self,
        image: np.ndarray,
        altitude: float | None = None,
    ) -> None:
        """Process frame asynchronously (results via callback).

        Args:
            image: Input image
            altitude: Optional altitude
        """
        # For now, just call synchronous version
        # Could be enhanced with threading/multiprocessing
        self.process(image, altitude)

    def on_result(self, callback: Callable[[LocalizationResult], None]) -> None:
        """Register callback for localization results.

        Args:
            callback: Function to call with results
        """
        self._on_result_callbacks.append(callback)

    def get_position_ned(self) -> tuple[float, float, float]:
        """Get current accumulated NED position.

        Returns:
            Tuple of (north, east, down)
        """
        return (
            float(self._position_ned[0]),
            float(self._position_ned[1]),
            float(self._position_ned[2]),
        )

    def reset(self) -> None:
        """Reset pipeline state."""
        self._prev_image = None
        self._prev_features = None
        self._position_ned = np.zeros(3)
        self._tracker.reset()
        self._vo.reset()
        self._frame_count = 0
        logger.info("LocalizationPipeline reset")

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "frame_count": self._frame_count,
            "vo_state": self._vo.state.value,
            "position_ned": self._position_ned.tolist(),
            "config": {
                "target_fps": self._config.target_fps,
                "max_latency_ms": self._config.max_latency_ms,
            },
        }


def create_default_pipeline(
    fov_deg: float = 90.0,
    image_width: int = 1280,
    image_height: int = 720,
    state_estimator: StateEstimator | None = None,
) -> LocalizationPipeline:
    """Create a localization pipeline with default settings.

    Args:
        fov_deg: Camera horizontal field of view in degrees
        image_width: Image width in pixels
        image_height: Image height in pixels
        state_estimator: Optional state estimator for fusion

    Returns:
        Configured LocalizationPipeline
    """
    intrinsics = CameraIntrinsics.from_fov(fov_deg, image_width, image_height)

    config = LocalizationConfig(
        camera_intrinsics=intrinsics,
        orb_config=ORBConfig(max_features=800, use_grid=True),
        tracker_config=TrackerConfig(use_bidirectional=True, use_ransac=True),
        vo_config=VOConfig(use_altitude_scale=True),
    )

    return LocalizationPipeline(config, state_estimator)
