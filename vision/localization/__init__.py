"""Visual Localization Package.

Provides visual odometry and localization for GPS-denied environments.
Uses ORB features for efficient CPU-based feature extraction and tracking.
"""

from vision.localization.features import (
    FeaturePoint,
    FrameFeatures,
    ORBConfig,
    ORBFeatureExtractor,
)
from vision.localization.pipeline import (
    LocalizationConfig,
    LocalizationPipeline,
    LocalizationResult,
)
from vision.localization.tracking import (
    FeatureMatch,
    FeatureTracker,
    TrackerConfig,
    TrackingResult,
)
from vision.localization.visual_odometry import (
    VisualOdometry,
    VOConfig,
    VOResult,
    VOState,
)

__all__ = [
    # Tracking
    "FeatureMatch",
    # Features
    "FeaturePoint",
    "FeatureTracker",
    "FrameFeatures",
    "LocalizationConfig",
    # Pipeline
    "LocalizationPipeline",
    "LocalizationResult",
    "ORBConfig",
    "ORBFeatureExtractor",
    "TrackerConfig",
    "TrackingResult",
    "VOConfig",
    "VOResult",
    "VOState",
    # Visual Odometry
    "VisualOdometry",
]
