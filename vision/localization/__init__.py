"""Visual Localization Package.

Provides visual odometry and localization for GPS-denied environments.
Uses ORB features for efficient CPU-based feature extraction and tracking.
"""

from vision.localization.features import (
    FeaturePoint,
    FrameFeatures,
    ORBFeatureExtractor,
    ORBConfig,
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
from vision.localization.pipeline import (
    LocalizationPipeline,
    LocalizationConfig,
    LocalizationResult,
)

__all__ = [
    # Features
    "FeaturePoint",
    "FrameFeatures",
    "ORBFeatureExtractor",
    "ORBConfig",
    # Tracking
    "FeatureMatch",
    "FeatureTracker",
    "TrackerConfig",
    "TrackingResult",
    # Visual Odometry
    "VisualOdometry",
    "VOConfig",
    "VOResult",
    "VOState",
    # Pipeline
    "LocalizationPipeline",
    "LocalizationConfig",
    "LocalizationResult",
]
