"""Target Detection Package.

Provides visual target detection and recognition for autonomous inspection.
"""

from vision.targets.detector import (
    DetectedTarget,
    TargetDetector,
    TargetDetectorConfig,
    TargetTemplate,
)

__all__ = [
    "DetectedTarget",
    "TargetDetector",
    "TargetDetectorConfig",
    "TargetTemplate",
]
