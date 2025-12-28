"""
Vision Subsystem

Computer vision capabilities for anomaly detection during drone inspections.

Includes:
- Camera abstraction layer (simulated and real cameras)
- Object detection models (YOLO-based)
- Image capture and management
- Detection result processing
"""

from vision.data_models import (
    BoundingBox,
    CameraState,
    CameraStatus,
    CaptureResult,
    Detection,
    DetectionClass,
    DetectionResult,
)

__all__ = [
    "BoundingBox",
    "CameraState",
    "CameraStatus",
    "CaptureResult",
    "Detection",
    "DetectionClass",
    "DetectionResult",
]
