"""
YOLO Detector

Object detection using YOLO (You Only Look Once) models.
Supports both real YOLO inference and mock detection for testing.
"""

import asyncio
import json
import logging
import random
import time
from pathlib import Path

from vision.data_models import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionResult,
)

logger = logging.getLogger(__name__)


class MockYOLODetector:
    """
    Mock YOLO detector for testing and simulation.

    Reads defect metadata from simulated camera captures and
    generates realistic detection results without running actual inference.
    """

    def __init__(
        self,
        model_variant: str = "yolov8n",
        confidence_threshold: float = 0.6,
        device: str = "cpu",
    ):
        """
        Initialize mock YOLO detector.

        Args:
            model_variant: YOLO variant (n/s/m/l/x)
            confidence_threshold: Minimum confidence for detections
            device: Device to use (cpu/cuda)
        """
        self.model_variant = model_variant
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._initialized = False
        self.logger = logger

    async def initialize(self) -> bool:
        """Initialize the mock detector."""
        self.logger.info(
            f"Initializing MockYOLODetector (variant={self.model_variant}, device={self.device})"
        )
        self._initialized = True
        return True

    async def analyze_image(self, image_path: Path) -> DetectionResult:
        """
        Analyze image for defects.

        Reads metadata from .json sidecar if available (from SimulatedCamera),
        otherwise returns empty result.

        Args:
            image_path (Path): Path to image file (:class:`pathlib.Path`).

        Returns:
            DetectionResult: Detection result with detections
            (:class:`vision.data_models.DetectionResult`).
        """
        if not self._initialized:
            raise RuntimeError("Detector not initialized - call initialize() first")

        start_time = time.time()

        # Look for metadata sidecar file
        metadata_path = image_path.with_suffix(".json")
        detections = []

        if metadata_path.exists():
            # Read metadata from simulated camera
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)

                # Check if defect was injected
                if "injected_defect" in metadata:
                    defect = metadata["injected_defect"]
                    detection = self._create_detection_from_metadata(defect)
                    if detection.confidence >= self.confidence_threshold:
                        detections.append(detection)
                        self.logger.info(
                            f"Detected {detection.detection_class.value} "
                            f"(confidence: {detection.confidence:.2f}, "
                            f"severity: {detection.severity:.2f})"
                        )
                else:
                    # No defect - might add "normal" detection
                    if random.random() < 0.8:  # 80% chance to classify as normal
                        detections.append(self._create_normal_detection())

            except Exception as e:
                self.logger.error(f"Failed to read metadata from {metadata_path}: {e}")

        else:
            # No metadata - simulate quick inference
            # Randomly detect "normal" with high confidence
            if random.random() < 0.7:
                detections.append(self._create_normal_detection())

        inference_time_ms = (time.time() - start_time) * 1000

        # Add small delay to simulate inference
        simulated_delay = random.uniform(20, 80)  # 20-80ms
        await asyncio.sleep(simulated_delay / 1000)
        inference_time_ms += simulated_delay

        return DetectionResult(
            detections=detections,
            image_path=image_path,
            model_name=f"mock_{self.model_variant}",
            inference_time_ms=inference_time_ms,
        )

    def _create_detection_from_metadata(self, defect_metadata: dict) -> Detection:
        """
        Create detection from defect metadata.

        Adds realistic confidence and bounding box.

        Args:
            defect_metadata: Metadata from simulated camera

        Returns:
            Detection object
        """
        defect_type_str = defect_metadata["type"]
        severity = defect_metadata["severity"]

        # Map to DetectionClass
        try:
            detection_class = DetectionClass(defect_type_str)
        except ValueError:
            detection_class = DetectionClass.UNKNOWN

        # Generate confidence based on severity (higher severity = higher confidence)
        # Add some randomness
        base_confidence = 0.5 + (severity * 0.4)  # 0.5 to 0.9 range
        confidence = base_confidence + random.uniform(-0.1, 0.1)
        confidence = max(0.4, min(0.95, confidence))  # Clamp to [0.4, 0.95]

        # Generate random but realistic bounding box
        bbox = self._generate_bounding_box()

        return Detection(
            detection_class=detection_class,
            confidence=confidence,
            bounding_box=bbox,
            severity=severity,
        )

    def _create_normal_detection(self) -> Detection:
        """Create a 'normal' detection (no defect found)."""
        bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
        return Detection(
            detection_class=DetectionClass.NORMAL,
            confidence=random.uniform(0.85, 0.95),
            bounding_box=bbox,
            severity=0.0,
        )

    def _generate_bounding_box(self) -> BoundingBox:
        """Generate a random but realistic bounding box."""
        # Random position
        x_center = random.uniform(0.2, 0.8)
        y_center = random.uniform(0.2, 0.8)

        # Random size (defects are usually small to medium)
        width = random.uniform(0.1, 0.4)
        height = random.uniform(0.1, 0.4)

        x_min = max(0.0, x_center - width / 2)
        x_max = min(1.0, x_center + width / 2)
        y_min = max(0.0, y_center - height / 2)
        y_max = min(1.0, y_center + height / 2)

        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "mock_yolo",
            "variant": self.model_variant,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "initialized": self._initialized,
        }

    async def shutdown(self) -> None:
        """Shutdown the detector."""
        self.logger.info("Shutting down MockYOLODetector")
        self._initialized = False


# For future real YOLO implementation
class YOLODetector:
    """
    Real YOLO detector using ultralytics library.

    TODO: Implement when ultralytics dependency is added.
    For now, use MockYOLODetector.
    """

    def __init__(
        self,
        weights_path: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.6,
        device: str = "cpu",
    ):
        """
        Initialize YOLO detector.

        Args:
            weights_path: Path to YOLO weights
            confidence_threshold: Minimum confidence
            device: Device (cpu/cuda)
        """
        raise NotImplementedError(
            "Real YOLODetector not yet implemented. Use MockYOLODetector for now."
        )
