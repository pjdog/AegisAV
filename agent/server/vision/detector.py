"""Server-Side Detectors.

Detailed vision analysis for server-side processing.
Includes simulated detector for testing and real YOLO for production.
"""

import asyncio
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


class SimulatedDetector:
    """Simulated detector for testing server-side vision.

    Generates realistic detection results based on:
    - Client-side detection confidence
    - Random defect detection with configurable probability
    - Simulated detailed analysis with higher accuracy
    """

    def __init__(
        self,
        defect_probability: float = 0.15,
        severity_range: tuple[float, float] = (0.3, 0.9),
        confidence_boost: float = 0.1,
    ) -> None:
        """Initialize simulated detector.

        Args:
            defect_probability: Probability of detecting a defect (0.0-1.0)
            severity_range: Range for defect severity (min, max)
            confidence_boost: Confidence boost over client detection
        """
        self.defect_probability = defect_probability
        self.severity_range = severity_range
        self.confidence_boost = confidence_boost
        self.logger = logger
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the detector."""
        self.logger.info("Initializing SimulatedDetector (server-side)")
        self._initialized = True
        return True

    async def analyze_image(
        self,
        image_path: Path,
        client_detection: DetectionResult | None = None,
    ) -> DetectionResult:
        """Perform detailed server-side analysis.

        Args:
            image_path (Path): Path to image (:class:`pathlib.Path`).
            client_detection (DetectionResult | None): Optional client-side detection result
                (a :class:`vision.data_models.DetectionResult`).

        Returns:
            DetectionResult: Detailed analysis result.
        """
        if not self._initialized:
            raise RuntimeError("Detector not initialized")

        start_time = time.time()

        # Simulate longer, more detailed inference
        inference_delay = random.uniform(200, 500)  # 200-500ms
        await asyncio.sleep(inference_delay / 1000)

        detections = []

        # If client detected something suspicious, analyze more carefully
        if client_detection and client_detection.detected_defects:
            # Refine client detection with higher confidence
            for client_det in client_detection.detected_defects:
                refined_confidence = min(1.0, client_det.confidence + self.confidence_boost)
                refined_severity = client_det.severity

                # Detailed analysis might increase severity
                if random.random() < 0.3:  # 30% chance to increase severity
                    refined_severity = min(1.0, refined_severity + random.uniform(0.1, 0.2))

                detections.append(
                    Detection(
                        detection_class=client_det.detection_class,
                        confidence=refined_confidence,
                        bounding_box=client_det.bounding_box,
                        severity=refined_severity,
                    )
                )

                self.logger.info(
                    f"Server refined detection: {client_det.detection_class.value} "
                    f"(confidence: {refined_confidence:.2f}, severity: {refined_severity:.2f})"
                )

        else:
            # No client detection - random sampling for defects
            if random.random() < self.defect_probability:
                defect_class, severity = self._sample_defect()
                confidence = random.uniform(0.7, 0.95)
                bbox = self._generate_bounding_box()

                detections.append(
                    Detection(
                        detection_class=defect_class,
                        confidence=confidence,
                        bounding_box=bbox,
                        severity=severity,
                    )
                )

                self.logger.info(
                    f"Server detected new defect: {defect_class.value} "
                    f"(confidence: {confidence:.2f}, severity: {severity:.2f})"
                )

        inference_time_ms = (time.time() - start_time) * 1000

        return DetectionResult(
            detections=detections,
            image_path=image_path,
            model_name="simulated_server_detector",
            inference_time_ms=inference_time_ms,
        )

    def _sample_defect(self) -> tuple[DetectionClass, float]:
        """Sample a random defect type and severity."""
        defect_types = [
            DetectionClass.CRACK,
            DetectionClass.CORROSION,
            DetectionClass.STRUCTURAL_DAMAGE,
            DetectionClass.DISCOLORATION,
            DetectionClass.VEGETATION_OVERGROWTH,
        ]

        defect_class = random.choice(defect_types)
        severity = random.uniform(*self.severity_range)

        return defect_class, severity

    def _generate_bounding_box(self) -> BoundingBox:
        """Generate a random but realistic bounding box."""
        x_center = random.uniform(0.2, 0.8)
        y_center = random.uniform(0.2, 0.8)
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
            "model_type": "simulated_detector",
            "defect_probability": self.defect_probability,
            "severity_range": self.severity_range,
            "confidence_boost": self.confidence_boost,
        }

    async def shutdown(self) -> None:
        """Shutdown the detector."""
        self.logger.info("Shutting down SimulatedDetector")
        self._initialized = False
