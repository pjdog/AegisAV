"""
Vision Client

Client-side vision orchestration.
Coordinates camera capture and quick detection during inspections.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vision.camera.base import CameraInterface
from vision.camera.simulated import DefectConfig, SimulatedCamera, SimulatedCameraConfig
from vision.data_models import CaptureResult, DetectionResult
from vision.image_manager import ImageManager
from vision.models.base import ModelInterface
from vision.models.yolo_detector import MockYOLODetector

logger = logging.getLogger(__name__)


@dataclass
class InspectionVisionResults:
    """
    Vision results from an inspection.

    Aggregates all captures and detections from a single inspection.
    """

    asset_id: str
    captures: list[CaptureResult] = field(default_factory=list)
    detections: list[DetectionResult] = field(default_factory=list)
    defects_detected: int = 0
    max_confidence: float = 0.0
    max_severity: float = 0.0
    best_detection_image: Path | None = None

    def needs_server_analysis(self) -> bool:
        """Whether any detection needs detailed server-side analysis."""
        return any(d.needs_detailed_analysis for d in self.detections)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for sending to server."""
        return {
            "asset_id": self.asset_id,
            "total_captures": len(self.captures),
            "total_detections": len(self.detections),
            "defects_detected": self.defects_detected,
            "max_confidence": self.max_confidence,
            "max_severity": self.max_severity,
            "needs_server_analysis": self.needs_server_analysis(),
            "best_detection_image": str(self.best_detection_image)
            if self.best_detection_image
            else None,
        }


@dataclass(frozen=True)
class VisionClientConfig:
    """Configuration for client-side vision capture.

    Attributes:
        capture_interval_s (float): Seconds between captures.
        max_captures_per_inspection (int): Maximum captures per inspection.
        simulated_inference_delay_ms (int): Extra delay per capture to simulate
            constrained edge compute.
        enabled (bool): Whether the vision client is enabled.
    """

    capture_interval_s: float = 2.0
    max_captures_per_inspection: int = 10
    simulated_inference_delay_ms: int = 0
    enabled: bool = True


class VisionClient:
    """
    Client-side vision orchestration.

    Manages:
    - Camera initialization and capture
    - Quick on-drone detection
    - Image storage
    - Results aggregation
    """

    def __init__(
        self,
        camera: CameraInterface | None = None,
        detector: ModelInterface | None = None,
        image_manager: ImageManager | None = None,
        config: VisionClientConfig | None = None,
    ):
        """
        Initialize vision client.

        Args:
            camera (CameraInterface | None): Camera instance (creates
                :class:`vision.camera.simulated.SimulatedCamera` if None).
            detector (ModelInterface | None): Detector instance (creates
                :class:`vision.models.yolo_detector.MockYOLODetector` if None).
            image_manager (ImageManager | None): Image manager (creates default if None).
            config (VisionClientConfig | None): Vision client configuration.
        """
        self.config = config or VisionClientConfig()
        self.logger = logger

        # Create default instances if not provided
        if camera is None and self.config.enabled:
            defect_config = DefectConfig()
            camera_config = SimulatedCameraConfig(defect_config=defect_config, save_images=True)
            camera = SimulatedCamera(config=camera_config)

        if detector is None and self.config.enabled:
            detector = MockYOLODetector(
                model_variant="yolov8n",
                confidence_threshold=0.4,  # Lower threshold for quick check
            )

        if image_manager is None and self.config.enabled:
            image_manager = ImageManager()

        self.camera = camera
        self.detector = detector
        self.image_manager = image_manager
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    async def initialize(self) -> bool:
        """
        Initialize camera and detector.

        Returns:
            bool: True if successful.
        """
        if not self.config.enabled:
            self.logger.info("Vision client disabled")
            return True

        self.logger.info("Initializing vision client")

        try:
            # Initialize camera
            if self.camera:
                camera_ok = await self.camera.initialize()
                if not camera_ok:
                    self.logger.error("Camera initialization failed")
                    return False

            # Initialize detector
            if self.detector:
                detector_ok = await self.detector.initialize()
                if not detector_ok:
                    self.logger.error("Detector initialization failed")
                    return False

            self._initialized = True
            self.logger.info("Vision client initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Vision client initialization failed: {e}")
            return False

    async def capture_during_inspection(
        self,
        asset_id: str,
        duration_s: float,
        vehicle_state_fn: Callable[[], dict[str, Any] | None] | None = None,
    ) -> InspectionVisionResults:
        """
        Capture images during an inspection maneuver.

        Captures at regular intervals for the duration of the inspection.

        Args:
            asset_id (str): Asset being inspected.
            duration_s (float): Duration of inspection in seconds.
            vehicle_state_fn (Callable[[], dict[str, Any] | None] | None): Optional
                callback that returns the current vehicle state payload.

        Returns:
            InspectionVisionResults: Aggregated captures and detections.
        """
        if not self.config.enabled or not self._initialized:
            # Vision disabled or not initialized - return empty results
            return InspectionVisionResults(asset_id=asset_id)

        self.logger.info(
            f"Starting vision capture for asset {asset_id} (duration: {duration_s:.1f}s)"
        )

        results = InspectionVisionResults(asset_id=asset_id)
        start_time = asyncio.get_event_loop().time()

        capture_count = 0
        while capture_count < self.config.max_captures_per_inspection:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= duration_s:
                break

            # Get current vehicle state if function provided
            vehicle_state = vehicle_state_fn() if vehicle_state_fn else None

            # Capture image
            capture_result = await self._capture_and_analyze(vehicle_state=vehicle_state)

            if capture_result:
                results.captures.append(capture_result[0])
                results.detections.append(capture_result[1])

                # Update statistics
                detection = capture_result[1]
                if detection.detected_defects:
                    results.defects_detected += len(detection.detected_defects)

                if detection.max_confidence > results.max_confidence:
                    results.max_confidence = detection.max_confidence
                    results.best_detection_image = detection.image_path

                results.max_severity = max(results.max_severity, detection.max_severity)

            if self.config.simulated_inference_delay_ms > 0:
                await asyncio.sleep(self.config.simulated_inference_delay_ms / 1000)

            capture_count += 1

            # Wait for next capture (or until duration expires)
            remaining_time = duration_s - (asyncio.get_event_loop().time() - start_time)
            wait_time = min(self.config.capture_interval_s, remaining_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.logger.info(
            "Vision capture complete: %s images, %s defects detected "
            "(max confidence: %.2f, max severity: %.2f)",
            len(results.captures),
            results.defects_detected,
            results.max_confidence,
            results.max_severity,
        )

        return results

    async def _capture_and_analyze(
        self, vehicle_state: dict | None
    ) -> tuple[CaptureResult, DetectionResult] | None:
        """
        Capture image and run quick detection.

        Args:
            vehicle_state (dict | None): Current vehicle state.

        Returns:
            tuple[CaptureResult, DetectionResult] | None: Capture and detection results,
            or None on failure.
        """
        try:
            # Capture image
            capture = await self.camera.capture(vehicle_state=vehicle_state)

            if not capture.success or not capture.image_path:
                self.logger.warning("Capture failed")
                return None

            # Run quick detection
            detection = await self.detector.analyze_image(capture.image_path)

            # Log detection results
            if detection.detected_defects:
                defect_types = [d.detection_class.value for d in detection.detected_defects]
                self.logger.info(
                    f"🔍 Defects detected: {', '.join(defect_types)} "
                    f"(confidence: {detection.max_confidence:.2f}, "
                    f"severity: {detection.max_severity:.2f})"
                )
            else:
                self.logger.debug(
                    f"No defects detected (inference time: {detection.inference_time_ms:.1f}ms)"
                )

            return (capture, detection)

        except Exception as e:
            self.logger.error(f"Capture and analyze failed: {e}")
            return None

    async def shutdown(self) -> None:
        """Shutdown camera and detector."""
        if not self.config.enabled:
            return

        self.logger.info("Shutting down vision client")

        if self.camera:
            await self.camera.shutdown()

        if self.detector:
            await self.detector.shutdown()

        self._initialized = False
