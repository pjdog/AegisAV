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
from typing import Any

import numpy as np

from vision.data_models import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionResult,
)

logger = logging.getLogger(__name__)

# Try to import ultralytics - optional dependency
try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# Try to import PIL for image loading
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


# Mapping from YOLO class names to our DetectionClass
# This maps common defect-related classes from custom trained models
YOLO_CLASS_MAPPING: dict[str, DetectionClass] = {
    # Infrastructure defects
    "crack": DetectionClass.CRACK,
    "corrosion": DetectionClass.CORROSION,
    "rust": DetectionClass.CORROSION,
    "hot_spot": DetectionClass.HOT_SPOT,
    "hotspot": DetectionClass.HOT_SPOT,
    "thermal_anomaly": DetectionClass.HOT_SPOT,
    "debris": DetectionClass.DEBRIS,
    "bird_dropping": DetectionClass.DEBRIS,
    "vegetation": DetectionClass.VEGETATION,
    "overgrowth": DetectionClass.VEGETATION,
    "damage": DetectionClass.DAMAGE,
    "broken": DetectionClass.DAMAGE,
    "missing": DetectionClass.DAMAGE,
    "deformation": DetectionClass.DEFORMATION,
    "bend": DetectionClass.DEFORMATION,
    "wear": DetectionClass.WEAR,
    "erosion": DetectionClass.WEAR,
    "stain": DetectionClass.STAIN,
    "discoloration": DetectionClass.STAIN,
    "leak": DetectionClass.LEAK,
    "oil_leak": DetectionClass.LEAK,
    # Normal/healthy
    "normal": DetectionClass.NORMAL,
    "healthy": DetectionClass.NORMAL,
    "ok": DetectionClass.NORMAL,
    # Solar panel specific
    "solar_panel": DetectionClass.NORMAL,
    "panel_crack": DetectionClass.CRACK,
    "cell_damage": DetectionClass.DAMAGE,
    # Wind turbine specific
    "blade": DetectionClass.NORMAL,
    "blade_crack": DetectionClass.CRACK,
    "blade_erosion": DetectionClass.WEAR,
    "ice": DetectionClass.DEBRIS,
    # Power line specific
    "insulator": DetectionClass.NORMAL,
    "insulator_damage": DetectionClass.DAMAGE,
    "conductor": DetectionClass.NORMAL,
    "conductor_damage": DetectionClass.DAMAGE,
}


class YOLODetector:
    """
    Real YOLO detector using ultralytics library.

    Supports YOLOv8 and YOLOv11 models for infrastructure defect detection.
    Can use pretrained models or custom trained models.

    Example:
        detector = YOLODetector(weights_path="best.pt", device="cuda")
        await detector.initialize()
        result = await detector.analyze_image(Path("image.jpg"))
    """

    def __init__(
        self,
        weights_path: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        image_size: int = 640,
        half_precision: bool = False,
    ):
        """
        Initialize YOLO detector.

        Args:
            weights_path: Path to YOLO weights (.pt file) or model name
                         (e.g., "yolov8n.pt", "yolov8s.pt", "yolov11n.pt")
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            device: Device to use ("cpu", "cuda", "cuda:0", or "auto")
            image_size: Input image size for inference
            half_precision: Use FP16 inference (faster on GPU)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics package not installed. Install with: pip install ultralytics"
            )

        self.weights_path = Path(weights_path) if isinstance(weights_path, str) else weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size
        self.half_precision = half_precision

        self.model: Any | None = None
        self._initialized = False
        self._class_names: dict[int, str] = {}
        self.logger = logger

    async def initialize(self) -> bool:
        """
        Initialize the YOLO model.

        Downloads pretrained weights if necessary.

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(
                f"Initializing YOLODetector (weights={self.weights_path}, device={self.device})"
            )

            # Load model (runs in thread pool to not block event loop)
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, self._load_model)

            # Get class names from model
            if hasattr(self.model, "names"):
                self._class_names = self.model.names
                self.logger.info(f"Model classes: {list(self._class_names.values())}")

            self._initialized = True
            self.logger.info("YOLODetector initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize YOLODetector: {e}")
            return False

    def _load_model(self) -> Any:
        """Load the YOLO model (blocking, run in executor)."""
        model = YOLO(str(self.weights_path))

        # Move to device
        if self.device != "auto":
            model.to(self.device)

        return model

    async def analyze_image(self, image_path: Path) -> DetectionResult:
        """
        Analyze image for defects using YOLO inference.

        Args:
            image_path: Path to image file

        Returns:
            DetectionResult with detected defects
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Detector not initialized - call initialize() first")

        start_time = time.time()

        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: self._run_inference(image_path)
            )

            # Process results
            detections = self._process_results(results)

            inference_time_ms = (time.time() - start_time) * 1000

            self.logger.debug(
                f"Analyzed {image_path.name}: {len(detections)} detections in {inference_time_ms:.1f}ms"
            )

            return DetectionResult(
                detections=detections,
                image_path=image_path,
                model_name=str(self.weights_path.stem),
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            self.logger.error(f"Failed to analyze image {image_path}: {e}")
            return DetectionResult(
                detections=[],
                image_path=image_path,
                model_name=str(self.weights_path.stem),
                inference_time_ms=(time.time() - start_time) * 1000,
            )

    def _run_inference(self, image_path: Path) -> Any:
        """Run YOLO inference (blocking, run in executor)."""
        return self.model(
            str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            half=self.half_precision,
            verbose=False,
        )

    def _process_results(self, results: Any) -> list[Detection]:
        """Process YOLO results into Detection objects."""
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            img_height, img_width = result.orig_shape

            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box

                # Normalize to 0-1 range
                x_min = float(x1 / img_width)
                y_min = float(y1 / img_height)
                x_max = float(x2 / img_width)
                y_max = float(y2 / img_height)

                # Get confidence
                confidence = float(boxes.conf[i].cpu().numpy())

                # Get class
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self._class_names.get(cls_id, "unknown").lower()

                # Map to our DetectionClass
                detection_class = YOLO_CLASS_MAPPING.get(cls_name, DetectionClass.UNKNOWN)

                # Estimate severity based on confidence and box size
                box_area = (x_max - x_min) * (y_max - y_min)
                severity = min(1.0, confidence * 0.7 + box_area * 0.3)

                detection = Detection(
                    detection_class=detection_class,
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                    severity=severity,
                    raw_class_name=cls_name,
                )
                detections.append(detection)

                if detection_class != DetectionClass.NORMAL:
                    self.logger.info(
                        f"Detected {detection_class.value} "
                        f"(confidence: {confidence:.2f}, severity: {severity:.2f})"
                    )

        return detections

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        info = {
            "model_type": "yolo",
            "weights_path": str(self.weights_path),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "initialized": self._initialized,
        }

        if self._initialized and self.model is not None:
            info["class_names"] = list(self._class_names.values())
            info["num_classes"] = len(self._class_names)

        return info

    async def shutdown(self) -> None:
        """Shutdown the detector and free resources."""
        self.logger.info("Shutting down YOLODetector")
        self.model = None
        self._initialized = False


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


def create_detector(
    use_real: bool = True,
    weights_path: str | Path = "yolov8n.pt",
    confidence_threshold: float = 0.5,
    device: str = "auto",
    **kwargs,
) -> YOLODetector | MockYOLODetector:
    """
    Factory function to create appropriate detector.

    Args:
        use_real: Use real YOLO if available, else mock
        weights_path: Path to weights (for real detector)
        confidence_threshold: Minimum confidence
        device: Device to use
        **kwargs: Additional arguments for detector

    Returns:
        YOLODetector or MockYOLODetector instance
    """
    if use_real and ULTRALYTICS_AVAILABLE:
        return YOLODetector(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            device=device,
            **kwargs,
        )
    else:
        if use_real and not ULTRALYTICS_AVAILABLE:
            logger.warning(
                "ultralytics not available, falling back to MockYOLODetector. "
                "Install with: pip install ultralytics"
            )
        return MockYOLODetector(
            model_variant=Path(weights_path).stem if isinstance(weights_path, (str, Path)) else "yolov8n",
            confidence_threshold=confidence_threshold,
            device=device,
        )
