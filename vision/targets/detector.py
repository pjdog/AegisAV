"""Target Detection and Recognition.

Detects and identifies inspection targets using visual features.
Supports template matching and learned embeddings for target recognition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TargetType(Enum):
    """Types of inspection targets."""

    SOLAR_PANEL = "solar_panel"
    WIND_TURBINE = "wind_turbine"
    SUBSTATION = "substation"
    POWER_LINE = "power_line"
    BUILDING = "building"
    MARKER = "marker"  # Visual marker/fiducial
    UNKNOWN = "unknown"


class DetectedTarget(BaseModel):
    """A detected target in an image."""

    target_id: str
    target_type: str
    confidence: float  # 0.0 to 1.0

    # Bounding box (pixel coordinates)
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int

    # Center point
    center_x: float
    center_y: float

    # Optional: matched template ID
    template_id: str | None = None

    # Optional: embedding similarity
    embedding_similarity: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x, y, w, h)."""
        return (self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height)

    @property
    def area(self) -> int:
        """Get bounding box area."""
        return self.bbox_width * self.bbox_height


@dataclass
class TargetTemplate:
    """A registered target template for matching."""

    template_id: str
    target_type: TargetType
    name: str

    # Template image (grayscale)
    template: np.ndarray | None = None

    # Feature descriptors for matching
    descriptors: np.ndarray | None = None
    keypoints: list = field(default_factory=list)

    # Optional embedding vector
    embedding: np.ndarray | None = None

    # Metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class TargetDetectorConfig:
    """Configuration for target detector."""

    # Detection method
    use_template_matching: bool = True
    use_feature_matching: bool = True
    use_color_detection: bool = False

    # Template matching
    template_threshold: float = 0.7
    template_scales: list[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])

    # Feature matching
    feature_min_matches: int = 10
    feature_ratio_threshold: float = 0.75

    # Non-maximum suppression
    nms_threshold: float = 0.3

    # Minimum detection confidence
    min_confidence: float = 0.5

    # Color detection (HSV ranges for common targets)
    solar_panel_hsv_lower: tuple[int, int, int] = (100, 50, 50)
    solar_panel_hsv_upper: tuple[int, int, int] = (130, 255, 255)


class TargetDetector:
    """Detects and recognizes inspection targets.

    Supports multiple detection methods:
    - Template matching: Fast, good for known targets
    - Feature matching: Robust to viewpoint changes
    - Color detection: Simple, for color-coded targets

    Example:
        detector = TargetDetector(config)

        # Register templates
        detector.register_template("solar_1", TargetType.SOLAR_PANEL, template_image)

        # Detect targets
        targets = detector.detect(image)
        for target in targets:
            print(f"Found {target.target_type} at ({target.center_x}, {target.center_y})")
    """

    def __init__(self, config: TargetDetectorConfig | None = None) -> None:
        """Initialize target detector.

        Args:
            config: Detector configuration
        """
        self._config = config or TargetDetectorConfig()
        self._templates: dict[str, TargetTemplate] = {}

        # Create ORB for feature matching
        self._orb = cv2.ORB_create(nfeatures=500)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Detection counter
        self._detection_count = 0

        logger.info("TargetDetector initialized")

    def register_template(
        self,
        template_id: str,
        target_type: TargetType,
        image: np.ndarray,
        name: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Register a target template for matching.

        Args:
            template_id: Unique template identifier
            target_type: Type of target
            image: Template image (BGR or grayscale)
            name: Human-readable name
            metadata: Optional metadata

        Returns:
            True if registration successful
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Extract features
            keypoints, descriptors = self._orb.detectAndCompute(gray, None)

            template = TargetTemplate(
                template_id=template_id,
                target_type=target_type,
                name=name or template_id,
                template=gray,
                descriptors=descriptors,
                keypoints=keypoints,
                metadata=metadata or {},
            )

            self._templates[template_id] = template
            logger.info(f"Registered template: {template_id} ({target_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to register template {template_id}: {e}")
            return False

    def register_template_from_file(
        self,
        template_id: str,
        target_type: TargetType,
        image_path: str | Path,
        **kwargs,
    ) -> bool:
        """Register template from image file.

        Args:
            template_id: Unique template identifier
            target_type: Type of target
            image_path: Path to template image

        Returns:
            True if registration successful
        """
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Template image not found: {path}")
            return False

        image = cv2.imread(str(path))
        if image is None:
            logger.error(f"Failed to load template image: {path}")
            return False

        return self.register_template(template_id, target_type, image, **kwargs)

    def detect(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[DetectedTarget]:
        """Detect targets in an image.

        Args:
            image: Input image (BGR)
            roi: Optional region of interest (x, y, w, h)

        Returns:
            List of detected targets
        """
        start_time = time.perf_counter()
        detections = []

        # Apply ROI if provided
        if roi:
            x, y, w, h = roi
            image = image[y : y + h, x : x + w]
            offset_x, offset_y = x, y
        else:
            offset_x, offset_y = 0, 0

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Template matching
        if self._config.use_template_matching and self._templates:
            template_detections = self._detect_templates(gray, offset_x, offset_y)
            detections.extend(template_detections)

        # Feature matching
        if self._config.use_feature_matching and self._templates:
            feature_detections = self._detect_features(gray, offset_x, offset_y)
            detections.extend(feature_detections)

        # Color detection
        if self._config.use_color_detection and len(image.shape) == 3:
            color_detections = self._detect_colors(image, offset_x, offset_y)
            detections.extend(color_detections)

        # Apply NMS
        detections = self._non_max_suppression(detections)

        # Filter by confidence
        detections = [d for d in detections if d.confidence >= self._config.min_confidence]

        self._detection_count += len(detections)

        logger.debug(
            f"Detected {len(detections)} targets in {(time.perf_counter() - start_time) * 1000:.1f}ms"
        )

        return detections

    def _detect_templates(
        self,
        gray: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[DetectedTarget]:
        """Detect targets using template matching."""
        detections = []
        h, w = gray.shape[:2]

        for template_id, template in self._templates.items():
            if template.template is None:
                continue

            th, tw = template.template.shape[:2]

            # Multi-scale matching
            for scale in self._config.template_scales:
                # Resize template
                scaled_w = int(tw * scale)
                scaled_h = int(th * scale)

                if scaled_w > w or scaled_h > h:
                    continue

                scaled_template = cv2.resize(template.template, (scaled_w, scaled_h))

                # Match
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self._config.template_threshold)

                for pt in zip(*locations[::-1], strict=False):
                    confidence = float(result[pt[1], pt[0]])

                    detections.append(
                        DetectedTarget(
                            target_id=f"{template_id}_{len(detections)}",
                            target_type=template.target_type.value,
                            confidence=confidence,
                            bbox_x=pt[0] + offset_x,
                            bbox_y=pt[1] + offset_y,
                            bbox_width=scaled_w,
                            bbox_height=scaled_h,
                            center_x=pt[0] + scaled_w / 2 + offset_x,
                            center_y=pt[1] + scaled_h / 2 + offset_y,
                            template_id=template_id,
                        )
                    )

        return detections

    def _detect_features(
        self,
        gray: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[DetectedTarget]:
        """Detect targets using feature matching."""
        detections = []

        # Extract features from query image
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < self._config.feature_min_matches:
            return detections

        for template_id, template in self._templates.items():
            if template.descriptors is None:
                continue

            # Match features
            matches = self._bf_matcher.knnMatch(template.descriptors, descriptors, k=2)

            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self._config.feature_ratio_threshold * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self._config.feature_min_matches:
                continue

            # Get matched point locations
            src_pts = np.float32([template.keypoints[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                continue

            # Get bounding box in query image
            th, tw = template.template.shape[:2]
            corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, M)

            # Calculate bounding box
            x_coords = transformed[:, 0, 0]
            y_coords = transformed[:, 0, 1]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Calculate confidence based on inlier ratio
            inlier_ratio = mask.sum() / len(good_matches) if len(good_matches) > 0 else 0
            confidence = inlier_ratio * (len(good_matches) / 50)  # Normalize by expected matches
            confidence = min(confidence, 1.0)

            detections.append(
                DetectedTarget(
                    target_id=f"{template_id}_feat_{len(detections)}",
                    target_type=template.target_type.value,
                    confidence=confidence,
                    bbox_x=x_min + offset_x,
                    bbox_y=y_min + offset_y,
                    bbox_width=x_max - x_min,
                    bbox_height=y_max - y_min,
                    center_x=(x_min + x_max) / 2 + offset_x,
                    center_y=(y_min + y_max) / 2 + offset_y,
                    template_id=template_id,
                )
            )

        return detections

    def _detect_colors(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[DetectedTarget]:
        """Detect targets using color segmentation."""
        detections = []

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect solar panels (blue-ish)
        lower = np.array(self._config.solar_panel_hsv_lower)
        upper = np.array(self._config.solar_panel_hsv_upper)
        mask = cv2.inRange(hsv, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area threshold
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Calculate confidence based on area and aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            # Solar panels typically have aspect ratio between 0.5 and 2
            aspect_score = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)
            confidence = min(area / 10000, 1.0) * aspect_score * 0.7

            detections.append(
                DetectedTarget(
                    target_id=f"color_{len(detections)}",
                    target_type=TargetType.SOLAR_PANEL.value,
                    confidence=confidence,
                    bbox_x=x + offset_x,
                    bbox_y=y + offset_y,
                    bbox_width=w,
                    bbox_height=h,
                    center_x=x + w / 2 + offset_x,
                    center_y=y + h / 2 + offset_y,
                )
            )

        return detections

    def _non_max_suppression(
        self,
        detections: list[DetectedTarget],
    ) -> list[DetectedTarget]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        suppressed = set()

        for i, det in enumerate(detections):
            if i in suppressed:
                continue

            keep.append(det)

            # Suppress overlapping detections
            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue

                other = detections[j]
                iou = self._compute_iou(det.bbox, other.bbox)

                if iou > self._config.nms_threshold:
                    suppressed.add(j)

        return keep

    def _compute_iou(
        self,
        bbox1: tuple[int, int, int, int],
        bbox2: tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union for two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    @property
    def num_templates(self) -> int:
        """Get number of registered templates."""
        return len(self._templates)

    @property
    def detection_count(self) -> int:
        """Get total detection count."""
        return self._detection_count
