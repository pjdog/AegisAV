"""ORB Feature Extraction.

Extracts ORB (Oriented FAST and Rotated BRIEF) features from images.
ORB is chosen for its speed on CPU and rotation invariance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FeaturePoint(BaseModel):
    """A single feature point with descriptor."""

    x: float  # x coordinate in pixels
    y: float  # y coordinate in pixels
    size: float  # Feature scale
    angle: float  # Orientation in degrees
    response: float  # Detector response (strength)
    octave: int  # Pyramid octave
    class_id: int = -1  # Optional class ID

    def to_cv_keypoint(self) -> cv2.KeyPoint:
        """Convert to OpenCV KeyPoint."""
        return cv2.KeyPoint(
            x=self.x,
            y=self.y,
            size=self.size,
            angle=self.angle,
            response=self.response,
            octave=self.octave,
            class_id=self.class_id,
        )

    @classmethod
    def from_cv_keypoint(cls, kp: cv2.KeyPoint) -> FeaturePoint:
        """Create from OpenCV KeyPoint."""
        return cls(
            x=kp.pt[0],
            y=kp.pt[1],
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
            class_id=kp.class_id,
        )


@dataclass
class FrameFeatures:
    """Features extracted from a single frame."""

    frame_id: int
    timestamp: float

    # Keypoints
    keypoints: list[FeaturePoint]

    # Descriptors (N x 32 for ORB)
    descriptors: np.ndarray | None = None

    # Image dimensions
    width: int = 0
    height: int = 0

    # Extraction metrics
    extraction_time_ms: float = 0.0

    @property
    def num_features(self) -> int:
        return len(self.keypoints)

    def get_keypoints_array(self) -> np.ndarray:
        """Get keypoints as Nx2 array."""
        if not self.keypoints:
            return np.array([]).reshape(0, 2)
        return np.array([[kp.x, kp.y] for kp in self.keypoints])

    def get_cv_keypoints(self) -> list[cv2.KeyPoint]:
        """Convert keypoints to OpenCV format."""
        return [kp.to_cv_keypoint() for kp in self.keypoints]


@dataclass
class ORBConfig:
    """Configuration for ORB feature extractor."""

    # Number of features
    max_features: int = 1000
    min_features: int = 100  # Warn if below this

    # ORB parameters
    scale_factor: float = 1.2  # Pyramid scale factor
    num_levels: int = 8  # Number of pyramid levels
    edge_threshold: int = 31  # Border margin
    first_level: int = 0  # Pyramid level of source image
    wta_k: int = 2  # Points for BRIEF descriptor
    patch_size: int = 31  # BRIEF patch size
    fast_threshold: int = 20  # FAST threshold

    # Grid-based extraction for better distribution
    use_grid: bool = True
    grid_rows: int = 4
    grid_cols: int = 6

    # Target extraction time
    target_time_ms: float = 30.0


class ORBFeatureExtractor:
    """Extracts ORB features from images.

    ORB (Oriented FAST and Rotated BRIEF) is a fast, rotation-invariant
    feature detector suitable for real-time visual odometry on edge devices.

    Example:
        extractor = ORBFeatureExtractor(config)
        features = extractor.extract(image)
        print(f"Extracted {features.num_features} features")
    """

    def __init__(self, config: ORBConfig | None = None) -> None:
        """Initialize ORB extractor.

        Args:
            config: Extraction configuration
        """
        self._config = config or ORBConfig()
        self._frame_counter = 0

        # Create ORB detector
        self._orb = cv2.ORB_create(
            nfeatures=self._config.max_features,
            scaleFactor=self._config.scale_factor,
            nlevels=self._config.num_levels,
            edgeThreshold=self._config.edge_threshold,
            firstLevel=self._config.first_level,
            WTA_K=self._config.wta_k,
            patchSize=self._config.patch_size,
            fastThreshold=self._config.fast_threshold,
        )

        logger.info(f"ORBFeatureExtractor initialized (max_features={self._config.max_features})")

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> FrameFeatures:
        """Extract ORB features from an image.

        Args:
            image: Input image (BGR or grayscale)
            mask: Optional mask (255 = extract, 0 = ignore)
            timestamp: Optional timestamp

        Returns:
            FrameFeatures with keypoints and descriptors
        """
        start_time = time.perf_counter()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape[:2]

        # Extract features
        if self._config.use_grid:
            keypoints, descriptors = self._extract_grid(gray, mask)
        else:
            keypoints, descriptors = self._orb.detectAndCompute(gray, mask)

        # Convert keypoints
        feature_points = []
        if keypoints:
            for kp in keypoints:
                feature_points.append(FeaturePoint.from_cv_keypoint(kp))

        extraction_time = (time.perf_counter() - start_time) * 1000

        # Increment frame counter
        self._frame_counter += 1

        # Log warnings
        if len(feature_points) < self._config.min_features:
            logger.warning(
                f"Low feature count: {len(feature_points)} < {self._config.min_features}"
            )

        if extraction_time > self._config.target_time_ms * 1.5:
            logger.warning(
                f"Slow feature extraction: {extraction_time:.1f}ms > {self._config.target_time_ms}ms"
            )

        return FrameFeatures(
            frame_id=self._frame_counter,
            timestamp=timestamp or time.time(),
            keypoints=feature_points,
            descriptors=descriptors,
            width=width,
            height=height,
            extraction_time_ms=extraction_time,
        )

    def _extract_grid(
        self,
        gray: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Extract features using grid-based approach for better distribution.

        Divides image into grid cells and extracts features from each cell,
        ensuring good spatial coverage.
        """
        height, width = gray.shape[:2]
        cell_h = height // self._config.grid_rows
        cell_w = width // self._config.grid_cols
        features_per_cell = self._config.max_features // (
            self._config.grid_rows * self._config.grid_cols
        )

        # Create per-cell ORB with fewer features
        cell_orb = cv2.ORB_create(
            nfeatures=features_per_cell * 2,  # Extract more, keep best
            scaleFactor=self._config.scale_factor,
            nlevels=self._config.num_levels,
            edgeThreshold=self._config.edge_threshold,
            fastThreshold=self._config.fast_threshold,
        )

        all_keypoints = []
        all_descriptors = []

        for row in range(self._config.grid_rows):
            for col in range(self._config.grid_cols):
                # Extract cell region
                y1 = row * cell_h
                y2 = (row + 1) * cell_h if row < self._config.grid_rows - 1 else height
                x1 = col * cell_w
                x2 = (col + 1) * cell_w if col < self._config.grid_cols - 1 else width

                cell = gray[y1:y2, x1:x2]

                # Cell mask
                cell_mask = None
                if mask is not None:
                    cell_mask = mask[y1:y2, x1:x2]

                # Extract features in cell
                kps, descs = cell_orb.detectAndCompute(cell, cell_mask)

                if kps:
                    # Keep top features by response
                    sorted_kps = sorted(kps, key=lambda k: k.response, reverse=True)
                    keep_kps = sorted_kps[:features_per_cell]

                    # Adjust keypoint coordinates to image frame
                    for kp in keep_kps:
                        kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
                        all_keypoints.append(kp)

                    # Get corresponding descriptors
                    if descs is not None:
                        indices = [kps.index(kp) for kp in keep_kps if kp in kps]
                        for idx in range(len(keep_kps)):
                            if idx < len(sorted_kps) and descs is not None:
                                orig_idx = (
                                    kps.index(sorted_kps[idx]) if sorted_kps[idx] in kps else -1
                                )
                                if orig_idx >= 0 and orig_idx < len(descs):
                                    all_descriptors.append(descs[orig_idx])

        # Stack descriptors
        descriptors = None
        if all_descriptors:
            descriptors = np.vstack(all_descriptors)

        return all_keypoints, descriptors

    def extract_from_pil(self, pil_image: Any) -> FrameFeatures:
        """Extract features from PIL Image.

        Args:
            pil_image: PIL Image object

        Returns:
            FrameFeatures
        """
        # Convert PIL to numpy
        image = np.array(pil_image)

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.extract(image)

    @property
    def config(self) -> ORBConfig:
        """Get current configuration."""
        return self._config
