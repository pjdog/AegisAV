"""Feature Tracking.

Tracks features across consecutive frames using optical flow and descriptor matching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from pydantic import BaseModel

from vision.localization.features import FrameFeatures, FeaturePoint

logger = logging.getLogger(__name__)


class TrackingMethod(Enum):
    """Feature tracking method."""

    OPTICAL_FLOW = "optical_flow"  # Lucas-Kanade optical flow
    DESCRIPTOR_MATCH = "descriptor_match"  # ORB descriptor matching
    HYBRID = "hybrid"  # Optical flow with descriptor verification


class FeatureMatch(BaseModel):
    """A matched feature pair between two frames."""

    # Query (previous) frame
    query_idx: int
    query_pt: tuple[float, float]

    # Train (current) frame
    train_idx: int
    train_pt: tuple[float, float]

    # Match quality
    distance: float = 0.0  # Descriptor distance or flow error
    confidence: float = 1.0  # Match confidence (0-1)

    @property
    def motion(self) -> tuple[float, float]:
        """Get motion vector (dx, dy)."""
        return (
            self.train_pt[0] - self.query_pt[0],
            self.train_pt[1] - self.query_pt[1],
        )


class TrackingResult(BaseModel):
    """Result of tracking features between frames."""

    frame_id: int
    timestamp: float

    # Matched features
    matches: list[FeatureMatch]

    # Tracking statistics
    num_tracked: int = 0
    num_lost: int = 0
    num_new: int = 0

    # Quality metrics
    mean_motion: tuple[float, float] = (0.0, 0.0)
    tracking_confidence: float = 0.0
    tracking_time_ms: float = 0.0

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    def get_matched_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Get matched point arrays.

        Returns:
            Tuple of (prev_pts, curr_pts) as Nx2 arrays
        """
        if not self.matches:
            empty = np.array([]).reshape(0, 2)
            return empty, empty

        prev_pts = np.array([m.query_pt for m in self.matches])
        curr_pts = np.array([m.train_pt for m in self.matches])
        return prev_pts, curr_pts


@dataclass
class TrackerConfig:
    """Configuration for feature tracker."""

    # Tracking method
    method: TrackingMethod = TrackingMethod.OPTICAL_FLOW

    # Optical flow parameters (Lucas-Kanade)
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_criteria: tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    lk_min_eig_threshold: float = 0.001

    # Descriptor matching parameters
    match_ratio_threshold: float = 0.7  # Lowe's ratio test
    max_descriptor_distance: float = 50.0

    # Bidirectional verification
    use_bidirectional: bool = True
    bidirectional_threshold: float = 1.0  # pixels

    # RANSAC for outlier rejection
    use_ransac: bool = True
    ransac_threshold: float = 3.0  # pixels
    ransac_confidence: float = 0.99

    # Minimum matches for valid tracking
    min_matches: int = 20


class FeatureTracker:
    """Tracks features across consecutive frames.

    Supports multiple tracking methods:
    - Optical flow (fast, good for small motion)
    - Descriptor matching (robust to large motion)
    - Hybrid (optical flow with descriptor verification)

    Example:
        tracker = FeatureTracker(config)

        # Track between frames
        result = tracker.track(prev_features, curr_features, prev_image, curr_image)
        print(f"Tracked {result.num_matches} features")
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        """Initialize feature tracker.

        Args:
            config: Tracker configuration
        """
        self._config = config or TrackerConfig()

        # Create BFMatcher for descriptor matching
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Tracking state
        self._prev_image: np.ndarray | None = None
        self._prev_features: FrameFeatures | None = None

        logger.info(f"FeatureTracker initialized (method={self._config.method.value})")

    def track(
        self,
        prev_features: FrameFeatures,
        curr_features: FrameFeatures,
        prev_image: np.ndarray | None = None,
        curr_image: np.ndarray | None = None,
    ) -> TrackingResult:
        """Track features between two frames.

        Args:
            prev_features: Features from previous frame
            curr_features: Features from current frame
            prev_image: Previous image (required for optical flow)
            curr_image: Current image (required for optical flow)

        Returns:
            TrackingResult with matched features
        """
        start_time = time.perf_counter()

        matches = []
        method = self._config.method

        # Choose tracking method
        if method == TrackingMethod.OPTICAL_FLOW:
            if prev_image is None or curr_image is None:
                logger.warning("Optical flow requires images, falling back to descriptor matching")
                method = TrackingMethod.DESCRIPTOR_MATCH
            else:
                matches = self._track_optical_flow(
                    prev_features, prev_image, curr_image
                )

        if method == TrackingMethod.DESCRIPTOR_MATCH:
            matches = self._track_descriptors(prev_features, curr_features)

        elif method == TrackingMethod.HYBRID:
            if prev_image is not None and curr_image is not None:
                matches = self._track_hybrid(
                    prev_features, curr_features, prev_image, curr_image
                )
            else:
                matches = self._track_descriptors(prev_features, curr_features)

        # Apply RANSAC outlier rejection if enabled
        if self._config.use_ransac and len(matches) >= 8:
            matches = self._ransac_filter(matches)

        # Calculate statistics
        num_tracked = len(matches)
        num_lost = prev_features.num_features - num_tracked

        # Calculate mean motion
        mean_motion = (0.0, 0.0)
        if matches:
            motions = [m.motion for m in matches]
            mean_motion = (
                sum(m[0] for m in motions) / len(motions),
                sum(m[1] for m in motions) / len(motions),
            )

        # Calculate tracking confidence
        tracking_confidence = num_tracked / max(prev_features.num_features, 1)

        tracking_time = (time.perf_counter() - start_time) * 1000

        return TrackingResult(
            frame_id=curr_features.frame_id,
            timestamp=curr_features.timestamp,
            matches=matches,
            num_tracked=num_tracked,
            num_lost=num_lost,
            num_new=curr_features.num_features - num_tracked,
            mean_motion=mean_motion,
            tracking_confidence=tracking_confidence,
            tracking_time_ms=tracking_time,
        )

    def _track_optical_flow(
        self,
        prev_features: FrameFeatures,
        prev_image: np.ndarray,
        curr_image: np.ndarray,
    ) -> list[FeatureMatch]:
        """Track using Lucas-Kanade optical flow."""
        if prev_features.num_features == 0:
            return []

        # Convert images to grayscale
        if len(prev_image.shape) == 3:
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_image

        if len(curr_image.shape) == 3:
            curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_image

        # Get previous points
        prev_pts = prev_features.get_keypoints_array().astype(np.float32)
        prev_pts = prev_pts.reshape(-1, 1, 2)

        # Forward optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=self._config.lk_win_size,
            maxLevel=self._config.lk_max_level,
            criteria=self._config.lk_criteria,
            minEigThreshold=self._config.lk_min_eig_threshold,
        )

        if curr_pts is None:
            return []

        # Bidirectional verification
        if self._config.use_bidirectional:
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray,
                prev_gray,
                curr_pts,
                None,
                winSize=self._config.lk_win_size,
                maxLevel=self._config.lk_max_level,
                criteria=self._config.lk_criteria,
            )

            if back_pts is not None:
                # Check bidirectional consistency
                diff = np.abs(prev_pts - back_pts).reshape(-1, 2).max(axis=1)
                status = status & (diff < self._config.bidirectional_threshold).reshape(-1, 1)

        # Build matches
        matches = []
        for i, (st, e) in enumerate(zip(status.flatten(), err.flatten())):
            if st == 1:
                prev_pt = tuple(prev_pts[i, 0])
                curr_pt = tuple(curr_pts[i, 0])

                matches.append(
                    FeatureMatch(
                        query_idx=i,
                        query_pt=prev_pt,
                        train_idx=i,  # Same index for optical flow
                        train_pt=curr_pt,
                        distance=float(e),
                        confidence=1.0 - min(float(e) / 50.0, 1.0),
                    )
                )

        return matches

    def _track_descriptors(
        self,
        prev_features: FrameFeatures,
        curr_features: FrameFeatures,
    ) -> list[FeatureMatch]:
        """Track using descriptor matching."""
        if (
            prev_features.descriptors is None
            or curr_features.descriptors is None
            or prev_features.num_features == 0
            or curr_features.num_features == 0
        ):
            return []

        # Match descriptors
        raw_matches = self._bf_matcher.knnMatch(
            prev_features.descriptors,
            curr_features.descriptors,
            k=2,
        )

        # Apply ratio test
        matches = []
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue

            m, n = match_pair
            if m.distance < self._config.match_ratio_threshold * n.distance:
                if m.distance < self._config.max_descriptor_distance:
                    prev_kp = prev_features.keypoints[m.queryIdx]
                    curr_kp = curr_features.keypoints[m.trainIdx]

                    matches.append(
                        FeatureMatch(
                            query_idx=m.queryIdx,
                            query_pt=(prev_kp.x, prev_kp.y),
                            train_idx=m.trainIdx,
                            train_pt=(curr_kp.x, curr_kp.y),
                            distance=m.distance,
                            confidence=1.0 - m.distance / self._config.max_descriptor_distance,
                        )
                    )

        return matches

    def _track_hybrid(
        self,
        prev_features: FrameFeatures,
        curr_features: FrameFeatures,
        prev_image: np.ndarray,
        curr_image: np.ndarray,
    ) -> list[FeatureMatch]:
        """Track using optical flow with descriptor verification."""
        # First, track with optical flow
        of_matches = self._track_optical_flow(prev_features, prev_image, curr_image)

        if not of_matches or prev_features.descriptors is None or curr_features.descriptors is None:
            return of_matches

        # Verify matches using descriptors
        verified_matches = []
        for match in of_matches:
            # Find nearest descriptor in current frame near the tracked point
            curr_pt = np.array(match.train_pt)

            # Search within a radius
            best_dist = float("inf")
            best_idx = -1

            for i, kp in enumerate(curr_features.keypoints):
                pt_dist = np.sqrt((kp.x - curr_pt[0]) ** 2 + (kp.y - curr_pt[1]) ** 2)
                if pt_dist < 10:  # Within 10 pixels
                    desc_dist = cv2.norm(
                        prev_features.descriptors[match.query_idx],
                        curr_features.descriptors[i],
                        cv2.NORM_HAMMING,
                    )
                    if desc_dist < best_dist:
                        best_dist = desc_dist
                        best_idx = i

            if best_idx >= 0 and best_dist < self._config.max_descriptor_distance:
                # Update match with verified point
                verified_matches.append(
                    FeatureMatch(
                        query_idx=match.query_idx,
                        query_pt=match.query_pt,
                        train_idx=best_idx,
                        train_pt=(
                            curr_features.keypoints[best_idx].x,
                            curr_features.keypoints[best_idx].y,
                        ),
                        distance=best_dist,
                        confidence=match.confidence * (1.0 - best_dist / self._config.max_descriptor_distance),
                    )
                )
            else:
                # Keep optical flow match if no descriptor match
                verified_matches.append(match)

        return verified_matches

    def _ransac_filter(self, matches: list[FeatureMatch]) -> list[FeatureMatch]:
        """Filter matches using RANSAC on fundamental matrix."""
        if len(matches) < 8:
            return matches

        prev_pts = np.array([m.query_pt for m in matches], dtype=np.float32)
        curr_pts = np.array([m.train_pt for m in matches], dtype=np.float32)

        # Find fundamental matrix with RANSAC
        _, mask = cv2.findFundamentalMat(
            prev_pts,
            curr_pts,
            cv2.FM_RANSAC,
            self._config.ransac_threshold,
            self._config.ransac_confidence,
        )

        if mask is None:
            return matches

        # Filter matches
        filtered = []
        for i, m in enumerate(matches):
            if mask[i]:
                filtered.append(m)

        return filtered

    def update(
        self,
        features: FrameFeatures,
        image: np.ndarray | None = None,
    ) -> TrackingResult | None:
        """Track from previous frame and update state.

        Convenience method that maintains internal state.

        Args:
            features: Features from current frame
            image: Current image (for optical flow)

        Returns:
            TrackingResult or None if no previous frame
        """
        result = None

        if self._prev_features is not None:
            result = self.track(
                self._prev_features,
                features,
                self._prev_image,
                image,
            )

        # Update state
        self._prev_features = features
        self._prev_image = image.copy() if image is not None else None

        return result

    def reset(self) -> None:
        """Reset tracker state."""
        self._prev_features = None
        self._prev_image = None
