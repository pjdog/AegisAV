"""
Tests for Vision Data Models

Validates all vision data structures including camera state,
detection results, and capture results.
"""

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from vision.data_models import (
    BoundingBox,
    CameraState,
    CameraStatus,
    CaptureResult,
    Detection,
    DetectionClass,
    DetectionResult,
)


class TestCameraStatus:
    """Tests for CameraStatus enum."""

    def test_camera_status_values(self):
        """Test that CameraStatus has expected values."""
        assert CameraStatus.INITIALIZING == "initializing"
        assert CameraStatus.READY == "ready"
        assert CameraStatus.CAPTURING == "capturing"
        assert CameraStatus.ERROR == "error"
        assert CameraStatus.OFFLINE == "offline"


class TestDetectionClass:
    """Tests for DetectionClass enum."""

    def test_detection_class_defects(self):
        """Test defect detection classes."""
        assert DetectionClass.CRACK == "crack"
        assert DetectionClass.CORROSION == "corrosion"
        assert DetectionClass.STRUCTURAL_DAMAGE == "structural_damage"

    def test_detection_class_normal(self):
        """Test normal/unknown classes."""
        assert DetectionClass.NORMAL == "normal"
        assert DetectionClass.UNKNOWN == "unknown"


class TestCameraState:
    """Tests for CameraState model."""

    def test_camera_state_creation(self):
        """Test creating a basic camera state."""
        state = CameraState(
            status=CameraStatus.READY,
            resolution=(1920, 1080),
            capture_format="RGB",
            total_captures=5,
        )

        assert state.status == CameraStatus.READY
        assert state.resolution == (1920, 1080)
        assert state.capture_format == "RGB"
        assert state.total_captures == 5
        assert state.last_capture_time is None
        assert state.error_message is None

    def test_camera_state_with_error(self):
        """Test camera state with error."""
        state = CameraState(status=CameraStatus.ERROR, error_message="Camera initialization failed")

        assert state.status == CameraStatus.ERROR
        assert state.error_message == "Camera initialization failed"

    def test_camera_state_immutable(self):
        """Test that CameraState is immutable."""
        state = CameraState(status=CameraStatus.READY)

        with pytest.raises(ValidationError):
            state.status = CameraStatus.ERROR  # type: ignore

    def test_camera_state_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now()
        state = CameraState(
            timestamp=now,
            status=CameraStatus.READY,
            resolution=(1920, 1080),
            total_captures=10,
        )

        data = state.to_dict()

        assert data["status"] == "ready"
        assert data["resolution"] == [1920, 1080]
        assert data["total_captures"] == 10
        assert "timestamp" in data


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_bounding_box_creation(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        assert bbox.x_min == 0.2
        assert bbox.y_min == 0.3
        assert bbox.x_max == 0.8
        assert bbox.y_max == 0.9

    def test_bounding_box_properties(self):
        """Test bounding box computed properties."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        assert bbox.width == pytest.approx(0.6)
        assert bbox.height == pytest.approx(0.6)
        assert bbox.area == pytest.approx(0.36)
        assert bbox.center == pytest.approx((0.5, 0.6))

    def test_bounding_box_validation_ranges(self):
        """Test that coordinates must be in [0, 1] range."""
        # Valid
        BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

        # Invalid - out of range
        with pytest.raises(ValidationError):
            BoundingBox(x_min=-0.1, y_min=0.0, x_max=0.5, y_max=0.5)

        with pytest.raises(ValidationError):
            BoundingBox(x_min=0.0, y_min=0.0, x_max=1.5, y_max=0.5)

    def test_bounding_box_validation_ordering(self):
        """Test that max > min for coordinates."""
        # Valid
        BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        # Invalid - x_max <= x_min
        with pytest.raises(ValidationError):
            BoundingBox(x_min=0.8, y_min=0.3, x_max=0.2, y_max=0.9)

        # Invalid - y_max <= y_min
        with pytest.raises(ValidationError):
            BoundingBox(x_min=0.2, y_min=0.9, x_max=0.8, y_max=0.3)

    def test_bounding_box_immutable(self):
        """Test that BoundingBox is immutable."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        with pytest.raises(ValidationError):
            bbox.x_min = 0.1  # type: ignore


class TestDetection:
    """Tests for Detection model."""

    def test_detection_creation(self):
        """Test creating a detection."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)
        detection = Detection(
            detection_class=DetectionClass.CRACK,
            confidence=0.85,
            bounding_box=bbox,
            severity=0.6,
        )

        assert detection.detection_class == DetectionClass.CRACK
        assert detection.confidence == 0.85
        assert detection.severity == 0.6
        assert detection.bounding_box == bbox

    def test_detection_is_defect(self):
        """Test is_defect property."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        # Defect detection
        defect = Detection(detection_class=DetectionClass.CRACK, confidence=0.85, bounding_box=bbox)
        assert defect.is_defect is True

        # Normal detection
        normal = Detection(
            detection_class=DetectionClass.NORMAL, confidence=0.95, bounding_box=bbox
        )
        assert normal.is_defect is False

        # Unknown detection
        unknown = Detection(
            detection_class=DetectionClass.UNKNOWN, confidence=0.5, bounding_box=bbox
        )
        assert unknown.is_defect is False

    def test_detection_confidence_validation(self):
        """Test that confidence must be in [0, 1] range."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        # Valid
        Detection(detection_class=DetectionClass.CRACK, confidence=0.0, bounding_box=bbox)
        Detection(detection_class=DetectionClass.CRACK, confidence=1.0, bounding_box=bbox)

        # Invalid
        with pytest.raises(ValidationError):
            Detection(detection_class=DetectionClass.CRACK, confidence=-0.1, bounding_box=bbox)

        with pytest.raises(ValidationError):
            Detection(detection_class=DetectionClass.CRACK, confidence=1.5, bounding_box=bbox)


class TestDetectionResult:
    """Tests for DetectionResult model."""

    def test_detection_result_empty(self):
        """Test empty detection result."""
        result = DetectionResult()

        assert result.detections == []
        assert result.detected_any is False
        assert result.detected_defects == []
        assert result.max_confidence == 0.0
        assert result.max_severity == 0.0
        assert result.needs_detailed_analysis is False

    def test_detection_result_with_detections(self):
        """Test detection result with multiple detections."""
        bbox1 = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3)
        bbox2 = BoundingBox(x_min=0.5, y_min=0.5, x_max=0.7, y_max=0.7)

        detection1 = Detection(
            detection_class=DetectionClass.CRACK,
            confidence=0.85,
            bounding_box=bbox1,
            severity=0.6,
        )
        detection2 = Detection(
            detection_class=DetectionClass.CORROSION,
            confidence=0.72,
            bounding_box=bbox2,
            severity=0.4,
        )

        result = DetectionResult(
            detections=[detection1, detection2],
            model_name="yolov8n",
            inference_time_ms=45.2,
        )

        assert result.detected_any is True
        assert len(result.detections) == 2
        assert len(result.detected_defects) == 2
        assert result.max_confidence == 0.85
        assert result.max_severity == 0.6
        assert result.model_name == "yolov8n"
        assert result.inference_time_ms == 45.2

    def test_detection_result_needs_detailed_analysis(self):
        """Test needs_detailed_analysis property."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        # Low confidence, low severity - no detailed analysis needed
        low_detection = Detection(
            detection_class=DetectionClass.CRACK,
            confidence=0.3,
            bounding_box=bbox,
            severity=0.2,
        )
        result_low = DetectionResult(detections=[low_detection])
        assert result_low.needs_detailed_analysis is False

        # High confidence - needs detailed analysis
        high_conf = Detection(
            detection_class=DetectionClass.CRACK,
            confidence=0.7,
            bounding_box=bbox,
            severity=0.2,
        )
        result_conf = DetectionResult(detections=[high_conf])
        assert result_conf.needs_detailed_analysis is True

        # High severity - needs detailed analysis
        high_sev = Detection(
            detection_class=DetectionClass.CRACK,
            confidence=0.3,
            bounding_box=bbox,
            severity=0.5,
        )
        result_sev = DetectionResult(detections=[high_sev])
        assert result_sev.needs_detailed_analysis is True

    def test_detection_result_filters_normal(self):
        """Test that detected_defects filters out normal/unknown."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)

        detections = [
            Detection(detection_class=DetectionClass.CRACK, confidence=0.85, bounding_box=bbox),
            Detection(detection_class=DetectionClass.NORMAL, confidence=0.95, bounding_box=bbox),
            Detection(detection_class=DetectionClass.CORROSION, confidence=0.75, bounding_box=bbox),
            Detection(detection_class=DetectionClass.UNKNOWN, confidence=0.50, bounding_box=bbox),
        ]

        result = DetectionResult(detections=detections)

        assert len(result.detections) == 4
        assert len(result.detected_defects) == 2  # Only CRACK and CORROSION
        assert all(d.is_defect for d in result.detected_defects)


class TestCaptureResult:
    """Tests for CaptureResult model."""

    def test_capture_result_success(self):
        """Test successful capture result."""
        camera_state = CameraState(status=CameraStatus.READY, total_captures=5)
        image_path = Path("/tmp/test_image.png")

        result = CaptureResult(success=True, image_path=image_path, camera_state=camera_state)

        assert result.success is True
        assert result.image_path == image_path
        assert result.camera_state == camera_state
        assert result.error_message is None
        assert result.is_valid is True

    def test_capture_result_failure(self):
        """Test failed capture result."""
        camera_state = CameraState(status=CameraStatus.ERROR, error_message="Hardware error")

        result = CaptureResult(
            success=False, camera_state=camera_state, error_message="Capture failed"
        )

        assert result.success is False
        assert result.image_path is None
        assert result.error_message == "Capture failed"
        assert result.is_valid is False

    def test_capture_result_with_metadata(self):
        """Test capture result with metadata."""
        camera_state = CameraState(status=CameraStatus.READY)
        metadata = {
            "vehicle_position": {"lat": 37.7749, "lon": -122.4194},
            "altitude_msl": 150.0,
        }

        result = CaptureResult(
            success=True,
            image_path=Path("/tmp/test.png"),
            camera_state=camera_state,
            metadata=metadata,
        )

        assert result.metadata == metadata
        assert result.metadata["altitude_msl"] == 150.0
