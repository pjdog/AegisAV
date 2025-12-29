"""Vision Data Models.

Core data structures for computer vision subsystem.
Includes camera state, detection results, and capture metadata.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class CameraStatus(str, Enum):
    """Camera operational status."""

    INITIALIZING = "initializing"
    READY = "ready"
    CAPTURING = "capturing"
    ERROR = "error"
    OFFLINE = "offline"


class DetectionClass(str, Enum):
    """Object/defect classes that can be detected."""

    # Infrastructure defects
    CRACK = "crack"
    CORROSION = "corrosion"
    STRUCTURAL_DAMAGE = "structural_damage"
    DISCOLORATION = "discoloration"
    DEFORMATION = "deformation"
    VEGETATION_OVERGROWTH = "vegetation_overgrowth"
    HOT_SPOT = "hot_spot"
    DAMAGE = "damage"
    WEAR = "wear"
    STAIN = "stain"
    LEAK = "leak"
    VEGETATION = "vegetation"

    # Equipment issues
    MISSING_COMPONENT = "missing_component"
    DAMAGED_INSULATOR = "damaged_insulator"
    RUST = "rust"

    # Environmental
    BIRD_NEST = "bird_nest"
    DEBRIS = "debris"

    # Normal/OK
    NORMAL = "normal"
    UNKNOWN = "unknown"


class CameraState(BaseModel, frozen=True):
    """Current state of the camera sensor.

    Immutable snapshot of camera status at a point in time.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    status: CameraStatus
    resolution: tuple[int, int] = Field(default=(1920, 1080))  # width, height
    capture_format: str = "RGB"  # RGB, RGBA, grayscale
    total_captures: int = 0
    last_capture_time: datetime | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            dict[str, Any]: Serialized camera state payload.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "resolution": list(self.resolution),
            "capture_format": self.capture_format,
            "total_captures": self.total_captures,
            "last_capture_time": self.last_capture_time.isoformat()
            if self.last_capture_time
            else None,
            "error_message": self.error_message,
        }


class BoundingBox(BaseModel, frozen=True):
    """Bounding box for detected object.

    Coordinates are normalized to [0, 1] range relative to image dimensions.
    """

    x_min: float = Field(ge=0.0, le=1.0, description="Left edge (normalized)")
    y_min: float = Field(ge=0.0, le=1.0, description="Top edge (normalized)")
    x_max: float = Field(ge=0.0, le=1.0, description="Right edge (normalized)")
    y_max: float = Field(ge=0.0, le=1.0, description="Bottom edge (normalized)")

    @field_validator("x_max")
    @classmethod
    def validate_x_max(cls, v: float, info: ValidationInfo) -> float:
        """Ensure x_max > x_min."""
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def validate_y_max(cls, v: float, info: ValidationInfo) -> float:
        """Ensure y_max > y_min."""
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return v

    @property
    def width(self) -> float:
        """Bounding box width (normalized)."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Bounding box height (normalized)."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Bounding box area (normalized)."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Bounding box center (x, y) normalized coordinates."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )


class Detection(BaseModel, frozen=True):
    """Single object detection result.

    Represents one detected object/defect in an image.
    """

    detection_class: DetectionClass
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox
    severity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Defect severity (0=minor, 1=critical)"
    )
    raw_class_name: str | None = Field(
        default=None, description="Original class name from detector model"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_defect(self) -> bool:
        """Whether this detection represents a defect (not normal/unknown)."""
        return self.detection_class not in (DetectionClass.NORMAL, DetectionClass.UNKNOWN)


class DetectionResult(BaseModel, frozen=True):
    """Complete detection result from analyzing an image.

    Contains all detections found in a single image.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    detections: list[Detection] = Field(default_factory=list)
    image_path: Path | None = None
    model_name: str = "unknown"
    inference_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def detected_any(self) -> bool:
        """Whether any objects were detected.

        Returns:
            bool: True if any detections exist.
        """
        return len(self.detections) > 0

    @property
    def detected_defects(self) -> list[Detection]:
        """Get only defect detections (excluding normal/unknown).

        Returns:
            list[Detection]: Defect detections.
        """
        return [d for d in self.detections if d.is_defect]

    @property
    def max_confidence(self) -> float:
        """Maximum confidence across all detections.

        Returns:
            float: Maximum confidence value.
        """
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)

    @property
    def max_severity(self) -> float:
        """Maximum severity across all defect detections.

        Returns:
            float: Maximum severity value.
        """
        defects = self.detected_defects
        if not defects:
            return 0.0
        return max(d.severity for d in defects)

    @property
    def needs_detailed_analysis(self) -> bool:
        """Whether this result should be sent for detailed server-side analysis.

        True if any detection has confidence > 0.4 or severity > 0.3.
        """
        for detection in self.detections:
            if detection.confidence > 0.4 or detection.severity > 0.3:
                return True
        return False


class CaptureResult(BaseModel, frozen=True):
    """Result from a camera capture operation.

    Contains captured image information and camera state.
    """

    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    image_path: Path | None = None
    camera_state: CameraState
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether the capture was successful and has a valid image path.

        Returns:
            bool: True if the capture succeeded and has an image path.
        """
        return self.success and self.image_path is not None
