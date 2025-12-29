"""
Server-Side Vision Models

Data models for vision observations, defects, and analysis results.
Used by the server for tracking and processing vision data.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autonomy.vehicle_state import Position


class DefectType(str, Enum):
    """
    Detailed defect classification for server-side processing.

    More granular than client-side DetectionClass.
    """

    # Structural defects
    CRACK_HAIRLINE = "crack_hairline"
    CRACK_MODERATE = "crack_moderate"
    CRACK_SEVERE = "crack_severe"
    CORROSION_SURFACE = "corrosion_surface"
    CORROSION_DEEP = "corrosion_deep"
    STRUCTURAL_DAMAGE_MINOR = "structural_damage_minor"
    STRUCTURAL_DAMAGE_MAJOR = "structural_damage_major"
    DEFORMATION = "deformation"

    # Surface issues
    DISCOLORATION = "discoloration"
    PAINT_DEGRADATION = "paint_degradation"
    RUST_LIGHT = "rust_light"
    RUST_HEAVY = "rust_heavy"

    # Component issues
    MISSING_BOLT = "missing_bolt"
    LOOSE_COMPONENT = "loose_component"
    DAMAGED_INSULATOR = "damaged_insulator"
    BROKEN_WIRE = "broken_wire"

    # Environmental
    VEGETATION_LIGHT = "vegetation_light"
    VEGETATION_HEAVY = "vegetation_heavy"
    BIRD_NEST = "bird_nest"
    DEBRIS = "debris"

    # Normal
    NORMAL = "normal"
    UNKNOWN = "unknown"


class CameraMetadata(BaseModel, frozen=True):
    """
    Metadata about camera and capture conditions.

    Stored with vision observations for analysis and debugging.

    Attributes:
        camera_type (str): Camera source identifier (e.g., ``simulated``).
        resolution (tuple[int, int]): Image resolution in pixels.
        capture_format (str): Color format (e.g., ``RGB``).
        focal_length_mm (float | None): Lens focal length in millimeters.
        sensor_size_mm (tuple[float, float] | None): Sensor size in millimeters.
        iso (int | None): ISO setting.
        shutter_speed (float | None): Shutter speed in seconds.
        aperture (float | None): Aperture value (f-stop).
    """

    camera_type: str = "unknown"  # simulated, mavlink, opencv, etc.
    resolution: tuple[int, int] = (1920, 1080)
    capture_format: str = "RGB"
    focal_length_mm: float | None = None
    sensor_size_mm: tuple[float, float] | None = None
    iso: int | None = None
    shutter_speed: float | None = None
    aperture: float | None = None


class VisionObservation(BaseModel):
    """
    Record of a vision observation during inspection.

    Tracks what was seen, where, when, and the analysis results.
    Used for building inspection history and anomaly tracking.

    Attributes:
        observation_id (str): Unique observation ID.
        asset_id (str): Asset being inspected.
        timestamp (datetime): Observation timestamp.
        image_path (Path | None): Image path (:class:`pathlib.Path`).
        camera_metadata (CameraMetadata | None): Capture metadata.
        position (Position | None): Drone position (:class:`autonomy.vehicle_state.Position`).
        altitude_agl (float | None): Altitude above ground level (meters).
        distance_to_asset (float | None): Distance to asset (meters).
        heading_deg (float | None): Heading in degrees.
        detections (list[dict[str, Any]]): Detection payloads.
        max_confidence (float): Maximum confidence across detections.
        max_severity (float): Maximum severity across detections.
        defect_detected (bool): Whether any defect was detected.
        model_name (str): Model identifier.
        inference_time_ms (float): Inference latency in milliseconds.
        processed_on_server (bool): Whether analysis ran on server.
        anomaly_created (bool): Whether an anomaly was created.
        anomaly_id (str | None): Created anomaly ID.
    """

    observation_id: str = Field(description="Unique observation ID")
    asset_id: str = Field(description="Asset being inspected")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Capture information
    image_path: Path | None = None
    camera_metadata: CameraMetadata | None = None

    # Location information
    position: Position | None = Field(default=None, description="Drone position during capture")
    altitude_agl: float | None = Field(default=None, description="Altitude above ground level")
    distance_to_asset: float | None = Field(default=None, description="Distance to asset in meters")
    heading_deg: float | None = Field(default=None, description="Drone heading in degrees")

    # Detection results
    detections: list[dict[str, Any]] = Field(
        default_factory=list, description="List of detection dictionaries"
    )
    max_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    defect_detected: bool = False

    # Analysis metadata
    model_name: str = "unknown"
    inference_time_ms: float = 0.0
    processed_on_server: bool = False

    # Anomaly tracking
    anomaly_created: bool = Field(
        default=False, description="Whether an anomaly was created from this observation"
    )
    anomaly_id: str | None = Field(default=None, description="ID of created anomaly")


class InspectionVisionSummary(BaseModel, frozen=True):
    """
    Summary of vision observations for a complete inspection.

    Aggregates results from all captures during an inspection.

    Attributes:
        asset_id (str): Asset being inspected.
        inspection_timestamp (datetime): Inspection timestamp.
        total_captures (int): Total captures.
        total_detections (int): Total detections.
        defects_found (int): Count of defect detections.
        max_confidence (float): Maximum confidence across captures.
        max_severity (float): Maximum severity across captures.
        observation_ids (list[str]): Observation IDs included.
        defect_types_detected (list[str]): Detected defect class names.
        anomaly_created (bool): Whether an anomaly was created.
        anomaly_id (str | None): Created anomaly ID.
    """

    asset_id: str
    inspection_timestamp: datetime
    total_captures: int = 0
    total_detections: int = 0
    defects_found: int = 0
    max_confidence: float = 0.0
    max_severity: float = 0.0
    observation_ids: list[str] = Field(default_factory=list)
    defect_types_detected: list[str] = Field(default_factory=list)
    anomaly_created: bool = False
    anomaly_id: str | None = None


class VisionAnalysisRequest(BaseModel, frozen=True):
    """
    Request for server-side vision analysis.

    Sent from client to server for detailed processing.

    Attributes:
        request_id (str): Request ID.
        asset_id (str): Asset ID.
        image_path (Path): Image path (:class:`pathlib.Path`).
        timestamp (datetime): Request timestamp.
        vehicle_state (dict[str, Any]): Vehicle state snapshot.
        quick_detection_result (dict[str, Any] | None): Client detection payload.
        priority (int): Analysis priority (lower is higher).
    """

    request_id: str
    asset_id: str
    image_path: Path
    timestamp: datetime = Field(default_factory=datetime.now)
    vehicle_state: dict[str, Any] = Field(default_factory=dict)
    quick_detection_result: dict[str, Any] | None = Field(
        default=None, description="Quick detection result from client"
    )
    priority: int = Field(default=10, description="Analysis priority (lower = higher priority)")


class VisionAnalysisResponse(BaseModel, frozen=True):
    """
    Response from server-side vision analysis.

    Contains detailed detection results and any anomalies created.

    Attributes:
        request_id (str): Request ID.
        success (bool): Whether analysis succeeded.
        timestamp (datetime): Response timestamp.
        observation_id (str | None): Observation ID created.
        detections (list[dict[str, Any]]): Detection payloads.
        defect_detected (bool): Whether any defect was detected.
        max_confidence (float): Maximum confidence across detections.
        max_severity (float): Maximum severity across detections.
        anomaly_created (bool): Whether an anomaly was created.
        anomaly_id (str | None): Created anomaly ID.
        processing_time_ms (float): Processing latency in milliseconds.
        error_message (str | None): Error details, if any.
    """

    request_id: str
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    observation_id: str | None = None
    detections: list[dict[str, Any]] = Field(default_factory=list)
    defect_detected: bool = False
    max_confidence: float = 0.0
    max_severity: float = 0.0
    anomaly_created: bool = False
    anomaly_id: str | None = None
    processing_time_ms: float = 0.0
    error_message: str | None = None
