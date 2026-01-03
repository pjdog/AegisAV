"""SLAM and Gaussian Splatting mapping pipeline.

This module provides:
- Keyframe selection for SLAM/splat training
- Dataset capture and replay tools
- Manifest generation for training pipelines
- Splat artifact storage with versioning
- Obstacle extraction from fused maps
- Decision context with map usage metadata
- Safety gating for map updates and planning
"""

from mapping.capture_replay import (
    CaptureFrame,
    CaptureIntegrityChecker,
    CaptureIntegrityResult,
    CaptureReplay,
    CaptureSequence,
    ReplayConfig,
)
from mapping.decision_context import (
    MapContext,
    MapDecisionEvent,
    MapDecisionLogger,
    TimedQuery,
    add_map_context_to_decision,
    map_decision_logger,
)
from mapping.keyframe_selector import FramePose, KeyframeSelectionConfig, KeyframeSelector
from mapping.manifest import (
    DatasetManifest,
    ManifestBuilder,
    ManifestEntry,
    SensorCalibration,
    validate_manifest,
)
from mapping.map_fusion import MapFusion, MapFusionConfig, MapFusionResult
from mapping.map_update import MapUpdateConfig, MapUpdateService
from mapping.obstacle_extraction import (
    ExtractedObstacle,
    ExtractionConfig,
    MapMetadataResult,
    ObstacleExtractor,
    Point3D,
    obstacles_to_navigation_map,
)
from mapping.safety_gates import (
    GateCheckResult,
    MapUpdateGate,
    PlannerSafetyGate,
    SafetyGateConfig,
    SafetyGateResult,
    validate_map_output,
    validate_slam_output,
)
from mapping.splat_storage import SplatMetadata, SplatScene, SplatStorage

__all__ = [
    "CaptureFrame",
    "CaptureIntegrityChecker",
    "CaptureIntegrityResult",
    # Capture replay
    "CaptureReplay",
    "CaptureSequence",
    # Manifest
    "DatasetManifest",
    "ExtractedObstacle",
    "ExtractionConfig",
    "FramePose",
    "GateCheckResult",
    "KeyframeSelectionConfig",
    # Keyframe selection
    "KeyframeSelector",
    "ManifestBuilder",
    "ManifestEntry",
    # Decision context
    "MapContext",
    "MapDecisionEvent",
    "MapDecisionLogger",
    # Map fusion/update
    "MapFusion",
    "MapFusionConfig",
    "MapFusionResult",
    "MapMetadataResult",
    "MapUpdateConfig",
    "MapUpdateGate",
    "MapUpdateService",
    # Obstacle extraction
    "ObstacleExtractor",
    "PlannerSafetyGate",
    "Point3D",
    "ReplayConfig",
    "SafetyGateConfig",
    # Safety gates
    "SafetyGateResult",
    "SensorCalibration",
    "SplatMetadata",
    "SplatScene",
    # Splat storage
    "SplatStorage",
    "TimedQuery",
    "add_map_context_to_decision",
    "map_decision_logger",
    "obstacles_to_navigation_map",
    "validate_manifest",
    "validate_map_output",
    "validate_slam_output",
]
