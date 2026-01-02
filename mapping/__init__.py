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

from mapping.keyframe_selector import KeyframeSelector, KeyframeSelectionConfig, FramePose
from mapping.capture_replay import (
    CaptureReplay,
    ReplayConfig,
    CaptureFrame,
    CaptureSequence,
    CaptureIntegrityChecker,
    CaptureIntegrityResult,
)
from mapping.manifest import DatasetManifest, ManifestEntry, ManifestBuilder, SensorCalibration, validate_manifest
from mapping.splat_storage import SplatStorage, SplatScene, SplatMetadata
from mapping.obstacle_extraction import (
    ObstacleExtractor,
    ExtractionConfig,
    ExtractedObstacle,
    Point3D,
    MapMetadataResult,
    obstacles_to_navigation_map,
)
from mapping.decision_context import (
    MapContext,
    MapDecisionEvent,
    MapDecisionLogger,
    TimedQuery,
    add_map_context_to_decision,
    map_decision_logger,
)
from mapping.map_fusion import MapFusion, MapFusionConfig, MapFusionResult
from mapping.map_update import MapUpdateService, MapUpdateConfig
from mapping.safety_gates import (
    SafetyGateResult,
    SafetyGateConfig,
    GateCheckResult,
    MapUpdateGate,
    PlannerSafetyGate,
    validate_map_output,
    validate_slam_output,
)

__all__ = [
    # Keyframe selection
    "KeyframeSelector",
    "KeyframeSelectionConfig",
    "FramePose",
    # Capture replay
    "CaptureReplay",
    "ReplayConfig",
    "CaptureFrame",
    "CaptureSequence",
    "CaptureIntegrityChecker",
    "CaptureIntegrityResult",
    # Manifest
    "DatasetManifest",
    "ManifestEntry",
    "ManifestBuilder",
    "SensorCalibration",
    "validate_manifest",
    # Splat storage
    "SplatStorage",
    "SplatScene",
    "SplatMetadata",
    # Obstacle extraction
    "ObstacleExtractor",
    "ExtractionConfig",
    "ExtractedObstacle",
    "Point3D",
    "MapMetadataResult",
    "obstacles_to_navigation_map",
    # Decision context
    "MapContext",
    "MapDecisionEvent",
    "MapDecisionLogger",
    "TimedQuery",
    "add_map_context_to_decision",
    "map_decision_logger",
    # Map fusion/update
    "MapFusion",
    "MapFusionConfig",
    "MapFusionResult",
    "MapUpdateService",
    "MapUpdateConfig",
    # Safety gates
    "SafetyGateResult",
    "SafetyGateConfig",
    "GateCheckResult",
    "MapUpdateGate",
    "PlannerSafetyGate",
    "validate_map_output",
    "validate_slam_output",
]
