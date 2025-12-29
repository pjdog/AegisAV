from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agent.client.action_executor import ExecutionResult, ExecutionState
from agent.client.edge_policy import apply_edge_config_to_vision_client, compute_anomaly_detected
from agent.client.feedback import build_feedback
from agent.client.vision_client import InspectionVisionResults, VisionClient, VisionClientConfig
from agent.edge_config import (
    EdgeComputeProfile,
    apply_edge_compute_update,
    default_edge_compute_config,
)
from vision.data_models import (
    BoundingBox,
    CameraState,
    CameraStatus,
    CaptureResult,
    Detection,
    DetectionClass,
    DetectionResult,
)
from vision.models.yolo_detector import MockYOLODetector


def _make_detection(
    image_path: Path,
    defect: bool,
    confidence: float,
    severity: float,
) -> DetectionResult:
    detections = []
    if defect:
        detections = [
            Detection(
                detection_class=DetectionClass.CRACK,
                confidence=confidence,
                bounding_box=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2),
                severity=severity,
            )
        ]
    return DetectionResult(
        timestamp=datetime.now(),
        detections=detections,
        image_path=image_path,
        model_name="mock",
        inference_time_ms=5.0,
    )


def _make_capture(image_path: Path) -> CaptureResult:
    return CaptureResult(
        success=True,
        timestamp=datetime.now(),
        image_path=image_path,
        camera_state=CameraState(status=CameraStatus.READY),
        metadata={
            "vehicle_state": {"position": {"latitude": 0.0, "longitude": 0.0, "altitude_msl": 0.0}}
        },
    )


def test_default_edge_profiles_smoke():
    for profile in EdgeComputeProfile:
        cfg = default_edge_compute_config(profile)
        assert cfg.profile == profile

    assert default_edge_compute_config(EdgeComputeProfile.FC_ONLY).vision_enabled is False
    assert default_edge_compute_config(EdgeComputeProfile.FC_ONLY).uplink.send_images is False


def test_apply_edge_compute_update_switch_profile_and_override():
    current = default_edge_compute_config(EdgeComputeProfile.SBC_CPU)

    updated = apply_edge_compute_update(current, {"profile": "mcu_heuristic"})
    assert updated.profile == EdgeComputeProfile.MCU_HEURISTIC
    assert updated.uplink.summary_only is True
    assert updated.uplink.send_images is False

    # Nested partial update should deep-merge
    updated2 = apply_edge_compute_update(updated, {"uplink": {"summary_only": False}})
    assert updated2.profile == EdgeComputeProfile.MCU_HEURISTIC
    assert updated2.uplink.summary_only is False


def test_apply_edge_config_to_vision_client_updates_threshold_and_cadence():
    detector = MockYOLODetector(confidence_threshold=0.2)
    vc = VisionClient(
        camera=None,
        detector=detector,
        image_manager=None,
        config=VisionClientConfig(enabled=False),
    )

    edge = default_edge_compute_config(EdgeComputeProfile.SBC_ACCEL)
    apply_edge_config_to_vision_client(vc, edge)

    assert vc.config.enabled == edge.vision_enabled
    assert vc.config.max_captures_per_inspection == edge.max_captures_per_inspection
    assert detector.confidence_threshold == edge.client_confidence_threshold


def test_anomaly_gate_n_of_m():
    edge = default_edge_compute_config(EdgeComputeProfile.SBC_CPU)  # n=2 of m=5

    img = Path("img.png")
    detections = [
        _make_detection(img, defect=True, confidence=0.7, severity=0.6),
        _make_detection(img, defect=True, confidence=0.65, severity=0.5),
        _make_detection(img, defect=False, confidence=0.0, severity=0.0),
        _make_detection(img, defect=True, confidence=0.4, severity=0.2),  # below threshold
        _make_detection(img, defect=False, confidence=0.0, severity=0.0),
    ]
    results = InspectionVisionResults(
        asset_id="asset-1", captures=[_make_capture(img)], detections=detections
    )

    assert compute_anomaly_detected(results, edge) is True


def test_anomaly_gate_severity_override():
    edge = default_edge_compute_config(EdgeComputeProfile.SBC_ACCEL)  # has severity override

    img = Path("img.png")
    detections = [
        _make_detection(img, defect=True, confidence=0.9, severity=0.8),  # override hit
    ]
    results = InspectionVisionResults(
        asset_id="asset-1", captures=[_make_capture(img)], detections=detections
    )

    assert compute_anomaly_detected(results, edge) is True


@pytest.mark.parametrize(
    ("profile", "expect_detection", "expect_image"),
    [
        (EdgeComputeProfile.FC_ONLY, False, False),
        (EdgeComputeProfile.MCU_HEURISTIC, False, False),
        (EdgeComputeProfile.MCU_TINY_CNN, True, False),
        (EdgeComputeProfile.SBC_CPU, True, True),
    ],
)
def test_feedback_payload_shaping(
    profile: EdgeComputeProfile, expect_detection: bool, expect_image: bool
):
    edge = default_edge_compute_config(profile)

    img = Path("img.png")
    results = InspectionVisionResults(
        asset_id="asset-1",
        captures=[_make_capture(img)],
        detections=[_make_detection(img, defect=True, confidence=0.9, severity=0.8)],
        defects_detected=1,
        max_confidence=0.9,
        max_severity=0.8,
        best_detection_image=img,
    )

    decision = {"decision_id": "dec-1", "action": "inspect", "parameters": {"asset_id": "asset-1"}}
    exec_result = ExecutionResult(
        decision_id="dec-1",
        action="inspect",
        state=ExecutionState.COMPLETED,
        message="",
        duration_s=1.0,
    )

    payload = build_feedback(decision, exec_result, results, edge=edge)

    if profile == EdgeComputeProfile.FC_ONLY:
        assert payload["anomaly_detected"] is False
        assert "inspection_data" not in payload
        return

    assert "inspection_data" in payload
    vision = payload["inspection_data"]["vision"]

    assert ("best_detection" in vision) is expect_detection
    assert (vision.get("best_image_path") is not None) is expect_image
