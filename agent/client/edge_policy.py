"""Edge Policy (Client-Side).

Applies edge-compute profiles (configured on the server) to the on-drone client:
- adjusts capture cadence and simulated inference latency
- gates anomaly flags from client detections
- shapes the feedback payload sent to the server
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.client.vision_client import InspectionVisionResults, VisionClient, VisionClientConfig
from agent.edge_config import AnomalyGateMode, EdgeComputeConfig
from vision.data_models import CaptureResult, DetectionResult

logger = logging.getLogger(__name__)


def apply_edge_config_to_vision_client(
    vision_client: VisionClient, edge: EdgeComputeConfig
) -> None:
    """Apply edge config knobs to an existing VisionClient instance."""
    vision_client.config = VisionClientConfig(
        capture_interval_s=edge.capture_interval_s,
        max_captures_per_inspection=edge.max_captures_per_inspection,
        simulated_inference_delay_ms=edge.simulated_inference_delay_ms,
        enabled=edge.vision_enabled,
    )

    detector = vision_client.detector
    if detector is not None and hasattr(detector, "confidence_threshold"):
        try:
            detector.confidence_threshold = edge.client_confidence_threshold  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as e:
            logger.debug("detector_threshold_update_failed: %s", e)


def _detection_meets_gate(detection: DetectionResult, edge: EdgeComputeConfig) -> bool:
    """Check if a detection meets the anomaly gate thresholds.

    Args:
        detection: Detection result to evaluate.
        edge: Edge compute configuration with gate thresholds.

    Returns:
        True if detection meets minimum confidence and severity thresholds.
    """
    gate = edge.anomaly_gate
    if not detection.detected_defects:
        return False
    return (
        detection.max_confidence >= gate.min_confidence
        and detection.max_severity >= gate.min_severity
    )


def compute_anomaly_detected(results: InspectionVisionResults, edge: EdgeComputeConfig) -> bool:
    """Convert client detections into a single anomaly flag per inspection."""
    if not edge.vision_enabled:
        return False

    gate = edge.anomaly_gate

    if gate.mode == AnomalyGateMode.ANY:
        return any(_detection_meets_gate(d, edge) for d in results.detections)

    if gate.mode == AnomalyGateMode.N_OF_M:
        n = gate.n or 0
        m = gate.m or 0
        subset = results.detections[: min(m, len(results.detections))]
        return sum(1 for d in subset if _detection_meets_gate(d, edge)) >= n

    # SEVERITY_OVERRIDE
    override = gate.min_severity_override or 1.0
    for detection in results.detections:
        if not detection.detected_defects:
            continue
        if detection.max_confidence >= gate.min_confidence and detection.max_severity >= override:
            return True

    # Fallback gate (either N-of-M if provided, else ANY).
    if gate.n is not None and gate.m is not None:
        subset = results.detections[: min(gate.m, len(results.detections))]
        return sum(1 for d in subset if _detection_meets_gate(d, edge)) >= gate.n

    return any(_detection_meets_gate(d, edge) for d in results.detections)


def select_best_detection(results: InspectionVisionResults) -> DetectionResult | None:
    """Select the best detection from inspection results based on confidence.

    Args:
        results: Inspection vision results containing detections.

    Returns:
        The detection with highest confidence, or None if no detections.
    """
    defect_detections = [d for d in results.detections if d.detected_defects]
    if defect_detections:
        return max(defect_detections, key=lambda d: d.max_confidence)
    if results.detections:
        return max(results.detections, key=lambda d: d.max_confidence)
    return None


def select_best_image_path(
    best_detection: DetectionResult | None, edge: EdgeComputeConfig
) -> Path | None:
    """Select the image path from the best detection if image upload is enabled.

    Args:
        best_detection: The best detection result, or None.
        edge: Edge compute configuration with uplink settings.

    Returns:
        Path to the best detection image, or None if images disabled or no detection.
    """
    if not edge.uplink.send_images or edge.uplink.max_images <= 0:
        return None
    if best_detection is None:
        return None
    return best_detection.image_path


def _find_capture_for_image(
    captures: list[CaptureResult], image_path: Path | None
) -> CaptureResult | None:
    """Find the capture result that produced the given image.

    Args:
        captures: List of capture results to search.
        image_path: Path of the image to find.

    Returns:
        The matching capture result, or None if not found.
    """
    if image_path is None:
        return None
    return next((c for c in captures if c.image_path == image_path), None)


def build_inspection_data(
    results: InspectionVisionResults, edge: EdgeComputeConfig
) -> dict[str, Any]:
    """Build the `inspection_data` payload attached to DecisionFeedback."""
    best_detection = select_best_detection(results)
    best_image_path = select_best_image_path(best_detection, edge)
    best_capture = _find_capture_for_image(results.captures, best_image_path)

    vehicle_state: dict[str, Any] | None = None
    if best_capture and isinstance(best_capture.metadata, dict):
        candidate = best_capture.metadata.get("vehicle_state")
        if isinstance(candidate, dict):
            vehicle_state = candidate

    vision_payload: dict[str, Any] = {
        "profile": edge.profile.value,
        "summary": results.to_dict(),
        "vehicle_state": vehicle_state,
    }

    if not edge.uplink.summary_only:
        vision_payload["best_detection"] = (
            best_detection.model_dump(mode="json") if best_detection else None
        )

    if best_image_path:
        vision_payload["best_image_path"] = str(best_image_path)
    else:
        vision_payload["best_image_path"] = None

    return {
        "asset_id": results.asset_id,
        "vehicle_state": vehicle_state,
        "vision": vision_payload,
    }
