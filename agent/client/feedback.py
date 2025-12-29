"""Client Feedback Builder.

Creates DecisionFeedback payloads to POST back to the server `/feedback` endpoint.
Includes optional `inspection_data` (vision summaries / evidence) shaped by the
current edge-compute profile.
"""

from __future__ import annotations

from typing import Any

from agent.client.action_executor import ExecutionResult, ExecutionState
from agent.client.edge_policy import build_inspection_data, compute_anomaly_detected
from agent.client.vision_client import InspectionVisionResults
from agent.edge_config import EdgeComputeConfig, EdgeComputeProfile, default_edge_compute_config


def build_feedback(
    decision: dict[str, Any],
    result: ExecutionResult,
    inspection_results: InspectionVisionResults | None,
    edge: EdgeComputeConfig | None,
) -> dict[str, Any]:
    """Create a feedback payload compatible with server-side DecisionFeedback."""
    # Map execution result to server-side ExecutionStatus enum values.
    status = "failed"
    if result.state == ExecutionState.COMPLETED:
        status = "success"
    elif result.state == ExecutionState.ABORTED:
        status = "aborted"

    decision_id = decision.get("decision_id", "unknown")
    action = decision.get("action", "none")
    parameters = (
        decision.get("parameters", {}) if isinstance(decision.get("parameters"), dict) else {}
    )

    feedback: dict[str, Any] = {
        "decision_id": decision_id,
        "status": status,
        "duration_s": result.duration_s,
        "mission_objective_achieved": status == "success" and action != "abort",
        "asset_inspected": parameters.get("asset_id") if action == "inspect" else None,
        "anomaly_detected": False,
        "errors": [result.message]
        if result.state == ExecutionState.FAILED and result.message
        else [],
        "notes": result.message or None,
    }

    edge = edge or default_edge_compute_config(EdgeComputeProfile.SBC_CPU)

    if action == "inspect" and inspection_results and edge.vision_enabled:
        feedback["anomaly_detected"] = compute_anomaly_detected(inspection_results, edge)
        feedback["inspection_data"] = build_inspection_data(inspection_results, edge)

    return feedback
