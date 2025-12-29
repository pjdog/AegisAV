"""Feedback persistence helpers.

Stores client feedback and derived outcomes in the configured persistence store
(Redis or in-memory) so the dashboard and API can query recent execution results.

Supports indexing by:
- Global recency (FEEDBACK_INDEX_KEY)
- Per-run (feedback:run:{run_id})
- Per-scenario (feedback:scenario:{scenario_id})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.server.models import DecisionFeedback, DecisionOutcome


class StateStore(Protocol):
    """Subset of the persistence interface used for feedback storage."""

    async def set_state(self, key: str, value: object) -> bool:
        """Set a state value by key."""

    async def get_state(self, key: str, default: object = None) -> object:
        """Get a state value by key."""


# Index keys
FEEDBACK_INDEX_KEY = "feedback:index"


def _feedback_key(decision_id: str) -> str:
    return f"feedback:{decision_id}"


def _outcome_key(decision_id: str) -> str:
    return f"outcome:{decision_id}"


def _run_index_key(run_id: str) -> str:
    return f"feedback:run:{run_id}"


def _scenario_index_key(scenario_id: str) -> str:
    return f"feedback:scenario:{scenario_id}"


def _run_meta_key(run_id: str) -> str:
    return f"feedback:run_meta:{run_id}"


@dataclass
class RunFeedbackSummary:
    """Summary statistics for feedback in a run."""

    run_id: str
    scenario_id: str | None
    total_feedback: int
    success_count: int
    failed_count: int
    anomalies_detected: int
    anomalies_resolved: int
    total_battery_consumed: float
    total_duration_s: float
    avg_duration_s: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "total_feedback": self.total_feedback,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "anomalies_detected": self.anomalies_detected,
            "anomalies_resolved": self.anomalies_resolved,
            "total_battery_consumed": round(self.total_battery_consumed, 2),
            "total_duration_s": round(self.total_duration_s, 2),
            "avg_duration_s": round(self.avg_duration_s, 2),
        }


async def persist_feedback(
    store: StateStore | None,
    feedback: DecisionFeedback,
    outcome: DecisionOutcome | None,
    *,
    run_id: str | None = None,
    scenario_id: str | None = None,
    max_items: int = 1000,
) -> None:
    """Persist the latest feedback/outcome for a decision and update indexes.

    Args:
        store: State store instance
        feedback: The feedback to persist
        outcome: Optional outcome to persist
        run_id: Optional run ID to index by
        scenario_id: Optional scenario ID to index by
        max_items: Maximum items in each index
    """
    if store is None:
        return

    decision_id = feedback.decision_id

    # Store the feedback with run/scenario metadata
    feedback_data = feedback.model_dump(mode="json")
    if run_id:
        feedback_data["run_id"] = run_id
    if scenario_id:
        feedback_data["scenario_id"] = scenario_id

    await store.set_state(_feedback_key(decision_id), feedback_data)
    if outcome is not None:
        outcome_data = outcome.model_dump(mode="json")
        if run_id:
            outcome_data["run_id"] = run_id
        if scenario_id:
            outcome_data["scenario_id"] = scenario_id
        await store.set_state(_outcome_key(decision_id), outcome_data)

    # Update global recency index
    await _update_index(store, FEEDBACK_INDEX_KEY, decision_id, max_items)

    # Update run-specific index
    if run_id:
        await _update_index(store, _run_index_key(run_id), decision_id, max_items)

    # Update scenario-specific index
    if scenario_id:
        await _update_index(store, _scenario_index_key(scenario_id), decision_id, max_items)


async def _update_index(
    store: StateStore,
    index_key: str,
    decision_id: str,
    max_items: int,
) -> None:
    """Update an index list with a new decision ID."""
    index_raw = await store.get_state(index_key, default=[])
    index: list[str] = []
    if isinstance(index_raw, list):
        index = [item for item in index_raw if isinstance(item, str)]

    index = [item for item in index if item != decision_id]
    index.append(decision_id)
    if max_items > 0:
        index = index[-max_items:]

    await store.set_state(index_key, index)


async def get_recent_feedback(store: StateStore | None, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent feedback entries (latest per decision)."""
    if store is None:
        return []

    index_raw = await store.get_state(FEEDBACK_INDEX_KEY, default=[])
    if not isinstance(index_raw, list):
        return []

    decision_ids = [item for item in index_raw if isinstance(item, str)]
    if limit > 0:
        decision_ids = decision_ids[-limit:]
    decision_ids = list(reversed(decision_ids))

    results: list[dict[str, Any]] = []
    for decision_id in decision_ids:
        payload = await store.get_state(_feedback_key(decision_id), default=None)
        if isinstance(payload, dict):
            results.append(payload)

    return results


async def get_recent_outcomes(store: StateStore | None, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent outcomes (latest per decision)."""
    if store is None:
        return []

    index_raw = await store.get_state(FEEDBACK_INDEX_KEY, default=[])
    if not isinstance(index_raw, list):
        return []

    decision_ids = [item for item in index_raw if isinstance(item, str)]
    if limit > 0:
        decision_ids = decision_ids[-limit:]
    decision_ids = list(reversed(decision_ids))

    results: list[dict[str, Any]] = []
    for decision_id in decision_ids:
        payload = await store.get_state(_outcome_key(decision_id), default=None)
        if isinstance(payload, dict):
            results.append(payload)

    return results


async def get_feedback_for_decision(
    store: StateStore | None, decision_id: str
) -> dict[str, Any] | None:
    """Fetch the latest stored feedback for a decision."""
    if store is None:
        return None
    payload = await store.get_state(_feedback_key(decision_id), default=None)
    return payload if isinstance(payload, dict) else None


async def get_outcome_for_decision(
    store: StateStore | None, decision_id: str
) -> dict[str, Any] | None:
    """Fetch the latest stored outcome for a decision."""
    if store is None:
        return None
    payload = await store.get_state(_outcome_key(decision_id), default=None)
    return payload if isinstance(payload, dict) else None


# ============================================================================
# Run-indexed queries
# ============================================================================


async def get_feedback_for_run(store: StateStore | None, run_id: str) -> list[dict[str, Any]]:
    """Get all feedback entries for a specific run.

    Args:
        store: State store instance
        run_id: The run ID to query

    Returns:
        List of feedback entries for this run (oldest first)
    """
    if store is None:
        return []

    index_raw = await store.get_state(_run_index_key(run_id), default=[])
    if not isinstance(index_raw, list):
        return []

    results: list[dict[str, Any]] = []
    for decision_id in index_raw:
        if isinstance(decision_id, str):
            payload = await store.get_state(_feedback_key(decision_id), default=None)
            if isinstance(payload, dict):
                results.append(payload)

    return results


async def get_outcomes_for_run(store: StateStore | None, run_id: str) -> list[dict[str, Any]]:
    """Get all outcomes for a specific run.

    Args:
        store: State store instance
        run_id: The run ID to query

    Returns:
        List of outcome entries for this run (oldest first)
    """
    if store is None:
        return []

    index_raw = await store.get_state(_run_index_key(run_id), default=[])
    if not isinstance(index_raw, list):
        return []

    results: list[dict[str, Any]] = []
    for decision_id in index_raw:
        if isinstance(decision_id, str):
            payload = await store.get_state(_outcome_key(decision_id), default=None)
            if isinstance(payload, dict):
                results.append(payload)

    return results


async def get_anomalies_for_run(store: StateStore | None, run_id: str) -> list[dict[str, Any]]:
    """Get all anomaly-related feedback for a specific run.

    Returns feedback entries where anomaly_detected or anomaly_resolved is True.

    Args:
        store: State store instance
        run_id: The run ID to query

    Returns:
        List of anomaly-related feedback entries
    """
    all_feedback = await get_feedback_for_run(store, run_id)

    anomalies = []
    for fb in all_feedback:
        if fb.get("anomaly_detected") or fb.get("anomaly_resolved"):
            anomalies.append({
                "decision_id": fb.get("decision_id"),
                "timestamp": fb.get("timestamp"),
                "status": fb.get("status"),
                "anomaly_detected": fb.get("anomaly_detected", False),
                "anomaly_resolved": fb.get("anomaly_resolved", False),
                "asset_inspected": fb.get("asset_inspected"),
                "inspection_data": fb.get("inspection_data"),
            })

    return anomalies


async def get_run_summary(
    store: StateStore | None, run_id: str, scenario_id: str | None = None
) -> RunFeedbackSummary | None:
    """Calculate summary statistics for a run's feedback.

    Args:
        store: State store instance
        run_id: The run ID to summarize
        scenario_id: Optional scenario ID for the summary

    Returns:
        RunFeedbackSummary or None if no feedback found
    """
    feedback_list = await get_feedback_for_run(store, run_id)

    if not feedback_list:
        return None

    success_count = 0
    failed_count = 0
    anomalies_detected = 0
    anomalies_resolved = 0
    total_battery = 0.0
    total_duration = 0.0

    for fb in feedback_list:
        status = fb.get("status", "")
        if status == "success":
            success_count += 1
        elif status in ("failed", "aborted", "timeout"):
            failed_count += 1

        if fb.get("anomaly_detected"):
            anomalies_detected += 1
        if fb.get("anomaly_resolved"):
            anomalies_resolved += 1

        total_battery += float(fb.get("battery_consumed", 0) or 0)
        total_duration += float(fb.get("duration_s", 0) or 0)

    # Get scenario_id from first feedback if not provided
    if not scenario_id and feedback_list:
        scenario_id = feedback_list[0].get("scenario_id")

    avg_duration = total_duration / len(feedback_list) if feedback_list else 0.0

    return RunFeedbackSummary(
        run_id=run_id,
        scenario_id=scenario_id,
        total_feedback=len(feedback_list),
        success_count=success_count,
        failed_count=failed_count,
        anomalies_detected=anomalies_detected,
        anomalies_resolved=anomalies_resolved,
        total_battery_consumed=total_battery,
        total_duration_s=total_duration,
        avg_duration_s=avg_duration,
    )


async def save_run_metadata(
    store: StateStore | None,
    run_id: str,
    scenario_id: str,
    start_time: str,
) -> None:
    """Save metadata for a run to enable later queries.

    Args:
        store: State store instance
        run_id: The run ID
        scenario_id: The scenario ID
        start_time: ISO format start time
    """
    if store is None:
        return

    await store.set_state(
        _run_meta_key(run_id),
        {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "start_time": start_time,
        },
    )
