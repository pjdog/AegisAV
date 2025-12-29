"""Feedback persistence helpers.

Stores client feedback and derived outcomes in the configured persistence store
(Redis or in-memory) so the dashboard and API can query recent execution results.
"""

from __future__ import annotations

from typing import Any, Protocol

from agent.server.models import DecisionFeedback, DecisionOutcome


class StateStore(Protocol):
    """Subset of the persistence interface used for feedback storage."""

    async def set_state(self, key: str, value: object) -> bool:
        """Set a state value by key."""

    async def get_state(self, key: str, default: object = None) -> object:
        """Get a state value by key."""


FEEDBACK_INDEX_KEY = "feedback:index"


def _feedback_key(decision_id: str) -> str:
    return f"feedback:{decision_id}"


def _outcome_key(decision_id: str) -> str:
    return f"outcome:{decision_id}"


async def persist_feedback(
    store: StateStore | None,
    feedback: DecisionFeedback,
    outcome: DecisionOutcome | None,
    *,
    max_items: int = 1000,
) -> None:
    """Persist the latest feedback/outcome for a decision and update recency index."""
    if store is None:
        return

    decision_id = feedback.decision_id

    await store.set_state(_feedback_key(decision_id), feedback.model_dump(mode="json"))
    if outcome is not None:
        await store.set_state(_outcome_key(decision_id), outcome.model_dump(mode="json"))

    index_raw = await store.get_state(FEEDBACK_INDEX_KEY, default=[])
    index: list[str] = []
    if isinstance(index_raw, list):
        index = [item for item in index_raw if isinstance(item, str)]

    index = [item for item in index if item != decision_id]
    index.append(decision_id)
    if max_items > 0:
        index = index[-max_items:]

    await store.set_state(FEEDBACK_INDEX_KEY, index)


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
