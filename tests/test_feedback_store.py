"""
Tests for server-side feedback persistence helpers.
"""

from datetime import datetime

import pytest

from agent.server.feedback_store import (
    get_anomalies_for_run,
    get_feedback_for_decision,
    get_feedback_for_run,
    get_outcome_for_decision,
    get_outcomes_for_run,
    get_recent_feedback,
    get_recent_outcomes,
    get_run_summary,
    persist_feedback,
    save_run_metadata,
)
from agent.server.models import DecisionFeedback, DecisionOutcome, ExecutionStatus
from agent.server.persistence import InMemoryStore


class TestFeedbackStore:
    """Test feedback persistence helper functions."""

    @pytest.mark.asyncio
    async def test_persist_and_fetch_feedback_and_outcome(self):
        """Persist a feedback record and fetch it back by decision id."""
        store = InMemoryStore()
        await store.connect()

        feedback = DecisionFeedback(
            decision_id="dec_1",
            status=ExecutionStatus.SUCCESS,
            battery_consumed=1.0,
            duration_s=2.0,
        )
        outcome = DecisionOutcome(
            decision_id="dec_1",
            execution_status=ExecutionStatus.SUCCESS,
            started_at=datetime.now(),
            actual_battery_consumed=1.0,
            actual_duration_s=2.0,
        )

        await persist_feedback(store, feedback, outcome, max_items=10)

        stored_feedback = await get_feedback_for_decision(store, "dec_1")
        assert stored_feedback is not None
        assert stored_feedback["decision_id"] == "dec_1"
        assert stored_feedback["status"] == "success"

        stored_outcome = await get_outcome_for_decision(store, "dec_1")
        assert stored_outcome is not None
        assert stored_outcome["decision_id"] == "dec_1"
        assert stored_outcome["execution_status"] == "success"

        recent_feedback = await get_recent_feedback(store, limit=10)
        assert len(recent_feedback) == 1
        assert recent_feedback[0]["decision_id"] == "dec_1"

        recent_outcomes = await get_recent_outcomes(store, limit=10)
        assert len(recent_outcomes) == 1
        assert recent_outcomes[0]["decision_id"] == "dec_1"

    @pytest.mark.asyncio
    async def test_recent_feedback_order_and_cap(self):
        """Recent feedback is ordered newest-first and caps the index size."""
        store = InMemoryStore()
        await store.connect()

        await persist_feedback(
            store,
            DecisionFeedback(decision_id="dec_1", status=ExecutionStatus.SUCCESS),
            None,
            max_items=2,
        )
        await persist_feedback(
            store,
            DecisionFeedback(decision_id="dec_2", status=ExecutionStatus.SUCCESS),
            None,
            max_items=2,
        )
        await persist_feedback(
            store,
            DecisionFeedback(decision_id="dec_3", status=ExecutionStatus.SUCCESS),
            None,
            max_items=2,
        )

        recent = await get_recent_feedback(store, limit=10)
        assert [item["decision_id"] for item in recent] == ["dec_3", "dec_2"]

        # Updating an existing decision id should move it to the front.
        await persist_feedback(
            store,
            DecisionFeedback(decision_id="dec_2", status=ExecutionStatus.FAILED),
            None,
            max_items=2,
        )
        recent = await get_recent_feedback(store, limit=10)
        assert [item["decision_id"] for item in recent] == ["dec_2", "dec_3"]

    @pytest.mark.asyncio
    async def test_recent_outcomes_skips_missing(self):
        """Outcomes list skips decisions with no stored outcome."""
        store = InMemoryStore()
        await store.connect()

        await persist_feedback(
            store,
            DecisionFeedback(decision_id="dec_1", status=ExecutionStatus.SUCCESS),
            None,
            max_items=10,
        )
        outcomes = await get_recent_outcomes(store, limit=10)
        assert outcomes == []

    @pytest.mark.asyncio
    async def test_run_indexed_queries_and_summary(self):
        """Run-specific queries return indexed feedback and summaries."""
        store = InMemoryStore()
        await store.connect()

        run_id = "run_001"
        scenario_id = "scenario_alpha"
        await save_run_metadata(store, run_id, scenario_id, "2025-01-01T00:00:00Z")

        feedback_one = DecisionFeedback(
            decision_id="dec_1",
            status=ExecutionStatus.SUCCESS,
            battery_consumed=5.0,
            duration_s=12.0,
            anomaly_detected=True,
            asset_inspected="asset_1",
            inspection_data={"note": "ok"},
        )
        outcome_one = DecisionOutcome(
            decision_id="dec_1",
            execution_status=ExecutionStatus.SUCCESS,
            started_at=datetime.now(),
            actual_battery_consumed=5.0,
            actual_duration_s=12.0,
            anomaly_detected=True,
            asset_inspected="asset_1",
        )

        feedback_two = DecisionFeedback(
            decision_id="dec_2",
            status=ExecutionStatus.FAILED,
            battery_consumed=1.0,
            duration_s=4.0,
            anomaly_resolved=True,
        )
        outcome_two = DecisionOutcome(
            decision_id="dec_2",
            execution_status=ExecutionStatus.FAILED,
            started_at=datetime.now(),
            actual_battery_consumed=1.0,
            actual_duration_s=4.0,
            anomaly_resolved=True,
        )

        await persist_feedback(
            store,
            feedback_one,
            outcome_one,
            run_id=run_id,
            scenario_id=scenario_id,
            max_items=10,
        )
        await persist_feedback(
            store,
            feedback_two,
            outcome_two,
            run_id=run_id,
            scenario_id=scenario_id,
            max_items=10,
        )

        run_feedback = await get_feedback_for_run(store, run_id)
        assert [entry["decision_id"] for entry in run_feedback] == ["dec_1", "dec_2"]

        run_outcomes = await get_outcomes_for_run(store, run_id)
        assert [entry["decision_id"] for entry in run_outcomes] == ["dec_1", "dec_2"]

        anomalies = await get_anomalies_for_run(store, run_id)
        assert {entry["decision_id"] for entry in anomalies} == {"dec_1", "dec_2"}

        summary = await get_run_summary(store, run_id)
        assert summary is not None
        assert summary.total_feedback == 2
        assert summary.success_count == 1
        assert summary.failed_count == 1
        assert summary.anomalies_detected == 1
        assert summary.anomalies_resolved == 1
        assert summary.total_battery_consumed == 6.0
        assert summary.total_duration_s == 16.0
        assert summary.avg_duration_s == 8.0

        run_meta = await store.get_state(f"feedback:run_meta:{run_id}")
        assert run_meta["scenario_id"] == scenario_id
