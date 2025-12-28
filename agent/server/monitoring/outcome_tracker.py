"""
Outcome Tracker

Tracks decision execution results and validates predictions for learning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.server.decision import Decision
from agent.server.models.outcome_models import (
    DecisionFeedback,
    DecisionOutcome,
    ExecutionStatus,
    OutcomeStatistics,
)

logger = logging.getLogger(__name__)


class OutcomeTracker:
    """
    Tracks outcomes of executed decisions.

    Responsibilities:
    - Create pending outcomes when decisions are made
    - Update outcomes with execution results from client feedback
    - Calculate prediction errors
    - Persist outcomes to disk for analysis
    - Generate outcome statistics
    """

    def __init__(self, log_dir: Path | str = "logs/outcomes"):
        """
        Initialize outcome tracker.

        Args:
            log_dir: Directory to store outcome logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory tracking of pending outcomes
        self.pending_outcomes: dict[str, DecisionOutcome] = {}

        # Statistics
        self.total_outcomes_tracked = 0
        self.successful_outcomes = 0
        self.failed_outcomes = 0

        logger.info(f"OutcomeTracker initialized: log_dir={self.log_dir}")

    def create_outcome(self, decision: Decision) -> DecisionOutcome:
        """
        Create a pending outcome for a decision.

        This is called immediately after a decision is made, before execution.

        Args:
            decision: The decision that was made

        Returns:
            DecisionOutcome in PENDING status
        """
        # Extract predicted values from decision parameters
        predicted_battery = decision.parameters.get("predicted_battery_consumption")
        predicted_duration = decision.parameters.get("estimated_duration_s")

        outcome = DecisionOutcome(
            decision_id=decision.decision_id,
            execution_status=ExecutionStatus.PENDING,
            started_at=datetime.now(),
            predicted_battery_consumed=predicted_battery,
            predicted_duration_s=predicted_duration,
        )

        # Store in pending outcomes
        self.pending_outcomes[decision.decision_id] = outcome

        logger.debug(
            f"Created pending outcome for decision {decision.decision_id} "
            f"(action: {decision.action.value})"
        )

        return outcome

    async def update_outcome(
        self, decision_id: str, status: ExecutionStatus, **kwargs
    ) -> DecisionOutcome | None:
        """
        Update outcome with execution results.

        This is called when client reports execution status or completion.

        Args:
            decision_id: ID of the decision
            status: New execution status
            **kwargs: Additional outcome fields (battery_consumed, duration, etc.)

        Returns:
            Updated DecisionOutcome, or None if decision_id not found
        """
        if decision_id not in self.pending_outcomes:
            logger.warning(f"Received update for unknown decision: {decision_id}")
            return None

        outcome = self.pending_outcomes[decision_id]
        outcome.execution_status = status
        outcome.updated_at = datetime.now()

        for kwarg_key, value in kwargs.items():
            self._apply_update(outcome, kwarg_key, value)

        # Calculate prediction errors if we have actuals and predictions
        if (
            outcome.actual_battery_consumed is not None
            and outcome.predicted_battery_consumed is not None
        ):
            outcome.prediction_error_battery = abs(
                outcome.actual_battery_consumed - outcome.predicted_battery_consumed
            )

        if outcome.actual_duration_s is not None and outcome.predicted_duration_s is not None:
            outcome.prediction_error_duration = abs(
                outcome.actual_duration_s - outcome.predicted_duration_s
            )

        # If outcome is terminal (success, failed, aborted), finalize it
        if status in {
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.ABORTED,
            ExecutionStatus.TIMEOUT,
        }:
            outcome.completed_at = datetime.now()
            await self._finalize_outcome(outcome)

        logger.info(
            f"Updated outcome for {decision_id}: status={status.value}, "
            f"battery_error={outcome.prediction_error_battery}, "
            f"duration_error={outcome.prediction_error_duration}"
        )

        return outcome

    @staticmethod
    def _apply_update(outcome: DecisionOutcome, key: str, value: Any) -> None:
        """
        Apply a single outcome update by key.

        Args:
            outcome (DecisionOutcome): Outcome instance to update.
            key (str): Update key from feedback.
            value (Any): Value to apply.
        """
        match key:
            case "battery_consumed":
                outcome.actual_battery_consumed = value
            case "distance_traveled":
                outcome.actual_distance_traveled = value
            case "duration_s":
                outcome.actual_duration_s = value
            case "mission_objective_achieved":
                outcome.mission_objective_achieved = value
            case "asset_inspected":
                outcome.asset_inspected = value
            case "anomaly_detected":
                outcome.anomaly_detected = value
            case "anomaly_resolved":
                outcome.anomaly_resolved = value
            case "errors":
                outcome.errors = value
            case "client_feedback":
                outcome.client_feedback = value
            case _:
                return

    async def process_feedback(self, feedback: DecisionFeedback) -> DecisionOutcome | None:
        """
        Process feedback from client.

        This is a convenience method that extracts fields from DecisionFeedback
        and calls update_outcome.

        Args:
            feedback: Feedback from client

        Returns:
            Updated DecisionOutcome
        """
        return await self.update_outcome(
            decision_id=feedback.decision_id,
            status=feedback.status,
            battery_consumed=feedback.battery_consumed,
            distance_traveled=feedback.distance_traveled,
            duration_s=feedback.duration_s,
            mission_objective_achieved=feedback.mission_objective_achieved,
            asset_inspected=feedback.asset_inspected,
            anomaly_detected=feedback.anomaly_detected,
            anomaly_resolved=feedback.anomaly_resolved,
            errors=feedback.errors,
            client_feedback=feedback.notes,
        )

    async def _finalize_outcome(self, outcome: DecisionOutcome) -> None:
        """
        Finalize a completed outcome.

        Actions:
        - Save to persistent storage
        - Remove from pending outcomes
        - Update statistics
        - Trigger learning hooks (future: feed to monitoring agent)

        Args:
            outcome: The completed outcome
        """
        # Update statistics
        self.total_outcomes_tracked += 1
        if outcome.execution_status == ExecutionStatus.SUCCESS:
            self.successful_outcomes += 1
        elif outcome.execution_status in {ExecutionStatus.FAILED, ExecutionStatus.ABORTED}:
            self.failed_outcomes += 1

        # Save to disk
        await self._save_outcome(outcome)

        # Remove from pending
        self.pending_outcomes.pop(outcome.decision_id, None)

        logger.info(
            f"Finalized outcome for {outcome.decision_id}: "
            f"status={outcome.execution_status.value}, "
            f"success={outcome.mission_objective_achieved}"
        )

        # TODO: In Phase 4, feed to monitoring agent for pattern analysis
        # await self.monitoring_agent.process_outcome(outcome)

    async def _save_outcome(self, outcome: DecisionOutcome) -> None:
        """
        Save outcome to persistent storage.

        Outcomes are saved as JSONL (one JSON per line) for easy loading and analysis.

        Args:
            outcome: Outcome to save
        """
        try:
            # Create filename based on date
            filename = f"outcomes_{datetime.now().strftime('%Y%m%d')}.jsonl"
            filepath = self.log_dir / filename

            # Append to file
            with open(filepath, "a", encoding="utf-8") as f:
                json_str = outcome.model_dump_json()
                f.write(json_str + "\n")

            logger.debug(f"Saved outcome to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save outcome: {e}", exc_info=True)

    def get_statistics(self, period_hours: int | None = None) -> OutcomeStatistics:
        """
        Get outcome statistics.

        Args:
            period_hours: If specified, only include outcomes from last N hours

        Returns:
            OutcomeStatistics with aggregated metrics
        """
        _ = period_hours
        # For Phase 1, return basic stats from in-memory counters
        # In future phases, load from disk and filter by time period

        stats = OutcomeStatistics(
            total_decisions=self.total_outcomes_tracked,
            successful=self.successful_outcomes,
            failed=self.failed_outcomes,
            aborted=0,  # TODO: Track separately
            mission_success_rate=(
                self.successful_outcomes / self.total_outcomes_tracked
                if self.total_outcomes_tracked > 0
                else 0.0
            ),
        )

        return stats

    def get_pending_count(self) -> int:
        """
        Get number of pending outcomes.

        Returns:
            Count of outcomes still in pending/in_progress state
        """
        return len(self.pending_outcomes)

    def get_pending_outcomes(self) -> dict[str, DecisionOutcome]:
        """
        Get all pending outcomes.

        Returns:
            Dictionary of decision_id -> DecisionOutcome
        """
        return self.pending_outcomes.copy()

    async def load_outcomes_from_file(self, filepath: Path | str) -> list[DecisionOutcome]:
        """
        Load outcomes from a JSONL file.

        Args:
            filepath: Path to outcomes file

        Returns:
            List of DecisionOutcome objects
        """
        outcomes = []
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        outcome_dict = json.loads(line)
                        outcome = DecisionOutcome.model_validate(outcome_dict)
                        outcomes.append(outcome)

            logger.info(f"Loaded {len(outcomes)} outcomes from {filepath}")
            return outcomes

        except Exception as e:
            logger.error(f"Failed to load outcomes from {filepath}: {e}", exc_info=True)
            return []

    def get_stats_dict(self) -> dict:
        """
        Get statistics as dictionary.

        Returns:
            Dictionary with tracker stats
        """
        return {
            "total_tracked": self.total_outcomes_tracked,
            "successful": self.successful_outcomes,
            "failed": self.failed_outcomes,
            "pending": len(self.pending_outcomes),
            "success_rate": (
                self.successful_outcomes / self.total_outcomes_tracked
                if self.total_outcomes_tracked > 0
                else 0.0
            ),
        }
