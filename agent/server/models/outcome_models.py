"""Data models for decision outcome tracking.

Defines structures for tracking execution results and validating predictions.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionStatus(Enum):
    """Status of decision execution."""

    PENDING = "pending"  # Decision sent, awaiting execution
    IN_PROGRESS = "in_progress"  # Currently executing
    SUCCESS = "success"  # Completed successfully
    FAILED = "failed"  # Execution failed
    ABORTED = "aborted"  # Aborted mid-execution
    TIMEOUT = "timeout"  # Timed out


class DecisionOutcome(BaseModel):
    """Outcome of an executed decision.

    Tracks actual execution results and compares against predictions
    to enable learning and model validation.
    """

    # Decision linkage
    decision_id: str
    execution_status: ExecutionStatus

    # Timestamps
    started_at: datetime
    completed_at: datetime | None = None

    # Actual results (reported by client)
    actual_battery_consumed: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Actual battery consumed (%)"
    )
    actual_distance_traveled: float | None = Field(
        default=None, ge=0.0, description="Actual distance traveled (m)"
    )
    actual_duration_s: float | None = Field(
        default=None, ge=0.0, description="Actual duration (seconds)"
    )

    # Predictions (from decision context)
    predicted_battery_consumed: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Predicted battery consumption (%)"
    )
    predicted_duration_s: float | None = Field(
        default=None, ge=0.0, description="Predicted duration (seconds)"
    )

    # Mission impact
    mission_objective_achieved: bool = Field(default=False)
    asset_inspected: str | None = Field(default=None, description="Asset ID if inspected")
    anomaly_detected: bool = Field(default=False)
    anomaly_resolved: bool = Field(default=False)

    # Validation metrics (computed)
    prediction_error_battery: float | None = Field(
        default=None, description="Absolute error in battery prediction"
    )
    prediction_error_duration: float | None = Field(
        default=None, description="Absolute error in duration prediction"
    )

    # Feedback
    client_feedback: str | None = Field(default=None, description="Free-form feedback from client")
    errors: list[str] = Field(default_factory=list, description="Error messages")

    # Metadata
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision_id": "dec_20250128_123456_abc123",
                "execution_status": "success",
                "started_at": "2025-01-28T12:34:56Z",
                "completed_at": "2025-01-28T12:37:30Z",
                "actual_battery_consumed": 8.5,
                "actual_duration_s": 154.0,
                "predicted_battery_consumed": 7.2,
                "predicted_duration_s": 180.0,
                "mission_objective_achieved": True,
                "asset_inspected": "asset_1",
                "anomaly_detected": False,
                "prediction_error_battery": 1.3,
                "prediction_error_duration": 26.0,
            }
        }
    }


class DecisionFeedback(BaseModel):
    """Client feedback about decision execution.

    Sent from agent client back to server after executing a decision.
    """

    decision_id: str
    status: ExecutionStatus

    # Optional execution metrics
    battery_consumed: float | None = Field(default=None, ge=0.0, le=100.0)
    distance_traveled: float | None = Field(default=None, ge=0.0)
    duration_s: float | None = Field(default=None, ge=0.0)

    # Mission outcomes
    mission_objective_achieved: bool = Field(default=False)
    asset_inspected: str | None = None
    anomaly_detected: bool = Field(default=False)
    anomaly_resolved: bool = Field(default=False)

    # Optional inspection payload (e.g., vision results)
    inspection_data: dict[str, Any] | None = Field(default=None)

    # Error reporting
    errors: list[str] = Field(default_factory=list)
    notes: str | None = None

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision_id": "dec_20250128_123456_abc123",
                "status": "success",
                "battery_consumed": 8.5,
                "duration_s": 154.0,
                "mission_objective_achieved": True,
                "asset_inspected": "asset_1",
            }
        }
    }


class OutcomeStatistics(BaseModel):
    """Aggregated statistics about decision outcomes."""

    total_decisions: int = 0
    successful: int = 0
    failed: int = 0
    aborted: int = 0

    # Prediction accuracy
    avg_battery_prediction_error: float | None = None
    avg_duration_prediction_error: float | None = None

    # Mission metrics
    mission_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    assets_inspected: int = 0
    anomalies_detected: int = 0
    anomalies_resolved: int = 0

    # Time window
    period_start: datetime | None = None
    period_end: datetime | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_decisions": 145,
                "successful": 132,
                "failed": 8,
                "aborted": 5,
                "avg_battery_prediction_error": 1.2,
                "avg_duration_prediction_error": 15.3,
                "mission_success_rate": 0.91,
                "assets_inspected": 42,
                "anomalies_detected": 3,
                "anomalies_resolved": 2,
            }
        }
    }
