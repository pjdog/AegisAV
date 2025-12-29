"""Data models for adaptive learning and pattern recognition.

Supports continuous improvement through pattern detection, insight generation,
and threshold optimization.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PatternType(Enum):
    """Types of patterns that can be detected."""

    SUCCESS_PATTERN = "success_pattern"  # Recurring successful decisions
    FAILURE_PATTERN = "failure_pattern"  # Recurring failures
    ANOMALY_PATTERN = "anomaly_pattern"  # Unusual or unexpected patterns
    EFFICIENCY_PATTERN = "efficiency_pattern"  # Resource optimization patterns
    RISK_PATTERN = "risk_pattern"  # Risk-related patterns


class LearningInsight(BaseModel):
    """Insight learned from decision history.

    Represents a discovered pattern and its implications for future decisions.
    """

    # Identification
    insight_id: str
    discovered_at: datetime = Field(default_factory=datetime.now)
    pattern_type: PatternType

    # Pattern details
    pattern_description: str = Field(..., description="Human-readable pattern description")
    conditions: dict[str, Any] = Field(
        ..., description="Conditions under which this pattern occurs"
    )
    frequency: int = Field(..., ge=1, description="Number of times observed")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate for this pattern")

    # Lesson learned
    lesson: str = Field(..., description="What should be done differently")

    # Recommended adjustments
    threshold_adjustments: dict[str, float] = Field(
        default_factory=dict, description="Suggested parameter adjustments"
    )
    confidence_in_adjustment: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the recommended adjustments"
    )

    # Statistical support
    sample_size: int = Field(..., ge=1, description="Number of samples supporting this insight")
    statistical_significance: float = Field(
        ..., ge=0.0, le=1.0, description="Statistical significance (p-value complement)"
    )

    # Metadata
    status: str = Field(
        default="pending_review", pattern="^(pending_review|approved|applied|rejected)$"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "insight_id": "insight_001",
                "pattern_type": "failure_pattern",
                "pattern_description": "Inspections fail when wind > 8 m/s and battery < 40%",
                "conditions": {"wind_speed_ms": ">8.0", "battery_percent": "<40.0"},
                "frequency": 12,
                "success_rate": 0.25,
                "lesson": "Avoid inspections in high wind with moderate battery",
                "threshold_adjustments": {"wind_abort_ms": 8.5, "battery_warning_percent": 45.0},
                "confidence_in_adjustment": 0.82,
                "sample_size": 48,
                "statistical_significance": 0.95,
            }
        }
    }


class ThresholdAdjustment(BaseModel):
    """Record of a threshold parameter adjustment."""

    # Parameter identification
    parameter_name: str = Field(..., description="Name of the parameter being adjusted")
    parameter_category: str = Field(..., description="Category (battery, wind, gps, etc.)")

    # Adjustment
    old_value: float = Field(..., description="Previous value")
    new_value: float = Field(..., description="New value")
    change_percent: float = Field(..., description="Percentage change")

    # Justification
    reason: str = Field(..., description="Why this adjustment was made")
    supporting_insight_id: str | None = Field(
        default=None, description="Insight that motivated this adjustment"
    )

    # Metadata
    applied_at: datetime = Field(default_factory=datetime.now)
    applied_by: str = Field(
        ..., description="Source of adjustment (monitoring_agent, manual, etc.)"
    )
    auto_applied: bool = Field(default=False, description="Whether this was automatically applied")

    # Validation
    within_safe_bounds: bool = Field(..., description="Whether adjustment is within safety limits")
    approved: bool = Field(default=False, description="Whether adjustment was approved")

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter_name": "battery_warning_percent",
                "parameter_category": "battery",
                "old_value": 30.0,
                "new_value": 35.0,
                "change_percent": 16.7,
                "reason": "Observed failures with battery between 30-35%",
                "supporting_insight_id": "insight_001",
                "applied_by": "monitoring_agent",
                "auto_applied": False,
                "within_safe_bounds": True,
                "approved": False,
            }
        }
    }


class PatternInstance(BaseModel):
    """Single instance of a detected pattern."""

    decision_id: str
    timestamp: datetime
    matched_conditions: dict[str, Any] = Field(
        ..., description="Conditions that matched the pattern"
    )
    outcome_successful: bool
    outcome_value: float | None = Field(
        default=None, description="Quantitative outcome metric if applicable"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision_id": "dec_20250128_123456_abc123",
                "timestamp": "2025-01-28T12:34:56Z",
                "matched_conditions": {"wind_speed_ms": 9.2, "battery_percent": 38.0},
                "outcome_successful": False,
                "outcome_value": 0.25,
            }
        }
    }


class PatternAnalysisReport(BaseModel):
    """Report from pattern analysis.

    Generated by MonitoringAgent after analyzing decision history.
    """

    # Time period
    analysis_period_start: datetime
    analysis_period_end: datetime
    decisions_analyzed: int

    # Patterns found
    patterns_detected: list[LearningInsight] = Field(default_factory=list)
    high_confidence_patterns: int = Field(
        default=0, description="Number of patterns with confidence > 0.8"
    )

    # Recommendations
    recommended_adjustments: list[ThresholdAdjustment] = Field(default_factory=list)
    urgent_recommendations: list[str] = Field(
        default_factory=list, description="Issues requiring immediate attention"
    )

    # Summary statistics
    overall_success_rate: float = Field(..., ge=0.0, le=1.0)
    prediction_accuracy_improvement: float | None = Field(
        default=None, description="Change in prediction accuracy since last analysis"
    )

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    analysis_version: str = Field(default="1.0")

    model_config = {
        "json_schema_extra": {
            "example": {
                "analysis_period_start": "2025-01-27T00:00:00Z",
                "analysis_period_end": "2025-01-28T00:00:00Z",
                "decisions_analyzed": 245,
                "patterns_detected": [],
                "high_confidence_patterns": 3,
                "recommended_adjustments": [],
                "overall_success_rate": 0.89,
                "prediction_accuracy_improvement": 0.05,
            }
        }
    }


class LearningConfiguration(BaseModel):
    """Configuration for learning and adaptive systems."""

    # Pattern detection
    min_pattern_frequency: int = Field(default=5, ge=1, description="Minimum pattern occurrences")
    min_confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for pattern acceptance"
    )
    min_sample_size: int = Field(
        default=20, ge=1, description="Minimum samples for statistical significance"
    )

    # Threshold adjustment
    auto_apply_adjustments: bool = Field(
        default=False, description="Automatically apply threshold adjustments"
    )
    max_adjustment_percent: float = Field(
        default=20.0, ge=0.0, le=100.0, description="Maximum allowed parameter change (%)"
    )
    require_approval_above_percent: float = Field(
        default=10.0, ge=0.0, le=100.0, description="Adjustment % requiring manual approval"
    )

    # Learning rate
    learning_rate: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Rate at which to adapt parameters"
    )
    experience_buffer_size: int = Field(
        default=500, ge=1, description="Number of recent experiences to keep"
    )

    # Analysis frequency
    pattern_analysis_interval_s: int = Field(
        default=3600, ge=60, description="How often to run pattern analysis (seconds)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "min_pattern_frequency": 5,
                "min_confidence_threshold": 0.7,
                "auto_apply_adjustments": False,
                "max_adjustment_percent": 20.0,
                "learning_rate": 0.05,
            }
        }
    }
