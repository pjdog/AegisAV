"""Data models for decision explainability and audit trails.

Supports transparency, debugging, and trust by capturing decision reasoning chains.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.server.models.critic_models import CriticResponse, EscalationDecision


class FactorContribution(BaseModel):
    """Contribution of a single factor to a decision."""

    factor_name: str = Field(..., description="Name of the factor (e.g., 'battery_level')")
    value: float = Field(..., description="Raw value of the factor")
    weight: float = Field(..., ge=0.0, le=1.0, description="Importance weight")
    contribution: float = Field(
        ..., description="Weighted contribution (weight * normalized_value)"
    )
    unit: str | None = Field(default=None, description="Unit of measurement")

    model_config = {
        "json_schema_extra": {
            "example": {
                "factor_name": "battery_remaining",
                "value": 75.0,
                "weight": 0.25,
                "contribution": 0.1875,
                "unit": "percent",
            }
        }
    }


class CounterfactualScenario(BaseModel):
    """What-if analysis of alternative scenarios."""

    scenario_name: str = Field(..., description="Description of the counterfactual")
    changed_factors: dict[str, Any] = Field(..., description="Factors that differ from actual")
    predicted_outcome: str = Field(..., description="Predicted outcome under this scenario")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    would_change_decision: bool = Field(..., description="Would this change the decision?")

    model_config = {
        "json_schema_extra": {
            "example": {
                "scenario_name": "If battery was at 50%",
                "changed_factors": {"battery_remaining": 50.0},
                "predicted_outcome": "Would trigger early return instead of continuing inspection",
                "confidence": 0.85,
                "would_change_decision": True,
            }
        }
    }


class ReasoningStep(BaseModel):
    """Single step in a reasoning chain."""

    step_number: int = Field(..., ge=1)
    description: str = Field(..., description="What happened in this step")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Input values")
    outputs: dict[str, Any] = Field(default_factory=dict, description="Output values")
    rationale: str | None = Field(default=None, description="Why this step was taken")

    model_config = {
        "json_schema_extra": {
            "example": {
                "step_number": 1,
                "description": "Evaluated battery safety margin",
                "inputs": {"battery_remaining": 75.0, "distance_to_dock": 450.0},
                "outputs": {"battery_needed": 15.0, "safety_margin": 60.0},
                "rationale": "Sufficient battery for return with 60% safety margin",
            }
        }
    }


class AuditTrail(BaseModel):
    """Complete audit trail for a decision.

    Captures the full reasoning chain, factor contributions, critic evaluations,
    and counterfactual analysis for transparency and debugging.
    """

    # Linkage
    decision_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Reasoning chain
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list, description="Step-by-step decision reasoning"
    )

    # Factor analysis
    factor_contributions: list[FactorContribution] = Field(
        default_factory=list, description="Contribution of each factor to the decision"
    )

    # Alternatives considered
    counterfactuals: list[CounterfactualScenario] = Field(
        default_factory=list, description="What-if scenarios analyzed"
    )

    # Critic evaluations
    critic_responses: list[CriticResponse] = Field(
        default_factory=list, description="Evaluations from critic agents"
    )
    escalation_path: EscalationDecision | None = Field(
        default=None, description="Escalation decision if applicable"
    )

    # Final validation
    approved: bool = Field(..., description="Whether decision was approved")
    approval_timestamp: datetime = Field(default_factory=datetime.now)
    approver: str = Field(default="system", description="Who/what approved the decision")

    # Summary
    summary: str | None = Field(
        default=None, description="Human-readable summary of the audit trail"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision_id": "dec_20250128_123456_abc123",
                "reasoning_steps": [
                    {"step_number": 1, "description": "Checked battery level"},
                    {"step_number": 2, "description": "Evaluated risk factors"},
                    {"step_number": 3, "description": "Selected inspection goal"},
                ],
                "approved": True,
                "approver": "critic_orchestrator",
                "summary": (
                    "Decision approved after safety and efficiency critics found no concerns"
                ),
            }
        }
    }


class ExplanationRequest(BaseModel):
    """Request for generating an explanation of a decision."""

    decision_id: str
    detail_level: str = Field(
        default="standard",
        pattern="^(brief|standard|detailed)$",
        description="Level of detail for explanation",
    )
    include_counterfactuals: bool = Field(default=True, description="Include what-if analysis")
    target_audience: str = Field(
        default="technical",
        pattern="^(technical|operator|management)$",
        description="Target audience for explanation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision_id": "dec_20250128_123456_abc123",
                "detail_level": "detailed",
                "include_counterfactuals": True,
                "target_audience": "operator",
            }
        }
    }


class ExplanationResponse(BaseModel):
    """Generated explanation of a decision."""

    decision_id: str
    explanation_text: str = Field(..., description="Human-readable explanation")
    key_factors: list[str] = Field(default_factory=list, description="Most important factors")
    confidence_level: str = Field(..., description="Overall confidence in decision")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for operator"
    )
    generated_at: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision_id": "dec_20250128_123456_abc123",
                "explanation_text": "Decision to inspect asset_1 was based on...",
                "key_factors": ["Battery at safe level (75%)", "Asset scheduled for inspection"],
                "confidence_level": "high",
                "recommendations": [
                    "Monitor battery during inspection",
                    "Return if wind exceeds 10 m/s",
                ],
            }
        }
    }
