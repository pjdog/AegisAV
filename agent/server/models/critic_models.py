"""
Data models for multi-agent critic system.

Defines critic responses, verdicts, escalation decisions, and related structures.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CriticType(Enum):
    """Types of critic agents."""

    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    GOAL_ALIGNMENT = "goal_alignment"


class CriticVerdict(Enum):
    """Verdict outcomes from critic evaluation."""

    APPROVE = "approve"  # Decision is acceptable
    APPROVE_WITH_CONCERNS = "approve_with_concerns"  # Advisory warnings
    REJECT = "reject"  # Block decision
    ESCALATE = "escalate"  # Needs hierarchical review


class CriticResponse(BaseModel):
    """Response from a single critic agent."""

    critic_type: CriticType
    verdict: CriticVerdict
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in verdict")

    # Analysis results
    concerns: list[str] = Field(default_factory=list, description="Issues identified")
    alternatives: list[str] = Field(default_factory=list, description="Alternative actions")
    reasoning: str = Field(default="", description="LLM-generated or rule-based explanation")

    # Metrics
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Critic's risk assessment")
    processing_time_ms: float = Field(default=0.0, description="Time taken to evaluate")

    # Metadata
    used_llm: bool = Field(default=False, description="Whether LLM was used")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "example": {
                "critic_type": "safety",
                "verdict": "reject",
                "confidence": 0.95,
                "concerns": ["Battery at 15% insufficient for 500m return distance"],
                "alternatives": ["Return to dock immediately", "Land at current position"],
                "reasoning": "Battery level critical with insufficient margin for safe return",
                "risk_score": 0.9,
                "processing_time_ms": 42.5,
                "used_llm": False,
            }
        }
    }


class EscalationLevel(Enum):
    """Levels of decision escalation."""

    NONE = 0  # No escalation (advisory)
    ADVISORY = 1  # Log warnings but approve
    BLOCKING = 2  # Can reject decisions
    HIERARCHICAL = 3  # Full multi-critic review with LLM


class EscalationDecision(BaseModel):
    """Decision from the escalation authority model."""

    escalation_level: EscalationLevel
    reason: str = Field(..., description="Why this escalation level was chosen")
    recommended_action: str = Field(..., description="What should be done")

    # Critic consensus
    critic_responses: list[CriticResponse]
    consensus_score: float = Field(..., ge=0.0, le=1.0, description="Agreement between critics")

    # Decision
    approved: bool = Field(..., description="Whether decision is approved")
    requires_human_review: bool = Field(default=False, description="Flag for human oversight")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "example": {
                "escalation_level": "blocking",
                "reason": "2 critics rejected decision due to safety concerns",
                "recommended_action": "Abort current goal and return to dock",
                "consensus_score": 0.83,
                "approved": False,
                "requires_human_review": False,
            }
        }
    }


class CriticConfig(BaseModel):
    """Configuration for a critic agent."""

    enabled: bool = Field(default=True)
    use_llm: bool = Field(default=True, description="Enable LLM evaluation")
    llm_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Risk score above which to use LLM"
    )
    timeout_ms: float = Field(default=2000.0, description="Max evaluation time")

    model_config = {
        "json_schema_extra": {
            "example": {
                "enabled": True,
                "use_llm": True,
                "llm_threshold": 0.6,
                "timeout_ms": 2000.0,
            }
        }
    }
