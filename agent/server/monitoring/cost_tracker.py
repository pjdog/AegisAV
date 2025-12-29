"""Cost Tracking and Monitoring for LLM Calls.

Tracks LLM usage, estimates costs, and provides budget monitoring.
Helps ensure we stay within the target cost budget (~$0.52/day).
"""

import logging
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_GLOBAL_COST_TRACKER: dict[str, "CostTracker"] = {}


# Model pricing (per 1M tokens as of January 2025)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15/$0.60 per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},  # $2.50/$10.00 per 1M tokens
    "gpt-4": {"input": 30.00, "output": 60.00},  # Legacy pricing
}


@dataclass
class TokenUsage:
    """Token usage for an LLM call."""

    prompt: int
    completion: int

    @property
    def total(self) -> int:
        """Total tokens used."""
        return self.prompt + self.completion


@dataclass
class CallOutcome:
    """Outcome details for an LLM call."""

    estimated_cost: float
    latency_ms: float
    success: bool
    error_message: str | None = None


@dataclass
class CallDetails:
    """Input details for recording an LLM call."""

    model: str
    context: str  # e.g., "safety_critic", "explanation_agent"
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    success: bool = True
    error_message: str | None = None


@dataclass
class LLMCallRecord:
    """Record of a single LLM call."""

    timestamp: datetime
    model: str
    context: str
    tokens: TokenUsage
    outcome: CallOutcome


@dataclass
class CallCounts:
    """Aggregate call counts."""

    total: int = 0
    successful: int = 0
    failed: int = 0


@dataclass
class TokenTotals:
    """Aggregate token totals."""

    prompt: int = 0
    completion: int = 0
    total: int = 0


@dataclass
class CostTotals:
    """Aggregate cost totals."""

    total_cost: float = 0.0
    average_latency_ms: float = 0.0


@dataclass
class CostStatistics:
    """Aggregated cost statistics."""

    counts: CallCounts = field(default_factory=CallCounts)
    tokens: TokenTotals = field(default_factory=TokenTotals)
    cost: CostTotals = field(default_factory=CostTotals)
    calls_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls_by_context: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cost_by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cost_by_context: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_calls": self.counts.total,
            "successful_calls": self.counts.successful,
            "failed_calls": self.counts.failed,
            "total_prompt_tokens": self.tokens.prompt,
            "total_completion_tokens": self.tokens.completion,
            "total_tokens": self.tokens.total,
            "total_cost": round(self.cost.total_cost, 4),
            "average_latency_ms": round(self.cost.average_latency_ms, 2),
            "calls_by_model": dict(self.calls_by_model),
            "calls_by_context": dict(self.calls_by_context),
            "cost_by_model": {k: round(v, 4) for k, v in self.cost_by_model.items()},
            "cost_by_context": {k: round(v, 4) for k, v in self.cost_by_context.items()},
        }


class CostTracker:
    """Tracks LLM usage and costs across the system.

    Thread-safe implementation for concurrent critic evaluations.
    Provides cost statistics, budget monitoring, and usage reports.
    """

    def __init__(self, daily_budget: float = 1.0) -> None:
        """Initialize cost tracker.

        Args:
            daily_budget: Daily cost budget in USD (default: $1.00/day)
        """
        self.daily_budget = daily_budget
        self._lock = Lock()
        self._calls: list[LLMCallRecord] = []
        logger.info(f"Initialized CostTracker with daily budget: ${daily_budget:.2f}")

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for an LLM call.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        # Extract base model name (remove "openai:" prefix if present)
        base_model = model.rsplit(":", maxsplit=1)[-1]

        if base_model not in MODEL_PRICING:
            logger.warning(
                f"Unknown model for pricing: {base_model}, using gpt-4o-mini as fallback"
            )
            base_model = "gpt-4o-mini"

        pricing = MODEL_PRICING[base_model]
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def record_call(self, details: CallDetails) -> LLMCallRecord:
        """Record an LLM call.

        Args:
            details: Call details to record

        Returns:
            LLMCallRecord for the recorded call
        """
        tokens = TokenUsage(details.prompt_tokens, details.completion_tokens)
        estimated_cost = self.estimate_cost(
            details.model, details.prompt_tokens, details.completion_tokens
        )

        record = LLMCallRecord(
            timestamp=datetime.now(),
            model=details.model,
            context=details.context,
            tokens=tokens,
            outcome=CallOutcome(
                estimated_cost=estimated_cost,
                latency_ms=details.latency_ms,
                success=details.success,
                error_message=details.error_message,
            ),
        )

        with self._lock:
            self._calls.append(record)

        logger.info(
            "LLM call recorded: %s using %s - %s tokens, $%.4f, %.1fms, success=%s",
            details.context,
            details.model,
            tokens.total,
            estimated_cost,
            details.latency_ms,
            details.success,
        )

        return record

    def get_statistics(self, since: datetime | None = None) -> CostStatistics:
        """Get aggregated cost statistics.

        Args:
            since: Only include calls since this timestamp (default: all calls)

        Returns:
            CostStatistics with aggregated metrics
        """
        with self._lock:
            calls = (
                self._calls if since is None else [c for c in self._calls if c.timestamp >= since]
            )

        if not calls:
            return CostStatistics()

        stats = CostStatistics()
        stats.counts.total = len(calls)
        stats.counts.successful = sum(1 for c in calls if c.outcome.success)
        stats.counts.failed = sum(1 for c in calls if not c.outcome.success)

        total_latency = 0.0
        for call in calls:
            stats.tokens.prompt += call.tokens.prompt
            stats.tokens.completion += call.tokens.completion
            stats.tokens.total += call.tokens.total
            stats.cost.total_cost += call.outcome.estimated_cost
            total_latency += call.outcome.latency_ms

            # Track by model
            stats.calls_by_model[call.model] += 1
            stats.cost_by_model[call.model] += call.outcome.estimated_cost

            # Track by context
            stats.calls_by_context[call.context] += 1
            stats.cost_by_context[call.context] += call.outcome.estimated_cost

        stats.cost.average_latency_ms = total_latency / len(calls) if calls else 0.0

        return stats

    def get_daily_statistics(self) -> CostStatistics:
        """Get statistics for the last 24 hours."""
        since = datetime.now() - timedelta(days=1)
        return self.get_statistics(since=since)

    def get_hourly_statistics(self) -> CostStatistics:
        """Get statistics for the last hour."""
        since = datetime.now() - timedelta(hours=1)
        return self.get_statistics(since=since)

    def check_budget(self, timeframe: str = "daily") -> dict[str, Any]:
        """Check budget usage.

        Args:
            timeframe: "daily" or "hourly"

        Returns:
            Dictionary with budget status
        """
        if timeframe == "daily":
            stats = self.get_daily_statistics()
            budget = self.daily_budget
            period = "24 hours"
        elif timeframe == "hourly":
            stats = self.get_hourly_statistics()
            budget = self.daily_budget / 24  # Hourly budget
            period = "1 hour"
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        usage_percent = (stats.cost.total_cost / budget * 100) if budget > 0 else 0.0
        remaining = budget - stats.cost.total_cost

        status = {
            "timeframe": timeframe,
            "period": period,
            "budget": budget,
            "spent": stats.cost.total_cost,
            "remaining": remaining,
            "usage_percent": usage_percent,
            "total_calls": stats.counts.total,
            "average_cost_per_call": stats.cost.total_cost / stats.counts.total
            if stats.counts.total > 0
            else 0.0,
            "on_track": usage_percent <= 100,
        }

        if usage_percent > 100:
            logger.warning(
                f"Budget exceeded for {timeframe}: ${stats.cost.total_cost:.4f} / ${budget:.4f} "
                f"({usage_percent:.1f}%)"
            )
        elif usage_percent > 80:
            logger.warning(
                f"Budget warning for {timeframe}: ${stats.cost.total_cost:.4f} / ${budget:.4f} "
                f"({usage_percent:.1f}%)"
            )

        return status

    def reset(self) -> None:
        """Reset all recorded calls (use with caution)."""
        with self._lock:
            num_calls = len(self._calls)
            self._calls.clear()
        logger.info(f"Cost tracker reset - cleared {num_calls} records")

    def export_calls(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """Export call records as dictionaries.

        Args:
            since: Only export calls since this timestamp

        Returns:
            List of call records as dictionaries
        """
        with self._lock:
            calls = (
                self._calls if since is None else [c for c in self._calls if c.timestamp >= since]
            )

        return [
            {
                "timestamp": call.timestamp.isoformat(),
                "model": call.model,
                "context": call.context,
                "prompt_tokens": call.tokens.prompt,
                "completion_tokens": call.tokens.completion,
                "total_tokens": call.tokens.total,
                "estimated_cost": round(call.outcome.estimated_cost, 6),
                "latency_ms": round(call.outcome.latency_ms, 2),
                "success": call.outcome.success,
                "error_message": call.outcome.error_message,
            }
            for call in calls
        ]


def get_cost_tracker(daily_budget: float = 1.0) -> CostTracker:
    """Get the global cost tracker instance (singleton pattern).

    Args:
        daily_budget: Daily budget (only used on first initialization)

    Returns:
        Global CostTracker instance
    """
    tracker = _GLOBAL_COST_TRACKER.get("instance")
    if tracker is None:
        tracker = CostTracker(daily_budget=daily_budget)
        _GLOBAL_COST_TRACKER["instance"] = tracker
    return tracker


@asynccontextmanager
async def track_llm_call(  # noqa: RUF029
    model: str, context: str, tracker: CostTracker | None = None
) -> "AsyncGenerator[_LLMTracking, None]":
    """Context manager for tracking LLM calls automatically.

    Usage:
        async with track_llm_call("gpt-4o-mini", "safety_critic") as tracking:
            result = await agent.run(prompt)
            tracking.set_tokens(result.usage.prompt_tokens, result.usage.completion_tokens)

    Args:
        model: Model name
        context: Context (e.g., "safety_critic")
        tracker: CostTracker instance (uses global if None)

    Yields:
        Tracking object with set_tokens() method
    """
    if tracker is None:
        tracker = get_cost_tracker()

    start_time = time.time()
    tracking = _LLMTracking()

    try:
        yield tracking
        success = True
        error_message = None
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        latency_ms = (time.time() - start_time) * 1000
        tracker.record_call(
            CallDetails(
                model=model,
                context=context,
                prompt_tokens=tracking.prompt_tokens,
                completion_tokens=tracking.completion_tokens,
                latency_ms=latency_ms,
                success=success,
                error_message=error_message,
            )
        )


class _LLMTracking:
    """Helper class for tracking within context manager."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def set_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Set token counts for the LLM call."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count for text.

    Uses ~4 characters per token heuristic (conservative estimate).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4
