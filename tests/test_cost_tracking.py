"""
Tests for Cost Tracking and Monitoring

Validates LLM cost tracking, budget monitoring, and usage statistics.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta

import pytest

from agent.server.monitoring.cost_tracker import (
    CostTracker,
    estimate_tokens,
    get_cost_tracker,
    track_llm_call,
)


@pytest.fixture
def cost_tracker():
    """Create a fresh cost tracker for testing."""
    return CostTracker(daily_budget=1.0)


def test_cost_estimation():
    """Test that cost estimation works correctly for different models."""
    tracker = CostTracker()

    # GPT-4o-mini: $0.15/$0.60 per 1M tokens
    cost_mini = tracker.estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    expected_mini = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
    assert abs(cost_mini - expected_mini) < 0.0001

    # GPT-4o: $2.50/$10.00 per 1M tokens
    cost_4o = tracker.estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    expected_4o = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
    assert abs(cost_4o - expected_4o) < 0.0001

    # Should be more expensive with GPT-4o
    assert cost_4o > cost_mini


def test_cost_estimation_with_prefix():
    """Test cost estimation handles model names with 'openai:' prefix."""
    tracker = CostTracker()

    cost1 = tracker.estimate_cost("gpt-4o-mini", 1000, 500)
    cost2 = tracker.estimate_cost("openai:gpt-4o-mini", 1000, 500)

    # Should be the same regardless of prefix
    assert abs(cost1 - cost2) < 0.0001


def test_record_call(cost_tracker):
    """Test recording individual LLM calls."""
    record = cost_tracker.record_call(
        model="gpt-4o-mini",
        context="safety_critic",
        prompt_tokens=500,
        completion_tokens=200,
        latency_ms=150.5,
        success=True,
    )

    assert record.model == "gpt-4o-mini"
    assert record.context == "safety_critic"
    assert record.prompt_tokens == 500
    assert record.completion_tokens == 200
    assert record.total_tokens == 700
    assert record.latency_ms == 150.5
    assert record.success is True
    assert record.error_message is None
    assert record.estimated_cost > 0


def test_record_failed_call(cost_tracker):
    """Test recording failed LLM calls."""
    record = cost_tracker.record_call(
        model="gpt-4o-mini",
        context="efficiency_critic",
        prompt_tokens=100,
        completion_tokens=0,
        latency_ms=50.0,
        success=False,
        error_message="API timeout",
    )

    assert record.success is False
    assert record.error_message == "API timeout"
    assert record.completion_tokens == 0


def test_get_statistics_empty(cost_tracker):
    """Test statistics with no recorded calls."""
    stats = cost_tracker.get_statistics()

    assert stats.total_calls == 0
    assert stats.successful_calls == 0
    assert stats.failed_calls == 0
    assert stats.total_cost == 0.0
    assert stats.average_latency_ms == 0.0


def test_get_statistics(cost_tracker):
    """Test aggregated statistics calculation."""
    # Record multiple calls
    cost_tracker.record_call("gpt-4o-mini", "safety_critic", 500, 200, 150.0, success=True)
    cost_tracker.record_call("gpt-4o-mini", "efficiency_critic", 600, 300, 200.0, success=True)
    cost_tracker.record_call("gpt-4o", "explanation_agent", 1000, 400, 300.0, success=True)
    cost_tracker.record_call("gpt-4o-mini", "safety_critic", 100, 0, 50.0, success=False)

    stats = cost_tracker.get_statistics()

    assert stats.total_calls == 4
    assert stats.successful_calls == 3
    assert stats.failed_calls == 1
    assert stats.total_prompt_tokens == 2200
    assert stats.total_completion_tokens == 900
    assert stats.total_tokens == 3100
    assert stats.total_cost > 0
    assert stats.average_latency_ms == (150.0 + 200.0 + 300.0 + 50.0) / 4

    # Check model breakdown
    assert stats.calls_by_model["gpt-4o-mini"] == 3
    assert stats.calls_by_model["gpt-4o"] == 1

    # Check context breakdown
    assert stats.calls_by_context["safety_critic"] == 2
    assert stats.calls_by_context["efficiency_critic"] == 1
    assert stats.calls_by_context["explanation_agent"] == 1


def test_get_statistics_with_since(cost_tracker):
    """Test filtering statistics by timestamp."""
    # Record a call 2 hours ago (simulated)
    old_call = cost_tracker.record_call("gpt-4o-mini", "old_context", 100, 50, 100.0)
    old_call.timestamp = datetime.now() - timedelta(hours=2)

    # Record recent calls
    cost_tracker.record_call("gpt-4o-mini", "new_context", 200, 100, 150.0)
    cost_tracker.record_call("gpt-4o-mini", "new_context", 300, 150, 200.0)

    # Get stats for last hour (should exclude old call)
    since = datetime.now() - timedelta(hours=1)
    stats = cost_tracker.get_statistics(since=since)

    assert stats.total_calls == 2
    assert stats.calls_by_context.get("old_context", 0) == 0
    assert stats.calls_by_context["new_context"] == 2


def test_daily_statistics(cost_tracker):
    """Test daily statistics retrieval."""
    cost_tracker.record_call("gpt-4o-mini", "test", 500, 200, 100.0)

    stats = cost_tracker.get_daily_statistics()
    assert stats.total_calls == 1


def test_hourly_statistics(cost_tracker):
    """Test hourly statistics retrieval."""
    cost_tracker.record_call("gpt-4o-mini", "test", 500, 200, 100.0)

    stats = cost_tracker.get_hourly_statistics()
    assert stats.total_calls == 1


def test_budget_check_under_budget(cost_tracker):
    """Test budget checking when under budget."""
    # Record a small call
    cost_tracker.record_call("gpt-4o-mini", "test", 100, 50, 100.0)

    status = cost_tracker.check_budget(timeframe="daily")

    assert status["timeframe"] == "daily"
    assert status["budget"] == 1.0
    assert status["spent"] < 1.0
    assert status["remaining"] > 0
    assert status["usage_percent"] < 100
    assert status["on_track"] is True


def test_budget_check_over_budget():
    """Test budget checking when over budget."""
    # Create tracker with very small budget
    tracker = CostTracker(daily_budget=0.001)

    # Record expensive call (simulated)
    tracker.record_call("gpt-4o", "test", 100000, 50000, 1000.0)

    status = tracker.check_budget(timeframe="daily")

    assert status["spent"] > status["budget"]
    assert status["remaining"] < 0
    assert status["usage_percent"] > 100
    assert status["on_track"] is False


def test_budget_check_hourly(cost_tracker):
    """Test hourly budget checking."""
    cost_tracker.record_call("gpt-4o-mini", "test", 1000, 500, 100.0)

    status = cost_tracker.check_budget(timeframe="hourly")

    assert status["timeframe"] == "hourly"
    assert status["period"] == "1 hour"
    assert status["budget"] == 1.0 / 24  # Hourly budget


def test_statistics_to_dict(cost_tracker):
    """Test statistics serialization to dictionary."""
    cost_tracker.record_call("gpt-4o-mini", "test", 500, 200, 150.0)

    stats = cost_tracker.get_statistics()
    stats_dict = stats.to_dict()

    assert isinstance(stats_dict, dict)
    assert stats_dict["total_calls"] == 1
    assert stats_dict["total_tokens"] == 700
    assert "total_cost" in stats_dict
    assert "calls_by_model" in stats_dict
    assert "cost_by_context" in stats_dict


def test_export_calls(cost_tracker):
    """Test exporting call records."""
    cost_tracker.record_call("gpt-4o-mini", "critic1", 500, 200, 150.0, success=True)
    cost_tracker.record_call(
        "gpt-4o", "critic2", 1000, 400, 300.0, success=False, error_message="timeout"
    )

    exports = cost_tracker.export_calls()

    assert len(exports) == 2
    assert exports[0]["model"] == "gpt-4o-mini"
    assert exports[0]["context"] == "critic1"
    assert exports[0]["success"] is True
    assert exports[1]["error_message"] == "timeout"


def test_export_calls_with_since(cost_tracker):
    """Test exporting calls with timestamp filter."""
    # Record old call
    old_record = cost_tracker.record_call("gpt-4o-mini", "old", 100, 50, 100.0)
    old_record.timestamp = datetime.now() - timedelta(hours=2)

    # Record new call
    cost_tracker.record_call("gpt-4o-mini", "new", 200, 100, 150.0)

    # Export only recent calls
    since = datetime.now() - timedelta(hours=1)
    exports = cost_tracker.export_calls(since=since)

    assert len(exports) == 1
    assert exports[0]["context"] == "new"


def test_reset(cost_tracker):
    """Test resetting the cost tracker."""
    cost_tracker.record_call("gpt-4o-mini", "test", 500, 200, 100.0)
    assert cost_tracker.get_statistics().total_calls == 1

    cost_tracker.reset()

    stats = cost_tracker.get_statistics()
    assert stats.total_calls == 0
    assert stats.total_cost == 0.0


def test_global_tracker_singleton():
    """Test that get_cost_tracker returns a singleton."""
    tracker1 = get_cost_tracker(daily_budget=2.0)
    tracker2 = get_cost_tracker(daily_budget=5.0)  # Budget only used on first init

    assert tracker1 is tracker2
    assert tracker1.daily_budget == 2.0  # First initialization wins


@pytest.mark.asyncio
async def test_track_llm_call_context_manager():
    """Test the track_llm_call context manager."""
    tracker = CostTracker(daily_budget=1.0)

    async with track_llm_call("gpt-4o-mini", "test_context", tracker=tracker) as tracking:
        # Simulate LLM call
        await asyncio.sleep(0.01)  # Small delay to simulate latency
        tracking.set_tokens(prompt_tokens=500, completion_tokens=200)

    stats = tracker.get_statistics()
    assert stats.total_calls == 1
    assert stats.total_prompt_tokens == 500
    assert stats.total_completion_tokens == 200
    assert stats.calls_by_context["test_context"] == 1
    assert stats.average_latency_ms > 0


@pytest.mark.asyncio
async def test_track_llm_call_with_error():
    """Test context manager records errors correctly."""
    tracker = CostTracker(daily_budget=1.0)

    with pytest.raises(ValueError):
        async with track_llm_call("gpt-4o-mini", "error_context", tracker=tracker) as tracking:
            tracking.set_tokens(100, 50)
            raise ValueError("Simulated API error")

    stats = tracker.get_statistics()
    assert stats.total_calls == 1
    assert stats.failed_calls == 1
    assert stats.successful_calls == 0

    exports = tracker.export_calls()
    assert exports[0]["error_message"] == "Simulated API error"


def test_estimate_tokens():
    """Test token estimation utility."""
    text = "This is a test string with approximately 50 characters"
    estimated = estimate_tokens(text)

    # Should use ~4 characters per token
    assert estimated == len(text) // 4


def test_concurrent_calls():
    """Test thread-safety of concurrent call recording."""
    tracker = CostTracker(daily_budget=10.0)

    def record_multiple():
        for _ in range(10):
            tracker.record_call("gpt-4o-mini", "concurrent", 100, 50, 100.0)
            time.sleep(0.001)

    threads = [threading.Thread(target=record_multiple) for _ in range(5)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    stats = tracker.get_statistics()
    assert stats.total_calls == 50  # 5 threads * 10 calls each
    assert stats.calls_by_context["concurrent"] == 50


def test_cost_breakdown_by_model(cost_tracker):
    """Test cost breakdown by model type."""
    # Record calls with different models
    cost_tracker.record_call("gpt-4o-mini", "test", 1000, 500, 100.0)
    cost_tracker.record_call("gpt-4o-mini", "test", 1000, 500, 100.0)
    cost_tracker.record_call("gpt-4o", "test", 1000, 500, 100.0)

    stats = cost_tracker.get_statistics()

    # GPT-4o should be significantly more expensive
    assert stats.cost_by_model["gpt-4o"] > stats.cost_by_model["gpt-4o-mini"]

    # Two gpt-4o-mini calls should cost about 2x one call
    single_call_cost = cost_tracker.estimate_cost("gpt-4o-mini", 1000, 500)
    assert abs(stats.cost_by_model["gpt-4o-mini"] - (single_call_cost * 2)) < 0.0001


def test_average_cost_per_call():
    """Test average cost per call calculation."""
    tracker = CostTracker(daily_budget=1.0)

    # Record 3 calls
    tracker.record_call("gpt-4o-mini", "test", 500, 200, 100.0)
    tracker.record_call("gpt-4o-mini", "test", 600, 250, 120.0)
    tracker.record_call("gpt-4o-mini", "test", 400, 150, 90.0)

    status = tracker.check_budget(timeframe="daily")

    expected_avg = status["spent"] / 3
    assert abs(status["average_cost_per_call"] - expected_avg) < 0.0001


@pytest.mark.asyncio
async def test_realistic_usage_scenario():
    """Test a realistic multi-critic usage scenario."""
    tracker = CostTracker(daily_budget=1.0)

    # Simulate 10 decisions with 3 critics each, 20% using LLM
    num_decisions = 10
    llm_usage_rate = 0.2

    for decision_num in range(num_decisions):
        for critic in ["safety", "efficiency", "goal_alignment"]:
            if decision_num < (num_decisions * llm_usage_rate):
                # LLM evaluation (20% of decisions)
                await asyncio.sleep(0.001)  # Simulate LLM latency
                tracker.record_call(
                    model="gpt-4o-mini",
                    context=f"{critic}_critic",
                    prompt_tokens=800,
                    completion_tokens=300,
                    latency_ms=1500.0,
                    success=True,
                )

    stats = tracker.get_statistics()

    # 10 decisions * 3 critics * 0.2 LLM rate = 6 LLM calls
    assert stats.total_calls == 6
    assert stats.successful_calls == 6

    # Check budget
    status = tracker.check_budget(timeframe="daily")
    assert status["spent"] < status["budget"]  # Should be well under budget
    assert status["on_track"] is True

    # Cost should be very low (< $0.01 for 6 calls with small token counts)
    assert stats.total_cost < 0.01
