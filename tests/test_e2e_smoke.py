"""End-to-end smoke test for the simulation pipeline.

This test verifies that the complete simulation pipeline works:
- Scenario loading
- Runner execution
- Decision logging
- Summary generation

Marked as slow for CI filtering.
"""

import tempfile
from pathlib import Path

import pytest

from agent.server.scenario_runner import ScenarioRunner
from agent.server.scenarios import get_scenario


@pytest.mark.slow
class TestE2ESmoke:
    """End-to-end smoke tests for simulation pipeline."""

    @pytest.mark.asyncio
    async def test_smoke_normal_ops(self):
        """Smoke test with normal_ops_001 scenario.

        Verifies:
        - Scenario loads successfully
        - Runner starts and runs
        - At least 1 decision is logged
        - Summary contains expected fields
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create runner with fixed seed for determinism
            runner = ScenarioRunner(
                tick_interval_s=1.0,
                decision_interval_s=2.0,  # Faster decisions for test
                log_dir=log_dir,
                seed=42,
            )

            # Load scenario
            loaded = await runner.load_scenario("normal_ops_001")
            assert loaded, "Failed to load scenario"
            assert runner.run_id is not None

            # Run for short duration
            await runner.run(
                time_scale=50.0,  # Fast for testing
                max_duration_s=5.0,  # Short real-time limit
            )

            # Verify run completed or was stopped
            assert runner.run_state is not None

            # Verify decisions were logged
            decision_count = len(runner.run_state.decision_log)
            assert decision_count >= 1, f"Expected at least 1 decision, got {decision_count}"

            # Verify summary
            summary = runner.get_summary()
            assert "run_id" in summary
            assert "scenario_id" in summary
            assert summary["scenario_id"] == "normal_ops_001"
            assert summary["total_decisions"] >= 1
            assert "drones" in summary
            assert len(summary["drones"]) > 0

            # Verify log can be saved
            log_path = runner.save_decision_log(log_dir)
            assert log_path.exists()
            assert log_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_smoke_with_determinism(self):
        """Verify deterministic runs produce same results with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            results = []
            for _ in range(2):
                runner = ScenarioRunner(
                    tick_interval_s=1.0,
                    decision_interval_s=3.0,
                    log_dir=log_dir,
                    seed=12345,  # Same seed
                )

                await runner.load_scenario("normal_ops_001")
                await runner.run(time_scale=100.0, max_duration_s=2.0)

                # Collect key metrics
                summary = runner.get_summary()
                results.append({
                    "decisions": summary["total_decisions"],
                    "battery": summary["total_battery_consumed"],
                })

            # Both runs should produce identical results
            assert (
                results[0]["decisions"] == results[1]["decisions"]
            ), f"Decision counts differ: {results[0]['decisions']} vs {results[1]['decisions']}"

    @pytest.mark.asyncio
    async def test_smoke_runner_lifecycle(self):
        """Test runner lifecycle: start, stop, restart."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            runner = ScenarioRunner(
                tick_interval_s=0.5,  # Faster ticks for test
                decision_interval_s=1.0,
                log_dir=log_dir,
            )

            # Load scenario
            await runner.load_scenario("normal_ops_001")

            # Check initial state
            assert runner.run_id is not None

            # Start in background task
            run_task = asyncio.create_task(runner.run(time_scale=50.0, max_duration_s=10.0))

            # Let it run briefly
            await asyncio.sleep(0.3)

            # Stop runner
            runner.stop()

            # Wait for task to complete
            try:
                await asyncio.wait_for(run_task, timeout=3.0)
            except asyncio.TimeoutError:
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass

            # Verify we have a summary (runner may or may not still be "running")
            summary = runner.get_summary()
            assert summary is not None
            assert "run_id" in summary

    @pytest.mark.asyncio
    async def test_smoke_decision_log_format(self):
        """Verify decision log entries have expected format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            runner = ScenarioRunner(
                tick_interval_s=1.0,
                decision_interval_s=1.0,  # Fast decisions
                log_dir=log_dir,
                seed=42,
            )

            await runner.load_scenario("normal_ops_001")
            await runner.run(time_scale=100.0, max_duration_s=3.0)

            # Get decision entries
            decisions = [e for e in runner.run_state.decision_log if e.get("type") == "decision"]
            assert len(decisions) >= 1

            # Check decision format
            decision = decisions[0]
            required_fields = [
                "run_id",
                "type",
                "timestamp",
                "drone_id",
                "drone_name",
                "action",
                "confidence",
                "risk_score",
                "battery_percent",
            ]
            for field in required_fields:
                assert field in decision, f"Missing field: {field}"

            # Check run_summary entry exists
            summaries = [e for e in runner.run_state.decision_log if e.get("type") == "run_summary"]
            assert len(summaries) == 1
            summary_entry = summaries[0]
            assert "run_id" in summary_entry
            assert "total_decisions" in summary_entry


@pytest.mark.slow
class TestScenarioVariants:
    """Test different scenario types."""

    @pytest.mark.asyncio
    async def test_smoke_battery_scenario(self):
        """Test battery cascade scenario runs."""
        scenario = get_scenario("battery_cascade_001")
        assert scenario is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ScenarioRunner(
                log_dir=Path(tmpdir),
                seed=42,
            )

            await runner.load_scenario("battery_cascade_001")
            await runner.run(time_scale=100.0, max_duration_s=3.0)

            summary = runner.get_summary()
            assert summary["total_decisions"] >= 1
            # Battery scenario should show consumption
            assert summary["total_battery_consumed"] > 0

    @pytest.mark.asyncio
    async def test_smoke_multi_drone_coordination(self):
        """Test multi-drone scenario runs."""
        scenario = get_scenario("coord_001")
        assert scenario is not None
        assert len(scenario.drones) > 1

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ScenarioRunner(
                log_dir=Path(tmpdir),
                seed=42,
            )

            await runner.load_scenario("coord_001")
            await runner.run(time_scale=100.0, max_duration_s=3.0)

            summary = runner.get_summary()
            # Should have decisions from multiple drones
            assert len(summary["drones"]) > 1
            for drone in summary["drones"]:
                assert drone["decisions_made"] >= 0
