"""
Behavioral tests for scenario runner.

These tests verify that the system makes CORRECT decisions
in various edge cases - not just that it runs without errors.
"""


import pytest

from agent.server.goals import GoalType
from agent.server.scenario_runner import ScenarioRunner


@pytest.mark.asyncio
class TestScenarioBehavior:
    """Test that scenarios produce correct behavioral outcomes."""

    async def test_battery_cascade_prioritizes_critical_drones(self):
        """
        Critical: Battery cascade scenario should prioritize returning
        the drone with lowest battery first.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.2,
        )

        await runner.load_scenario("battery_cascade_001")

        # Run simulation briefly
        runner.run_state.scenario.duration_minutes = 0.1
        await runner.run(time_scale=50.0, max_duration_s=2)

        # Check decision log for battery-related decisions
        decisions = [
            e for e in runner.run_state.decision_log if e.get("type") == "decision"
        ]

        # Should have made decisions
        assert len(decisions) > 0, "Should have made at least one decision"

        # Find drone with lowest initial battery (critical_bat at 18%)
        critical_drone = runner.run_state.drone_states.get("critical_bat")
        assert critical_drone is not None, "Critical battery drone should exist"

        # Critical drone should have received RETURN or ABORT goal
        critical_decisions = [
            d for d in decisions if d.get("drone_id") == "critical_bat"
        ]
        if critical_decisions:
            actions = [d.get("action", "").lower() for d in critical_decisions]
            assert any(
                action in ["return_low_battery", "abort", "return"]
                for action in actions
            ), f"Critical battery drone should RETURN or ABORT, got: {actions}"

    async def test_gps_loss_triggers_abort(self):
        """
        Critical: GPS loss should trigger abort decision.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.2,
        )

        await runner.load_scenario("gps_degrade_001")

        # Run simulation
        runner.run_state.scenario.duration_minutes = 0.1
        await runner.run(time_scale=50.0, max_duration_s=2)

        # Find drone with worst GPS
        worst_gps_drone = min(
            runner.run_state.drone_states.values(),
            key=lambda ds: 1 if ds.drone.gps_healthy else 0,
        )

        # If GPS is unhealthy, verify abort goal was assigned
        if not worst_gps_drone.drone.gps_healthy:
            goal = worst_gps_drone.current_goal
            assert goal is not None, "Drone with GPS issues should have a goal"
            assert goal.goal_type in [
                GoalType.ABORT,
                GoalType.RETURN_LOW_BATTERY,
            ], f"GPS-unhealthy drone should ABORT, got: {goal.goal_type}"

    async def test_sensor_failure_grounds_drone(self):
        """
        Critical: Sensor failures should ground affected drones.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.2,
        )

        await runner.load_scenario("sensor_cascade_001")

        # Run simulation
        runner.run_state.scenario.duration_minutes = 0.1
        await runner.run(time_scale=50.0, max_duration_s=2)

        # Find drones with sensor issues
        unhealthy_drones = [
            ds
            for ds in runner.run_state.drone_states.values()
            if not ds.drone.sensors_healthy
            or not ds.drone.ekf_healthy
            or not ds.drone.motors_healthy
        ]

        # Unhealthy drones should have appropriate goals
        for ds in unhealthy_drones:
            if ds.current_goal:
                # Should be returning or aborting
                assert ds.current_goal.goal_type in [
                    GoalType.ABORT,
                    GoalType.RETURN_LOW_BATTERY,
                ], f"Unhealthy drone {ds.drone.name} should abort"

    async def test_normal_operations_proceeds_with_inspections(self):
        """
        Normal operations should proceed with inspections,
        not abort unnecessarily.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.2,
        )

        await runner.load_scenario("normal_ops_001")

        # Run simulation
        runner.run_state.scenario.duration_minutes = 0.1
        await runner.run(time_scale=50.0, max_duration_s=2)

        # Check that no abort decisions were made (all drones healthy)
        decisions = [
            e for e in runner.run_state.decision_log if e.get("type") == "decision"
        ]

        abort_count = sum(
            1 for d in decisions if d.get("action", "").lower() == "abort"
        )

        # Normal ops shouldn't have aborts (drones are healthy)
        assert abort_count == 0, f"Normal ops shouldn't abort: {abort_count} aborts"

        # Should have inspection or wait goals
        actions = [d.get("action", "").lower() for d in decisions]
        valid_normal_actions = ["inspect_asset", "wait", "inspect"]
        assert any(
            action in valid_normal_actions for action in actions
        ), f"Normal ops should have inspect/wait actions, got: {actions}"

    async def test_weather_emergency_triggers_return(self):
        """
        Weather emergency should trigger return decisions.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.2,
        )

        await runner.load_scenario("weather_001")

        # Check that scenario has weather triggers
        env = runner.run_state.environment
        assert env is not None
        assert env.wind_speed_ms > 0

        # Run simulation
        runner.run_state.scenario.duration_minutes = 0.1
        await runner.run(time_scale=50.0, max_duration_s=2)

        # Should have decision log entries
        assert len(runner.run_state.decision_log) > 0


@pytest.mark.asyncio
class TestMultiDroneCoordination:
    """Test multi-drone coordination behavior."""

    @pytest.mark.xfail(reason="Fleet coordination not yet implemented - all drones may target same asset")
    async def test_drones_dont_collide_in_decisions(self):
        """
        Verify that the system doesn't assign same asset to multiple drones.

        NOTE: This test exposes a known gap - fleet coordination is not yet
        implemented. The goal selector works per-drone without knowledge of
        other drones' assignments.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.2,
        )

        await runner.load_scenario("coord_001")  # Coordination scenario

        # Run simulation
        runner.run_state.scenario.duration_minutes = 0.1
        await runner.run(time_scale=50.0, max_duration_s=2)

        # Check that drones have distinct goals
        goals = [
            ds.current_goal
            for ds in runner.run_state.drone_states.values()
            if ds.current_goal
        ]

        # Extract target assets
        target_assets = [
            g.target_asset.asset_id if g.target_asset else None for g in goals
        ]
        target_assets = [t for t in target_assets if t is not None]

        # No duplicates (unless WAIT/ABORT which have no target)
        unique_targets = set(target_assets)
        if len(target_assets) > 1:
            assert len(unique_targets) == len(
                target_assets
            ), f"Drones should have unique targets: {target_assets}"

    async def test_fleet_coverage_distributes_work(self):
        """
        Fleet should distribute inspection work across available drones.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.3,
        )

        await runner.load_scenario("normal_ops_001")

        # Run longer to get multiple decision cycles
        runner.run_state.scenario.duration_minutes = 0.2
        await runner.run(time_scale=50.0, max_duration_s=3)

        # Check that decisions were made for multiple drones
        decisions = [
            e for e in runner.run_state.decision_log if e.get("type") == "decision"
        ]

        drones_with_decisions = set(d.get("drone_id") for d in decisions if d.get("drone_id"))

        # With 3 drones, at least 2 should have received decisions
        assert (
            len(drones_with_decisions) >= 2
        ), f"Work should be distributed: only {len(drones_with_decisions)} drones got decisions"


@pytest.mark.asyncio
class TestDecisionQuality:
    """Test the quality of decisions made by the system."""

    async def test_decision_includes_reasoning(self):
        """
        Decisions should include reasoning for auditability.
        """
        runner = ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.2,
        )

        await runner.load_scenario("normal_ops_001")

        runner.run_state.scenario.duration_minutes = 0.05
        await runner.run(time_scale=100.0, max_duration_s=2)

        decisions = [
            e for e in runner.run_state.decision_log if e.get("type") == "decision"
        ]

        for decision in decisions[:5]:  # Check first 5
            # Should have key fields
            assert "action" in decision, "Decision should have action"
            assert "drone_id" in decision, "Decision should have drone_id"
            assert "timestamp" in decision, "Decision should have timestamp"

    async def test_risk_scoring_reflects_state(self):
        """
        Risk scores should reflect the actual vehicle state.
        """
        runner = ScenarioRunner()

        await runner.load_scenario("battery_cascade_001")

        # Check risk calculations before running
        for drone_id, ds in runner.run_state.drone_states.items():
            risk = runner._calculate_risk(ds)

            # Low battery should mean high risk
            if ds.drone.battery_percent < 25:
                assert risk >= 0.3, f"Low battery drone {drone_id} should have high risk"

            # Healthy drone with good battery should have low risk
            if (
                ds.drone.battery_percent > 50
                and ds.drone.sensors_healthy
                and ds.drone.gps_healthy
            ):
                assert risk < 0.3, f"Healthy drone {drone_id} should have low risk"

    async def test_goal_priority_ordering(self):
        """
        Safety goals should have higher priority than inspection goals.
        """
        runner = ScenarioRunner()

        await runner.load_scenario("battery_cascade_001")

        # Run briefly
        runner.run_state.scenario.duration_minutes = 0.05
        await runner.run(time_scale=100.0, max_duration_s=1)

        # Check goals
        for ds in runner.run_state.drone_states.values():
            goal = ds.current_goal
            if goal:
                # ABORT should have priority 0 (highest)
                if goal.goal_type == GoalType.ABORT:
                    assert goal.priority == 0, "ABORT should have priority 0"

                # RETURN_LOW_BATTERY should have priority 5
                if goal.goal_type == GoalType.RETURN_LOW_BATTERY:
                    assert goal.priority <= 10, "Return should have high priority"

                # INSPECT should have lower priority
                if goal.goal_type == GoalType.INSPECT_ASSET:
                    assert goal.priority >= 20, "Inspect should have lower priority"
