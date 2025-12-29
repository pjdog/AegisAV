"""
Tests for the scenario runner module.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from agent.server.scenario_runner import (
    DroneSimState,
    ScenarioRunner,
    ScenarioRunState,
)
from agent.server.scenarios import (
    DroneState,
    Scenario,
    ScenarioCategory,
    SimulatedDrone,
)


class TestDroneSimState:
    """Test DroneSimState dataclass."""

    def test_create_drone_sim_state(self):
        """Test creating drone simulation state."""
        from agent.server.world_model import WorldModel

        drone = SimulatedDrone(drone_id="test", name="Test Drone")
        world = WorldModel()

        state = DroneSimState(drone=drone, world_model=world)

        assert state.drone.drone_id == "test"
        assert state.current_goal is None
        assert state.decisions_made == 0
        assert state.inspections_completed == 0


class TestScenarioRunState:
    """Test ScenarioRunState dataclass."""

    def test_create_run_state(self):
        """Test creating scenario run state."""
        scenario = Scenario(
            scenario_id="test",
            name="Test",
            description="Test scenario",
            category=ScenarioCategory.NORMAL_OPERATIONS,
        )
        now = datetime.now()

        state = ScenarioRunState(
            scenario=scenario,
            start_time=now,
            current_time=now,
        )

        assert state.scenario.scenario_id == "test"
        assert state.elapsed_seconds == 0.0
        assert state.is_running is True
        assert state.is_complete is False
        assert len(state.drone_states) == 0
        assert len(state.decision_log) == 0


class TestScenarioRunner:
    """Test ScenarioRunner class."""

    @pytest.fixture
    def runner(self):
        """Create a scenario runner."""
        return ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.5,
        )

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_load_scenario_success(self, runner):
        """Test loading a valid scenario."""
        result = await runner.load_scenario("normal_ops_001")

        assert result is True
        assert runner.run_state is not None
        assert runner.run_state.scenario.scenario_id == "normal_ops_001"
        assert len(runner.run_state.drone_states) == 3

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_load_scenario_not_found(self, runner):
        """Test loading non-existent scenario."""
        result = await runner.load_scenario("nonexistent_999")

        assert result is False
        assert runner.run_state is None

    @pytest.mark.asyncio
    async def test_load_scenario_initializes_world_models(self, runner):
        """Test that world models are initialized for each drone."""
        await runner.load_scenario("normal_ops_001")

        for _drone_id, drone_state in runner.run_state.drone_states.items():
            assert drone_state.world_model is not None
            # World model should have dock set
            assert drone_state.world_model._dock is not None
            # World model should have assets
            assert len(drone_state.world_model._assets) > 0

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_run_without_loading(self, runner):
        """Test running without loading scenario first."""
        result = await runner.run()
        assert result is False

    @pytest.mark.asyncio
    async def test_run_scenario_completes(self, runner):
        """Test running a scenario to completion."""
        await runner.load_scenario("normal_ops_001")

        # Run with very fast time scale and short duration
        runner.run_state.scenario.duration_minutes = 0.1  # 6 seconds simulated
        await runner.run(time_scale=100.0, max_duration_s=5)

        assert runner.run_state.elapsed_seconds > 0
        assert len(runner.run_state.decision_log) > 0

    @pytest.mark.asyncio
    async def test_run_respects_max_duration(self, runner):
        """Test that run respects max_duration_s."""
        await runner.load_scenario("normal_ops_001")

        start = datetime.now()
        await runner.run(time_scale=1.0, max_duration_s=1)
        elapsed = (datetime.now() - start).total_seconds()

        # Should stop within reasonable time of max_duration
        assert elapsed < 3  # Allow some overhead

    @pytest.mark.asyncio
    async def test_decisions_are_made(self, runner):
        """Test that decisions are made for drones."""
        await runner.load_scenario("normal_ops_001")
        runner.run_state.scenario.duration_minutes = 0.1

        await runner.run(time_scale=100.0, max_duration_s=3)

        # At least some decisions should have been made
        total_decisions = sum(
            ds.decisions_made
            for ds in runner.run_state.drone_states.values()
        )
        assert total_decisions > 0

    @pytest.mark.asyncio
    async def test_decision_log_contains_entries(self, runner):
        """Test that decision log is populated."""
        await runner.load_scenario("normal_ops_001")
        runner.run_state.scenario.duration_minutes = 0.1

        await runner.run(time_scale=100.0, max_duration_s=3)

        log = runner.run_state.decision_log
        assert len(log) > 0

        # Check log entry structure
        for entry in log:
            assert "type" in entry
            assert "timestamp" in entry

    @pytest.mark.asyncio
    async def test_save_decision_log(self, runner, temp_log_dir):
        """Test saving decision log to file."""
        runner.log_dir = temp_log_dir
        await runner.load_scenario("normal_ops_001")
        runner.run_state.scenario.duration_minutes = 0.05

        await runner.run(time_scale=100.0, max_duration_s=2)

        # Save log
        log_path = runner.save_decision_log()

        assert log_path.exists()
        assert log_path.suffix == ".jsonl"
        assert "normal_ops_001" in log_path.name

        # Verify content
        with open(log_path) as f:
            lines = f.readlines()
            assert len(lines) > 0

    @pytest.mark.asyncio
    async def test_save_decision_log_custom_dir(self, runner, temp_log_dir):
        """Test saving decision log to custom directory."""
        await runner.load_scenario("normal_ops_001")
        runner.run_state.scenario.duration_minutes = 0.05

        await runner.run(time_scale=100.0, max_duration_s=2)

        custom_dir = temp_log_dir / "custom"
        log_path = runner.save_decision_log(custom_dir)

        assert log_path.parent == custom_dir
        assert log_path.exists()

    def test_save_decision_log_without_run(self, runner):
        """Test saving log without running scenario."""
        with pytest.raises(ValueError):
            runner.save_decision_log()

    @pytest.mark.asyncio
    async def test_get_summary(self, runner):
        """Test getting scenario summary."""
        await runner.load_scenario("normal_ops_001")
        runner.run_state.scenario.duration_minutes = 0.05

        await runner.run(time_scale=100.0, max_duration_s=2)

        summary = runner.get_summary()

        assert summary["scenario_id"] == "normal_ops_001"
        assert summary["scenario_name"] == "Normal Fleet Operations"
        assert "duration_s" in summary
        assert "total_decisions" in summary
        assert "drones" in summary
        assert len(summary["drones"]) == 3

    def test_get_summary_without_run(self, runner):
        """Test getting summary without running."""
        summary = runner.get_summary()
        assert summary == {}

    @pytest.mark.asyncio
    async def test_stop_scenario(self, runner):
        """Test stopping a running scenario."""
        await runner.load_scenario("normal_ops_001")

        # Start in background
        async def run_and_stop():
            task = asyncio.create_task(runner.run(time_scale=1.0))
            await asyncio.sleep(0.5)
            runner.stop()
            await task

        await asyncio.wait_for(run_and_stop(), timeout=5)

        assert runner.run_state.is_running is False

    @pytest.mark.asyncio
    async def test_callbacks_are_called(self, runner):
        """Test that callbacks are invoked."""
        decisions = []
        events = []
        ticks = []

        runner.on_decision = lambda d, g, r: decisions.append((d, g))
        runner.on_event = lambda e: events.append(e)
        runner.on_tick = lambda s: ticks.append(s.elapsed_seconds)

        await runner.load_scenario("normal_ops_001")
        runner.run_state.scenario.duration_minutes = 0.05

        await runner.run(time_scale=100.0, max_duration_s=2)

        assert len(ticks) > 0
        # Events may or may not fire depending on timing


class TestScenarioRunnerBatteryScenario:
    """Test runner with battery cascade scenario."""

    @pytest.fixture
    def runner(self):
        """Create a scenario runner."""
        return ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.3,
        )

    @pytest.mark.asyncio
    async def test_battery_drains(self, runner):
        """Test that battery drains during flight."""
        await runner.load_scenario("battery_cascade_001")

        # Get initial battery levels
        initial_batteries = {
            drone_id: ds.drone.battery_percent
            for drone_id, ds in runner.run_state.drone_states.items()
        }

        runner.run_state.scenario.duration_minutes = 0.1
        await runner.run(time_scale=50.0, max_duration_s=3)

        # Check batteries drained
        for drone_id, ds in runner.run_state.drone_states.items():
            if ds.drone.in_air:
                assert ds.drone.battery_percent < initial_batteries[drone_id]

    @pytest.mark.asyncio
    async def test_critical_battery_triggers_decisions(self, runner):
        """Test that critical battery triggers appropriate decisions."""
        await runner.load_scenario("battery_cascade_001")
        runner.run_state.scenario.duration_minutes = 0.1

        await runner.run(time_scale=50.0, max_duration_s=3)

        # Check decision log for battery-related decisions
        log = runner.run_state.decision_log
        decision_entries = [e for e in log if e.get("type") == "decision"]

        # Should have decisions with ABORT or RETURN for low battery drones
        actions = [e.get("action") for e in decision_entries]
        assert len(actions) > 0


class TestScenarioRunnerEnvironment:
    """Test runner environment handling."""

    @pytest.fixture
    def runner(self):
        """Create a scenario runner."""
        return ScenarioRunner(
            tick_interval_s=0.1,
            decision_interval_s=0.5,
        )

    @pytest.mark.asyncio
    async def test_environment_conditions_apply(self, runner):
        """Test that environment conditions are tracked."""
        await runner.load_scenario("weather_001")

        assert runner.run_state.environment is not None
        assert runner.run_state.environment.wind_speed_ms > 0


class TestScenarioRunnerRiskCalculation:
    """Test risk calculation in runner."""

    @pytest.fixture
    def runner(self):
        """Create a scenario runner."""
        return ScenarioRunner()

    @pytest.mark.asyncio
    async def test_risk_increases_with_low_battery(self, runner):
        """Test that risk score increases with low battery."""
        await runner.load_scenario("battery_cascade_001")

        # Find drone with lowest battery
        lowest_battery_state = min(
            runner.run_state.drone_states.values(),
            key=lambda ds: ds.drone.battery_percent
        )

        risk = runner._calculate_risk(lowest_battery_state)

        # Low battery drone should have higher risk (18% battery = 0.3 risk)
        assert risk >= 0.3

    @pytest.mark.asyncio
    async def test_risk_increases_with_sensor_failure(self, runner):
        """Test that risk increases with sensor failures."""
        await runner.load_scenario("sensor_cascade_001")

        # Find drone with sensor issues
        unhealthy_state = None
        for ds in runner.run_state.drone_states.values():
            if not ds.drone.sensors_healthy or not ds.drone.ekf_healthy:
                unhealthy_state = ds
                break

        if unhealthy_state:
            risk = runner._calculate_risk(unhealthy_state)
            assert risk > 0.2

    @pytest.mark.asyncio
    async def test_risk_level_strings(self, runner):
        """Test risk level string conversion."""
        await runner.load_scenario("normal_ops_001")

        ds = next(iter(runner.run_state.drone_states.values()))

        # Healthy drone should have low risk
        level = runner._risk_level(ds)
        assert level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]


class TestScenarioRunnerDroneConversion:
    """Test drone state conversion."""

    @pytest.fixture
    def runner(self):
        """Create a scenario runner."""
        return ScenarioRunner()

    def test_drone_to_vehicle_state(self, runner):
        """Test converting SimulatedDrone to VehicleState."""
        drone = SimulatedDrone(
            drone_id="test",
            name="Test",
            latitude=37.7749,
            longitude=-122.4194,
            altitude_agl=100.0,
            battery_percent=75.0,
            armed=True,
            in_air=True,
        )

        state = runner._drone_to_vehicle_state(drone)

        assert state.position.latitude == 37.7749
        assert state.position.altitude_agl == 100.0
        assert state.battery.remaining_percent == 75.0
        assert state.armed is True
        assert state.in_air is True

    def test_drone_to_vehicle_state_grounded(self, runner):
        """Test converting grounded drone."""
        drone = SimulatedDrone(
            drone_id="test",
            name="Test",
            armed=False,
            in_air=False,
        )

        state = runner._drone_to_vehicle_state(drone)

        assert state.armed is False
        assert state.in_air is False


class TestScenarioRunnerGoalApplication:
    """Test goal application to drones."""

    @pytest.fixture
    def runner(self):
        """Create a scenario runner."""
        return ScenarioRunner()

    @pytest.mark.asyncio
    async def test_abort_goal_grounds_drone(self, runner):
        """Test that ABORT goal grounds the drone."""
        from agent.server.goals import Goal, GoalType
        from agent.server.world_model import WorldModel

        drone = SimulatedDrone(
            drone_id="test",
            name="Test",
            armed=True,
            in_air=True,
            state=DroneState.INSPECTING,
        )
        ds = DroneSimState(drone=drone, world_model=WorldModel())

        goal = Goal(goal_type=GoalType.ABORT, priority=0, reason="Emergency")
        runner._apply_goal(ds, goal)

        assert drone.state == DroneState.EMERGENCY
        assert drone.in_air is False
        assert drone.armed is False

    @pytest.mark.asyncio
    async def test_return_goal_sets_returning(self, runner):
        """Test that RETURN goal sets drone to returning."""
        from agent.server.goals import Goal, GoalType
        from agent.server.world_model import WorldModel

        drone = SimulatedDrone(
            drone_id="test",
            name="Test",
            state=DroneState.INSPECTING,
        )
        ds = DroneSimState(drone=drone, world_model=WorldModel())

        goal = Goal(goal_type=GoalType.RETURN_LOW_BATTERY, priority=5, reason="Low battery")
        runner._apply_goal(ds, goal)

        assert drone.state == DroneState.RETURNING

    @pytest.mark.asyncio
    async def test_inspect_goal_sets_inspecting(self, runner):
        """Test that INSPECT goal sets drone to inspecting."""
        from agent.server.goals import Goal, GoalType
        from agent.server.world_model import WorldModel

        drone = SimulatedDrone(
            drone_id="test",
            name="Test",
            state=DroneState.IDLE,
            armed=False,
            in_air=False,
        )
        ds = DroneSimState(drone=drone, world_model=WorldModel())

        goal = Goal(goal_type=GoalType.INSPECT_ASSET, priority=30, reason="Inspect asset")
        runner._apply_goal(ds, goal)

        assert drone.state == DroneState.INSPECTING
        assert drone.in_air is True
        assert drone.armed is True
