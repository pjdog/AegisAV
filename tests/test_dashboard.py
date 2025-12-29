"""
Tests for dashboard routes and functionality.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from agent.server import dashboard as dashboard_module


class TestDashboardHelpers:
    """Test dashboard helper functions."""

    def test_calculate_relative_pos_same_location(self):
        """Test relative position calculation for same location."""
        result = dashboard_module._calculate_relative_pos(37.7749, -122.4194, 37.7749, -122.4194)
        assert result["x"] == pytest.approx(0.0, abs=0.01)
        assert result["y"] == pytest.approx(0.0, abs=0.01)

    def test_calculate_relative_pos_north(self):
        """Test relative position calculation for point to the north."""
        # 1 degree north should be ~111km
        result = dashboard_module._calculate_relative_pos(37.0, -122.0, 38.0, -122.0)
        assert result["x"] == pytest.approx(0.0, abs=1.0)
        assert result["y"] == pytest.approx(111111, rel=0.01)

    def test_calculate_relative_pos_east(self):
        """Test relative position calculation for point to the east."""
        result = dashboard_module._calculate_relative_pos(37.0, -122.0, 37.0, -121.0)
        # At 37 degrees latitude, 1 degree longitude is about 88km
        assert result["x"] > 80000
        assert result["y"] == pytest.approx(0.0, abs=1.0)

    def test_load_entries_empty_file(self):
        """Test loading entries from empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        entries = dashboard_module._load_entries(temp_path)
        assert entries == []
        temp_path.unlink()

    def test_load_entries_with_data(self):
        """Test loading entries from file with data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"action": "INSPECT", "risk_score": 0.3}\n')
            f.write('{"action": "WAIT", "risk_score": 0.1}\n')
            f.write("\n")  # Empty line should be skipped
            f.write('{"action": "RETURN", "risk_score": 0.8}\n')
            temp_path = Path(f.name)

        entries = dashboard_module._load_entries(temp_path)
        assert len(entries) == 3
        assert entries[0]["action"] == "INSPECT"
        assert entries[1]["action"] == "WAIT"
        assert entries[2]["action"] == "RETURN"
        temp_path.unlink()

    def test_load_entries_nonexistent_file(self):
        """Test loading entries from non-existent file returns empty list."""
        entries = dashboard_module._load_entries(Path("/nonexistent/path.jsonl"))
        assert entries == []

    def test_list_runs_empty_dir(self):
        """Test listing runs from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = dashboard_module._list_runs(Path(tmpdir))
            assert runs == []

    def test_list_runs_with_files(self):
        """Test listing runs from directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "decisions_20240101_120000.jsonl").touch()
            (log_dir / "decisions_20240102_120000.jsonl").touch()
            (log_dir / "other_file.txt").touch()  # Should be ignored

            runs = dashboard_module._list_runs(log_dir)
            assert len(runs) == 2
            assert "20240101_120000" in runs
            assert "20240102_120000" in runs
            # Should be sorted
            assert runs == sorted(runs)

    def test_summarize_empty_entries(self):
        """Test summarize with empty entries."""
        summary = dashboard_module._summarize([])
        assert summary["total"] == 0
        assert summary["avg_risk"] == 0.0
        assert summary["max_risk"] == 0.0
        assert summary["time_start"] is None
        assert summary["time_end"] is None

    def test_summarize_with_entries(self):
        """Test summarize with entries."""
        entries = [
            {"risk_score": 0.2, "timestamp": "2024-01-01T12:00:00"},
            {"risk_score": 0.4, "timestamp": "2024-01-01T12:01:00"},
            {"risk_score": 0.6, "timestamp": "2024-01-01T12:02:00"},
        ]
        summary = dashboard_module._summarize(entries)
        assert summary["total"] == 3
        assert summary["avg_risk"] == pytest.approx(0.4, abs=0.01)
        assert summary["max_risk"] == 0.6
        assert summary["time_start"] == "2024-01-01T12:00:00"
        assert summary["time_end"] == "2024-01-01T12:02:00"

    def test_series_creates_risk_and_battery(self):
        """Test series creates risk and battery series."""
        entries = [
            {"timestamp": "T1", "risk_score": 0.3, "battery_percent": 80},
            {"timestamp": "T2", "risk_score": 0.5, "battery_percent": 75},
        ]
        risk_series, battery_series = dashboard_module._series(entries)

        assert len(risk_series) == 2
        assert len(battery_series) == 2
        assert risk_series[0] == {"t": "T1", "value": 0.3}
        assert battery_series[0] == {"t": "T1", "value": 80}

    def test_action_counts(self):
        """Test action counting."""
        entries = [
            {"action": "INSPECT"},
            {"action": "INSPECT"},
            {"action": "WAIT"},
            {"action": "RETURN"},
        ]
        counts = dashboard_module._action_counts(entries)

        # Convert to dict for easier testing
        count_dict = {c["action"]: c["count"] for c in counts}
        assert count_dict["INSPECT"] == 2
        assert count_dict["WAIT"] == 1
        assert count_dict["RETURN"] == 1

    def test_recent_limits_entries(self):
        """Test recent limits entries to specified amount."""
        entries = [{"action": f"action_{i}"} for i in range(20)]
        recent = dashboard_module._recent(entries, limit=5)

        assert len(recent) == 5
        # Should be in reverse order (most recent first)
        assert recent[0]["action"] == "action_19"

    def test_recent_includes_spatial_context(self):
        """Test recent includes spatial context when available."""
        entries = [
            {
                "action": "INSPECT",
                "vehicle_position": {"lat": 37.7749, "lon": -122.4194},
                "assets": [
                    {"id": "asset1", "type": "solar_panel", "lat": 37.7750, "lon": -122.4193}
                ],
            }
        ]
        recent = dashboard_module._recent(entries)

        assert len(recent) == 1
        assert len(recent[0]["spatial_context"]) == 1
        assert recent[0]["spatial_context"][0]["id"] == "asset1"


class TestDashboardRoutes:
    """Test dashboard API routes."""

    @pytest.fixture
    def app_with_dashboard(self, tmp_path):
        """Create FastAPI app with dashboard routes."""
        app = FastAPI()
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        dashboard_module.add_dashboard_routes(app, log_dir)
        return app, log_dir

    @pytest_asyncio.fixture
    async def client(self, app_with_dashboard):
        """Create async test client."""
        app, _ = app_with_dashboard
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_dashboard_route_no_build(self, app_with_dashboard):
        """Test dashboard route when build is missing."""
        app, _ = app_with_dashboard
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/dashboard")
        # Should return either HTML file or missing message
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_runs_empty(self, client):
        """Test listing runs when empty."""
        response = await client.get("/api/dashboard/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["latest"] is None

    @pytest.mark.asyncio
    async def test_list_runs_with_data(self, app_with_dashboard):
        """Test listing runs with data."""
        app, log_dir = app_with_dashboard
        transport = httpx.ASGITransport(app=app)

        # Create some run files
        (log_dir / "decisions_run1.jsonl").write_text('{"action": "WAIT"}\n')
        (log_dir / "decisions_run2.jsonl").write_text('{"action": "INSPECT"}\n')

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/dashboard/runs")
        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 2
        assert data["latest"] is not None

    @pytest.mark.asyncio
    async def test_run_data_not_found(self, client):
        """Test getting run data for non-existent run."""
        response = await client.get("/api/dashboard/run/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_run_data_success(self, app_with_dashboard):
        """Test getting run data successfully."""
        app, log_dir = app_with_dashboard
        transport = httpx.ASGITransport(app=app)

        # Create run file with entries
        run_file = log_dir / "decisions_test_run.jsonl"
        entries = [
            {"action": "INSPECT", "risk_score": 0.3, "timestamp": "T1", "battery_percent": 80},
            {"action": "WAIT", "risk_score": 0.2, "timestamp": "T2", "battery_percent": 78},
        ]
        run_file.write_text("\n".join(json.dumps(e) for e in entries))

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/dashboard/run/test_run")
        assert response.status_code == 200
        data = response.json()

        assert data["run_id"] == "test_run"
        assert data["summary"]["total"] == 2
        assert len(data["actions"]) > 0
        assert len(data["risk_series"]) == 2
        assert len(data["battery_series"]) == 2
        assert len(data["recent"]) == 2


class TestScenarioRoutes:
    """Test scenario API routes."""

    @pytest.fixture
    def app_with_scenarios(self, tmp_path):
        """Create FastAPI app with dashboard routes including scenarios."""
        app = FastAPI()
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        dashboard_module.add_dashboard_routes(app, log_dir)
        return app

    @pytest_asyncio.fixture
    async def client(self, app_with_scenarios):
        """Create async test client."""
        transport = httpx.ASGITransport(app=app_with_scenarios)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_list_scenarios(self, client):
        """Test listing all scenarios."""
        response = await client.get("/api/dashboard/scenarios")
        assert response.status_code == 200
        data = response.json()

        assert "scenarios" in data
        assert "total" in data
        assert "categories" in data
        assert "difficulties" in data
        assert data["total"] >= 7  # At least 7 preloaded scenarios
        assert len(data["scenarios"]) == data["total"]

    @pytest.mark.asyncio
    async def test_list_scenarios_by_category(self, client):
        """Test filtering scenarios by category."""
        response = await client.get("/api/dashboard/scenarios?category=battery_critical")
        assert response.status_code == 200
        data = response.json()

        assert data["total"] >= 1
        for scenario in data["scenarios"]:
            assert scenario["category"] == "battery_critical"

    @pytest.mark.asyncio
    async def test_list_scenarios_invalid_category(self, client):
        """Test filtering with invalid category."""
        response = await client.get("/api/dashboard/scenarios?category=invalid_category")
        assert response.status_code == 400
        assert "Invalid category" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_scenarios_by_difficulty(self, client):
        """Test filtering scenarios by difficulty."""
        response = await client.get("/api/dashboard/scenarios?difficulty=hard")
        assert response.status_code == 200
        data = response.json()

        for scenario in data["scenarios"]:
            assert scenario["difficulty"] == "hard"

    @pytest.mark.asyncio
    async def test_get_scenario_detail(self, client):
        """Test getting specific scenario details."""
        response = await client.get("/api/dashboard/scenarios/normal_ops_001")
        assert response.status_code == 200
        data = response.json()

        assert data["scenario_id"] == "normal_ops_001"
        assert data["name"] == "Normal Fleet Operations"
        assert data["category"] == "normal_operations"
        assert "drones" in data
        assert "assets" in data
        assert "environment" in data
        assert "events" in data
        assert len(data["drones"]) == 3
        assert len(data["assets"]) == 3

    @pytest.mark.asyncio
    async def test_get_scenario_not_found(self, client):
        """Test getting non-existent scenario."""
        response = await client.get("/api/dashboard/scenarios/nonexistent_999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_scenario_drone_details(self, client):
        """Test that drone details are included."""
        response = await client.get("/api/dashboard/scenarios/battery_cascade_001")
        assert response.status_code == 200
        data = response.json()

        # Check drone structure
        for drone in data["drones"]:
            assert "drone_id" in drone
            assert "name" in drone
            assert "battery_percent" in drone
            assert "state" in drone
            assert "latitude" in drone
            assert "sensors_healthy" in drone

        # Verify varying battery levels in battery cascade
        batteries = [d["battery_percent"] for d in data["drones"]]
        assert min(batteries) < 30  # At least one critical
        assert max(batteries) > 50  # At least one healthy

    @pytest.mark.asyncio
    async def test_scenario_asset_details(self, client):
        """Test that asset details are included."""
        response = await client.get("/api/dashboard/scenarios/multi_anom_001")
        assert response.status_code == 200
        data = response.json()

        # Check asset structure
        for asset in data["assets"]:
            assert "asset_id" in asset
            assert "name" in asset
            assert "asset_type" in asset
            assert "has_anomaly" in asset
            assert "anomaly_severity" in asset

        # Verify anomalies present
        anomaly_assets = [a for a in data["assets"] if a["has_anomaly"]]
        assert len(anomaly_assets) >= 3

    @pytest.mark.asyncio
    async def test_scenario_environment_details(self, client):
        """Test that environment details are included."""
        response = await client.get("/api/dashboard/scenarios/weather_001")
        assert response.status_code == 200
        data = response.json()

        env = data["environment"]
        assert "wind_speed_ms" in env
        assert "visibility_m" in env
        assert "precipitation" in env
        assert "is_daylight" in env

    @pytest.mark.asyncio
    async def test_scenario_events_timeline(self, client):
        """Test that events timeline is included."""
        response = await client.get("/api/dashboard/scenarios/normal_ops_001")
        assert response.status_code == 200
        data = response.json()

        assert len(data["events"]) > 0
        for event in data["events"]:
            assert "timestamp_offset_s" in event
            assert "event_type" in event
            assert "description" in event

        # Events should be in chronological order
        offsets = [e["timestamp_offset_s"] for e in data["events"]]
        assert offsets == sorted(offsets)


class TestRunnerRoutes:
    """Test scenario runner API routes."""

    @pytest.fixture
    def app_with_runner(self, tmp_path):
        """Create FastAPI app with dashboard and runner routes."""
        app = FastAPI()
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        dashboard_module.add_dashboard_routes(app, log_dir)
        return app, log_dir

    @pytest_asyncio.fixture
    async def client(self, app_with_runner):
        """Create async test client."""
        app, _ = app_with_runner
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest_asyncio.fixture(autouse=True)
    async def reset_runner_state(self):
        """Reset global runner state before each test."""
        import agent.server.dashboard as dashboard_module

        loop = asyncio.get_running_loop()

        # Cancel any running task first
        if (
            dashboard_module._runner_state.run_task
            and not dashboard_module._runner_state.run_task.done()
        ):
            dashboard_module._runner_state.run_task.cancel()
            if dashboard_module._runner_state.run_task.get_loop() is loop:
                try:
                    await dashboard_module._runner_state.run_task
                except asyncio.CancelledError:
                    pass

        # Stop runner if active
        if dashboard_module._runner_state.runner:
            dashboard_module._runner_state.runner.stop()

        dashboard_module._runner_state.runner = None
        dashboard_module._runner_state.run_task = None
        dashboard_module._runner_state.is_running = False
        dashboard_module._runner_state.last_error = None
        yield
        # Cleanup after test
        if (
            dashboard_module._runner_state.run_task
            and not dashboard_module._runner_state.run_task.done()
        ):
            dashboard_module._runner_state.run_task.cancel()
            if dashboard_module._runner_state.run_task.get_loop() is loop:
                try:
                    await dashboard_module._runner_state.run_task
                except asyncio.CancelledError:
                    pass

        if dashboard_module._runner_state.runner:
            dashboard_module._runner_state.runner.stop()

        dashboard_module._runner_state.runner = None
        dashboard_module._runner_state.run_task = None
        dashboard_module._runner_state.is_running = False
        dashboard_module._runner_state.last_error = None

    @pytest.mark.asyncio
    async def test_runner_status_initial(self, client):
        """Test runner status when not running."""
        response = await client.get("/api/dashboard/runner/status")
        assert response.status_code == 200
        data = response.json()

        assert data["is_running"] is False
        assert data["scenario_id"] is None
        assert data["drones"] == []

    @pytest.mark.asyncio
    async def test_runner_start_scenario(self, client):
        """Test starting a scenario."""
        response = await client.post(
            "/api/dashboard/runner/start",
            json={"scenario_id": "normal_ops_001", "time_scale": 10.0, "max_duration_s": 1.0},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "started"
        assert data["scenario_id"] == "normal_ops_001"

        # Check status - the scenario should be loaded
        status_response = await client.get("/api/dashboard/runner/status")
        status_data = status_response.json()
        assert status_data["scenario_id"] == "normal_ops_001"

    @pytest.mark.allow_error_logs
    @pytest.mark.asyncio
    async def test_runner_start_not_found(self, client):
        """Test starting non-existent scenario."""
        response = await client.post(
            "/api/dashboard/runner/start",
            json={"scenario_id": "nonexistent_999"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_runner_start_already_running(self, client):
        """Test starting when already running."""
        import agent.server.dashboard as dashboard_module

        # Start first scenario
        response1 = await client.post(
            "/api/dashboard/runner/start",
            json={"scenario_id": "normal_ops_001", "time_scale": 10.0, "max_duration_s": 5.0},
        )
        assert response1.status_code == 200

        # Manually set is_running to simulate the runner still active
        dashboard_module._runner_state.is_running = True

        # Try to start another
        response = await client.post(
            "/api/dashboard/runner/start",
            json={"scenario_id": "battery_cascade_001"},
        )
        assert response.status_code == 409
        assert "already active" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_runner_stop(self, app_with_runner):
        """Test stopping a running scenario."""
        import agent.server.dashboard as dashboard_module
        from agent.server.scenario_runner import ScenarioRunner

        app, log_dir = app_with_runner

        # Manually set up a runner in "running" state without using the endpoint
        # This avoids async task issues with TestClient
        runner = ScenarioRunner(log_dir=log_dir)
        dashboard_module._runner_state.runner = runner
        dashboard_module._runner_state.is_running = True
        dashboard_module._runner_state.run_task = None  # No actual task

        transport = httpx.ASGITransport(app=app)

        # Stop it
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/dashboard/runner/stop")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "stopped"
        assert "summary" in data

    @pytest.mark.asyncio
    async def test_runner_stop_not_running(self, client):
        """Test stopping when not running."""
        response = await client.post("/api/dashboard/runner/stop")
        assert response.status_code == 409
        assert "No runner active" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_runner_decisions_empty(self, client):
        """Test getting decisions when not running."""
        response = await client.get("/api/dashboard/runner/decisions")
        assert response.status_code == 200
        data = response.json()

        assert data["decisions"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_runner_decisions_with_data(self, client):
        """Test getting decisions from running scenario."""
        import agent.server.dashboard as dashboard_module

        # Start scenario with fast execution
        response = await client.post(
            "/api/dashboard/runner/start",
            json={"scenario_id": "normal_ops_001", "time_scale": 50.0, "max_duration_s": 2.0},
        )
        assert response.status_code == 200

        # Simulate adding some decisions to the log
        if (
            dashboard_module._runner_state.runner
            and dashboard_module._runner_state.runner.run_state
        ):
            dashboard_module._runner_state.runner.run_state.decision_log.append({
                "type": "decision",
                "action": "WAIT",
                "timestamp": "2024-01-01T12:00:00",
            })

        response = await client.get("/api/dashboard/runner/decisions?limit=10")
        assert response.status_code == 200
        data = response.json()

        # Should have response structure
        assert "decisions" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_runner_status_with_drones(self, client):
        """Test runner status includes drone details."""
        # Start scenario
        response = await client.post(
            "/api/dashboard/runner/start",
            json={"scenario_id": "battery_cascade_001", "time_scale": 10.0, "max_duration_s": 2.0},
        )
        assert response.status_code == 200

        response = await client.get("/api/dashboard/runner/status")
        assert response.status_code == 200
        data = response.json()

        # Should have drones loaded from scenario
        assert len(data["drones"]) == 3
        for drone in data["drones"]:
            assert "drone_id" in drone
            assert "name" in drone
            assert "battery_percent" in drone
            assert "state" in drone
            assert "sensors_healthy" in drone
