"""
Tests for dashboard routes and functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.server.dashboard import (
    _action_counts,
    _calculate_relative_pos,
    _list_runs,
    _load_entries,
    _recent,
    _series,
    _summarize,
    add_dashboard_routes,
)


class TestDashboardHelpers:
    """Test dashboard helper functions."""

    def test_calculate_relative_pos_same_location(self):
        """Test relative position calculation for same location."""
        result = _calculate_relative_pos(37.7749, -122.4194, 37.7749, -122.4194)
        assert result["x"] == pytest.approx(0.0, abs=0.01)
        assert result["y"] == pytest.approx(0.0, abs=0.01)

    def test_calculate_relative_pos_north(self):
        """Test relative position calculation for point to the north."""
        # 1 degree north should be ~111km
        result = _calculate_relative_pos(37.0, -122.0, 38.0, -122.0)
        assert result["x"] == pytest.approx(0.0, abs=1.0)
        assert result["y"] == pytest.approx(111111, rel=0.01)

    def test_calculate_relative_pos_east(self):
        """Test relative position calculation for point to the east."""
        result = _calculate_relative_pos(37.0, -122.0, 37.0, -121.0)
        # At 37 degrees latitude, 1 degree longitude is about 88km
        assert result["x"] > 80000
        assert result["y"] == pytest.approx(0.0, abs=1.0)

    def test_load_entries_empty_file(self):
        """Test loading entries from empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        entries = _load_entries(temp_path)
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

        entries = _load_entries(temp_path)
        assert len(entries) == 3
        assert entries[0]["action"] == "INSPECT"
        assert entries[1]["action"] == "WAIT"
        assert entries[2]["action"] == "RETURN"
        temp_path.unlink()

    def test_load_entries_nonexistent_file(self):
        """Test loading entries from non-existent file returns empty list."""
        entries = _load_entries(Path("/nonexistent/path.jsonl"))
        assert entries == []

    def test_list_runs_empty_dir(self):
        """Test listing runs from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = _list_runs(Path(tmpdir))
            assert runs == []

    def test_list_runs_with_files(self):
        """Test listing runs from directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "decisions_20240101_120000.jsonl").touch()
            (log_dir / "decisions_20240102_120000.jsonl").touch()
            (log_dir / "other_file.txt").touch()  # Should be ignored

            runs = _list_runs(log_dir)
            assert len(runs) == 2
            assert "20240101_120000" in runs
            assert "20240102_120000" in runs
            # Should be sorted
            assert runs == sorted(runs)

    def test_summarize_empty_entries(self):
        """Test summarize with empty entries."""
        summary = _summarize([])
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
        summary = _summarize(entries)
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
        risk_series, battery_series = _series(entries)

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
        counts = _action_counts(entries)

        # Convert to dict for easier testing
        count_dict = {c["action"]: c["count"] for c in counts}
        assert count_dict["INSPECT"] == 2
        assert count_dict["WAIT"] == 1
        assert count_dict["RETURN"] == 1

    def test_recent_limits_entries(self):
        """Test recent limits entries to specified amount."""
        entries = [{"action": f"action_{i}"} for i in range(20)]
        recent = _recent(entries, limit=5)

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
        recent = _recent(entries)

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
        add_dashboard_routes(app, log_dir)
        return app, log_dir

    @pytest.fixture
    def client(self, app_with_dashboard):
        """Create test client."""
        app, _ = app_with_dashboard
        return TestClient(app)

    def test_dashboard_route_no_build(self, app_with_dashboard):
        """Test dashboard route when build is missing."""
        app, _ = app_with_dashboard
        client = TestClient(app)
        response = client.get("/dashboard")
        # Should return either HTML file or missing message
        assert response.status_code == 200

    def test_list_runs_empty(self, client):
        """Test listing runs when empty."""
        response = client.get("/api/dashboard/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["latest"] is None

    def test_list_runs_with_data(self, app_with_dashboard):
        """Test listing runs with data."""
        app, log_dir = app_with_dashboard
        client = TestClient(app)

        # Create some run files
        (log_dir / "decisions_run1.jsonl").write_text('{"action": "WAIT"}\n')
        (log_dir / "decisions_run2.jsonl").write_text('{"action": "INSPECT"}\n')

        response = client.get("/api/dashboard/runs")
        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 2
        assert data["latest"] is not None

    def test_run_data_not_found(self, client):
        """Test getting run data for non-existent run."""
        response = client.get("/api/dashboard/run/nonexistent")
        assert response.status_code == 404

    def test_run_data_success(self, app_with_dashboard):
        """Test getting run data successfully."""
        app, log_dir = app_with_dashboard
        client = TestClient(app)

        # Create run file with entries
        run_file = log_dir / "decisions_test_run.jsonl"
        entries = [
            {"action": "INSPECT", "risk_score": 0.3, "timestamp": "T1", "battery_percent": 80},
            {"action": "WAIT", "risk_score": 0.2, "timestamp": "T2", "battery_percent": 78},
        ]
        run_file.write_text("\n".join(json.dumps(e) for e in entries))

        response = client.get("/api/dashboard/run/test_run")
        assert response.status_code == 200
        data = response.json()

        assert data["run_id"] == "test_run"
        assert data["summary"]["total"] == 2
        assert len(data["actions"]) > 0
        assert len(data["risk_series"]) == 2
        assert len(data["battery_series"]) == 2
        assert len(data["recent"]) == 2
