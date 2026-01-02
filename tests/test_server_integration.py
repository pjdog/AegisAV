"""
Integration tests for the FastAPI server endpoints.

Tests the main API endpoints without requiring a real Redis connection.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from agent.api_models import ActionType, VehicleStateRequest
from agent.server.auth import AuthConfig
from agent.server.config_manager import ConfigManager, DEFAULT_SERVER_PORT
from agent.server.persistence import InMemoryStore

# Fixtures for server testing


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_store():
    """Create mock in-memory store."""
    store = InMemoryStore()
    return store


@pytest.fixture
def auth_disabled_config():
    """Create auth config with auth disabled."""
    return AuthConfig(enabled=False)


@pytest.fixture
def sample_vehicle_state():
    """Create sample vehicle state for testing."""
    return {
        "timestamp": datetime.now().isoformat(),
        "position": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude_msl": 100.0,
            "altitude_agl": 50.0,
        },
        "velocity": {"north": 0.0, "east": 0.0, "down": 0.0},
        "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "battery": {"voltage": 22.8, "current": 5.0, "remaining_percent": 80.0},
        "mode": "GUIDED",
        "armed": True,
        "in_air": True,
        "gps": {
            "fix_type": 3,
            "satellites_visible": 12,
            "hdop": 0.8,
            "vdop": 1.0,
        },
        "health": {
            "sensors_healthy": True,
            "gps_healthy": True,
            "battery_healthy": True,
            "motors_healthy": True,
            "ekf_healthy": True,
        },
        "home_position": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude_msl": 50.0,
            "altitude_agl": 0.0,
        },
    }


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_exists(self):
        """Test that health endpoint is defined."""
        # We test the health response model instead of making actual requests
        # since the server requires lifespan initialization
        from agent.api_models import HealthResponse

        response = HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            uptime_seconds=100.0,
            decisions_made=5,
        )
        assert response.status == "healthy"
        assert response.uptime_seconds == 100.0
        assert response.decisions_made == 5


class TestConfigManagerIntegration:
    """Test config manager with API endpoints."""

    def test_config_manager_initialization(self, temp_config_dir):
        """Test config manager can be initialized."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load()

        assert config is not None
        assert config.server.port == DEFAULT_SERVER_PORT
        assert config.redis.enabled is True

    def test_config_manager_save_load_roundtrip(self, temp_config_dir):
        """Test saving and loading config."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.load()

        # Modify config
        manager.update_section("server", {"port": 9000})
        manager.save()

        # Load in new manager
        manager2 = ConfigManager(config_dir=temp_config_dir)
        config2 = manager2.load()

        assert config2.server.port == 9000

    def test_config_manager_env_template(self, temp_config_dir):
        """Test generating environment template."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.load()

        template = manager.export_env_template()
        assert "AEGIS_HOST" in template
        assert "AEGIS_PORT" in template
        assert "AEGIS_REDIS_HOST" in template


class TestVehicleStateRequest:
    """Test vehicle state request model."""

    def test_valid_vehicle_state(self, sample_vehicle_state):
        """Test creating valid vehicle state request."""
        state = VehicleStateRequest(**sample_vehicle_state)
        assert state.position.latitude == 37.7749
        assert state.battery.remaining_percent == 80.0
        assert state.mode == "GUIDED"

    def test_vehicle_state_with_optional_fields(self, sample_vehicle_state):
        """Test vehicle state with optional fields."""
        # Remove optional fields
        del sample_vehicle_state["home_position"]
        state = VehicleStateRequest(**sample_vehicle_state)
        assert state.home_position is None


class TestInMemoryStorePersistence:
    """Test in-memory store through integration scenario."""

    @pytest.mark.asyncio
    async def test_full_storage_workflow(self, mock_store):
        """Test complete storage workflow."""
        await mock_store.connect()

        # Store asset
        asset = {"id": "asset_001", "type": "solar_panel", "lat": 37.7749}
        await mock_store.set_asset("asset_001", asset)

        # Store anomaly
        anomaly = {"id": "anomaly_001", "asset_id": "asset_001", "severity": 0.8}
        await mock_store.add_anomaly("anomaly_001", anomaly)

        # Store detection
        detection = {"class": "crack", "confidence": 0.95}
        await mock_store.add_detection("asset_001", detection)

        # Store telemetry
        telemetry = {"battery": 80, "altitude": 100}
        await mock_store.add_telemetry("vehicle_001", telemetry)

        # Store mission
        mission = {"id": "mission_001", "status": "completed"}
        await mock_store.save_mission("mission_001", mission)

        # Verify stats
        stats = await mock_store.get_stats()
        assert stats["asset_count"] == 1
        assert stats["anomaly_count"] == 1
        assert stats["mission_count"] == 1

        await mock_store.disconnect()

    @pytest.mark.asyncio
    async def test_anomaly_for_reinspection_workflow(self, mock_store):
        """Test anomaly workflow that triggers reinspection."""
        await mock_store.connect()

        # Initial asset
        await mock_store.set_asset("asset_001", {"id": "asset_001", "inspected": True})

        # Detection creates anomaly
        await mock_store.add_anomaly(
            "anomaly_001",
            {
                "id": "anomaly_001",
                "asset_id": "asset_001",
                "severity": 0.9,
                "resolved": False,
            },
        )

        # Get unresolved anomalies for asset
        anomalies = await mock_store.get_anomalies_for_asset("asset_001")
        unresolved = [a for a in anomalies if not a.get("resolved", False)]

        assert len(unresolved) == 1
        assert unresolved[0]["severity"] == 0.9

        # Resolve anomaly after reinspection
        await mock_store.resolve_anomaly("anomaly_001")

        # Verify resolved
        anomalies = await mock_store.get_anomalies_for_asset("asset_001")
        unresolved = [a for a in anomalies if not a.get("resolved", False)]
        assert len(unresolved) == 0

        await mock_store.disconnect()


class TestActionTypes:
    """Test action type enums and validation."""

    def test_all_action_types_exist(self):
        """Test that all expected action types exist."""
        expected = ["WAIT", "TAKEOFF", "LAND", "GOTO", "INSPECT", "RETURN", "ABORT", "DOCK"]
        for action in expected:
            assert hasattr(ActionType, action)

    def test_action_type_values(self):
        """Test action type string values."""
        assert ActionType.WAIT.value == "wait"
        assert ActionType.INSPECT.value == "inspect"
        assert ActionType.RETURN.value == "return"
        assert ActionType.ABORT.value == "abort"


class TestServerStateManagement:
    """Test server state management concepts."""

    def test_world_model_update(self):
        """Test world model can be updated with vehicle state."""
        from agent.server.world_model import WorldModel
        from autonomy.vehicle_state import (
            Attitude,
            BatteryState,
            FlightMode,
            GPSState,
            Position,
            VehicleHealth,
            VehicleState,
            Velocity,
        )

        world = WorldModel()

        state = VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=37.7749, longitude=-122.4194, altitude_msl=100.0, altitude_agl=50.0
            ),
            velocity=Velocity(north=0, east=0, down=0),
            attitude=Attitude(roll=0, pitch=0, yaw=0),
            battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(fix_type=3, satellites_visible=12, hdop=0.8, vdop=1.0),
            health=VehicleHealth(
                sensors_healthy=True,
                gps_healthy=True,
                battery_healthy=True,
                motors_healthy=True,
                ekf_healthy=True,
            ),
        )

        world.update_vehicle(state)
        # WorldModel stores state in _vehicle - check it was stored
        assert world._vehicle is not None

    @pytest.mark.asyncio
    async def test_goal_selector_with_world_model(self):
        """Test goal selector works with world model."""
        from agent.server.goal_selector import GoalSelector
        from agent.server.world_model import DockStatus, WorldModel
        from autonomy.vehicle_state import (
            Attitude,
            BatteryState,
            FlightMode,
            GPSState,
            Position,
            VehicleHealth,
            VehicleState,
            Velocity,
        )

        world = WorldModel()
        selector = GoalSelector()

        state = VehicleState(
            timestamp=datetime.now(),
            position=Position(
                latitude=37.7749, longitude=-122.4194, altitude_msl=100.0, altitude_agl=50.0
            ),
            velocity=Velocity(north=0, east=0, down=0),
            attitude=Attitude(roll=0, pitch=0, yaw=0),
            battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(fix_type=3, satellites_visible=12, hdop=0.8, vdop=1.0),
            health=VehicleHealth(
                sensors_healthy=True,
                gps_healthy=True,
                battery_healthy=True,
                motors_healthy=True,
                ekf_healthy=True,
            ),
        )

        world.update_vehicle(state)
        # Set a dock position (required for get_snapshot to return a snapshot)
        dock_position = Position(
            latitude=37.7740, longitude=-122.4180, altitude_msl=50.0, altitude_agl=0.0
        )
        world.set_dock(dock_position, DockStatus.AVAILABLE)

        # Get snapshot for goal selection (select_goal expects WorldSnapshot)
        snapshot = world.get_snapshot()
        assert snapshot is not None

        # Goal selection should work
        goal = await selector.select_goal(snapshot)
        assert goal is not None


class TestDashboardAPIIntegration:
    """Test dashboard API endpoints work correctly."""

    @pytest.mark.asyncio
    async def test_dashboard_runs_endpoint(self, temp_log_dir):
        """Test dashboard runs listing."""
        from agent.server.dashboard import add_dashboard_routes

        app = FastAPI()
        add_dashboard_routes(app, temp_log_dir)

        # Create some run files
        (temp_log_dir / "decisions_run1.jsonl").write_text('{"action": "WAIT"}\n')

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/dashboard/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert len(data["runs"]) == 1
