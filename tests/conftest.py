"""
Integration test configuration and fixtures.
"""

import asyncio
import logging
import sys
import tempfile
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from agent.server.goal_selector import GoalSelector
from agent.server.risk_evaluator import RiskEvaluator, RiskThresholds
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

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration paths
TEST_CONFIG_DIR = Path(__file__).parent / "configs"
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_LOGS_DIR = Path(__file__).parent / "logs"

# Test server configuration
TEST_SERVER_HOST = "127.0.0.1"
TEST_SERVER_PORT = 8765  # Different from default to avoid conflicts
TEST_SERVER_URL = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}"

# Mock MAVLink configuration for testing
TEST_MAVLINK_CONNECTION = "udpin:127.0.0.1:14551"  # Different port for testing
TEST_MAVLINK_SYSTEM_ID = 1
TEST_MAVLINK_COMPONENT_ID = 1

# Test timing
TEST_LOOP_RATE_HZ = 10
TEST_TIMEOUT_S = 30
TEST_STEP_DELAY_S = 0.1

# Test mission parameters
TEST_HOME_POSITION = {"lat": 47.397742, "lon": 8.545594, "alt": 488.0}
TEST_ASSET_POSITION = {"lat": 47.398500, "lon": 8.546500, "alt": 495.0}
TEST_DOCK_POSITION = {"lat": 47.397742, "lon": 8.545594, "alt": 488.0}

# Test thresholds
TEST_BATTERY_WARNING = 30.0
TEST_BATTERY_CRITICAL = 20.0
TEST_BATTERY_ABORT = 15.0

# Test vehicle parameters
TEST_ARMING_TIMEOUT_S = 5
TEST_TAKEOFF_ALTITUDE = 10.0
TEST_CRUISE_SPEED = 5.0
TEST_ACCEPTANCE_RADIUS = 5.0


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Async HTTP Client Fixtures
# ============================================================================


@pytest.fixture
def make_async_client() -> Callable[[FastAPI], "AsyncClientContextManager"]:
    """
    Factory fixture that creates httpx.AsyncClient instances for testing FastAPI apps.

    Usage:
        @pytest.mark.asyncio
        async def test_something(self, make_async_client, my_app):
            async with make_async_client(my_app) as client:
                response = await client.get("/endpoint")
                assert response.status_code == 200
    """

    @asynccontextmanager
    async def _make_client(
        app: FastAPI, base_url: str = "http://test"
    ) -> AsyncGenerator[httpx.AsyncClient, None]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
            yield client

    return _make_client


# Type alias for the async client context manager
AsyncClientContextManager = Callable[[FastAPI], AsyncGenerator[httpx.AsyncClient, None]]


@pytest_asyncio.fixture
async def async_client_for_app():
    """
    Async fixture that provides a reusable async client creator.

    This is an alternative to make_async_client that can be used when you
    need to create multiple clients within the same async context.

    Usage:
        @pytest.mark.asyncio
        async def test_something(self, async_client_for_app, my_app):
            client = await async_client_for_app(my_app)
            response = await client.get("/endpoint")
    """
    clients: list[httpx.AsyncClient] = []

    async def create_client(app: FastAPI, base_url: str = "http://test") -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url=base_url)
        clients.append(client)
        await asyncio.sleep(0)
        return client

    yield create_client

    # Cleanup all created clients
    for client in clients:
        await client.aclose()


@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_agent_config(temp_config_dir: Path) -> Path:
    """Create test agent configuration."""
    config_content = f"""
agent:
  name: "test-agent"
  loop_rate_hz: {TEST_LOOP_RATE_HZ}

server:
  host: "{TEST_SERVER_HOST}"
  port: {TEST_SERVER_PORT}

mavlink:
  connection: "{TEST_MAVLINK_CONNECTION}"
  system_id: {TEST_MAVLINK_SYSTEM_ID}
  component_id: {TEST_MAVLINK_COMPONENT_ID}
  timeout_ms: 1000

decision:
  confidence_threshold: 0.7
  max_replan_attempts: 3
"""
    config_file = temp_config_dir / "agent_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def test_mission_config(temp_config_dir: Path) -> Path:
    """Create test mission configuration."""
    config_content = f"""
mission:
  mission_id: "test-integration"
  mission_name: "Integration Test Mission"
  home_position:
    latitude: {TEST_HOME_POSITION["lat"]}
    longitude: {TEST_HOME_POSITION["lon"]}
    altitude: {TEST_HOME_POSITION["alt"]}

  dock:
    position:
      latitude: {TEST_DOCK_POSITION["lat"]}
      longitude: {TEST_DOCK_POSITION["lon"]}
      altitude: {TEST_DOCK_POSITION["alt"]}

assets:
  - asset_id: "test-asset-001"
    name: "Test Asset 1"
    asset_type: "solar_panel"
    position:
      latitude: {TEST_ASSET_POSITION["lat"]}
      longitude: {TEST_ASSET_POSITION["lon"]}
      altitude: {TEST_ASSET_POSITION["alt"]}
    priority: 1
    inspection_interval_minutes: 60
"""
    config_file = temp_config_dir / "mission_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def test_risk_config(temp_config_dir: Path) -> Path:
    """Create test risk thresholds configuration."""
    config_content = f"""
battery:
  warning_percent: {TEST_BATTERY_WARNING}
  critical_percent: {TEST_BATTERY_CRITICAL}
  abort_percent: {TEST_BATTERY_ABORT}

wind:
  warning_ms: 8.0
  abort_ms: 12.0

gps:
  min_satellites: 6
  max_hdop: 2.0
  abort_hdop: 3.0

connectivity:
  heartbeat_timeout_s: 5
  abort_timeout_s: 30
"""
    config_file = temp_config_dir / "risk_thresholds.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_mavlink_connection():
    """Create a mock MAVLink connection for testing."""
    mock = MagicMock()

    # Mock connection methods
    mock.wait_heartbeat = MagicMock(return_value=True)
    mock.target_system = TEST_MAVLINK_SYSTEM_ID
    mock.target_component = TEST_MAVLINK_COMPONENT_ID
    mock.recv_match = MagicMock(return_value=None)

    # Mock vehicle state
    mock.messages = {
        "HEARTBEAT": MagicMock(type=2, base_mode=217, custom_mode=4),  # GUIDED mode
        "GLOBAL_POSITION_INT": MagicMock(
            lat=int(TEST_HOME_POSITION["lat"] * 1e7),
            lon=int(TEST_HOME_POSITION["lon"] * 1e7),
            alt=int(TEST_HOME_POSITION["alt"] * 1000),
            relative_alt=(12 * 1000),
            vx=0,
            vy=0,
            vz=0,
        ),
        "ATTITUDE": MagicMock(roll=0.0, pitch=0.0, yaw=0.0),
        "SYS_STATUS": MagicMock(
            voltage_battery=22800,  # 22.8V
            current_battery=500,  # 0.5A
            battery_remaining=80,  # 80%
        ),
        "GPS_RAW_INT": MagicMock(
            fix_type=3,  # 3D fix
            satellites_visible=8,
            hdop=80,  # 0.8
            eph=100,
            epv=100,
        ),
    }

    # Mock command sending
    mock.mav.command_long_send = MagicMock()
    mock.mav.set_position_target_global_int_send = MagicMock()

    return mock


@pytest.fixture
def temp_log_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def integration_test_setup():
    """Setup common integration test components."""
    # Create test components
    world_model = WorldModel()
    goal_selector = GoalSelector()

    thresholds = RiskThresholds(
        battery_warning_percent=TEST_BATTERY_WARNING,
        battery_critical_percent=TEST_BATTERY_CRITICAL,
        wind_warning_ms=8.0,
        wind_abort_ms=12.0,
    )
    risk_evaluator = RiskEvaluator(thresholds)

    # Create test vehicle state
    vehicle_state = VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude_msl=TEST_HOME_POSITION["alt"],
            altitude_agl=12.0,
        ),
        velocity=Velocity(north=0, east=0, down=0),
        attitude=Attitude(roll=0, pitch=0, yaw=0),
        battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
        gps=GPSState(fix_type=3, satellites_visible=8, hdop=0.8, vdop=0.8),
        health=VehicleHealth(
            sensors_healthy=True,
            gps_healthy=True,
            battery_healthy=True,
            motors_healthy=True,
            ekf_healthy=True,
        ),
        home_position=Position(
            latitude=TEST_HOME_POSITION["lat"],
            longitude=TEST_HOME_POSITION["lon"],
            altitude_msl=TEST_HOME_POSITION["alt"],
            altitude_agl=0.0,
        ),
    )

    # Update world model with vehicle state
    world_model.update_vehicle(vehicle_state)

    return {
        "world_model": world_model,
        "goal_selector": goal_selector,
        "risk_evaluator": risk_evaluator,
        "vehicle_state": vehicle_state,
    }


# Ensure directories exist
for directory in [TEST_CONFIG_DIR, TEST_DATA_DIR, TEST_LOGS_DIR]:
    directory.mkdir(exist_ok=True)


class ErrorLogCapture(logging.Handler):
    """Handler that captures ERROR and above log records."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def get_errors(self) -> list[dict[str, Any]]:
        """Return captured errors as dicts for assertion messages."""
        return [
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "filename": record.filename,
                "lineno": record.lineno,
            }
            for record in self.records
        ]


@pytest.fixture(autouse=True)
def fail_on_error_logs(request):
    """
    Fixture that fails tests if any ERROR level logs are emitted.

    This helps catch silent failures where code logs an error but continues
    to return a fallback value instead of raising an exception.

    To skip this check for a specific test, mark it with:
        @pytest.mark.allow_error_logs
    """
    # Skip if test is marked to allow error logs
    if request.node.get_closest_marker("allow_error_logs"):
        yield
        return

    # Install the error capture handler
    handler = ErrorLogCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    yield handler

    # Remove handler
    root_logger.removeHandler(handler)

    # Check for errors
    errors = handler.get_errors()
    if errors:
        error_details = "\n".join(
            f"  [{e['level']}] {e['logger']}: {e['message']} ({e['filename']}:{e['lineno']})"
            for e in errors
        )
        pytest.fail(
            f"Test emitted {len(errors)} error log(s):\n{error_details}\n\n"
            "If this error is expected, mark the test with @pytest.mark.allow_error_logs"
        )


# Register the custom marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_error_logs: mark test to allow ERROR level log messages"
    )
