"""
Integration test configuration and fixtures.
"""

import asyncio
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Generator, AsyncGenerator
import sys
import os

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
    latitude: {TEST_HOME_POSITION['lat']}
    longitude: {TEST_HOME_POSITION['lon']}
    altitude: {TEST_HOME_POSITION['alt']}
  
  dock:
    position:
      latitude: {TEST_DOCK_POSITION['lat']}
      longitude: {TEST_DOCK_POSITION['lon']}
      altitude: {TEST_DOCK_POSITION['alt']}

assets:
  - asset_id: "test-asset-001"
    name: "Test Asset 1"
    asset_type: "solar_panel"
    position:
      latitude: {TEST_ASSET_POSITION['lat']}
      longitude: {TEST_ASSET_POSITION['lon']}
      altitude: {TEST_ASSET_POSITION['alt']}
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
    mock.wait_heartbeat = AsyncMock(return_value=True)
    mock.target_system = TEST_MAVLINK_SYSTEM_ID
    mock.target_component = TEST_MAVLINK_COMPONENT_ID
    
    # Mock vehicle state
    mock.messages = {
        'HEARTBEAT': MagicMock(type=2, base_mode=217, custom_mode=4),  # GUIDED mode
        'GLOBAL_POSITION_INT': MagicMock(
            lat=int(TEST_HOME_POSITION['lat'] * 1e7),
            lon=int(TEST_HOME_POSITION['lon'] * 1e7),
            alt=int(TEST_HOME_POSITION['alt'] * 1000),
            relative_alt=int(12 * 1000),
            vx=0, vy=0, vz=0
        ),
        'ATTITUDE': MagicMock(roll=0.0, pitch=0.0, yaw=0.0),
        'SYS_STATUS': MagicMock(
            voltage_battery=22800,  # 22.8V
            current_battery=500,     # 0.5A
            battery_remaining=80     # 80%
        ),
        'GPS_RAW_INT': MagicMock(
            fix_type=3,  # 3D fix
            satellites_visible=8,
            hdop=80,  # 0.8
            eph=100, epv=100
        )
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
async def integration_test_setup():
    """Setup common integration test components."""
    # Import here to avoid module import issues during pytest collection
    try:
        from agent.server.world_model import WorldModel
        from agent.server.goal_selector import GoalSelector
        from agent.server.risk_evaluator import RiskEvaluator, RiskThresholds
        from autonomy.vehicle_state import (
            VehicleState, Position, Velocity, Attitude, BatteryState,
            FlightMode, GPSState, VehicleHealth
        )
        from datetime import datetime
        
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
                latitude=TEST_HOME_POSITION['lat'],
                longitude=TEST_HOME_POSITION['lon'],
                altitude_msl=TEST_HOME_POSITION['alt'],
                altitude_agl=12.0
            ),
            velocity=Velocity(0, 0, 0),
            attitude=Attitude(0, 0, 0),
            battery=BatteryState(22.8, 5.0, 80.0),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(3, 8, 0.8, 0.8),
            health=VehicleHealth(True, True, True, True, True),
            home_position=Position(
                latitude=TEST_HOME_POSITION['lat'],
                longitude=TEST_HOME_POSITION['lon'],
                altitude_msl=TEST_HOME_POSITION['alt'],
                altitude_agl=0.0
            ),
        )
        
        yield {
            'world_model': world_model,
            'goal_selector': goal_selector,
            'risk_evaluator': risk_evaluator,
            'vehicle_state': vehicle_state,
        }
        
    except ImportError as e:
        pytest.skip(f"Cannot import required modules for integration test: {e}")


# Ensure directories exist
for directory in [TEST_CONFIG_DIR, TEST_DATA_DIR, TEST_LOGS_DIR]:
    directory.mkdir(exist_ok=True)