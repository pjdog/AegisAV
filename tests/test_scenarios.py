"""
Tests for the multi-drone simulation scenarios module.
"""

from datetime import datetime, timedelta

from agent.server.scenarios import (
    PRELOADED_SCENARIOS,
    DroneState,
    EnvironmentConditions,
    Scenario,
    ScenarioCategory,
    ScenarioEvent,
    SimulatedAsset,
    SimulatedDrone,
    create_battery_cascade_scenario,
    create_coordination_scenario,
    create_gps_degradation_scenario,
    create_multi_anomaly_scenario,
    create_normal_operations_scenario,
    create_sensor_cascade_scenario,
    create_weather_emergency_scenario,
    get_all_scenarios,
    get_scenario,
    get_scenarios_by_category,
    get_scenarios_by_difficulty,
    register_scenario,
)


class TestSimulatedDrone:
    """Test SimulatedDrone model."""

    def test_create_drone_defaults(self):
        """Test creating drone with defaults."""
        drone = SimulatedDrone(drone_id="test", name="Test Drone")
        assert drone.drone_id == "test"
        assert drone.name == "Test Drone"
        assert drone.battery_percent == 100.0
        assert drone.gps_fix_type == 3
        assert drone.state == DroneState.DOCKED
        assert drone.armed is False

    def test_create_drone_custom_values(self):
        """Test creating drone with custom values."""
        drone = SimulatedDrone(
            drone_id="custom",
            name="Custom Drone",
            battery_percent=75.0,
            gps_hdop=2.0,
            state=DroneState.INSPECTING,
            armed=True,
            in_air=True,
        )
        assert drone.battery_percent == 75.0
        assert drone.gps_hdop == 2.0
        assert drone.state == DroneState.INSPECTING
        assert drone.armed is True

    def test_drone_to_vehicle_state(self):
        """Test converting drone to vehicle state dict."""
        drone = SimulatedDrone(
            drone_id="test",
            name="Test Drone",
            latitude=37.7749,
            longitude=-122.4194,
            altitude_agl=100.0,
            battery_percent=80.0,
        )
        timestamp = datetime.now()
        state = drone.to_vehicle_state(timestamp)

        assert state["drone_id"] == "test"
        assert state["position"]["latitude"] == 37.7749
        assert state["position"]["altitude_agl"] == 100.0
        assert state["battery"]["remaining_percent"] == 80.0
        assert state["armed"] is False

    def test_drone_edge_case_triggers(self):
        """Test drone with edge case triggers."""
        trigger_time = datetime.now() + timedelta(minutes=5)
        drone = SimulatedDrone(
            drone_id="edge",
            name="Edge Case Drone",
            battery_failure_at=25.0,
            gps_loss_at=trigger_time,
            sensor_failure_at=trigger_time,
        )
        assert drone.battery_failure_at == 25.0
        assert drone.gps_loss_at == trigger_time


class TestSimulatedAsset:
    """Test SimulatedAsset model."""

    def test_create_asset(self):
        """Test creating asset."""
        asset = SimulatedAsset(
            asset_id="solar_001",
            name="Solar Farm",
            asset_type="solar_panel",
            latitude=37.77,
            longitude=-122.42,
        )
        assert asset.asset_id == "solar_001"
        assert asset.asset_type == "solar_panel"
        assert asset.has_anomaly is False

    def test_asset_with_anomaly(self):
        """Test asset with detected anomaly."""
        asset = SimulatedAsset(
            asset_id="damaged_001",
            name="Damaged Panel",
            asset_type="solar_panel",
            latitude=37.77,
            longitude=-122.42,
            has_anomaly=True,
            anomaly_severity=0.85,
        )
        assert asset.has_anomaly is True
        assert asset.anomaly_severity == 0.85


class TestEnvironmentConditions:
    """Test EnvironmentConditions model."""

    def test_default_conditions(self):
        """Test default environmental conditions."""
        env = EnvironmentConditions()
        assert env.wind_speed_ms == 3.0
        assert env.visibility_m == 10000.0
        assert env.precipitation == "none"
        assert env.is_daylight is True

    def test_adverse_conditions(self):
        """Test adverse weather conditions."""
        env = EnvironmentConditions(
            wind_speed_ms=15.0,
            visibility_m=500.0,
            precipitation="heavy_rain",
        )
        assert env.wind_speed_ms == 15.0
        assert env.visibility_m == 500.0

    def test_weather_change_triggers(self):
        """Test weather change triggers."""
        trigger_time = datetime.now() + timedelta(minutes=10)
        env = EnvironmentConditions(
            wind_increase_at=trigger_time,
            wind_increase_to=18.0,
            visibility_drop_at=trigger_time,
            visibility_drop_to=800.0,
        )
        assert env.wind_increase_to == 18.0
        assert env.visibility_drop_to == 800.0


class TestScenarioEvent:
    """Test ScenarioEvent model."""

    def test_create_event(self):
        """Test creating scenario event."""
        event = ScenarioEvent(
            timestamp_offset_s=60.0,
            event_type="decision",
            description="Drone assigned to inspect solar farm",
            data={"drone_id": "alpha", "target": "solar_001"},
        )
        assert event.timestamp_offset_s == 60.0
        assert event.event_type == "decision"
        assert event.data["drone_id"] == "alpha"


class TestScenario:
    """Test Scenario model."""

    def test_create_minimal_scenario(self):
        """Test creating minimal scenario."""
        scenario = Scenario(
            scenario_id="test_001",
            name="Test Scenario",
            description="A test scenario",
            category=ScenarioCategory.NORMAL_OPERATIONS,
        )
        assert scenario.scenario_id == "test_001"
        assert scenario.category == ScenarioCategory.NORMAL_OPERATIONS
        assert scenario.duration_minutes == 30.0
        assert len(scenario.drones) == 0

    def test_create_full_scenario(self):
        """Test creating full scenario with all components."""
        scenario = Scenario(
            scenario_id="full_001",
            name="Full Scenario",
            description="Complete scenario test",
            category=ScenarioCategory.BATTERY_CRITICAL,
            duration_minutes=20.0,
            difficulty="hard",
            tags=["test", "battery"],
            drones=[
                SimulatedDrone(drone_id="d1", name="Drone 1"),
                SimulatedDrone(drone_id="d2", name="Drone 2"),
            ],
            assets=[
                SimulatedAsset(
                    asset_id="a1",
                    name="Asset 1",
                    asset_type="solar_panel",
                    latitude=37.77,
                    longitude=-122.42,
                ),
            ],
            events=[
                ScenarioEvent(0.0, "start", "Mission begins"),
            ],
        )
        assert len(scenario.drones) == 2
        assert len(scenario.assets) == 1
        assert len(scenario.events) == 1
        assert scenario.difficulty == "hard"

    def test_scenario_to_dict(self):
        """Test converting scenario to dictionary."""
        scenario = Scenario(
            scenario_id="dict_test",
            name="Dict Test",
            description="Test serialization",
            category=ScenarioCategory.GPS_DEGRADED,
            drones=[SimulatedDrone(drone_id="d1", name="D1")],
            assets=[
                SimulatedAsset(
                    asset_id="a1", name="A1", asset_type="tower", latitude=37.77, longitude=-122.42
                ),
            ],
            events=[ScenarioEvent(0.0, "start", "Begin")],
        )
        data = scenario.to_dict()

        assert data["scenario_id"] == "dict_test"
        assert data["category"] == "gps_degraded"
        assert data["drone_count"] == 1
        assert data["asset_count"] == 1
        assert data["event_count"] == 1


class TestPreloadedScenarios:
    """Test preloaded scenario factories."""

    def test_normal_operations_scenario(self):
        """Test normal operations scenario."""
        scenario = create_normal_operations_scenario()
        assert scenario.scenario_id == "normal_ops_001"
        assert scenario.category == ScenarioCategory.NORMAL_OPERATIONS
        assert len(scenario.drones) == 3
        assert len(scenario.assets) == 3
        assert scenario.difficulty == "easy"

    def test_battery_cascade_scenario(self):
        """Test battery cascade scenario."""
        scenario = create_battery_cascade_scenario()
        assert scenario.scenario_id == "battery_cascade_001"
        assert scenario.category == ScenarioCategory.BATTERY_CRITICAL
        assert len(scenario.drones) == 3
        # Check drone battery levels vary
        batteries = [d.battery_percent for d in scenario.drones]
        assert min(batteries) < 30  # At least one low battery
        assert max(batteries) > 50  # At least one healthy

    def test_gps_degradation_scenario(self):
        """Test GPS degradation scenario."""
        scenario = create_gps_degradation_scenario()
        assert scenario.scenario_id == "gps_degrade_001"
        assert scenario.category == ScenarioCategory.GPS_DEGRADED
        # Check GPS quality varies
        hdops = [d.gps_hdop for d in scenario.drones]
        assert min(hdops) < 1.0  # Good GPS
        assert max(hdops) > 2.0  # Degraded GPS

    def test_weather_emergency_scenario(self):
        """Test weather emergency scenario."""
        scenario = create_weather_emergency_scenario()
        assert scenario.scenario_id == "weather_001"
        assert scenario.category == ScenarioCategory.WEATHER_ADVERSE
        # Check weather change triggers
        assert scenario.environment.wind_increase_at is not None
        assert scenario.environment.wind_increase_to > 15.0

    def test_sensor_cascade_scenario(self):
        """Test sensor failure cascade scenario."""
        scenario = create_sensor_cascade_scenario()
        assert scenario.scenario_id == "sensor_cascade_001"
        assert scenario.category == ScenarioCategory.SENSOR_FAILURE
        assert scenario.difficulty == "extreme"
        # Check sensor issues
        unhealthy_sensors = [
            d
            for d in scenario.drones
            if not d.sensors_healthy or not d.ekf_healthy or not d.motors_healthy
        ]
        assert len(unhealthy_sensors) >= 2

    def test_multi_anomaly_scenario(self):
        """Test multi-anomaly scenario."""
        scenario = create_multi_anomaly_scenario()
        assert scenario.scenario_id == "multi_anom_001"
        assert scenario.category == ScenarioCategory.MULTI_ANOMALY
        # Check multiple anomalies
        anomaly_assets = [a for a in scenario.assets if a.has_anomaly]
        assert len(anomaly_assets) >= 3
        # Check varying severities
        severities = [a.anomaly_severity for a in anomaly_assets]
        assert max(severities) > 0.7  # High severity
        assert min(severities) < 0.5  # Low severity

    def test_coordination_scenario(self):
        """Test fleet coordination scenario."""
        scenario = create_coordination_scenario()
        assert scenario.scenario_id == "coord_001"
        assert scenario.category == ScenarioCategory.COORDINATION
        assert len(scenario.drones) == 4  # Multi-drone
        assert len(scenario.assets) == 4  # Dense asset field


class TestScenarioRegistry:
    """Test scenario registry functions."""

    def test_preloaded_scenarios_initialized(self):
        """Test that preloaded scenarios are auto-initialized."""
        assert len(PRELOADED_SCENARIOS) >= 7

    def test_get_scenario_by_id(self):
        """Test getting scenario by ID."""
        scenario = get_scenario("normal_ops_001")
        assert scenario is not None
        assert scenario.name == "Normal Fleet Operations"

    def test_get_scenario_not_found(self):
        """Test getting non-existent scenario."""
        scenario = get_scenario("nonexistent_999")
        assert scenario is None

    def test_get_all_scenarios(self):
        """Test getting all scenarios."""
        scenarios = get_all_scenarios()
        assert len(scenarios) >= 7
        assert all(isinstance(s, Scenario) for s in scenarios)

    def test_get_scenarios_by_category(self):
        """Test filtering scenarios by category."""
        battery_scenarios = get_scenarios_by_category(ScenarioCategory.BATTERY_CRITICAL)
        assert len(battery_scenarios) >= 1
        assert all(s.category == ScenarioCategory.BATTERY_CRITICAL for s in battery_scenarios)

    def test_get_scenarios_by_difficulty(self):
        """Test filtering scenarios by difficulty."""
        easy_scenarios = get_scenarios_by_difficulty("easy")
        hard_scenarios = get_scenarios_by_difficulty("hard")

        assert len(easy_scenarios) >= 1
        assert len(hard_scenarios) >= 1
        assert all(s.difficulty == "easy" for s in easy_scenarios)
        assert all(s.difficulty == "hard" for s in hard_scenarios)

    def test_register_custom_scenario(self):
        """Test registering a custom scenario."""
        custom = Scenario(
            scenario_id="custom_test_999",
            name="Custom Test",
            description="Custom test scenario",
            category=ScenarioCategory.NORMAL_OPERATIONS,
        )
        register_scenario(custom)

        retrieved = get_scenario("custom_test_999")
        assert retrieved is not None
        assert retrieved.name == "Custom Test"


class TestScenarioCategories:
    """Test scenario categories."""

    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        expected = [
            "NORMAL_OPERATIONS",
            "BATTERY_CRITICAL",
            "GPS_DEGRADED",
            "WEATHER_ADVERSE",
            "SENSOR_FAILURE",
            "MULTI_ANOMALY",
            "EMERGENCY_ABORT",
            "MISSION_COMPLETE",
            "COORDINATION",
        ]
        for cat in expected:
            assert hasattr(ScenarioCategory, cat)

    def test_category_values(self):
        """Test category string values."""
        assert ScenarioCategory.BATTERY_CRITICAL.value == "battery_critical"
        assert ScenarioCategory.GPS_DEGRADED.value == "gps_degraded"


class TestDroneStates:
    """Test drone state enum."""

    def test_all_states_exist(self):
        """Test that all expected states exist."""
        expected = [
            "IDLE",
            "TAKEOFF",
            "INSPECTING",
            "RETURNING",
            "LANDING",
            "CHARGING",
            "EMERGENCY",
            "OFFLINE",
        ]
        for state in expected:
            assert hasattr(DroneState, state)
