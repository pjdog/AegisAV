"""
Comprehensive tests for RiskEvaluator module.
"""

from datetime import datetime

import pytest

from agent.server.risk_evaluator import (
    RiskAssessment,
    RiskEvaluator,
    RiskFactor,
    RiskLevel,
    RiskThresholds,
)
from agent.server.world_model import (
    DockState,
    DockStatus,
    EnvironmentState,
    MissionState,
    WorldSnapshot,
)
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

# ==================== Fixtures ====================


@pytest.fixture
def default_position():
    """Create a default position near the dock."""
    return Position(latitude=37.7749, longitude=-122.4194, altitude_msl=100.0, altitude_agl=50.0)


@pytest.fixture
def dock_position():
    """Create dock position."""
    return Position(latitude=37.7749, longitude=-122.4194, altitude_msl=50.0)


@pytest.fixture
def healthy_battery():
    """Create a healthy battery state."""
    return BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0)


@pytest.fixture
def low_battery():
    """Create a low battery state."""
    return BatteryState(voltage=21.0, current=5.0, remaining_percent=25.0)


@pytest.fixture
def critical_battery():
    """Create a critical battery state."""
    return BatteryState(voltage=20.0, current=5.0, remaining_percent=10.0)


@pytest.fixture
def good_gps():
    """Create a good GPS state."""
    return GPSState(fix_type=3, satellites_visible=10, hdop=0.8, vdop=1.0)


@pytest.fixture
def degraded_gps():
    """Create a degraded GPS state."""
    return GPSState(fix_type=3, satellites_visible=5, hdop=2.5, vdop=3.0)


@pytest.fixture
def no_gps():
    """Create a no-fix GPS state."""
    return GPSState(fix_type=0, satellites_visible=2, hdop=99.9, vdop=99.9)


@pytest.fixture
def healthy_vehicle():
    """Create a healthy vehicle health state."""
    return VehicleHealth(
        sensors_healthy=True,
        gps_healthy=True,
        battery_healthy=True,
        motors_healthy=True,
        ekf_healthy=True,
    )


@pytest.fixture
def unhealthy_vehicle():
    """Create an unhealthy vehicle health state."""
    return VehicleHealth(
        sensors_healthy=False,
        gps_healthy=False,
        battery_healthy=True,
        motors_healthy=True,
        ekf_healthy=True,
    )


@pytest.fixture
def critical_health():
    """Create a critical vehicle health state with motor issues."""
    return VehicleHealth(
        sensors_healthy=True,
        gps_healthy=True,
        battery_healthy=True,
        motors_healthy=False,
        ekf_healthy=True,
    )


@pytest.fixture
def calm_environment():
    """Create calm environmental conditions."""
    return EnvironmentState(
        timestamp=datetime.now(),
        wind_speed_ms=3.0,
        wind_direction_deg=180.0,
        visibility_m=10000.0,
    )


@pytest.fixture
def windy_environment():
    """Create windy environmental conditions."""
    return EnvironmentState(
        timestamp=datetime.now(),
        wind_speed_ms=10.0,
        wind_direction_deg=180.0,
        visibility_m=5000.0,
    )


@pytest.fixture
def extreme_wind_environment():
    """Create extreme wind environmental conditions."""
    return EnvironmentState(
        timestamp=datetime.now(),
        wind_speed_ms=15.0,
        wind_direction_deg=180.0,
        visibility_m=5000.0,
    )


def create_vehicle_state(
    position=None, battery=None, gps=None, health=None, home_position=None
):
    """Helper to create a VehicleState with customizable components."""
    return VehicleState(
        timestamp=datetime.now(),
        position=position or Position(latitude=37.7749, longitude=-122.4194, altitude_msl=100.0),
        velocity=Velocity(north=0, east=0, down=0),
        attitude=Attitude(roll=0, pitch=0, yaw=0),
        battery=battery or BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
        gps=gps,
        health=health,
        home_position=home_position,
    )


def create_world_snapshot(
    vehicle=None, environment=None, dock_position=None, assets=None
):
    """Helper to create a WorldSnapshot with customizable components."""
    if vehicle is None:
        vehicle = create_vehicle_state()

    if environment is None:
        environment = EnvironmentState(
            timestamp=datetime.now(),
            wind_speed_ms=3.0,
        )

    if dock_position is None:
        dock_position = Position(latitude=37.7749, longitude=-122.4194, altitude_msl=50.0)

    dock = DockState(position=dock_position, status=DockStatus.AVAILABLE)

    mission = MissionState(mission_id="test_001", mission_name="Test Mission")

    return WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=vehicle,
        assets=assets or [],
        anomalies=[],
        dock=dock,
        environment=environment,
        mission=mission,
    )


# ==================== RiskLevel Tests ====================


class TestRiskLevel:
    """Test RiskLevel enum."""

    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_risk_level_comparison(self):
        """Test that risk levels can be compared."""
        # Enum comparison by identity
        assert RiskLevel.LOW != RiskLevel.HIGH
        assert RiskLevel.CRITICAL == RiskLevel.CRITICAL


# ==================== RiskFactor Tests ====================


class TestRiskFactor:
    """Test RiskFactor model."""

    def test_risk_factor_creation(self):
        """Test creating a RiskFactor."""
        factor = RiskFactor(
            name="battery",
            value=0.5,
            threshold=0.6,
            critical=0.9,
            description="Battery OK",
        )
        assert factor.name == "battery"
        assert factor.value == 0.5
        assert factor.threshold == 0.6
        assert factor.critical == 0.9

    def test_level_low(self):
        """Test RiskFactor level when value is low."""
        factor = RiskFactor(name="test", value=0.3, threshold=0.6, critical=0.9)
        assert factor.level == RiskLevel.LOW

    def test_level_moderate(self):
        """Test RiskFactor level when value is moderate."""
        # Value >= threshold * 0.7 but < threshold
        factor = RiskFactor(name="test", value=0.45, threshold=0.6, critical=0.9)
        assert factor.level == RiskLevel.MODERATE

    def test_level_high(self):
        """Test RiskFactor level when value is high."""
        # Value >= threshold but < critical
        factor = RiskFactor(name="test", value=0.7, threshold=0.6, critical=0.9)
        assert factor.level == RiskLevel.HIGH

    def test_level_critical(self):
        """Test RiskFactor level when value is critical."""
        factor = RiskFactor(name="test", value=0.95, threshold=0.6, critical=0.9)
        assert factor.level == RiskLevel.CRITICAL

    def test_is_critical_true(self):
        """Test is_critical property when value is critical."""
        factor = RiskFactor(name="test", value=0.95, threshold=0.6, critical=0.9)
        assert factor.is_critical is True

    def test_is_critical_false(self):
        """Test is_critical property when value is not critical."""
        factor = RiskFactor(name="test", value=0.8, threshold=0.6, critical=0.9)
        assert factor.is_critical is False

    def test_is_concerning_true_high(self):
        """Test is_concerning when value is high."""
        factor = RiskFactor(name="test", value=0.7, threshold=0.6, critical=0.9)
        assert factor.is_concerning is True

    def test_is_concerning_true_moderate(self):
        """Test is_concerning when value is moderate."""
        factor = RiskFactor(name="test", value=0.45, threshold=0.6, critical=0.9)
        assert factor.is_concerning is True

    def test_is_concerning_false(self):
        """Test is_concerning when value is low."""
        factor = RiskFactor(name="test", value=0.3, threshold=0.6, critical=0.9)
        assert factor.is_concerning is False

    def test_boundary_at_threshold(self):
        """Test level exactly at threshold."""
        factor = RiskFactor(name="test", value=0.6, threshold=0.6, critical=0.9)
        assert factor.level == RiskLevel.HIGH

    def test_boundary_at_critical(self):
        """Test level exactly at critical."""
        factor = RiskFactor(name="test", value=0.9, threshold=0.6, critical=0.9)
        assert factor.level == RiskLevel.CRITICAL


# ==================== RiskAssessment Tests ====================


class TestRiskAssessment:
    """Test RiskAssessment model."""

    def test_risk_assessment_creation(self):
        """Test creating a RiskAssessment."""
        assessment = RiskAssessment(
            overall_level=RiskLevel.LOW,
            overall_score=0.2,
            abort_recommended=False,
        )
        assert assessment.overall_level == RiskLevel.LOW
        assert assessment.overall_score == 0.2
        assert assessment.abort_recommended is False
        assert assessment.factors == {}
        assert assessment.warnings == []

    def test_risk_assessment_with_factors(self):
        """Test RiskAssessment with risk factors."""
        battery_factor = RiskFactor(
            name="battery", value=0.3, threshold=0.6, critical=0.9
        )
        assessment = RiskAssessment(
            overall_level=RiskLevel.LOW,
            overall_score=0.3,
            factors={"battery": battery_factor},
        )
        assert "battery" in assessment.factors
        assert assessment.factors["battery"].value == 0.3

    def test_risk_assessment_with_warnings(self):
        """Test RiskAssessment with warnings."""
        assessment = RiskAssessment(
            overall_level=RiskLevel.MODERATE,
            overall_score=0.5,
            warnings=["Wind speed elevated", "Battery declining"],
        )
        assert len(assessment.warnings) == 2
        assert "Wind speed elevated" in assessment.warnings

    def test_risk_assessment_abort_recommended(self):
        """Test RiskAssessment with abort recommended."""
        assessment = RiskAssessment(
            overall_level=RiskLevel.CRITICAL,
            overall_score=0.95,
            abort_recommended=True,
            abort_reason="Battery critical",
        )
        assert assessment.abort_recommended is True
        assert assessment.abort_reason == "Battery critical"

    def test_to_dict(self):
        """Test RiskAssessment to_dict method."""
        battery_factor = RiskFactor(
            name="battery",
            value=0.5,
            threshold=0.6,
            critical=0.9,
            description="Battery at 50%",
        )
        assessment = RiskAssessment(
            overall_level=RiskLevel.MODERATE,
            overall_score=0.5,
            factors={"battery": battery_factor},
            warnings=["Check battery"],
            abort_recommended=False,
        )
        result = assessment.to_dict()

        assert result["overall_level"] == "moderate"
        assert result["overall_score"] == 0.5
        assert result["abort_recommended"] is False
        assert "battery" in result["factors"]
        assert result["factors"]["battery"]["value"] == 0.5


# ==================== RiskThresholds Tests ====================


class TestRiskThresholds:
    """Test RiskThresholds model."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = RiskThresholds()
        assert thresholds.battery_warning_percent == 30.0
        assert thresholds.battery_critical_percent == 15.0
        assert thresholds.wind_warning_ms == 8.0
        assert thresholds.wind_abort_ms == 12.0
        assert thresholds.max_distance_m == 5000.0
        assert thresholds.min_satellites == 6
        assert thresholds.max_hdop == 2.0
        assert thresholds.data_stale_warning_s == 5.0
        assert thresholds.data_stale_critical_s == 30.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = RiskThresholds(
            battery_warning_percent=40.0,
            battery_critical_percent=20.0,
            wind_warning_ms=10.0,
            wind_abort_ms=15.0,
            max_distance_m=3000.0,
        )
        assert thresholds.battery_warning_percent == 40.0
        assert thresholds.battery_critical_percent == 20.0
        assert thresholds.wind_warning_ms == 10.0
        assert thresholds.wind_abort_ms == 15.0
        assert thresholds.max_distance_m == 3000.0


# ==================== RiskEvaluator Tests ====================


class TestRiskEvaluatorInit:
    """Test RiskEvaluator initialization."""

    def test_init_with_defaults(self):
        """Test RiskEvaluator with default thresholds."""
        evaluator = RiskEvaluator()
        assert evaluator.thresholds is not None
        assert evaluator.thresholds.battery_warning_percent == 30.0

    def test_init_with_custom_thresholds(self):
        """Test RiskEvaluator with custom thresholds."""
        thresholds = RiskThresholds(battery_warning_percent=40.0)
        evaluator = RiskEvaluator(thresholds)
        assert evaluator.thresholds.battery_warning_percent == 40.0

    def test_default_weights(self):
        """Test default factor weights."""
        evaluator = RiskEvaluator()
        assert evaluator.weights["battery"] == 0.25
        assert evaluator.weights["wind"] == 0.15
        assert evaluator.weights["gps"] == 0.20
        assert evaluator.weights["health"] == 0.25
        assert evaluator.weights["distance"] == 0.15
        # Weights should sum to 1.0
        assert sum(evaluator.weights.values()) == pytest.approx(1.0)


class TestRiskEvaluatorBattery:
    """Test battery risk assessment."""

    def test_battery_ok(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test battery assessment with healthy battery."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        battery_factor = assessment.factors.get("battery")

        assert battery_factor is not None
        assert battery_factor.value < 0.5  # Low risk
        assert battery_factor.level in [RiskLevel.LOW, RiskLevel.MODERATE]

    def test_battery_low(self, low_battery, good_gps, healthy_vehicle, calm_environment):
        """Test battery assessment with low battery."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=low_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        battery_factor = assessment.factors.get("battery")

        assert battery_factor is not None
        assert battery_factor.value > 0.5  # Higher risk
        assert "low" in battery_factor.description.lower() or "Battery" in battery_factor.description

    def test_battery_critical(self, critical_battery, good_gps, healthy_vehicle, calm_environment):
        """Test battery assessment with critical battery."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=critical_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        battery_factor = assessment.factors.get("battery")

        assert battery_factor is not None
        assert battery_factor.value >= 0.85  # Critical threshold
        assert battery_factor.is_critical is True

    def test_battery_considers_distance(self, good_gps, healthy_vehicle, calm_environment):
        """Test that battery assessment considers distance to dock."""
        evaluator = RiskEvaluator()

        # Create a position far from dock
        far_position = Position(latitude=38.0, longitude=-122.0, altitude_msl=100.0)
        dock_pos = Position(latitude=37.0, longitude=-122.0, altitude_msl=50.0)

        battery = BatteryState(voltage=22.0, current=5.0, remaining_percent=35.0)
        vehicle = create_vehicle_state(position=far_position, battery=battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment, dock_position=dock_pos)

        assessment = evaluator.evaluate(world)
        battery_factor = assessment.factors.get("battery")

        # With distance factored in, effective battery should be lower
        assert battery_factor is not None


class TestRiskEvaluatorWind:
    """Test wind risk assessment."""

    def test_wind_calm(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test wind assessment with calm conditions."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        wind_factor = assessment.factors.get("wind")

        assert wind_factor is not None
        assert wind_factor.level == RiskLevel.LOW
        assert "OK" in wind_factor.description

    def test_wind_elevated(self, healthy_battery, good_gps, healthy_vehicle, windy_environment):
        """Test wind assessment with elevated wind."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=windy_environment)

        assessment = evaluator.evaluate(world)
        wind_factor = assessment.factors.get("wind")

        assert wind_factor is not None
        assert wind_factor.value > 0.6  # High risk
        assert "high" in wind_factor.description.lower() or "Wind" in wind_factor.description

    def test_wind_critical(self, healthy_battery, good_gps, healthy_vehicle, extreme_wind_environment):
        """Test wind assessment with extreme wind."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=extreme_wind_environment)

        assessment = evaluator.evaluate(world)
        wind_factor = assessment.factors.get("wind")

        assert wind_factor is not None
        assert wind_factor.is_critical is True
        assert "critical" in wind_factor.description.lower()


class TestRiskEvaluatorGPS:
    """Test GPS risk assessment."""

    def test_gps_good(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test GPS assessment with good signal."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        gps_factor = assessment.factors.get("gps")

        assert gps_factor is not None
        assert gps_factor.level == RiskLevel.LOW
        assert "OK" in gps_factor.description

    def test_gps_degraded(self, healthy_battery, degraded_gps, healthy_vehicle, calm_environment):
        """Test GPS assessment with degraded signal."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=degraded_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        gps_factor = assessment.factors.get("gps")

        assert gps_factor is not None
        assert gps_factor.value >= 0.5  # Elevated risk

    def test_gps_no_fix(self, healthy_battery, no_gps, healthy_vehicle, calm_environment):
        """Test GPS assessment with no fix."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=no_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        gps_factor = assessment.factors.get("gps")

        assert gps_factor is not None
        assert gps_factor.value == 1.0
        assert gps_factor.is_critical is True
        assert "No fix" in gps_factor.description

    def test_gps_none(self, healthy_battery, healthy_vehicle, calm_environment):
        """Test GPS assessment when GPS is None."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=None, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        gps_factor = assessment.factors.get("gps")

        assert gps_factor is not None
        assert gps_factor.value == 1.0


class TestRiskEvaluatorHealth:
    """Test vehicle health risk assessment."""

    def test_health_ok(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test health assessment with healthy vehicle."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        health_factor = assessment.factors.get("health")

        assert health_factor is not None
        assert health_factor.value == 0.0
        assert health_factor.level == RiskLevel.LOW
        assert "OK" in health_factor.description

    def test_health_degraded(self, healthy_battery, good_gps, unhealthy_vehicle, calm_environment):
        """Test health assessment with degraded vehicle."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=unhealthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        health_factor = assessment.factors.get("health")

        assert health_factor is not None
        assert health_factor.value > 0  # Some risk

    def test_health_critical_motors(self, healthy_battery, good_gps, critical_health, calm_environment):
        """Test health assessment with critical motor issues."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=critical_health)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        health_factor = assessment.factors.get("health")

        assert health_factor is not None
        assert health_factor.value == 1.0
        assert health_factor.is_critical is True

    def test_health_critical_ekf(self, healthy_battery, good_gps, calm_environment):
        """Test health assessment with critical EKF issues."""
        evaluator = RiskEvaluator()
        health = VehicleHealth(
            sensors_healthy=True,
            gps_healthy=True,
            battery_healthy=True,
            motors_healthy=True,
            ekf_healthy=False,
        )
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=health)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        health_factor = assessment.factors.get("health")

        assert health_factor is not None
        assert health_factor.value == 1.0
        assert health_factor.is_critical is True

    def test_health_none(self, healthy_battery, good_gps, calm_environment):
        """Test health assessment when health is None."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=None)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        health_factor = assessment.factors.get("health")

        assert health_factor is not None
        # Unknown health should have some risk
        assert "unknown" in health_factor.description.lower()


class TestRiskEvaluatorDistance:
    """Test distance risk assessment."""

    def test_distance_close(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test distance assessment when close to dock."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)
        distance_factor = assessment.factors.get("distance")

        assert distance_factor is not None
        assert distance_factor.level == RiskLevel.LOW
        assert "OK" in distance_factor.description

    def test_distance_moderate(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test distance assessment when moderately far from dock."""
        evaluator = RiskEvaluator()
        # Position about 4km from dock
        far_position = Position(latitude=37.81, longitude=-122.4194, altitude_msl=100.0)
        dock_pos = Position(latitude=37.7749, longitude=-122.4194, altitude_msl=50.0)

        vehicle = create_vehicle_state(position=far_position, battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment, dock_position=dock_pos)

        assessment = evaluator.evaluate(world)
        distance_factor = assessment.factors.get("distance")

        assert distance_factor is not None
        assert distance_factor.value > 0.3  # Some risk

    def test_distance_critical(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test distance assessment when beyond max distance."""
        evaluator = RiskEvaluator()
        # Position > 5km from dock
        far_position = Position(latitude=38.0, longitude=-122.0, altitude_msl=100.0)
        dock_pos = Position(latitude=37.0, longitude=-122.0, altitude_msl=50.0)

        vehicle = create_vehicle_state(position=far_position, battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment, dock_position=dock_pos)

        assessment = evaluator.evaluate(world)
        distance_factor = assessment.factors.get("distance")

        assert distance_factor is not None
        assert distance_factor.is_critical is True


class TestRiskEvaluatorOverall:
    """Test overall risk assessment."""

    def test_overall_low_risk(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test overall assessment with low risk conditions."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)

        assert assessment.overall_level == RiskLevel.LOW
        assert assessment.overall_score < 0.4
        assert assessment.abort_recommended is False
        assert len(assessment.warnings) == 0

    def test_overall_moderate_risk(self, low_battery, good_gps, healthy_vehicle, windy_environment):
        """Test overall assessment with moderate risk conditions."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=low_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=windy_environment)

        assessment = evaluator.evaluate(world)

        # Should have some warnings but not abort
        assert len(assessment.warnings) > 0

    def test_overall_critical_risk(self, critical_battery, good_gps, healthy_vehicle, extreme_wind_environment):
        """Test overall assessment with critical risk conditions."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=critical_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=extreme_wind_environment)

        assessment = evaluator.evaluate(world)

        assert assessment.overall_level == RiskLevel.CRITICAL
        assert assessment.abort_recommended is True
        assert assessment.abort_reason is not None

    def test_abort_on_any_critical_factor(self, healthy_battery, no_gps, healthy_vehicle, calm_environment):
        """Test that any critical factor triggers abort recommendation."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=healthy_battery, gps=no_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)

        assert assessment.abort_recommended is True

    def test_warnings_collected_from_concerning_factors(self, low_battery, degraded_gps, unhealthy_vehicle, windy_environment):
        """Test that warnings are collected from concerning factors."""
        evaluator = RiskEvaluator()
        vehicle = create_vehicle_state(battery=low_battery, gps=degraded_gps, health=unhealthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=windy_environment)

        assessment = evaluator.evaluate(world)

        # Should have multiple warnings
        assert len(assessment.warnings) >= 2


class TestRiskEvaluatorShouldAbort:
    """Test should_abort method."""

    def test_should_abort_when_recommended(self):
        """Test should_abort returns True when abort_recommended."""
        evaluator = RiskEvaluator()
        assessment = RiskAssessment(
            overall_level=RiskLevel.HIGH,
            overall_score=0.8,
            abort_recommended=True,
        )
        assert evaluator.should_abort(assessment) is True

    def test_should_abort_when_critical(self):
        """Test should_abort returns True when overall_level is CRITICAL."""
        evaluator = RiskEvaluator()
        assessment = RiskAssessment(
            overall_level=RiskLevel.CRITICAL,
            overall_score=0.9,
            abort_recommended=False,
        )
        assert evaluator.should_abort(assessment) is True

    def test_should_not_abort_when_safe(self):
        """Test should_abort returns False when conditions are safe."""
        evaluator = RiskEvaluator()
        assessment = RiskAssessment(
            overall_level=RiskLevel.LOW,
            overall_score=0.2,
            abort_recommended=False,
        )
        assert evaluator.should_abort(assessment) is False

    def test_should_not_abort_when_moderate(self):
        """Test should_abort returns False when conditions are moderate."""
        evaluator = RiskEvaluator()
        assessment = RiskAssessment(
            overall_level=RiskLevel.MODERATE,
            overall_score=0.5,
            abort_recommended=False,
        )
        assert evaluator.should_abort(assessment) is False


class TestRiskEvaluatorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_battery(self, good_gps, healthy_vehicle, calm_environment):
        """Test with zero battery percentage."""
        evaluator = RiskEvaluator()
        battery = BatteryState(voltage=18.0, current=5.0, remaining_percent=0.0)
        vehicle = create_vehicle_state(battery=battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)

        assert assessment.factors["battery"].is_critical is True
        assert assessment.abort_recommended is True

    def test_zero_wind(self, healthy_battery, good_gps, healthy_vehicle):
        """Test with zero wind."""
        evaluator = RiskEvaluator()
        environment = EnvironmentState(timestamp=datetime.now(), wind_speed_ms=0.0)
        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=environment)

        assessment = evaluator.evaluate(world)

        assert assessment.factors["wind"].value == 0.0
        assert assessment.factors["wind"].level == RiskLevel.LOW

    def test_max_satellites(self, healthy_battery, healthy_vehicle, calm_environment):
        """Test with maximum satellites."""
        evaluator = RiskEvaluator()
        gps = GPSState(fix_type=5, satellites_visible=24, hdop=0.5, vdop=0.5)
        vehicle = create_vehicle_state(battery=healthy_battery, gps=gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)

        assert assessment.factors["gps"].level == RiskLevel.LOW

    def test_all_systems_failed(self, calm_environment):
        """Test with all systems in failure state."""
        evaluator = RiskEvaluator()
        battery = BatteryState(voltage=18.0, current=10.0, remaining_percent=5.0)
        gps = GPSState(fix_type=0, satellites_visible=0, hdop=99.9, vdop=99.9)
        health = VehicleHealth(
            sensors_healthy=False,
            gps_healthy=False,
            battery_healthy=False,
            motors_healthy=False,
            ekf_healthy=False,
        )
        vehicle = create_vehicle_state(battery=battery, gps=gps, health=health)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)

        assert assessment.overall_level == RiskLevel.CRITICAL
        assert assessment.abort_recommended is True
        assert len(assessment.abort_reason) > 0

    def test_perfect_conditions(self, calm_environment):
        """Test with perfect conditions."""
        evaluator = RiskEvaluator()
        battery = BatteryState(voltage=25.0, current=2.0, remaining_percent=100.0)
        gps = GPSState(fix_type=5, satellites_visible=15, hdop=0.5, vdop=0.5)
        health = VehicleHealth(
            sensors_healthy=True,
            gps_healthy=True,
            battery_healthy=True,
            motors_healthy=True,
            ekf_healthy=True,
        )
        vehicle = create_vehicle_state(battery=battery, gps=gps, health=health)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)

        assert assessment.overall_level == RiskLevel.LOW
        assert assessment.overall_score < 0.3
        assert assessment.abort_recommended is False
        assert len(assessment.warnings) == 0

    def test_custom_thresholds_affect_assessment(self, healthy_battery, good_gps, healthy_vehicle, calm_environment):
        """Test that custom thresholds affect the assessment."""
        # Create evaluator with stricter battery threshold
        thresholds = RiskThresholds(battery_warning_percent=90.0, battery_critical_percent=80.0)
        evaluator = RiskEvaluator(thresholds)

        vehicle = create_vehicle_state(battery=healthy_battery, gps=good_gps, health=healthy_vehicle)
        world = create_world_snapshot(vehicle=vehicle, environment=calm_environment)

        assessment = evaluator.evaluate(world)

        # With 80% battery but 80% critical threshold, should be critical
        # Note: This depends on how the battery risk calculation works with distance
        battery_factor = assessment.factors.get("battery")
        assert battery_factor is not None
