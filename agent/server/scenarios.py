"""Multi-Drone Simulation Scenarios.

Preloaded scenarios for demonstrating the AegisAV decision system with
multiple drones, each with their own edge cases and operational challenges.
These scenarios can be viewed in the dashboard for training and testing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class ScenarioCategory(str, Enum):
    """Categories of simulation scenarios."""

    NORMAL_OPERATIONS = "normal_operations"
    BATTERY_CRITICAL = "battery_critical"
    GPS_DEGRADED = "gps_degraded"
    WEATHER_ADVERSE = "weather_adverse"
    SENSOR_FAILURE = "sensor_failure"
    MULTI_ANOMALY = "multi_anomaly"
    EMERGENCY_ABORT = "emergency_abort"
    MISSION_COMPLETE = "mission_complete"
    COORDINATION = "coordination"  # Multi-drone coordination scenarios


class DroneState(str, Enum):
    """Operational state of a simulated drone."""

    IDLE = "idle"
    TAKEOFF = "takeoff"
    INSPECTING = "inspecting"
    RETURNING = "returning"
    LANDING = "landing"
    CHARGING = "charging"
    EMERGENCY = "emergency"
    OFFLINE = "offline"


@dataclass
class SimulatedDrone:
    """Configuration for a simulated drone with edge cases."""

    drone_id: str
    name: str

    # Initial position
    latitude: float = 37.7749
    longitude: float = -122.4194
    altitude_agl: float = 0.0

    # Battery configuration
    battery_percent: float = 100.0
    battery_drain_rate: float = 0.5  # percent per minute
    battery_critical_threshold: float = 15.0

    # GPS configuration
    gps_fix_type: int = 3  # 3D fix
    gps_hdop: float = 0.8
    satellites_visible: int = 12

    # Health status
    sensors_healthy: bool = True
    gps_healthy: bool = True
    motors_healthy: bool = True
    ekf_healthy: bool = True

    # Current state
    state: DroneState = DroneState.IDLE
    armed: bool = False
    in_air: bool = False

    # Edge case triggers
    battery_failure_at: float | None = None  # Trigger at this battery %
    gps_loss_at: datetime | None = None  # Trigger GPS loss at time
    sensor_failure_at: datetime | None = None  # Trigger sensor failure
    motor_issue_at: datetime | None = None  # Trigger motor problem

    def to_vehicle_state(self, timestamp: datetime) -> dict[str, Any]:
        """Convert to vehicle state dictionary."""
        return {
            "timestamp": timestamp.isoformat(),
            "drone_id": self.drone_id,
            "position": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude_msl": self.altitude_agl + 50.0,
                "altitude_agl": self.altitude_agl,
            },
            "velocity": {"north": 0.0, "east": 0.0, "down": 0.0},
            "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "battery": {
                "voltage": 22.2 + (self.battery_percent / 100) * 3.0,
                "current": 5.0,
                "remaining_percent": self.battery_percent,
            },
            "mode": "GUIDED",
            "armed": self.armed,
            "in_air": self.in_air,
            "state": self.state.value,
            "gps": {
                "fix_type": self.gps_fix_type,
                "satellites_visible": self.satellites_visible,
                "hdop": self.gps_hdop,
                "vdop": 1.0,
            },
            "health": {
                "sensors_healthy": self.sensors_healthy,
                "gps_healthy": self.gps_healthy,
                "battery_healthy": self.battery_percent > 10,
                "motors_healthy": self.motors_healthy,
                "ekf_healthy": self.ekf_healthy,
            },
        }


@dataclass
class SimulatedAsset:
    """Configuration for a simulated infrastructure asset."""

    asset_id: str
    name: str
    asset_type: str  # solar_panel, wind_turbine, power_line, etc.
    latitude: float
    longitude: float
    priority: int = 1
    has_anomaly: bool = False
    anomaly_severity: float = 0.0
    last_inspected: datetime | None = None


@dataclass
class EnvironmentConditions:
    """Environmental conditions for scenario."""

    wind_speed_ms: float = 3.0
    wind_direction_deg: float = 180.0
    visibility_m: float = 10000.0
    temperature_c: float = 20.0
    precipitation: str = "none"
    is_daylight: bool = True

    # Edge case triggers
    wind_increase_at: datetime | None = None
    wind_increase_to: float = 15.0
    visibility_drop_at: datetime | None = None
    visibility_drop_to: float = 500.0


@dataclass
class ScenarioEvent:
    """An event that occurs during scenario execution."""

    timestamp_offset_s: float  # Seconds from scenario start
    event_type: str  # decision, action, anomaly, alert, etc.
    description: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """A complete simulation scenario with drones, assets, and events."""

    scenario_id: str
    name: str
    description: str
    category: ScenarioCategory
    duration_minutes: float = 30.0

    # Scenario components
    drones: list[SimulatedDrone] = field(default_factory=list)
    assets: list[SimulatedAsset] = field(default_factory=list)
    environment: EnvironmentConditions = field(default_factory=EnvironmentConditions)

    # Pre-scripted events (for replay)
    events: list[ScenarioEvent] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    difficulty: str = "normal"  # easy, normal, hard, extreme
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to dictionary for JSON serialization."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "duration_minutes": self.duration_minutes,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "drone_count": len(self.drones),
            "asset_count": len(self.assets),
            "event_count": len(self.events),
            "created_at": self.created_at.isoformat(),
        }


# Preloaded Scenarios


def create_normal_operations_scenario() -> Scenario:
    """Standard multi-drone inspection scenario - everything works well."""
    return Scenario(
        scenario_id="normal_ops_001",
        name="Normal Fleet Operations",
        description="Three drones perform routine inspections with no issues.",
        category=ScenarioCategory.NORMAL_OPERATIONS,
        duration_minutes=20.0,
        difficulty="easy",
        tags=["training", "baseline", "multi-drone"],
        drones=[
            SimulatedDrone(
                drone_id="alpha",
                name="Alpha-1",
                latitude=37.7749,
                longitude=-122.4194,
                battery_percent=95.0,
            ),
            SimulatedDrone(
                drone_id="bravo",
                name="Bravo-2",
                latitude=37.7759,
                longitude=-122.4184,
                battery_percent=88.0,
            ),
            SimulatedDrone(
                drone_id="charlie",
                name="Charlie-3",
                latitude=37.7739,
                longitude=-122.4204,
                battery_percent=92.0,
            ),
        ],
        assets=[
            SimulatedAsset(
                asset_id="solar_farm_a",
                name="Solar Farm Alpha",
                asset_type="solar_panel",
                latitude=37.7760,
                longitude=-122.4180,
                priority=1,
            ),
            SimulatedAsset(
                asset_id="solar_farm_b",
                name="Solar Farm Beta",
                asset_type="solar_panel",
                latitude=37.7770,
                longitude=-122.4170,
                priority=2,
            ),
            SimulatedAsset(
                asset_id="substation_1",
                name="Substation One",
                asset_type="substation",
                latitude=37.7755,
                longitude=-122.4195,
                priority=1,
            ),
        ],
        events=[
            ScenarioEvent(0.0, "mission_start", "Fleet begins inspection mission"),
            ScenarioEvent(30.0, "decision", "Alpha-1 assigned to Solar Farm Alpha"),
            ScenarioEvent(35.0, "decision", "Bravo-2 assigned to Solar Farm Beta"),
            ScenarioEvent(40.0, "decision", "Charlie-3 assigned to Substation One"),
            ScenarioEvent(300.0, "action", "Alpha-1 inspection complete - no issues"),
            ScenarioEvent(320.0, "action", "Bravo-2 inspection complete - no issues"),
            ScenarioEvent(360.0, "action", "Charlie-3 inspection complete - no issues"),
            ScenarioEvent(600.0, "mission_end", "All inspections complete, fleet returning"),
        ],
    )


def create_battery_cascade_scenario() -> Scenario:
    """Multiple drones experience battery issues requiring coordinated response."""
    return Scenario(
        scenario_id="battery_cascade_001",
        name="Battery Cascade Emergency",
        description="Three drones with varying battery levels. One critical, one low, one "
        "healthy. Tests priority-based return decisions.",
        category=ScenarioCategory.BATTERY_CRITICAL,
        duration_minutes=15.0,
        difficulty="hard",
        tags=["emergency", "battery", "prioritization", "multi-drone"],
        drones=[
            SimulatedDrone(
                drone_id="critical_bat",
                name="Critical Battery Drone",
                latitude=37.7780,  # Far from dock
                longitude=-122.4150,
                battery_percent=18.0,
                battery_drain_rate=1.5,  # Fast drain
                battery_critical_threshold=15.0,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="low_bat",
                name="Low Battery Drone",
                latitude=37.7765,
                longitude=-122.4175,
                battery_percent=28.0,
                battery_drain_rate=0.8,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="healthy_bat",
                name="Healthy Battery Drone",
                latitude=37.7755,
                longitude=-122.4185,
                battery_percent=72.0,
                battery_drain_rate=0.4,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
        ],
        assets=[
            SimulatedAsset(
                asset_id="incomplete_solar",
                name="Incomplete Inspection Site",
                asset_type="solar_panel",
                latitude=37.7785,
                longitude=-122.4145,
                priority=1,
                has_anomaly=True,  # Critical asset with detected anomaly
                anomaly_severity=0.75,
            ),
        ],
        environment=EnvironmentConditions(
            wind_speed_ms=8.0,  # Moderate wind increases battery drain
        ),
        events=[
            ScenarioEvent(0.0, "alert", "Battery cascade scenario initiated"),
            ScenarioEvent(60.0, "decision", "Critical Battery Drone: ABORT - battery critical"),
            ScenarioEvent(65.0, "action", "Critical Battery Drone begins emergency return"),
            ScenarioEvent(180.0, "decision", "Low Battery Drone: RETURN - battery low"),
            ScenarioEvent(185.0, "decision", "Healthy Battery Drone: CONTINUE - battery OK"),
            ScenarioEvent(300.0, "action", "Critical Battery Drone landed safely"),
            ScenarioEvent(420.0, "decision", "Healthy Battery Drone: INSPECT - cover for others"),
            ScenarioEvent(600.0, "action", "Low Battery Drone returned to dock"),
        ],
    )


def create_gps_degradation_scenario() -> Scenario:
    """GPS issues affect navigation, testing fallback behaviors."""
    return Scenario(
        scenario_id="gps_degrade_001",
        name="GPS Signal Degradation",
        description="Drones experience progressive GPS degradation. Tests graceful "
        "degradation and safe return protocols.",
        category=ScenarioCategory.GPS_DEGRADED,
        duration_minutes=25.0,
        difficulty="hard",
        tags=["gps", "navigation", "degradation", "safety"],
        drones=[
            SimulatedDrone(
                drone_id="gps_fail",
                name="GPS Failure Drone",
                latitude=37.7770,
                longitude=-122.4160,
                battery_percent=75.0,
                gps_fix_type=3,
                gps_hdop=0.9,
                satellites_visible=14,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
                # GPS will degrade progressively
                gps_loss_at=datetime.now() + timedelta(minutes=5),
            ),
            SimulatedDrone(
                drone_id="gps_weak",
                name="Weak GPS Drone",
                latitude=37.7755,
                longitude=-122.4180,
                battery_percent=82.0,
                gps_fix_type=2,  # Already degraded
                gps_hdop=2.5,  # High error
                satellites_visible=6,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="gps_good",
                name="Strong GPS Drone",
                latitude=37.7745,
                longitude=-122.4190,
                battery_percent=68.0,
                gps_fix_type=3,
                gps_hdop=0.6,
                satellites_visible=18,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
        ],
        assets=[
            SimulatedAsset(
                asset_id="urban_asset",
                name="Urban Installation",
                asset_type="substation",
                latitude=37.7775,
                longitude=-122.4155,
                priority=1,
            ),
        ],
        events=[
            ScenarioEvent(0.0, "mission_start", "GPS degradation scenario begins"),
            ScenarioEvent(120.0, "alert", "Weak GPS Drone: HDOP exceeding threshold"),
            ScenarioEvent(180.0, "decision", "Weak GPS Drone: HOLD - waiting for signal"),
            ScenarioEvent(300.0, "alert", "GPS Failure Drone: Satellites dropping"),
            ScenarioEvent(360.0, "decision", "GPS Failure Drone: RETURN - GPS unreliable"),
            ScenarioEvent(420.0, "decision", "Strong GPS Drone: CONTINUE - assist others"),
            ScenarioEvent(540.0, "alert", "GPS Failure Drone: FIX LOST"),
            ScenarioEvent(545.0, "decision", "GPS Failure Drone: ABORT - GPS failed"),
        ],
    )


def create_weather_emergency_scenario() -> Scenario:
    """Weather conditions deteriorate during mission."""
    return Scenario(
        scenario_id="weather_001",
        name="Sudden Weather Change",
        description="Weather deteriorates mid-mission. Wind increases, visibility drops. "
        "Tests coordinated fleet recall.",
        category=ScenarioCategory.WEATHER_ADVERSE,
        duration_minutes=20.0,
        difficulty="normal",
        tags=["weather", "emergency", "coordination", "recall"],
        drones=[
            SimulatedDrone(
                drone_id="far_drone",
                name="Far Field Drone",
                latitude=37.7800,  # Furthest from dock
                longitude=-122.4120,
                battery_percent=65.0,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="mid_drone",
                name="Mid Field Drone",
                latitude=37.7775,
                longitude=-122.4165,
                battery_percent=72.0,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="near_drone",
                name="Near Field Drone",
                latitude=37.7755,
                longitude=-122.4190,
                battery_percent=58.0,
                state=DroneState.RETURNING,
                armed=True,
                in_air=True,
            ),
        ],
        assets=[
            SimulatedAsset(
                asset_id="wind_farm",
                name="Wind Farm Complex",
                asset_type="wind_turbine",
                latitude=37.7810,
                longitude=-122.4110,
                priority=2,
            ),
        ],
        environment=EnvironmentConditions(
            wind_speed_ms=5.0,
            visibility_m=8000.0,
            wind_increase_at=datetime.now() + timedelta(minutes=8),
            wind_increase_to=18.0,  # Exceeds safe threshold
            visibility_drop_at=datetime.now() + timedelta(minutes=10),
            visibility_drop_to=800.0,
        ),
        events=[
            ScenarioEvent(0.0, "mission_start", "Fleet operating in favorable weather"),
            ScenarioEvent(480.0, "alert", "Weather station: Wind speed increasing"),
            ScenarioEvent(500.0, "decision", "FLEET RECALL: Weather deteriorating"),
            ScenarioEvent(510.0, "action", "Near Field Drone: Prioritized landing"),
            ScenarioEvent(520.0, "action", "Mid Field Drone: RTL initiated"),
            ScenarioEvent(530.0, "action", "Far Field Drone: Emergency RTL"),
            ScenarioEvent(600.0, "alert", "Visibility below minimum"),
            ScenarioEvent(720.0, "action", "Near Field Drone: Landed safely"),
            ScenarioEvent(840.0, "action", "Mid Field Drone: Landed safely"),
            ScenarioEvent(960.0, "action", "Far Field Drone: Landed (close call)"),
        ],
    )


def create_sensor_cascade_scenario() -> Scenario:
    """Multiple sensor failures test redundancy and decision-making."""
    return Scenario(
        scenario_id="sensor_cascade_001",
        name="Sensor Failure Cascade",
        description="Progressive sensor failures across fleet. Tests redundancy, "
        "graceful degradation, and abort decisions.",
        category=ScenarioCategory.SENSOR_FAILURE,
        duration_minutes=18.0,
        difficulty="extreme",
        tags=["sensor", "failure", "cascade", "safety", "extreme"],
        drones=[
            SimulatedDrone(
                drone_id="sens_critical",
                name="Critical Sensor Drone",
                latitude=37.7785,
                longitude=-122.4140,
                battery_percent=55.0,
                sensors_healthy=False,  # Already degraded
                ekf_healthy=False,  # EKF also failing
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
                sensor_failure_at=datetime.now() + timedelta(minutes=2),
            ),
            SimulatedDrone(
                drone_id="sens_partial",
                name="Partial Sensor Drone",
                latitude=37.7760,
                longitude=-122.4170,
                battery_percent=70.0,
                sensors_healthy=True,
                motors_healthy=False,  # Motor vibration detected
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
                motor_issue_at=datetime.now() + timedelta(minutes=5),
            ),
            SimulatedDrone(
                drone_id="sens_healthy",
                name="Healthy Sensor Drone",
                latitude=37.7745,
                longitude=-122.4195,
                battery_percent=83.0,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
        ],
        assets=[
            SimulatedAsset(
                asset_id="critical_infra",
                name="Critical Infrastructure",
                asset_type="substation",
                latitude=37.7790,
                longitude=-122.4135,
                priority=1,
                has_anomaly=True,
                anomaly_severity=0.85,
            ),
        ],
        events=[
            ScenarioEvent(0.0, "alert", "Sensor cascade scenario - high risk"),
            ScenarioEvent(30.0, "alert", "Critical Sensor Drone: EKF variance high"),
            ScenarioEvent(60.0, "decision", "Critical Sensor Drone: ABORT - EKF failed"),
            ScenarioEvent(120.0, "alert", "Partial Sensor Drone: Motor vibration detected"),
            ScenarioEvent(180.0, "decision", "Partial Sensor Drone: RETURN - motor issue"),
            ScenarioEvent(240.0, "decision", "Healthy Sensor Drone: TAKE OVER mission"),
            ScenarioEvent(300.0, "action", "Critical Sensor Drone: Emergency landing"),
            ScenarioEvent(420.0, "action", "Partial Sensor Drone: Precautionary landing"),
            ScenarioEvent(540.0, "action", "Healthy Sensor Drone: Completing inspection"),
        ],
    )


def create_multi_anomaly_scenario() -> Scenario:
    """Multiple anomalies detected requiring prioritized response."""
    return Scenario(
        scenario_id="multi_anom_001",
        name="Multiple Anomaly Detection",
        description="Several assets have detected anomalies of varying severity. "
        "Tests prioritization and resource allocation.",
        category=ScenarioCategory.MULTI_ANOMALY,
        duration_minutes=35.0,
        difficulty="normal",
        tags=["anomaly", "prioritization", "multi-target", "vision"],
        drones=[
            SimulatedDrone(
                drone_id="anom_hunter_1",
                name="Anomaly Hunter 1",
                latitude=37.7749,
                longitude=-122.4194,
                battery_percent=90.0,
                state=DroneState.IDLE,
            ),
            SimulatedDrone(
                drone_id="anom_hunter_2",
                name="Anomaly Hunter 2",
                latitude=37.7752,
                longitude=-122.4191,
                battery_percent=85.0,
                state=DroneState.IDLE,
            ),
        ],
        assets=[
            SimulatedAsset(
                asset_id="crit_solar",
                name="Critical Solar Array",
                asset_type="solar_panel",
                latitude=37.7780,
                longitude=-122.4150,
                priority=1,
                has_anomaly=True,
                anomaly_severity=0.92,  # Critical
            ),
            SimulatedAsset(
                asset_id="mod_substation",
                name="Moderate Substation",
                asset_type="substation",
                latitude=37.7765,
                longitude=-122.4170,
                priority=2,
                has_anomaly=True,
                anomaly_severity=0.55,  # Moderate
            ),
            SimulatedAsset(
                asset_id="low_tower",
                name="Low Priority Tower",
                asset_type="power_line",
                latitude=37.7790,
                longitude=-122.4140,
                priority=3,
                has_anomaly=True,
                anomaly_severity=0.25,  # Low
            ),
            SimulatedAsset(
                asset_id="new_anomaly",
                name="Newly Detected Issue",
                asset_type="wind_turbine",
                latitude=37.7775,
                longitude=-122.4160,
                priority=2,
                has_anomaly=True,
                anomaly_severity=0.78,  # High, detected mid-mission
            ),
        ],
        events=[
            ScenarioEvent(0.0, "mission_start", "Multi-anomaly response mission"),
            ScenarioEvent(30.0, "decision", "Hunter 1: Assigned to Critical Solar (0.92)"),
            ScenarioEvent(35.0, "decision", "Hunter 2: Assigned to Moderate Substation (0.55)"),
            ScenarioEvent(300.0, "alert", "NEW ANOMALY: Turbine issue detected (0.78)"),
            ScenarioEvent(330.0, "decision", "Hunter 2: REASSIGN to Turbine (higher priority)"),
            ScenarioEvent(600.0, "action", "Hunter 1: Critical Solar inspection complete"),
            ScenarioEvent(650.0, "decision", "Hunter 1: Assigned to Low Priority Tower"),
            ScenarioEvent(900.0, "action", "Hunter 2: Turbine inspection complete"),
            ScenarioEvent(1200.0, "mission_end", "All anomalies inspected"),
        ],
    )


def create_coordination_scenario() -> Scenario:
    """Tests multi-drone coordination and deconfliction."""
    return Scenario(
        scenario_id="coord_001",
        name="Fleet Coordination Challenge",
        description="Four drones must coordinate inspections in close proximity. "
        "Tests collision avoidance and task deconfliction.",
        category=ScenarioCategory.COORDINATION,
        duration_minutes=25.0,
        difficulty="hard",
        tags=["coordination", "deconfliction", "multi-drone", "proximity"],
        drones=[
            SimulatedDrone(
                drone_id="coord_a",
                name="Coordinator Alpha",
                latitude=37.7749,
                longitude=-122.4194,
                battery_percent=88.0,
                state=DroneState.IDLE,
            ),
            SimulatedDrone(
                drone_id="coord_b",
                name="Coordinator Beta",
                latitude=37.7751,
                longitude=-122.4192,
                battery_percent=82.0,
                state=DroneState.IDLE,
            ),
            SimulatedDrone(
                drone_id="coord_c",
                name="Coordinator Charlie",
                latitude=37.7747,
                longitude=-122.4196,
                battery_percent=79.0,
                state=DroneState.IDLE,
            ),
            SimulatedDrone(
                drone_id="coord_d",
                name="Coordinator Delta",
                latitude=37.7753,
                longitude=-122.4190,
                battery_percent=91.0,
                state=DroneState.IDLE,
            ),
        ],
        assets=[
            SimulatedAsset(
                asset_id="dense_array_1",
                name="Dense Array Section 1",
                asset_type="solar_panel",
                latitude=37.7760,
                longitude=-122.4175,
                priority=1,
            ),
            SimulatedAsset(
                asset_id="dense_array_2",
                name="Dense Array Section 2",
                asset_type="solar_panel",
                latitude=37.7762,
                longitude=-122.4173,  # Very close to section 1
                priority=1,
            ),
            SimulatedAsset(
                asset_id="dense_array_3",
                name="Dense Array Section 3",
                asset_type="solar_panel",
                latitude=37.7758,
                longitude=-122.4177,  # Also close
                priority=2,
            ),
            SimulatedAsset(
                asset_id="dense_array_4",
                name="Dense Array Section 4",
                asset_type="solar_panel",
                latitude=37.7764,
                longitude=-122.4171,
                priority=2,
            ),
        ],
        events=[
            ScenarioEvent(0.0, "mission_start", "Dense inspection coordination test"),
            ScenarioEvent(30.0, "decision", "Sequencing drones to avoid conflicts"),
            ScenarioEvent(60.0, "action", "Alpha: Takeoff"),
            ScenarioEvent(90.0, "action", "Beta: Takeoff (staggered)"),
            ScenarioEvent(120.0, "action", "Charlie: Takeoff (staggered)"),
            ScenarioEvent(150.0, "action", "Delta: Takeoff (staggered)"),
            ScenarioEvent(300.0, "alert", "Alpha/Beta: Proximity warning"),
            ScenarioEvent(310.0, "decision", "Beta: HOLD - Alpha has priority"),
            ScenarioEvent(450.0, "action", "Alpha: Section 1 complete, clearing"),
            ScenarioEvent(460.0, "action", "Beta: Proceeding to Section 2"),
            ScenarioEvent(600.0, "alert", "Charlie/Delta: Path conflict"),
            ScenarioEvent(610.0, "decision", "Delta: Altitude separation"),
            ScenarioEvent(900.0, "mission_end", "All sections inspected, no incidents"),
        ],
    )


# Scenario Registry

PRELOADED_SCENARIOS: dict[str, Scenario] = {}


def register_scenario(scenario: Scenario) -> None:
    """Register a scenario in the global registry."""
    PRELOADED_SCENARIOS[scenario.scenario_id] = scenario


def get_scenario(scenario_id: str) -> Scenario | None:
    """Get a scenario by ID."""
    return PRELOADED_SCENARIOS.get(scenario_id)


def get_all_scenarios() -> list[Scenario]:
    """Get all registered scenarios."""
    return list(PRELOADED_SCENARIOS.values())


def get_scenarios_by_category(category: ScenarioCategory) -> list[Scenario]:
    """Get scenarios filtered by category."""
    return [s for s in PRELOADED_SCENARIOS.values() if s.category == category]


def get_scenarios_by_difficulty(difficulty: str) -> list[Scenario]:
    """Get scenarios filtered by difficulty."""
    return [s for s in PRELOADED_SCENARIOS.values() if s.difficulty == difficulty]


def initialize_preloaded_scenarios() -> None:
    """Initialize all preloaded scenarios."""
    scenarios = [
        create_normal_operations_scenario(),
        create_battery_cascade_scenario(),
        create_gps_degradation_scenario(),
        create_weather_emergency_scenario(),
        create_sensor_cascade_scenario(),
        create_multi_anomaly_scenario(),
        create_coordination_scenario(),
    ]
    for scenario in scenarios:
        register_scenario(scenario)


# Auto-initialize on import
initialize_preloaded_scenarios()
