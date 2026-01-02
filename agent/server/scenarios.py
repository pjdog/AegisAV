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

    DOCKED = "docked"  # On dock, ready for mission
    IDLE = "idle"
    TAKEOFF = "takeoff"
    INSPECTING = "inspecting"
    RETURNING = "returning"
    LANDING = "landing"
    CHARGING = "charging"
    EMERGENCY = "emergency"
    OFFLINE = "offline"


# =============================================================================
# Standard Dock Position (all scenarios use this as home base)
# =============================================================================

# Default dock location - San Francisco area
DOCK_LATITUDE = 37.7749
DOCK_LONGITUDE = -122.4194
DOCK_ALTITUDE = 0.0


@dataclass
class SimulatedDrone:
    """Configuration for a simulated drone with edge cases."""

    drone_id: str
    name: str

    # Initial position (defaults to dock)
    latitude: float = DOCK_LATITUDE
    longitude: float = DOCK_LONGITUDE
    altitude_agl: float = DOCK_ALTITUDE

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

    # Current state (default: docked and ready)
    state: DroneState = DroneState.DOCKED
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
    altitude_m: float = 0.0
    priority: int = 1
    has_anomaly: bool = False
    anomaly_severity: float = 0.0
    last_inspected: datetime | None = None

    # Inspection profile (for flight planning + demos)
    inspection_altitude_agl: float = 30.0
    orbit_radius_m: float = 20.0
    dwell_time_s: float = 20.0

    # Unreal visualization tuning
    scale: float = 1.0
    rotation_deg: float = 0.0


@dataclass
class AssetDefect:
    """A defect on an asset that can be visualized in Unreal Engine.

    Defects are spawned as decals/materials on the asset meshes in the 3D
    scene. The drone's camera can "detect" these during inspection.
    """

    defect_id: str
    asset_id: str  # Which asset this defect is on
    defect_type: str  # crack, corrosion, hotspot, broken_cell, debris, bird_damage
    severity: float  # 0.0-1.0
    confidence: float = 0.85  # Detection confidence when found

    # UV position on asset mesh (0.0-1.0, normalized)
    uv_x: float = 0.5
    uv_y: float = 0.5

    # Defect size (normalized, 0.0-1.0 of asset)
    size: float = 0.1

    # Description for anomaly creation
    description: str = ""

    def to_spawn_message(self) -> dict[str, Any]:
        """Convert to message format for Unreal spawning."""
        return {
            "defect_id": self.defect_id,
            "asset_id": self.asset_id,
            "defect_type": self.defect_type,
            "severity": self.severity,
            "uv_x": self.uv_x,
            "uv_y": self.uv_y,
            "size": self.size,
        }


@dataclass
class EnvironmentConditions:
    """Environmental conditions for scenario."""

    wind_speed_ms: float = 3.0
    wind_direction_deg: float = 180.0
    visibility_m: float = 10000.0
    temperature_c: float = 20.0
    precipitation: str = "none"
    is_daylight: bool = True
    hour: int = 12  # 0-23 hour for AirSim time of day

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
    defects: list[AssetDefect] = field(default_factory=list)  # Defects to spawn in Unreal
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
            "defect_count": len(self.defects),
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
                battery_percent=95.0,
                # Starts at dock (default position)
            ),
            SimulatedDrone(
                drone_id="bravo",
                name="Bravo-2",
                battery_percent=88.0,
                # Starts at dock (default position)
            ),
            SimulatedDrone(
                drone_id="charlie",
                name="Charlie-3",
                battery_percent=92.0,
                # Starts at dock (default position)
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
                inspection_altitude_agl=28.0,
                orbit_radius_m=18.0,
                dwell_time_s=18.0,
                scale=1.35,
                rotation_deg=15.0,
            ),
            SimulatedAsset(
                asset_id="solar_farm_b",
                name="Solar Farm Beta",
                asset_type="solar_panel",
                latitude=37.7770,
                longitude=-122.4170,
                priority=2,
                inspection_altitude_agl=28.0,
                orbit_radius_m=18.0,
                dwell_time_s=18.0,
                scale=1.3,
                rotation_deg=32.0,
            ),
            SimulatedAsset(
                asset_id="substation_1",
                name="Substation One",
                asset_type="substation",
                latitude=37.7755,
                longitude=-122.4195,
                priority=1,
                inspection_altitude_agl=36.0,
                orbit_radius_m=26.0,
                dwell_time_s=22.0,
                scale=1.2,
                rotation_deg=8.0,
            ),
        ],
        defects=[
            # Minor debris on solar panel - good for training detection
            AssetDefect(
                defect_id="defect_norm_001",
                asset_id="solar_farm_b",
                defect_type="debris",
                severity=0.3,
                confidence=0.92,
                uv_x=0.65,
                uv_y=0.40,
                size=0.08,
                description="Minor leaf debris on panel surface",
            ),
        ],
        environment=EnvironmentConditions(
            hour=7,  # Golden hour morning light
            precipitation="mist",
            visibility_m=6000.0,
            wind_speed_ms=2.0,  # Calm morning
        ),
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
        # NOTE: This scenario starts MID-MISSION with drones already in the air
        # Drones departed from dock and are now at various inspection positions
        drones=[
            SimulatedDrone(
                drone_id="critical_bat",
                name="Critical Battery Drone",
                latitude=DOCK_LATITUDE + 0.0031,  # ~350m north of dock
                longitude=DOCK_LONGITUDE + 0.0044,  # ~370m east of dock
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
                latitude=DOCK_LATITUDE + 0.0016,  # ~180m north of dock
                longitude=DOCK_LONGITUDE + 0.0019,  # ~160m east of dock
                battery_percent=28.0,
                battery_drain_rate=0.8,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="healthy_bat",
                name="Healthy Battery Drone",
                latitude=DOCK_LATITUDE + 0.0006,  # ~65m north of dock
                longitude=DOCK_LONGITUDE + 0.0009,  # ~75m east of dock
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
                inspection_altitude_agl=30.0,
                orbit_radius_m=20.0,
                dwell_time_s=24.0,
                scale=1.4,
                rotation_deg=20.0,
            ),
        ],
        defects=[
            # Critical hotspot that needs urgent attention
            AssetDefect(
                defect_id="defect_bat_001",
                asset_id="incomplete_solar",
                defect_type="hotspot",
                severity=0.75,
                confidence=0.88,
                uv_x=0.35,
                uv_y=0.55,
                size=0.15,
                description="Thermal hotspot indicating cell failure",
            ),
            # Secondary crack near the hotspot
            AssetDefect(
                defect_id="defect_bat_002",
                asset_id="incomplete_solar",
                defect_type="crack",
                severity=0.60,
                confidence=0.82,
                uv_x=0.40,
                uv_y=0.52,
                size=0.12,
                description="Micro-crack in panel glass",
            ),
        ],
        environment=EnvironmentConditions(
            hour=14,  # Overcast afternoon
            precipitation="overcast",
            wind_speed_ms=10.0,  # Strong wind increases battery drain
            visibility_m=7000.0,
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
        # NOTE: This scenario starts MID-MISSION with drones already in the air
        drones=[
            SimulatedDrone(
                drone_id="gps_fail",
                name="GPS Failure Drone",
                latitude=DOCK_LATITUDE + 0.0021,  # ~230m north of dock
                longitude=DOCK_LONGITUDE + 0.0034,  # ~285m east of dock
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
                latitude=DOCK_LATITUDE + 0.0006,  # ~65m north of dock
                longitude=DOCK_LONGITUDE + 0.0014,  # ~120m east of dock
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
                latitude=DOCK_LATITUDE - 0.0004,  # ~45m south of dock
                longitude=DOCK_LONGITUDE + 0.0004,  # ~35m east of dock
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
                inspection_altitude_agl=34.0,
                orbit_radius_m=24.0,
                dwell_time_s=26.0,
                scale=1.25,
                rotation_deg=12.0,
            ),
        ],
        defects=[
            # Corrosion on substation - requires precise GPS for documentation
            AssetDefect(
                defect_id="defect_gps_001",
                asset_id="urban_asset",
                defect_type="corrosion",
                severity=0.55,
                confidence=0.78,
                uv_x=0.25,
                uv_y=0.70,
                size=0.18,
                description="Surface corrosion on transformer housing",
            ),
        ],
        environment=EnvironmentConditions(
            hour=18,  # Dusk - warm sunset colors
            precipitation="haze",
            visibility_m=4000.0,  # Reduced visibility adds to GPS challenge theme
            wind_speed_ms=4.0,
        ),
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
        # NOTE: This scenario starts MID-MISSION with drones already in the air
        drones=[
            SimulatedDrone(
                drone_id="far_drone",
                name="Far Field Drone",
                latitude=DOCK_LATITUDE + 0.0051,  # ~570m north (furthest from dock)
                longitude=DOCK_LONGITUDE + 0.0074,  # ~620m east
                battery_percent=65.0,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="mid_drone",
                name="Mid Field Drone",
                latitude=DOCK_LATITUDE + 0.0026,  # ~290m north
                longitude=DOCK_LONGITUDE + 0.0029,  # ~245m east
                battery_percent=72.0,
                state=DroneState.INSPECTING,
                armed=True,
                in_air=True,
            ),
            SimulatedDrone(
                drone_id="near_drone",
                name="Near Field Drone",
                latitude=DOCK_LATITUDE + 0.0006,  # ~65m north
                longitude=DOCK_LONGITUDE + 0.0004,  # ~35m east
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
                inspection_altitude_agl=48.0,
                orbit_radius_m=34.0,
                dwell_time_s=24.0,
                scale=1.15,
                rotation_deg=0.0,
            ),
            SimulatedAsset(
                asset_id="wind_farm_west",
                name="Wind Farm West",
                asset_type="wind_turbine",
                latitude=37.7804,
                longitude=-122.4132,
                priority=1,
                inspection_altitude_agl=48.0,
                orbit_radius_m=34.0,
                dwell_time_s=24.0,
                scale=1.12,
                rotation_deg=8.0,
            ),
            SimulatedAsset(
                asset_id="wind_farm_east",
                name="Wind Farm East",
                asset_type="wind_turbine",
                latitude=37.7816,
                longitude=-122.4093,
                priority=3,
                inspection_altitude_agl=46.0,
                orbit_radius_m=32.0,
                dwell_time_s=22.0,
                scale=1.1,
                rotation_deg=352.0,
            ),
        ],
        defects=[
            # Bird damage on turbine blade - visible in deteriorating weather
            AssetDefect(
                defect_id="defect_wth_001",
                asset_id="wind_farm",
                defect_type="bird_damage",
                severity=0.45,
                confidence=0.75,
                uv_x=0.80,
                uv_y=0.30,
                size=0.10,
                description="Bird strike damage on turbine blade leading edge",
            ),
        ],
        environment=EnvironmentConditions(
            hour=11,  # Late morning, storm approaches
            precipitation="light_rain",  # Starting conditions
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
        # NOTE: This scenario starts MID-MISSION with drones already in the air
        drones=[
            SimulatedDrone(
                drone_id="sens_critical",
                name="Critical Sensor Drone",
                latitude=DOCK_LATITUDE + 0.0036,  # ~400m north
                longitude=DOCK_LONGITUDE + 0.0054,  # ~450m east
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
                latitude=DOCK_LATITUDE + 0.0011,  # ~120m north
                longitude=DOCK_LONGITUDE + 0.0024,  # ~200m east
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
                latitude=DOCK_LATITUDE - 0.0004,  # ~45m south
                longitude=DOCK_LONGITUDE - 0.0001,  # Near dock longitude
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
                inspection_altitude_agl=38.0,
                orbit_radius_m=28.0,
                dwell_time_s=26.0,
                scale=1.3,
                rotation_deg=18.0,
            ),
        ],
        defects=[
            # Critical broken cell that must be documented despite sensor issues
            AssetDefect(
                defect_id="defect_sens_001",
                asset_id="critical_infra",
                defect_type="broken_cell",
                severity=0.85,
                confidence=0.70,  # Lower confidence due to sensor degradation
                uv_x=0.50,
                uv_y=0.45,
                size=0.20,
                description="Damaged cell requiring immediate attention",
            ),
            # Secondary corrosion, hard to see at night
            AssetDefect(
                defect_id="defect_sens_002",
                asset_id="critical_infra",
                defect_type="corrosion",
                severity=0.50,
                confidence=0.65,
                uv_x=0.70,
                uv_y=0.80,
                size=0.14,
                description="Surface corrosion visible under IR",
            ),
        ],
        environment=EnvironmentConditions(
            hour=22,  # Night operations
            is_daylight=False,
            precipitation="dust",
            visibility_m=3000.0,  # Reduced night visibility
            wind_speed_ms=6.0,
        ),
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
                battery_percent=90.0,
                # Starts at dock (default position)
            ),
            SimulatedDrone(
                drone_id="anom_hunter_2",
                name="Anomaly Hunter 2",
                battery_percent=85.0,
                # Starts at dock (default position)
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
                inspection_altitude_agl=30.0,
                orbit_radius_m=20.0,
                dwell_time_s=22.0,
                scale=1.45,
                rotation_deg=25.0,
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
                inspection_altitude_agl=36.0,
                orbit_radius_m=26.0,
                dwell_time_s=24.0,
                scale=1.2,
                rotation_deg=6.0,
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
                inspection_altitude_agl=40.0,
                orbit_radius_m=32.0,
                dwell_time_s=16.0,
                scale=1.05,
                rotation_deg=40.0,
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
                inspection_altitude_agl=48.0,
                orbit_radius_m=34.0,
                dwell_time_s=26.0,
                scale=1.1,
                rotation_deg=12.0,
            ),
        ],
        defects=[
            # Critical hotspot on solar array - highest priority
            AssetDefect(
                defect_id="defect_multi_001",
                asset_id="crit_solar",
                defect_type="hotspot",
                severity=0.92,
                confidence=0.95,
                uv_x=0.45,
                uv_y=0.35,
                size=0.22,
                description="Critical thermal hotspot - potential fire risk",
            ),
            # Moderate corrosion on substation
            AssetDefect(
                defect_id="defect_multi_002",
                asset_id="mod_substation",
                defect_type="corrosion",
                severity=0.55,
                confidence=0.88,
                uv_x=0.30,
                uv_y=0.65,
                size=0.16,
                description="Moderate surface corrosion on equipment housing",
            ),
            # Minor debris on power line
            AssetDefect(
                defect_id="defect_multi_003",
                asset_id="low_tower",
                defect_type="debris",
                severity=0.25,
                confidence=0.90,
                uv_x=0.55,
                uv_y=0.20,
                size=0.08,
                description="Bird nest debris on tower crossarm",
            ),
            # Crack on turbine blade - detected mid-mission
            AssetDefect(
                defect_id="defect_multi_004",
                asset_id="new_anomaly",
                defect_type="crack",
                severity=0.78,
                confidence=0.85,
                uv_x=0.75,
                uv_y=0.40,
                size=0.18,
                description="Structural crack in blade root section",
            ),
        ],
        environment=EnvironmentConditions(
            hour=12,  # Harsh midday sun for maximum contrast
            visibility_m=15000.0,  # Crystal clear
            wind_speed_ms=3.0,  # Light breeze
        ),
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
                battery_percent=88.0,
                # Starts at dock (default position)
            ),
            SimulatedDrone(
                drone_id="coord_b",
                name="Coordinator Beta",
                battery_percent=82.0,
                # Starts at dock (default position)
            ),
            SimulatedDrone(
                drone_id="coord_c",
                name="Coordinator Charlie",
                battery_percent=79.0,
                # Starts at dock (default position)
            ),
            SimulatedDrone(
                drone_id="coord_d",
                name="Coordinator Delta",
                battery_percent=91.0,
                # Starts at dock (default position)
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
                inspection_altitude_agl=26.0,
                orbit_radius_m=16.0,
                dwell_time_s=16.0,
                scale=1.2,
                rotation_deg=20.0,
            ),
            SimulatedAsset(
                asset_id="dense_array_2",
                name="Dense Array Section 2",
                asset_type="solar_panel",
                latitude=37.7762,
                longitude=-122.4173,  # Very close to section 1
                priority=1,
                inspection_altitude_agl=26.0,
                orbit_radius_m=16.0,
                dwell_time_s=16.0,
                scale=1.2,
                rotation_deg=32.0,
            ),
            SimulatedAsset(
                asset_id="dense_array_3",
                name="Dense Array Section 3",
                asset_type="solar_panel",
                latitude=37.7758,
                longitude=-122.4177,  # Also close
                priority=2,
                inspection_altitude_agl=26.0,
                orbit_radius_m=16.0,
                dwell_time_s=16.0,
                scale=1.2,
                rotation_deg=8.0,
            ),
            SimulatedAsset(
                asset_id="dense_array_4",
                name="Dense Array Section 4",
                asset_type="solar_panel",
                latitude=37.7764,
                longitude=-122.4171,
                priority=2,
                inspection_altitude_agl=26.0,
                orbit_radius_m=16.0,
                dwell_time_s=16.0,
                scale=1.2,
                rotation_deg=42.0,
            ),
        ],
        defects=[
            # Multiple defects across the dense array for coordination testing
            AssetDefect(
                defect_id="defect_coord_001",
                asset_id="dense_array_1",
                defect_type="crack",
                severity=0.50,
                confidence=0.87,
                uv_x=0.30,
                uv_y=0.50,
                size=0.12,
                description="Surface crack in panel glass",
            ),
            AssetDefect(
                defect_id="defect_coord_002",
                asset_id="dense_array_2",
                defect_type="hotspot",
                severity=0.65,
                confidence=0.91,
                uv_x=0.60,
                uv_y=0.45,
                size=0.14,
                description="Thermal anomaly in junction box area",
            ),
            AssetDefect(
                defect_id="defect_coord_003",
                asset_id="dense_array_3",
                defect_type="debris",
                severity=0.30,
                confidence=0.94,
                uv_x=0.40,
                uv_y=0.25,
                size=0.10,
                description="Accumulated debris affecting output",
            ),
        ],
        environment=EnvironmentConditions(
            hour=17,  # Golden sunset hour
            precipitation="light_fog",
            visibility_m=5000.0,  # Light fog adds atmosphere
            wind_speed_ms=4.0,
        ),
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


def create_showcase_scenario() -> Scenario:
    """High-polish demo scenario with diverse assets and clear visual beats."""
    return Scenario(
        scenario_id="showcase_001",
        name="AegisAV Showcase",
        description=(
            "Curated multi-drone inspection run spanning solar, substation, wind, "
            "and power line assets with staged anomalies."
        ),
        category=ScenarioCategory.NORMAL_OPERATIONS,
        duration_minutes=18.0,
        difficulty="normal",
        tags=["showcase", "demo", "multi-drone", "vision"],
        drones=[
            SimulatedDrone(drone_id="delta", name="Delta-1", battery_percent=94.0),
            SimulatedDrone(drone_id="echo", name="Echo-2", battery_percent=90.0),
            SimulatedDrone(drone_id="foxtrot", name="Foxtrot-3", battery_percent=92.0),
        ],
        assets=[
            SimulatedAsset(
                asset_id="show_solar_west",
                name="West Solar Array",
                asset_type="solar_panel",
                latitude=DOCK_LATITUDE + 0.0011,
                longitude=DOCK_LONGITUDE + 0.0012,
                priority=1,
                has_anomaly=True,
                anomaly_severity=0.42,
                inspection_altitude_agl=28.0,
                orbit_radius_m=18.0,
                dwell_time_s=18.0,
                scale=1.45,
                rotation_deg=22.0,
            ),
            SimulatedAsset(
                asset_id="show_substation",
                name="Grid Substation Node",
                asset_type="substation",
                latitude=DOCK_LATITUDE + 0.0002,
                longitude=DOCK_LONGITUDE + 0.0016,
                priority=1,
                has_anomaly=True,
                anomaly_severity=0.68,
                inspection_altitude_agl=38.0,
                orbit_radius_m=26.0,
                dwell_time_s=24.0,
                scale=1.25,
                rotation_deg=12.0,
            ),
            SimulatedAsset(
                asset_id="show_wind_ridge",
                name="Ridge Wind Turbine",
                asset_type="wind_turbine",
                latitude=DOCK_LATITUDE + 0.0020,
                longitude=DOCK_LONGITUDE + 0.0024,
                priority=2,
                has_anomaly=True,
                anomaly_severity=0.75,
                inspection_altitude_agl=50.0,
                orbit_radius_m=36.0,
                dwell_time_s=26.0,
                scale=1.15,
                rotation_deg=8.0,
            ),
            SimulatedAsset(
                asset_id="show_power_corridor",
                name="Transmission Corridor",
                asset_type="power_line",
                latitude=DOCK_LATITUDE + 0.0016,
                longitude=DOCK_LONGITUDE + 0.0006,
                priority=3,
                has_anomaly=False,
                inspection_altitude_agl=42.0,
                orbit_radius_m=32.0,
                dwell_time_s=16.0,
                scale=1.05,
                rotation_deg=46.0,
            ),
        ],
        defects=[
            AssetDefect(
                defect_id="defect_show_001",
                asset_id="show_solar_west",
                defect_type="debris",
                severity=0.42,
                confidence=0.9,
                uv_x=0.62,
                uv_y=0.35,
                size=0.1,
                description="Debris shading a cell string on the array edge",
            ),
            AssetDefect(
                defect_id="defect_show_002",
                asset_id="show_substation",
                defect_type="corrosion",
                severity=0.68,
                confidence=0.86,
                uv_x=0.28,
                uv_y=0.55,
                size=0.14,
                description="Visible corrosion on substation panel housing",
            ),
            AssetDefect(
                defect_id="defect_show_003",
                asset_id="show_wind_ridge",
                defect_type="crack",
                severity=0.75,
                confidence=0.82,
                uv_x=0.74,
                uv_y=0.32,
                size=0.18,
                description="Blade root crack detected under thermal sweep",
            ),
        ],
        environment=EnvironmentConditions(
            hour=16,
            precipitation="haze",
            visibility_m=9000.0,
            wind_speed_ms=3.2,
            wind_direction_deg=210.0,
        ),
        events=[
            ScenarioEvent(0.0, "mission_start", "Showcase mission begins"),
            ScenarioEvent(20.0, "decision", "Delta-1: Assigned to West Solar Array"),
            ScenarioEvent(35.0, "decision", "Echo-2: Assigned to Substation Node"),
            ScenarioEvent(50.0, "decision", "Foxtrot-3: Assigned to Ridge Turbine"),
            ScenarioEvent(300.0, "alert", "Anomaly confirmed on substation housing"),
            ScenarioEvent(420.0, "action", "Delta-1: Solar inspection complete"),
            ScenarioEvent(540.0, "action", "Echo-2: Substation inspection complete"),
            ScenarioEvent(700.0, "action", "Foxtrot-3: Turbine inspection complete"),
            ScenarioEvent(900.0, "mission_end", "Showcase run complete - fleet returning"),
        ],
    )


def create_patrol_recharge_scenario() -> Scenario:
    """Extended patrol scenario where drones observe, return for battery, then resume.

    This scenario demonstrates the full operational cycle:
    1. Drones launch from dock and begin inspections
    2. Battery drains faster than normal (2.5%/min) to accelerate the cycle
    3. When battery drops below return threshold, drones RTL
    4. Drones charge at dock (2%/min)
    5. Once charged (~95%), drones resume patrol
    6. Cycle continues throughout mission duration

    Duration: 45 minutes to allow multiple charge cycles
    """
    return Scenario(
        scenario_id="patrol_recharge_001",
        name="Patrol & Recharge Cycle",
        description=(
            "Extended autonomous patrol demonstrating full operational cycle: "
            "inspect assets, return to dock when battery low, recharge, and resume patrol. "
            "Drones will complete multiple charge cycles over the mission duration."
        ),
        category=ScenarioCategory.NORMAL_OPERATIONS,
        duration_minutes=45.0,
        difficulty="normal",
        tags=["patrol", "recharge", "endurance", "multi-drone", "autonomous"],
        drones=[
            SimulatedDrone(
                drone_id="patrol_a",
                name="Patrol Alpha",
                battery_percent=55.0,  # Start at 55% to trigger first return sooner
                battery_drain_rate=2.5,  # Faster drain (2.5% per minute) for demo
                battery_critical_threshold=18.0,
                # Starts at dock
            ),
            SimulatedDrone(
                drone_id="patrol_b",
                name="Patrol Bravo",
                battery_percent=70.0,  # Staggered start - B has more battery
                battery_drain_rate=2.2,  # Slightly slower drain
                battery_critical_threshold=18.0,
                # Starts at dock
            ),
            SimulatedDrone(
                drone_id="patrol_c",
                name="Patrol Charlie",
                battery_percent=40.0,  # Low start - will return first
                battery_drain_rate=2.8,  # Fastest drain
                battery_critical_threshold=18.0,
                # Starts at dock
            ),
        ],
        assets=[
            # Multiple assets spread across the area for continuous patrol
            SimulatedAsset(
                asset_id="patrol_solar_north",
                name="North Solar Array",
                asset_type="solar_panel",
                latitude=DOCK_LATITUDE + 0.0018,  # ~200m north
                longitude=DOCK_LONGITUDE + 0.0008,  # ~70m east
                priority=1,
                has_anomaly=True,
                anomaly_severity=0.45,
                inspection_altitude_agl=28.0,
                orbit_radius_m=18.0,
                dwell_time_s=20.0,
                scale=1.3,
                rotation_deg=15.0,
            ),
            SimulatedAsset(
                asset_id="patrol_solar_east",
                name="East Solar Field",
                asset_type="solar_panel",
                latitude=DOCK_LATITUDE + 0.0008,  # ~90m north
                longitude=DOCK_LONGITUDE + 0.0020,  # ~170m east
                priority=1,
                has_anomaly=False,
                inspection_altitude_agl=26.0,
                orbit_radius_m=16.0,
                dwell_time_s=18.0,
                scale=1.25,
                rotation_deg=25.0,
            ),
            SimulatedAsset(
                asset_id="patrol_substation",
                name="Central Substation",
                asset_type="substation",
                latitude=DOCK_LATITUDE + 0.0012,  # ~130m north
                longitude=DOCK_LONGITUDE + 0.0014,  # ~120m east
                priority=1,
                has_anomaly=True,
                anomaly_severity=0.55,
                inspection_altitude_agl=36.0,
                orbit_radius_m=24.0,
                dwell_time_s=22.0,
                scale=1.2,
                rotation_deg=8.0,
            ),
            SimulatedAsset(
                asset_id="patrol_turbine",
                name="Patrol Wind Turbine",
                asset_type="wind_turbine",
                latitude=DOCK_LATITUDE + 0.0024,  # ~270m north
                longitude=DOCK_LONGITUDE + 0.0006,  # ~50m east
                priority=2,
                has_anomaly=False,
                inspection_altitude_agl=48.0,
                orbit_radius_m=32.0,
                dwell_time_s=24.0,
                scale=1.1,
                rotation_deg=0.0,
            ),
            SimulatedAsset(
                asset_id="patrol_powerline",
                name="South Power Corridor",
                asset_type="power_line",
                latitude=DOCK_LATITUDE - 0.0005,  # ~55m south
                longitude=DOCK_LONGITUDE + 0.0018,  # ~150m east
                priority=3,
                has_anomaly=False,
                inspection_altitude_agl=40.0,
                orbit_radius_m=28.0,
                dwell_time_s=16.0,
                scale=1.0,
                rotation_deg=45.0,
            ),
        ],
        defects=[
            AssetDefect(
                defect_id="defect_patrol_001",
                asset_id="patrol_solar_north",
                defect_type="hotspot",
                severity=0.45,
                confidence=0.88,
                uv_x=0.55,
                uv_y=0.42,
                size=0.12,
                description="Thermal hotspot on north array panel",
            ),
            AssetDefect(
                defect_id="defect_patrol_002",
                asset_id="patrol_substation",
                defect_type="corrosion",
                severity=0.55,
                confidence=0.85,
                uv_x=0.32,
                uv_y=0.60,
                size=0.14,
                description="Surface corrosion on substation housing",
            ),
        ],
        environment=EnvironmentConditions(
            hour=10,  # Late morning - good lighting for patrol
            precipitation="none",
            visibility_m=12000.0,
            wind_speed_ms=3.5,
            wind_direction_deg=180.0,  # Southerly wind
        ),
        events=[
            ScenarioEvent(0.0, "mission_start", "Patrol & Recharge mission begins"),
            ScenarioEvent(20.0, "decision", "Patrol Alpha: Assigned to North Solar Array"),
            ScenarioEvent(30.0, "decision", "Patrol Bravo: Assigned to East Solar Field"),
            ScenarioEvent(40.0, "decision", "Patrol Charlie: Assigned to Central Substation"),
            # Charlie runs low first due to low start + fast drain
            ScenarioEvent(180.0, "alert", "Patrol Charlie: Battery at return threshold - RTL"),
            ScenarioEvent(220.0, "action", "Patrol Charlie: Docked and charging"),
            # Alpha returns next
            ScenarioEvent(360.0, "alert", "Patrol Alpha: Battery at return threshold - RTL"),
            ScenarioEvent(400.0, "action", "Patrol Alpha: Docked and charging"),
            # Charlie resumes after charging
            ScenarioEvent(480.0, "decision", "Patrol Charlie: Charged - resuming patrol"),
            # Bravo returns
            ScenarioEvent(540.0, "alert", "Patrol Bravo: Battery at return threshold - RTL"),
            ScenarioEvent(580.0, "action", "Patrol Bravo: Docked and charging"),
            # Alpha resumes
            ScenarioEvent(660.0, "decision", "Patrol Alpha: Charged - resuming patrol"),
            # Second cycle continues...
            ScenarioEvent(780.0, "decision", "Patrol Bravo: Charged - resuming patrol"),
            ScenarioEvent(900.0, "alert", "Patrol Charlie: Second RTL for battery"),
            ScenarioEvent(1200.0, "decision", "Patrol Charlie: Second charge complete"),
            ScenarioEvent(1500.0, "alert", "Patrol Alpha: Second RTL for battery"),
            ScenarioEvent(1800.0, "alert", "Patrol Bravo: Second RTL for battery"),
            ScenarioEvent(2400.0, "mission_end", "Patrol mission complete - all drones to dock"),
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
        create_showcase_scenario(),
        create_patrol_recharge_scenario(),  # Patrol with battery recharge cycles
    ]
    for scenario in scenarios:
        register_scenario(scenario)


# Auto-initialize on import
initialize_preloaded_scenarios()
