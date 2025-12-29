"""World Model.

Maintains a unified, consistent view of the operational environment
for decision-making. Fuses data from vehicle telemetry, asset database,
and environmental sources.
"""

from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from autonomy.vehicle_state import Position, VehicleState


class AssetType(Enum):
    """Types of infrastructure assets."""

    SOLAR_PANEL = "solar_panel"
    WIND_TURBINE = "wind_turbine"
    SUBSTATION = "substation"
    POWER_LINE = "power_line"
    BUILDING = "building"
    CROP_FIELD = "crop_field"
    OTHER = "other"


class AssetStatus(Enum):
    """Current status of an asset."""

    NORMAL = "normal"
    WARNING = "warning"
    ANOMALY = "anomaly"
    UNKNOWN = "unknown"


class DockStatus(Enum):
    """Dock availability status."""

    AVAILABLE = "available"
    OCCUPIED = "occupied"
    CHARGING = "charging"
    OFFLINE = "offline"


class Asset(BaseModel):
    """An infrastructure asset being monitored."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    asset_id: str
    name: str
    asset_type: AssetType
    position: Position

    # Inspection parameters
    inspection_altitude_agl: float = 20.0
    orbit_radius_m: float = 20.0
    dwell_time_s: float = 30.0

    # Status
    status: AssetStatus = AssetStatus.UNKNOWN
    priority: int = 1  # Lower = higher priority

    # Timing
    last_inspection: datetime | None = None
    next_scheduled: datetime | None = None

    @property
    def time_since_inspection(self) -> timedelta | None:
        """Time elapsed since last inspection."""
        if self.last_inspection is None:
            return None
        return datetime.now() - self.last_inspection

    @property
    def needs_inspection(self) -> bool:
        """Check if asset is due for inspection."""
        if self.next_scheduled is None:
            return True
        return datetime.now() >= self.next_scheduled


class Anomaly(BaseModel):
    """A detected anomaly at an asset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    anomaly_id: str
    asset_id: str
    detected_at: datetime
    severity: float = Field(..., ge=0.0, le=1.0)
    description: str

    # Location
    position: Position | None = None

    # Status
    acknowledged: bool = False
    resolved: bool = False


class DockState(BaseModel):
    """State of the charging dock."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: Position
    status: DockStatus

    # Charging info
    charge_rate_percent_per_minute: float = 1.5
    current_vehicle_id: str | None = None

    # Approach parameters
    approach_altitude_m: float = 10.0


class EnvironmentState(BaseModel):
    """Environmental conditions."""

    timestamp: datetime

    # Weather
    wind_speed_ms: float = 0.0
    wind_direction_deg: float = 0.0
    temperature_c: float = 20.0
    visibility_m: float = 10000.0

    # Conditions
    precipitation: str = "none"  # none, light_rain, heavy_rain, snow, etc.
    is_daylight: bool = True

    @property
    def is_flyable(self) -> bool:
        """Check if conditions allow flight."""
        return (
            self.wind_speed_ms < 12.0
            and self.visibility_m > 1000.0
            and self.precipitation in ("none", "light_rain")
        )


class MissionState(BaseModel):
    """Current mission progress."""

    mission_id: str
    mission_name: str

    # Progress
    started_at: datetime | None = None
    current_phase: str = "idle"
    assets_inspected: int = 0
    assets_total: int = 0

    # Status
    is_active: bool = False
    is_complete: bool = False
    abort_reason: str | None = None

    @property
    def progress_percent(self) -> float:
        """Calculate mission progress percentage."""
        if self.assets_total == 0:
            return 0.0
        return (self.assets_inspected / self.assets_total) * 100


class WorldSnapshot(BaseModel):
    """Immutable snapshot of the world state at a point in time.

    This is passed to the goal selector and risk evaluator for
    decision-making. Creating snapshots ensures consistent state
    during decision computation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime

    # Vehicle
    vehicle: VehicleState

    # Assets
    assets: list[Asset]
    anomalies: list[Anomaly]

    # Infrastructure
    dock: DockState

    # Environment
    environment: EnvironmentState

    # Mission
    mission: MissionState

    # Computed
    overall_confidence: float = 1.0

    def get_asset(self, asset_id: str) -> Asset | None:
        """Get asset by ID.

        Args:
            asset_id: Asset identifier.

        Returns:
            Asset if found, None otherwise.
        """
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        return None

    def get_pending_assets(self) -> list[Asset]:
        """Get assets that need inspection, sorted by priority.

        Returns:
            List of assets needing inspection, sorted by priority.
        """
        pending = [a for a in self.assets if a.needs_inspection]
        return sorted(pending, key=lambda a: a.priority)

    def get_anomaly_assets(self) -> list[Asset]:
        """Get assets with active anomalies.

        Returns:
            List of assets that have unresolved anomalies.
        """
        anomaly_asset_ids = {a.asset_id for a in self.anomalies if not a.resolved}
        return [a for a in self.assets if a.asset_id in anomaly_asset_ids]

    def distance_to_dock(self) -> float:
        """Calculate distance from current position to dock.

        Returns:
            Distance in meters to the dock.
        """
        return self.vehicle.position.distance_to(self.dock.position)


class WorldModel:
    """Maintains and updates the world state.

    The WorldModel is the single source of truth for the agent's
    understanding of the operational environment. It:

    - Aggregates vehicle telemetry
    - Tracks asset status and inspection history
    - Monitors environmental conditions
    - Provides consistent snapshots for decision-making

    Example:
        model = WorldModel()
        model.load_assets_from_config(mission_config)
        model.set_dock(dock_position)

        # Update from telemetry
        model.update_vehicle(vehicle_state)

        # Get snapshot for decisions
        snapshot = model.get_snapshot()
    """

    def __init__(self) -> None:
        """Initialize the WorldModel."""
        self._vehicle: VehicleState | None = None
        self._assets: list[Asset] = []
        self._anomalies: list[Anomaly] = []
        self._dock: DockState | None = None
        self._environment = EnvironmentState(timestamp=datetime.now())
        self._mission = MissionState(mission_id="", mission_name="")

        self._last_update: datetime | None = None

    def update_vehicle(self, state: VehicleState) -> None:
        """Update vehicle state from telemetry.

        Args:
            state: New vehicle state from telemetry.
        """
        self._vehicle = state
        self._last_update = datetime.now()

    def set_dock(self, position: Position, status: DockStatus = DockStatus.AVAILABLE) -> None:
        """Set dock position and status.

        Args:
            position: Dock position.
            status: Dock availability status.
        """
        self._dock = DockState(position=position, status=status)

    def add_asset(self, asset: Asset) -> None:
        """Add or update an asset.

        Args:
            asset: Asset to add or update.
        """
        # Replace if exists
        self._assets = [a for a in self._assets if a.asset_id != asset.asset_id]
        self._assets.append(asset)

    def load_assets_from_config(self, config: dict) -> None:
        """Load assets from mission configuration.

        Args:
            config: Mission configuration dictionary
        """
        assets_config = config.get("assets", [])

        for asset_data in assets_config:
            pos = asset_data.get("position", {})
            inspection = asset_data.get("inspection", {})

            asset = Asset(
                asset_id=asset_data["id"],
                name=asset_data.get("name", asset_data["id"]),
                asset_type=AssetType(asset_data.get("type", "other")),
                position=Position(
                    latitude=pos.get("latitude", 0),
                    longitude=pos.get("longitude", 0),
                    altitude_msl=pos.get("altitude_m", 0),
                ),
                inspection_altitude_agl=inspection.get("altitude_agl_m", 20),
                orbit_radius_m=inspection.get("orbit_radius_m", 20),
                dwell_time_s=inspection.get("dwell_time_s", 30),
                priority=asset_data.get("priority", 1),
            )
            self.add_asset(asset)

    def record_inspection(self, asset_id: str, cadence_minutes: float = 30) -> None:
        """Record that an asset was inspected.

        Args:
            asset_id: Asset identifier that was inspected.
            cadence_minutes: Minutes until next scheduled inspection.
        """
        for asset in self._assets:
            if asset.asset_id == asset_id:
                asset.last_inspection = datetime.now()
                asset.next_scheduled = datetime.now() + timedelta(minutes=cadence_minutes)
                break

    def add_anomaly(self, anomaly: Anomaly) -> None:
        """Add a detected anomaly.

        Args:
            anomaly: Anomaly to add.
        """
        self._anomalies.append(anomaly)

        # Update asset status
        for asset in self._assets:
            if asset.asset_id == anomaly.asset_id:
                asset.status = AssetStatus.ANOMALY
                break

    def resolve_anomaly(self, anomaly_id: str) -> None:
        """Mark an anomaly as resolved.

        Args:
            anomaly_id: Anomaly identifier to resolve.
        """
        for anomaly in self._anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.resolved = True
                break

    def update_environment(self, env: EnvironmentState) -> None:
        """Update environmental conditions.

        Args:
            env: New environmental state.
        """
        self._environment = env

    def start_mission(self, mission_id: str, mission_name: str) -> None:
        """Start a new mission.

        Args:
            mission_id: Unique mission identifier.
            mission_name: Human-readable mission name.
        """
        self._mission = MissionState(
            mission_id=mission_id,
            mission_name=mission_name,
            started_at=datetime.now(),
            is_active=True,
            assets_total=len(self._assets),
        )

    def update_asset_status(self, asset_id: str, status: AssetStatus) -> bool:
        """Update an asset status by asset ID.

        Args:
            asset_id (str): Asset identifier.
            status (AssetStatus): New status to apply.

        Returns:
            bool: True if the asset was found and updated.
        """
        for asset in self._assets:
            if asset.asset_id == asset_id:
                asset.status = status
                return True
        return False

    def get_anomaly_assets(self) -> list[str]:
        """Get asset IDs with active anomalies.

        Returns:
            List of asset IDs that have unresolved anomalies
        """
        return [a.asset_id for a in self._anomalies if not a.resolved]

    def get_snapshot(self) -> WorldSnapshot | None:
        """Get an immutable snapshot of current world state.

        Returns:
            WorldSnapshot if sufficient data available, None otherwise
        """
        if self._vehicle is None or self._dock is None:
            return None

        return WorldSnapshot(
            timestamp=datetime.now(),
            vehicle=self._vehicle,
            assets=list(self._assets),
            anomalies=[a for a in self._anomalies if not a.resolved],
            dock=self._dock,
            environment=self._environment,
            mission=self._mission,
        )

    def time_since_update(self) -> timedelta | None:
        """Time since last vehicle state update.

        Returns:
            Timedelta since last update, or None if never updated.
        """
        if self._last_update is None:
            return None
        return datetime.now() - self._last_update
