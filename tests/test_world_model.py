"""
Comprehensive tests for WorldModel to increase coverage.
"""

from datetime import datetime, timedelta

import pytest

from agent.server.world_model import (
    Anomaly,
    Asset,
    AssetStatus,
    AssetType,
    DockState,
    DockStatus,
    EnvironmentState,
    WorldModel,
    WorldSnapshot,
)
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    Position,
    VehicleState,
    Velocity,
)


@pytest.fixture
def world_model():
    """Create a fresh WorldModel."""
    return WorldModel()


@pytest.fixture
def sample_vehicle_state():
    """Create a sample vehicle state."""
    return VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=100.0,
            altitude_agl=50.0,
        ),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=80.0),
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
    )


@pytest.fixture
def sample_asset():
    """Create a sample asset."""
    return Asset(
        asset_id="test_asset_001",
        name="Test Asset",
        asset_type=AssetType.BUILDING,
        position=Position(
            latitude=37.7750,
            longitude=-122.4195,
            altitude_msl=0,
        ),
        priority=1,
    )


@pytest.fixture
def sample_dock():
    """Create a sample dock position."""
    return Position(
        latitude=37.7749,
        longitude=-122.4194,
        altitude_msl=0,
    )


class TestWorldModelBasics:
    """Test basic WorldModel operations."""

    def test_create_empty_world_model(self, world_model):
        """Test creating empty world model."""
        assert world_model._vehicle is None
        assert world_model._dock is None
        assert len(world_model._assets) == 0
        assert len(world_model._anomalies) == 0

    def test_set_dock(self, world_model, sample_dock):
        """Test setting dock position."""
        world_model.set_dock(sample_dock)
        assert world_model._dock is not None
        assert isinstance(world_model._dock, DockState)
        assert world_model._dock.position == sample_dock
        assert world_model._dock.status == DockStatus.AVAILABLE

    def test_update_vehicle(self, world_model, sample_vehicle_state):
        """Test updating vehicle state."""
        world_model.update_vehicle(sample_vehicle_state)
        assert world_model._vehicle == sample_vehicle_state
        assert world_model._last_update is not None


class TestAssetOperations:
    """Test asset-related operations."""

    def test_add_asset(self, world_model, sample_asset):
        """Test adding an asset."""
        world_model.add_asset(sample_asset)
        assert len(world_model._assets) == 1
        assert world_model._assets[0].asset_id == "test_asset_001"

    def test_add_duplicate_asset_replaces(self, world_model, sample_asset):
        """Test that adding duplicate asset replaces the old one."""
        world_model.add_asset(sample_asset)

        # Modify and add again
        updated_asset = Asset(
            asset_id="test_asset_001",
            name="Updated Asset",
            asset_type=AssetType.POWER_LINE,
            position=sample_asset.position,
            priority=2,
        )
        world_model.add_asset(updated_asset)

        assert len(world_model._assets) == 1
        assert world_model._assets[0].name == "Updated Asset"
        assert world_model._assets[0].priority == 2

    def test_get_asset_by_id(self, world_model, sample_asset):
        """Test accessing an asset by ID from internal list."""
        world_model.add_asset(sample_asset)

        # Find asset from internal list
        found = None
        for asset in world_model._assets:
            if asset.asset_id == "test_asset_001":
                found = asset
                break

        assert found is not None
        assert found.name == "Test Asset"

        # Check for non-existent asset
        not_found = None
        for asset in world_model._assets:
            if asset.asset_id == "nonexistent":
                not_found = asset
                break
        assert not_found is None

    def test_asset_needs_inspection(self, world_model, sample_asset):
        """Test asset needs_inspection property."""
        world_model.add_asset(sample_asset)

        # Asset should need inspection initially (no next_scheduled)
        asset = world_model._assets[0]
        assert asset.needs_inspection is True

        # Record inspection to set next_scheduled in the future
        world_model.record_inspection("test_asset_001", cadence_minutes=60)

        # Now it shouldn't need inspection
        assert asset.needs_inspection is False

    def test_update_asset_status(self, world_model, sample_asset):
        """Test updating asset status."""
        world_model.add_asset(sample_asset)

        result = world_model.update_asset_status("test_asset_001", AssetStatus.WARNING)
        assert result is True
        assert world_model._assets[0].status == AssetStatus.WARNING

        # Non-existent asset
        result = world_model.update_asset_status("nonexistent", AssetStatus.NORMAL)
        assert result is False

    def test_record_inspection(self, world_model, sample_asset):
        """Test recording an inspection."""
        world_model.add_asset(sample_asset)

        world_model.record_inspection("test_asset_001", cadence_minutes=60)

        asset = world_model._assets[0]
        assert asset.last_inspection is not None
        assert asset.next_scheduled is not None
        # Next scheduled should be about 60 minutes from now
        expected = datetime.now() + timedelta(minutes=60)
        assert abs((asset.next_scheduled - expected).total_seconds()) < 5

    def test_record_inspection_nonexistent(self, world_model):
        """Test recording inspection for non-existent asset."""
        # Should not raise, just do nothing
        world_model.record_inspection("nonexistent", cadence_minutes=30)


class TestLoadAssetsFromConfig:
    """Test loading assets from configuration."""

    def test_load_assets_from_config_basic(self, world_model):
        """Test loading assets from basic config."""
        config = {
            "assets": [
                {
                    "id": "building_001",
                    "name": "Office Building",
                    "type": "building",
                    "position": {
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "altitude_m": 10,
                    },
                    "priority": 1,
                },
                {
                    "id": "substation_001",
                    "name": "Power Substation",
                    "type": "substation",
                    "position": {
                        "latitude": 37.7760,
                        "longitude": -122.4200,
                    },
                    "priority": 2,
                },
            ]
        }

        world_model.load_assets_from_config(config)

        assert len(world_model._assets) == 2
        # Find assets by ID
        building = next(a for a in world_model._assets if a.asset_id == "building_001")
        substation = next(a for a in world_model._assets if a.asset_id == "substation_001")
        assert building.name == "Office Building"
        assert substation.asset_type == AssetType.SUBSTATION

    def test_load_assets_from_config_with_inspection(self, world_model):
        """Test loading assets with inspection parameters."""
        config = {
            "assets": [
                {
                    "id": "solar_001",
                    "name": "Solar Farm",
                    "type": "solar_panel",
                    "position": {"latitude": 37.77, "longitude": -122.42},
                    "inspection": {
                        "altitude_agl_m": 30,
                        "orbit_radius_m": 25,
                        "dwell_time_s": 45,
                    },
                }
            ]
        }

        world_model.load_assets_from_config(config)

        asset = next(a for a in world_model._assets if a.asset_id == "solar_001")
        assert asset.inspection_altitude_agl == 30
        assert asset.orbit_radius_m == 25
        assert asset.dwell_time_s == 45

    def test_load_assets_from_config_empty(self, world_model):
        """Test loading empty config."""
        world_model.load_assets_from_config({})
        assert len(world_model._assets) == 0

        world_model.load_assets_from_config({"assets": []})
        assert len(world_model._assets) == 0

    def test_load_assets_from_config_defaults(self, world_model):
        """Test loading assets with minimal config uses defaults."""
        config = {
            "assets": [
                {
                    "id": "minimal_001",
                    "position": {},
                }
            ]
        }

        world_model.load_assets_from_config(config)

        asset = next(a for a in world_model._assets if a.asset_id == "minimal_001")
        assert asset.name == "minimal_001"  # Defaults to ID
        assert asset.asset_type == AssetType.OTHER
        assert asset.priority == 1


class TestAnomalyOperations:
    """Test anomaly-related operations."""

    def test_add_anomaly(self, world_model, sample_asset):
        """Test adding an anomaly."""
        world_model.add_asset(sample_asset)

        anomaly = Anomaly(
            anomaly_id="anom_001",
            asset_id="test_asset_001",
            severity=0.8,
            detected_at=datetime.now(),
            description="Crack detected",
        )
        world_model.add_anomaly(anomaly)

        assert len(world_model._anomalies) == 1
        # Asset status should be updated
        asset = world_model._assets[0]
        assert asset.status == AssetStatus.ANOMALY

    def test_resolve_anomaly(self, world_model, sample_asset):
        """Test resolving an anomaly."""
        world_model.add_asset(sample_asset)

        anomaly = Anomaly(
            anomaly_id="anom_001",
            asset_id="test_asset_001",
            severity=0.5,
            detected_at=datetime.now(),
            description="Minor issue",
        )
        world_model.add_anomaly(anomaly)
        assert not world_model._anomalies[0].resolved

        world_model.resolve_anomaly("anom_001")
        assert world_model._anomalies[0].resolved

    def test_resolve_nonexistent_anomaly(self, world_model):
        """Test resolving non-existent anomaly does nothing."""
        world_model.resolve_anomaly("nonexistent")
        # Should not raise

    def test_get_anomaly_assets(self, world_model, sample_asset):
        """Test getting assets with anomalies."""
        world_model.add_asset(sample_asset)

        # No anomalies initially
        assert world_model.get_anomaly_assets() == []

        # Add anomaly
        anomaly = Anomaly(
            anomaly_id="anom_001",
            asset_id="test_asset_001",
            severity=0.7,
            detected_at=datetime.now(),
            description="Issue",
        )
        world_model.add_anomaly(anomaly)

        assert "test_asset_001" in world_model.get_anomaly_assets()

        # Resolve anomaly
        world_model.resolve_anomaly("anom_001")
        assert world_model.get_anomaly_assets() == []


class TestEnvironmentAndMission:
    """Test environment and mission operations."""

    def test_update_environment(self, world_model):
        """Test updating environment state."""
        env = EnvironmentState(
            timestamp=datetime.now(),
            wind_speed_ms=5.0,
            wind_direction_deg=180.0,
            temperature_c=25.0,
            visibility_m=5000.0,
        )
        world_model.update_environment(env)

        assert world_model._environment is not None
        assert world_model._environment.wind_speed_ms == 5.0

    def test_start_mission(self, world_model, sample_asset):
        """Test starting a mission."""
        world_model.add_asset(sample_asset)

        world_model.start_mission("mission_001", "Test Mission")

        assert world_model._mission is not None
        assert world_model._mission.mission_id == "mission_001"
        assert world_model._mission.mission_name == "Test Mission"
        assert world_model._mission.is_active is True
        assert world_model._mission.assets_total == 1


class TestWorldSnapshot:
    """Test WorldSnapshot creation."""

    def test_get_snapshot_without_vehicle(self, world_model, sample_dock):
        """Test getting snapshot without vehicle returns None."""
        world_model.set_dock(sample_dock)
        assert world_model.get_snapshot() is None

    def test_get_snapshot_without_dock(self, world_model, sample_vehicle_state):
        """Test getting snapshot without dock returns None."""
        world_model.update_vehicle(sample_vehicle_state)
        assert world_model.get_snapshot() is None

    def test_get_snapshot_complete(
        self, world_model, sample_vehicle_state, sample_dock, sample_asset
    ):
        """Test getting complete snapshot."""
        world_model.set_dock(sample_dock)
        world_model.update_vehicle(sample_vehicle_state)
        world_model.add_asset(sample_asset)

        snapshot = world_model.get_snapshot()

        assert snapshot is not None
        assert isinstance(snapshot, WorldSnapshot)
        assert snapshot.vehicle == sample_vehicle_state
        assert isinstance(snapshot.dock, DockState)
        assert snapshot.dock.position == sample_dock
        assert len(snapshot.assets) == 1

    def test_get_snapshot_excludes_resolved_anomalies(
        self, world_model, sample_vehicle_state, sample_dock, sample_asset
    ):
        """Test that snapshot excludes resolved anomalies."""
        world_model.set_dock(sample_dock)
        world_model.update_vehicle(sample_vehicle_state)
        world_model.add_asset(sample_asset)

        # Add two anomalies
        anomaly1 = Anomaly(
            anomaly_id="anom_001",
            asset_id="test_asset_001",
            severity=0.5,
            detected_at=datetime.now(),
            description="Issue 1",
        )
        anomaly2 = Anomaly(
            anomaly_id="anom_002",
            asset_id="test_asset_001",
            severity=0.7,
            detected_at=datetime.now(),
            description="Issue 2",
        )
        world_model.add_anomaly(anomaly1)
        world_model.add_anomaly(anomaly2)

        # Resolve one
        world_model.resolve_anomaly("anom_001")

        snapshot = world_model.get_snapshot()
        assert len(snapshot.anomalies) == 1
        assert snapshot.anomalies[0].anomaly_id == "anom_002"


class TestTimeSinceUpdate:
    """Test time since update functionality."""

    def test_time_since_update_none(self, world_model):
        """Test time since update when never updated."""
        assert world_model.time_since_update() is None

    def test_time_since_update_recent(self, world_model, sample_vehicle_state):
        """Test time since recent update."""
        world_model.update_vehicle(sample_vehicle_state)

        elapsed = world_model.time_since_update()
        assert elapsed is not None
        assert elapsed.total_seconds() < 1  # Should be very recent


class TestGetAnomalyAssetsDuplicate:
    """Test the get_anomaly_assets method."""

    def test_get_anomaly_assets_method(self, world_model, sample_asset):
        """Test get_anomaly_assets returns correct asset IDs."""
        world_model.add_asset(sample_asset)

        # Initially empty (no anomalies)
        result = world_model.get_anomaly_assets()
        assert result == []

        # Add an anomaly for the asset
        anomaly = Anomaly(
            anomaly_id="test_anom",
            asset_id="test_asset_001",
            severity=0.5,
            detected_at=datetime.now(),
            description="Test anomaly",
        )
        world_model.add_anomaly(anomaly)

        # Now should return the asset ID
        result = world_model.get_anomaly_assets()
        assert "test_asset_001" in result

        # Resolve the anomaly
        world_model.resolve_anomaly("test_anom")
        result = world_model.get_anomaly_assets()
        assert result == []
