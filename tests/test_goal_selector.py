"""
Tests for goal selector logic.
"""

from datetime import datetime, timedelta

import pytest

from agent.server.goal_selector import GoalSelector, GoalType
from agent.server.world_model import (
    Asset,
    AssetType,
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


def create_test_vehicle(
    battery_percent: float = 80.0,
    armed: bool = True,
    healthy: bool = True,
    has_gps: bool = True,
) -> VehicleState:
    """Create a test vehicle state."""
    return VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=47.398000, longitude=8.546000, altitude_msl=500.0, altitude_agl=12.0
        ),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(voltage=22.8, current=5.0, remaining_percent=battery_percent),
        mode=FlightMode.GUIDED,
        armed=armed,
        in_air=armed,
        gps=GPSState(
            fix_type=3 if has_gps else 0,
            satellites_visible=12 if has_gps else 0,
            hdop=0.9,
            vdop=0.9,
        ),
        health=VehicleHealth(
            sensors_healthy=healthy,
            gps_healthy=healthy,
            battery_healthy=healthy,
            motors_healthy=healthy,
            ekf_healthy=healthy,
        ),
        home_position=Position(latitude=47.397742, longitude=8.545594, altitude_msl=488.0),
    )


def create_test_snapshot(
    vehicle: VehicleState,
    assets: list[Asset] | None = None,
    environment: EnvironmentState | None = None,
) -> WorldSnapshot:
    """Create a test world snapshot."""
    if assets is None:
        assets = [
            Asset(
                asset_id="test-001",
                name="Test Asset",
                asset_type=AssetType.SOLAR_PANEL,
                position=Position(latitude=47.398500, longitude=8.546500, altitude_msl=495.0),
                priority=1,
            )
        ]

    if environment is None:
        environment = EnvironmentState(timestamp=datetime.now())

    return WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=vehicle,
        assets=assets,
        anomalies=[],
        dock=DockState(
            position=Position(latitude=47.397742, longitude=8.545594, altitude_msl=488.0),
            status=DockStatus.AVAILABLE,
        ),
        environment=environment,
        mission=MissionState(
            mission_id="test",
            mission_name="Test Mission",
            is_active=True,
            assets_total=len(assets),
        ),
    )


class TestGoalSelector:
    """Tests for GoalSelector."""

    @pytest.mark.asyncio
    async def test_select_wait_when_no_assets_pending(self):
        """Should select WAIT when no assets need inspection."""
        selector = GoalSelector()
        vehicle = create_test_vehicle()

        # Create asset that was recently inspected
        asset = Asset(
            asset_id="test-001",
            name="Test Asset",
            asset_type=AssetType.SOLAR_PANEL,
            position=Position(latitude=47.398500, longitude=8.546500, altitude_msl=495.0),
            last_inspection=datetime.now(),
            next_scheduled=datetime.now() + timedelta(hours=1),
        )

        snapshot = create_test_snapshot(vehicle, assets=[asset])
        goal = await selector.select_goal(snapshot)

        assert goal.goal_type == GoalType.WAIT

    @pytest.mark.asyncio
    async def test_select_inspect_when_asset_pending(self):
        """Should select INSPECT_ASSET when asset needs inspection."""
        selector = GoalSelector()
        vehicle = create_test_vehicle()

        # Create asset that needs inspection
        asset = Asset(
            asset_id="test-001",
            name="Test Asset",
            asset_type=AssetType.SOLAR_PANEL,
            position=Position(latitude=47.398500, longitude=8.546500, altitude_msl=495.0),
            # No last_inspection, so needs inspection
        )

        snapshot = create_test_snapshot(vehicle, assets=[asset])
        goal = await selector.select_goal(snapshot)

        assert goal.goal_type == GoalType.INSPECT_ASSET
        assert goal.target_asset.asset_id == "test-001"

    @pytest.mark.asyncio
    async def test_select_return_on_low_battery(self):
        """Should select RETURN when battery is low."""
        selector = GoalSelector(battery_return_threshold=30.0)
        vehicle = create_test_vehicle(battery_percent=25.0)

        snapshot = create_test_snapshot(vehicle)
        goal = await selector.select_goal(snapshot)

        assert goal.goal_type == GoalType.RETURN_LOW_BATTERY

    @pytest.mark.asyncio
    async def test_select_abort_on_critical_battery(self):
        """Should select ABORT when battery is critical."""
        selector = GoalSelector()
        vehicle = create_test_vehicle(battery_percent=10.0)

        snapshot = create_test_snapshot(vehicle)
        goal = await selector.select_goal(snapshot)

        assert goal.goal_type == GoalType.ABORT

    @pytest.mark.asyncio
    async def test_select_abort_on_gps_loss(self):
        """Should select ABORT when GPS is lost."""
        selector = GoalSelector()
        vehicle = create_test_vehicle(has_gps=False)

        snapshot = create_test_snapshot(vehicle)
        goal = await selector.select_goal(snapshot)

        assert goal.goal_type == GoalType.ABORT

    @pytest.mark.asyncio
    async def test_select_return_on_bad_weather(self):
        """Should select RETURN when weather is unflyable."""
        selector = GoalSelector()
        vehicle = create_test_vehicle()

        bad_weather = EnvironmentState(
            timestamp=datetime.now(),
            wind_speed_ms=15.0,  # High wind
        )

        snapshot = create_test_snapshot(vehicle, environment=bad_weather)
        goal = await selector.select_goal(snapshot)

        assert goal.goal_type == GoalType.RETURN_WEATHER

    @pytest.mark.asyncio
    async def test_priority_order(self):
        """Should respect priority order (battery before inspection)."""
        selector = GoalSelector(battery_return_threshold=30.0)
        vehicle = create_test_vehicle(battery_percent=25.0)

        # Asset needs inspection but battery is low
        asset = Asset(
            asset_id="test-001",
            name="Test Asset",
            asset_type=AssetType.SOLAR_PANEL,
            position=Position(latitude=47.398500, longitude=8.546500, altitude_msl=495.0),
        )

        snapshot = create_test_snapshot(vehicle, assets=[asset])
        goal = await selector.select_goal(snapshot)

        # Should return due to battery, not inspect
        assert goal.goal_type == GoalType.RETURN_LOW_BATTERY


class TestAsset:
    """Tests for Asset model."""

    def test_needs_inspection_when_never_inspected(self):
        """Asset should need inspection if never inspected."""
        asset = Asset(
            asset_id="test",
            name="Test",
            asset_type=AssetType.OTHER,
            position=Position(latitude=0, longitude=0, altitude_msl=0),
        )

        assert asset.needs_inspection is True

    def test_needs_inspection_when_scheduled(self):
        """Asset should need inspection if past scheduled time."""
        asset = Asset(
            asset_id="test",
            name="Test",
            asset_type=AssetType.OTHER,
            position=Position(latitude=0, longitude=0, altitude_msl=0),
            last_inspection=datetime.now() - timedelta(hours=2),
            next_scheduled=datetime.now() - timedelta(hours=1),
        )

        assert asset.needs_inspection is True

    def test_no_inspection_needed_when_recent(self):
        """Asset should not need inspection if not yet scheduled."""
        asset = Asset(
            asset_id="test",
            name="Test",
            asset_type=AssetType.OTHER,
            position=Position(latitude=0, longitude=0, altitude_msl=0),
            last_inspection=datetime.now(),
            next_scheduled=datetime.now() + timedelta(hours=1),
        )

        assert asset.needs_inspection is False
