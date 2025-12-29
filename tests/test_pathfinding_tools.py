from datetime import datetime

import pytest

from agent.server.advanced_decision import (
    _build_neighbor_graph,
    _collect_nodes,
    _dijkstra_path,
    _markov_rank_assets,
    _neural_rank_assets,
)
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


@pytest.fixture
def sample_world() -> WorldSnapshot:
    vehicle = VehicleState(
        timestamp=datetime.now(),
        position=Position(latitude=45.0, longitude=-75.0, altitude_msl=100.0),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(voltage=24.0, current=1.0, remaining_percent=80.0),
        mode=FlightMode.GUIDED,
        armed=True,
        health=VehicleHealth(
            sensors_healthy=True,
            gps_healthy=True,
            battery_healthy=True,
            motors_healthy=True,
            ekf_healthy=True,
        ),
        gps=GPSState(fix_type=3, satellites_visible=10, hdop=1.0),
    )

    assets = [
        Asset(
            asset_id="asset_a",
            name="Asset A",
            asset_type=AssetType.SOLAR_PANEL,
            position=Position(latitude=45.001, longitude=-75.0, altitude_msl=100.0),
            priority=1,
        ),
        Asset(
            asset_id="asset_b",
            name="Asset B",
            asset_type=AssetType.SOLAR_PANEL,
            position=Position(latitude=45.002, longitude=-75.001, altitude_msl=100.0),
            priority=2,
        ),
        Asset(
            asset_id="asset_c",
            name="Asset C",
            asset_type=AssetType.SOLAR_PANEL,
            position=Position(latitude=45.003, longitude=-75.002, altitude_msl=100.0),
            priority=3,
        ),
    ]

    return WorldSnapshot(
        timestamp=datetime.now(),
        vehicle=vehicle,
        assets=assets,
        anomalies=[],
        dock=DockState(
            position=Position(latitude=45.0, longitude=-75.0, altitude_msl=0.0),
            status=DockStatus.AVAILABLE,
        ),
        environment=EnvironmentState(timestamp=datetime.now()),
        mission=MissionState(mission_id="test", mission_name="Test Mission"),
    )


def test_dijkstra_path_basic(sample_world: WorldSnapshot) -> None:
    nodes = _collect_nodes(sample_world, include_vehicle=True, include_dock=True)
    graph = _build_neighbor_graph(nodes, neighbor_k=3)
    path, distance_m = _dijkstra_path(graph, "vehicle", "asset_b")

    assert path[0] == "vehicle"
    assert path[-1] == "asset_b"
    assert distance_m > 0


def test_markov_rank_assets_sorted(sample_world: WorldSnapshot) -> None:
    ranked = _markov_rank_assets(sample_world, current_id="vehicle")

    assert ranked
    assert ranked[0]["probability"] >= ranked[-1]["probability"]


def test_neural_rank_assets_sorted(sample_world: WorldSnapshot) -> None:
    ranked = _neural_rank_assets(sample_world, current_id="vehicle")

    assert ranked
    assert ranked[0]["score"] >= ranked[-1]["score"]
