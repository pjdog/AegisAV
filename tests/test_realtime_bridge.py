"""Tests for AirSim realtime bridge helpers."""

from simulation import realtime_bridge


def test_select_vehicle_name_prefers_requested() -> None:
    selected, note = realtime_bridge._select_vehicle_name("Drone1", ["Drone1", "SimpleFlight"])
    assert selected == "Drone1"
    assert note is None


def test_select_vehicle_name_falls_back_to_first() -> None:
    selected, note = realtime_bridge._select_vehicle_name("Drone1", ["SimpleFlight"])
    assert selected == "SimpleFlight"
    assert "Requested vehicle" in (note or "")


def test_select_vehicle_name_empty_list() -> None:
    selected, note = realtime_bridge._select_vehicle_name("Drone1", [])
    assert selected == "Drone1"
    assert note is None
