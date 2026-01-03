"""Navigation map helpers for obstacle avoidance."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

DEFAULT_AVOIDANCE_RULES = {
    "building": {"radius_m": 22.0, "height_m": 12.0},
    "house": {"radius_m": 18.0, "height_m": 10.0},
    "warehouse": {"radius_m": 26.0, "height_m": 14.0},
    "solar_panel": {"radius_m": 6.0, "height_m": 2.0},
    "substation": {"radius_m": 20.0, "height_m": 8.0},
    "power_line": {"radius_m": 12.0, "height_m": 6.0},
    "wind_turbine": {"radius_m": 14.0, "height_m": 80.0},
}


def _geo_to_ned(
    lat: float, lon: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    """Convert lat/lon to NED (North-East-Down) coordinates relative to origin.

    Returns (x_ned, y_ned) where x is North and y is East in meters.
    """
    # 1 deg lat ~ 111,111 meters
    # 1 deg lon ~ 111,111 * cos(lat) meters
    lat_diff = lat - origin_lat
    lon_diff = lon - origin_lon
    x_ned = lat_diff * 111111.0  # North
    y_ned = lon_diff * 111111.0 * math.cos(math.radians(origin_lat))  # East
    return x_ned, y_ned


def _get_asset_field(asset: Any, field: str, default: Any = None) -> Any:
    if isinstance(asset, dict):
        return asset.get(field, default)
    return getattr(asset, field, default)


def _normalize_asset_type(asset_type: str | None) -> str:
    if not asset_type:
        return ""
    return str(asset_type).strip().lower()


def build_navigation_map(
    assets: list[Any],
    scenario_id: str,
    source: str = "scenario_assets",
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> dict[str, Any]:
    """Build a simple navigation map for obstacle avoidance.

    Args:
        assets: List of scenario assets to convert to obstacles.
        scenario_id: ID of the scenario.
        source: Source of the obstacles (e.g., 'scenario_assets', 'slam').
        origin_lat: Origin latitude for NED conversion. If None, uses first asset or dock default.
        origin_lon: Origin longitude for NED conversion. If None, uses first asset or dock default.

    Returns:
        Navigation map with obstacles in both lat/lon and NED coordinates.
    """
    # Default dock coordinates (from scenarios.py)
    DEFAULT_DOCK_LAT = 37.7749
    DEFAULT_DOCK_LON = -122.4194

    obstacles: list[dict[str, Any]] = []

    # Determine origin for NED conversion
    if origin_lat is None or origin_lon is None:
        # Use dock as origin for NED coordinates
        origin_lat = DEFAULT_DOCK_LAT
        origin_lon = DEFAULT_DOCK_LON

    # Track bounds for metadata
    min_x, max_x, min_y, max_y = float("inf"), float("-inf"), float("inf"), float("-inf")

    for asset in assets:
        asset_type = _normalize_asset_type(_get_asset_field(asset, "asset_type", ""))
        if asset_type not in DEFAULT_AVOIDANCE_RULES:
            continue
        rules = DEFAULT_AVOIDANCE_RULES[asset_type]

        lat = float(_get_asset_field(asset, "latitude", 0.0))
        lon = float(_get_asset_field(asset, "longitude", 0.0))

        # Convert to NED coordinates
        x_ned, y_ned = _geo_to_ned(lat, lon, origin_lat, origin_lon)

        # Update bounds
        min_x = min(min_x, x_ned)
        max_x = max(max_x, x_ned)
        min_y = min(min_y, y_ned)
        max_y = max(max_y, y_ned)

        obstacles.append({
            "obstacle_id": _get_asset_field(asset, "asset_id", "unknown"),
            "asset_id": _get_asset_field(asset, "asset_id", "unknown"),
            "name": _get_asset_field(asset, "name", asset_type),
            "obstacle_type": asset_type,
            "asset_type": asset_type,
            "latitude": lat,
            "longitude": lon,
            "x_ned": x_ned,
            "y_ned": y_ned,
            "z_ned": 0.0,  # Ground level
            "x": x_ned,  # Alias for overlay compatibility
            "y": y_ned,  # Alias for overlay compatibility
            "radius_m": float(rules["radius_m"]),
            "height_m": float(rules["height_m"]),
            "confidence": 1.0,
            "source": source,
            "detected_at": datetime.now().isoformat(),
        })

    # Handle empty obstacle case for bounds
    if not obstacles:
        min_x, max_x, min_y, max_y = 0.0, 0.0, 0.0, 0.0

    return {
        "scenario_id": scenario_id,
        "generated_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "source": source,
        "obstacles": obstacles,
        "metadata": {
            "map_id": f"nav_{scenario_id}",
            "version": 1,
            "coordinate_frame": "NED",
            "origin_latitude": origin_lat,
            "origin_longitude": origin_lon,
            "bounds_min_x": min_x,
            "bounds_max_x": max_x,
            "bounds_min_y": min_y,
            "bounds_max_y": max_y,
            "bounds_min_z": 0.0,
            "bounds_max_z": 100.0,  # Default max height
            "resolution_m": 1.0,
            "obstacle_count": len(obstacles),
            "map_quality_score": 1.0,
            "coverage_percent": 100.0,
        },
    }
