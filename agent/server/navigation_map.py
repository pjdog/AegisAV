"""Navigation map helpers for obstacle avoidance."""

from __future__ import annotations

from datetime import datetime
from typing import Any

DEFAULT_AVOIDANCE_RULES = {
    "building": {"radius_m": 22.0, "height_m": 12.0},
    "house": {"radius_m": 18.0, "height_m": 10.0},
    "warehouse": {"radius_m": 26.0, "height_m": 14.0},
    "substation": {"radius_m": 20.0, "height_m": 8.0},
    "power_line": {"radius_m": 12.0, "height_m": 6.0},
}


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
) -> dict[str, Any]:
    """Build a simple navigation map for obstacle avoidance."""
    obstacles: list[dict[str, Any]] = []
    for asset in assets:
        asset_type = _normalize_asset_type(_get_asset_field(asset, "asset_type", ""))
        if asset_type not in DEFAULT_AVOIDANCE_RULES:
            continue
        rules = DEFAULT_AVOIDANCE_RULES[asset_type]
        obstacles.append({
            "asset_id": _get_asset_field(asset, "asset_id", "unknown"),
            "name": _get_asset_field(asset, "name", asset_type),
            "asset_type": asset_type,
            "latitude": float(_get_asset_field(asset, "latitude", 0.0)),
            "longitude": float(_get_asset_field(asset, "longitude", 0.0)),
            "radius_m": float(rules["radius_m"]),
            "height_m": float(rules["height_m"]),
            "source": source,
        })

    return {
        "scenario_id": scenario_id,
        "generated_at": datetime.now().isoformat(),
        "source": source,
        "obstacles": obstacles,
    }
