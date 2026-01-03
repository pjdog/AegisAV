"""Splat-based change detection stub.

Compares recent depth obstacles against a splat-derived navigation map
to flag potential scene changes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ChangeDetectionConfig:
    """Configuration for change detection heuristics."""

    match_distance_m: float = 12.0
    min_new_obstacles: int = 1
    max_changes: int = 5


@dataclass
class SplatChangeConfig:
    """Configuration for splat-based change detection."""

    enabled: bool = False
    min_age_s: float = 600.0
    max_age_s: float = 3600.0


@dataclass
class SplatChangeResult:
    """Result from a splat change detection pass."""

    available: bool
    change_score: float
    confidence: float
    reason: str
    scene_path: str | None = None
    image_path: str | None = None
    age_s: float | None = None
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-safe dictionary."""
        return {
            "available": self.available,
            "change_score": self.change_score,
            "confidence": self.confidence,
            "reason": self.reason,
            "scene_path": self.scene_path,
            "image_path": self.image_path,
            "age_s": self.age_s,
            "evaluated_at": self.evaluated_at,
            "details": self.details,
        }


def _geo_to_ned(
    lat: float, lon: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    """Convert lat/lon to NED (North-East-Down) coordinates relative to origin."""
    lat_diff = lat - origin_lat
    lon_diff = lon - origin_lon
    x_ned = lat_diff * 111111.0
    y_ned = lon_diff * 111111.0 * math.cos(math.radians(origin_lat))
    return x_ned, y_ned


def _extract_xy(
    obstacle: dict[str, Any],
    *,
    origin_lat: float | None,
    origin_lon: float | None,
) -> tuple[float, float] | None:
    if "x_ned" in obstacle and "y_ned" in obstacle:
        return float(obstacle["x_ned"]), float(obstacle["y_ned"])
    if "x" in obstacle and "y" in obstacle:
        return float(obstacle["x"]), float(obstacle["y"])

    lat = obstacle.get("latitude")
    lon = obstacle.get("longitude")
    if lat is None or lon is None or origin_lat is None or origin_lon is None:
        return None

    return _geo_to_ned(float(lat), float(lon), origin_lat, origin_lon)


def detect_splat_changes(
    nav_map: dict[str, Any],
    depth_capture: dict[str, Any] | None,
    config: ChangeDetectionConfig | None = None,
) -> list[dict[str, Any]]:
    """Detect new obstacles compared to a splat-derived map."""
    if not depth_capture:
        return []

    cfg = config or ChangeDetectionConfig()
    depth_obstacles = depth_capture.get("obstacles") or []
    if not depth_obstacles:
        return []

    metadata = nav_map.get("metadata", {})
    origin_lat = metadata.get("origin_latitude")
    origin_lon = metadata.get("origin_longitude")

    reference_points: list[tuple[float, float, dict[str, Any]]] = []
    for obstacle in nav_map.get("obstacles", []):
        xy = _extract_xy(obstacle, origin_lat=origin_lat, origin_lon=origin_lon)
        if xy:
            reference_points.append((xy[0], xy[1], obstacle))

    changes: list[dict[str, Any]] = []
    for obstacle in depth_obstacles:
        xy = _extract_xy(obstacle, origin_lat=origin_lat, origin_lon=origin_lon)
        if xy is None:
            continue

        nearest_distance = None
        nearest_id = None
        for ref_x, ref_y, ref_obs in reference_points:
            distance = math.hypot(xy[0] - ref_x, xy[1] - ref_y)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_id = ref_obs.get("obstacle_id") or ref_obs.get("asset_id")

        if nearest_distance is None or nearest_distance > cfg.match_distance_m:
            changes.append({
                "source": "airsim_depth",
                "x_ned": xy[0],
                "y_ned": xy[1],
                "latitude": obstacle.get("latitude"),
                "longitude": obstacle.get("longitude"),
                "radius_m": obstacle.get("radius_m"),
                "height_m": obstacle.get("height_m"),
                "distance_to_reference_m": nearest_distance,
                "nearest_obstacle_id": nearest_id,
            })
        if len(changes) >= cfg.max_changes:
            break

    if len(changes) < cfg.min_new_obstacles:
        return []

    return changes


def detect_splat_change(
    image_path: Path,
    splat_scene_path: Path,
    config: SplatChangeConfig,
) -> SplatChangeResult:
    """Detect scene change using splat artifacts.

    This is a stub that uses splat artifact age as a proxy for change risk.
    """
    if not config.enabled:
        return SplatChangeResult(
            available=False,
            change_score=0.0,
            confidence=0.0,
            reason="disabled",
        )

    if not image_path.exists() or not splat_scene_path.exists():
        return SplatChangeResult(
            available=False,
            change_score=0.0,
            confidence=0.0,
            reason="missing_inputs",
            scene_path=str(splat_scene_path) if splat_scene_path else None,
            image_path=str(image_path) if image_path else None,
        )

    age_s = max(0.0, time.time() - splat_scene_path.stat().st_mtime)
    min_age = max(0.0, config.min_age_s)
    max_age = max(min_age + 1.0, config.max_age_s)

    if age_s <= min_age:
        change_score = 0.0
    else:
        change_score = min(1.0, (age_s - min_age) / (max_age - min_age))

    confidence = min(0.85, 0.4 + (change_score * 0.4))

    return SplatChangeResult(
        available=True,
        change_score=change_score,
        confidence=confidence,
        reason="age_based_stub",
        scene_path=str(splat_scene_path),
        image_path=str(image_path),
        age_s=age_s,
        details={
            "min_age_s": min_age,
            "max_age_s": max_age,
        },
    )
