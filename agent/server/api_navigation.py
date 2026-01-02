"""Navigation map and SLAM API routes.

Phase 0 Worker B: API contracts for map outputs and updates.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.server.config_manager import get_config_manager
from agent.server.state import server_state
from mapping.decision_context import MapContext
from mapping.map_storage import MapArtifactStore, MapArtifactStoreConfig
from mapping.real_capture import RealCaptureConfig, RealSensorCapture
from mapping.safety_gates import MapUpdateGate, SafetyGateConfig

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Schemas (JSON contracts)
# -----------------------------------------------------------------------------


class ObstacleEntry(BaseModel):
    """Schema for a single obstacle in the navigation map."""

    obstacle_id: str = Field(..., description="Unique obstacle identifier")
    asset_id: str | None = Field(None, description="Associated asset ID if from scenario")
    name: str = Field("", description="Human-readable name")
    obstacle_type: str = Field("unknown", description="Type: building, tree, vehicle, dynamic, etc.")

    # Position (lat/lon for geo, or local NED)
    latitude: float | None = Field(None, description="WGS84 latitude")
    longitude: float | None = Field(None, description="WGS84 longitude")
    x_ned: float | None = Field(None, description="North position in NED frame (meters)")
    y_ned: float | None = Field(None, description="East position in NED frame (meters)")
    z_ned: float | None = Field(None, description="Down position in NED frame (meters)")

    # Geometry
    radius_m: float = Field(5.0, description="Avoidance radius in meters")
    height_m: float = Field(10.0, description="Obstacle height in meters")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")

    # Source tracking
    source: str = Field("unknown", description="Source: scenario, slam, splat, depth, manual")
    detected_at: str | None = Field(None, description="ISO timestamp of detection")
    last_updated: str | None = Field(None, description="ISO timestamp of last update")


class MapTile(BaseModel):
    """Schema for a 2D occupancy grid tile."""

    tile_id: str = Field(..., description="Tile identifier (e.g., 'tile_0_0')")
    x_index: int = Field(..., description="Tile X index in grid")
    y_index: int = Field(..., description="Tile Y index in grid")
    resolution_m: float = Field(1.0, description="Meters per cell")
    width_cells: int = Field(100, description="Tile width in cells")
    height_cells: int = Field(100, description="Tile height in cells")

    # Origin in world coordinates
    origin_x_ned: float = Field(0.0, description="North origin of tile")
    origin_y_ned: float = Field(0.0, description="East origin of tile")

    # Occupancy data (flattened row-major, values 0-255)
    occupancy: list[int] = Field(default_factory=list, description="Occupancy values 0-255")
    timestamp: str = Field(..., description="ISO timestamp of tile generation")


class VoxelCell(BaseModel):
    """Schema for a single voxel in 3D map."""

    x: int = Field(..., description="Voxel X index")
    y: int = Field(..., description="Voxel Y index")
    z: int = Field(..., description="Voxel Z index")
    occupancy: float = Field(0.0, ge=0.0, le=1.0, description="Occupancy probability")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Observation confidence")
    color_rgb: list[int] | None = Field(None, description="RGB color if available")


class MapMetadata(BaseModel):
    """Schema for navigation map metadata."""

    map_id: str = Field(..., description="Unique map identifier")
    version: int = Field(1, description="Map version number")
    scenario_id: str | None = Field(None, description="Associated scenario ID")
    generated_at: str = Field(..., description="ISO timestamp of generation")
    last_updated: str = Field(..., description="ISO timestamp of last update")

    # Coordinate frame info
    coordinate_frame: str = Field("NED", description="Frame: NED, ENU, or GPS")
    origin_latitude: float | None = Field(None, description="Geo origin latitude")
    origin_longitude: float | None = Field(None, description="Geo origin longitude")
    origin_altitude_m: float | None = Field(None, description="Geo origin altitude MSL")

    # Bounds
    bounds_min_x: float = Field(0.0, description="Min X (North) in meters")
    bounds_max_x: float = Field(0.0, description="Max X (North) in meters")
    bounds_min_y: float = Field(0.0, description="Min Y (East) in meters")
    bounds_max_y: float = Field(0.0, description="Max Y (East) in meters")
    bounds_min_z: float = Field(0.0, description="Min Z (Down) in meters")
    bounds_max_z: float = Field(0.0, description="Max Z (Down) in meters")

    # Resolution
    resolution_m: float = Field(1.0, description="Grid resolution in meters")
    voxel_size_m: float | None = Field(None, description="3D voxel size if applicable")

    # Stats
    obstacle_count: int = Field(0, description="Number of obstacles")
    tile_count: int = Field(0, description="Number of 2D tiles")
    voxel_count: int = Field(0, description="Number of occupied voxels")

    # Quality metrics
    map_quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall quality score")
    coverage_percent: float = Field(0.0, ge=0.0, le=100.0, description="Area coverage percentage")


class NavigationMapResponse(BaseModel):
    """Full navigation map response."""

    metadata: MapMetadata
    obstacles: list[ObstacleEntry] = Field(default_factory=list)
    tiles: list[MapTile] | None = Field(None, description="2D occupancy tiles if available")
    voxels: list[VoxelCell] | None = Field(None, description="3D voxels if available")


class SLAMStatus(BaseModel):
    """SLAM pipeline status."""

    enabled: bool = Field(False, description="Whether SLAM is enabled")
    running: bool = Field(False, description="Whether SLAM is actively processing")
    backend: str = Field("none", description="SLAM backend: orb_slam3, vins_fusion, none")

    # Tracking state
    tracking_state: str = Field("not_initialized", description="Tracking state")
    keyframe_count: int = Field(0, description="Number of keyframes")
    map_point_count: int = Field(0, description="Number of sparse map points")
    loop_closure_count: int = Field(0, description="Number of loop closures")
    loop_closure_rate: float = Field(0.0, description="Loop closure rate per keyframe")

    # Quality metrics
    pose_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Current pose confidence")
    reprojection_error: float = Field(0.0, description="Average reprojection error in pixels")
    drift_estimate_m: float = Field(0.0, description="Estimated drift in meters")

    # Timing
    last_frame_ms: float = Field(0.0, description="Last frame processing time in ms")
    avg_frame_ms: float = Field(0.0, description="Average frame processing time")
    last_update: str | None = Field(None, description="ISO timestamp of last update")


class MapStatusResponse(BaseModel):
    """Combined map status response."""

    map_available: bool = Field(False, description="Whether a valid map exists")
    map_age_s: float = Field(0.0, description="Seconds since last map update")
    map_version: int = Field(0, description="Current map version")
    obstacle_count: int = Field(0, description="Number of obstacles in map")
    map_quality_score: float = Field(0.0, description="Overall quality score")
    slam_status: SLAMStatus | None = Field(None, description="SLAM pipeline status")
    splat_available: bool = Field(False, description="Whether splat reconstruction exists")
    last_update: str | None = Field(None, description="ISO timestamp of last update")
    map_update_error: str | None = Field(None, description="Last map update error message")
    map_update_error_at: str | None = Field(None, description="Timestamp of last map update error")


# -----------------------------------------------------------------------------
# Server State Integration Points
# -----------------------------------------------------------------------------

# server_state.navigation_map is already defined in state.py (line 118)
# Structure expected:
# {
#     "scenario_id": str,
#     "generated_at": str (ISO),
#     "source": str,
#     "obstacles": list[dict],
#     "metadata": dict (optional),
#     "slam": dict (optional),
#     "splat": dict (optional),
# }

# Additional state fields to be added in future phases:
# server_state.slam_status: dict | None  # SLAM pipeline state
# server_state.splat_artifacts: dict | None  # Gaussian splat data


def _get_navigation_map() -> dict[str, Any] | None:
    """Get current navigation map from server state."""
    return server_state.navigation_map


def _get_map_age_seconds() -> float:
    """Calculate age of current map in seconds."""
    nav_map = _get_navigation_map()
    if not nav_map:
        return float("inf")

    generated_at = nav_map.get("generated_at") or nav_map.get("last_updated")
    if not generated_at:
        return float("inf")

    try:
        gen_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        age = (datetime.now(gen_time.tzinfo) - gen_time).total_seconds()
        return max(0.0, age)
    except Exception:
        return float("inf")


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------


async def get_navigation_map_latest() -> dict:
    """GET /api/navigation/map/latest - Get the current navigation map.

    Returns the full navigation map including obstacles, tiles, and metadata.
    """
    nav_map = _get_navigation_map()

    if not nav_map:
        return {
            "status": "no_map",
            "message": "No navigation map available",
            "metadata": None,
            "obstacles": [],
        }

    # Build metadata
    obstacles = nav_map.get("obstacles", [])
    metadata = nav_map.get("metadata", {})

    response_metadata = {
        "map_id": metadata.get("map_id", nav_map.get("scenario_id", "unknown")),
        "version": metadata.get("version", 1),
        "scenario_id": nav_map.get("scenario_id"),
        "generated_at": nav_map.get("generated_at", datetime.now().isoformat()),
        "last_updated": nav_map.get("last_updated", nav_map.get("generated_at")),
        "coordinate_frame": metadata.get("coordinate_frame", "NED"),
        "origin_latitude": metadata.get("origin_latitude"),
        "origin_longitude": metadata.get("origin_longitude"),
        "origin_altitude_m": metadata.get("origin_altitude_m"),
        "resolution_m": metadata.get("resolution_m", 1.0),
        "obstacle_count": len(obstacles),
        "map_quality_score": metadata.get("map_quality_score", 1.0),
        "coverage_percent": metadata.get("coverage_percent", 0.0),
    }

    return {
        "status": "ok",
        "metadata": response_metadata,
        "obstacles": obstacles,
        "source": nav_map.get("source", "unknown"),
    }


async def get_navigation_obstacles() -> dict:
    """GET /api/navigation/obstacles - Get obstacle list only.

    Returns just the obstacles for lightweight queries.
    """
    nav_map = _get_navigation_map()

    if not nav_map:
        return {
            "status": "no_map",
            "obstacles": [],
            "count": 0,
        }

    obstacles = nav_map.get("obstacles", [])

    return {
        "status": "ok",
        "obstacles": obstacles,
        "count": len(obstacles),
        "source": nav_map.get("source", "unknown"),
        "map_age_s": _get_map_age_seconds(),
    }


async def get_navigation_map_metadata() -> dict:
    """GET /api/navigation/map/metadata - Get map metadata only.

    Returns metadata without the full obstacle/tile data.
    """
    nav_map = _get_navigation_map()

    if not nav_map:
        return {
            "status": "no_map",
            "metadata": None,
        }

    obstacles = nav_map.get("obstacles", [])
    metadata = nav_map.get("metadata", {})

    return {
        "status": "ok",
        "metadata": {
            "map_id": metadata.get("map_id", nav_map.get("scenario_id", "unknown")),
            "version": metadata.get("version", 1),
            "scenario_id": nav_map.get("scenario_id"),
            "generated_at": nav_map.get("generated_at"),
            "last_updated": nav_map.get("last_updated", nav_map.get("generated_at")),
            "source": nav_map.get("source", "unknown"),
            "coordinate_frame": metadata.get("coordinate_frame", "NED"),
            "origin_latitude": metadata.get("origin_latitude"),
            "origin_longitude": metadata.get("origin_longitude"),
            "resolution_m": metadata.get("resolution_m", 1.0),
            "obstacle_count": len(obstacles),
            "map_quality_score": metadata.get("map_quality_score", 1.0),
        },
        "map_age_s": _get_map_age_seconds(),
    }


async def get_slam_status() -> dict:
    """GET /api/slam/status - Get SLAM pipeline status.

    Returns current state of the SLAM system.
    """
    # TODO: Phase 2 - integrate with actual SLAM runner
    slam_data = getattr(server_state, "slam_status", None)

    if not slam_data:
        return {
            "enabled": False,
            "running": False,
            "backend": "none",
            "tracking_state": "not_initialized",
            "keyframe_count": 0,
            "map_point_count": 0,
            "loop_closure_count": 0,
            "pose_confidence": 0.0,
            "reprojection_error": 0.0,
            "drift_estimate_m": 0.0,
            "last_frame_ms": 0.0,
            "avg_frame_ms": 0.0,
            "last_update": None,
        }

    if "loop_closure_rate" not in slam_data:
        keyframes = int(slam_data.get("keyframe_count", 0))
        loops = int(slam_data.get("loop_closure_count", 0))
        slam_data["loop_closure_rate"] = loops / max(1, keyframes)
    if "reprojection_error" not in slam_data:
        slam_data["reprojection_error"] = 0.0
    pose_graph_summary = getattr(server_state, "slam_pose_graph_summary", None)
    if pose_graph_summary:
        slam_data["pose_graph_summary"] = pose_graph_summary
    return slam_data


async def get_map_status() -> dict:
    """GET /api/navigation/map/status - Get combined map status.

    Returns high-level status for dashboard display.
    """
    nav_map = _get_navigation_map()
    slam_status = await get_slam_status()
    splat_available = getattr(server_state, "splat_artifacts", None) is not None

    if not nav_map:
        return {
            "map_available": False,
            "map_age_s": float("inf"),
            "map_version": 0,
            "obstacle_count": 0,
            "map_quality_score": 0.0,
            "slam_status": slam_status,
            "splat_available": splat_available,
            "last_update": None,
            "map_update_error": server_state.map_update_last_error,
            "map_update_error_at": server_state.map_update_last_error_at,
        }

    obstacles = nav_map.get("obstacles", [])
    metadata = nav_map.get("metadata", {})

    return {
        "map_available": True,
        "map_age_s": _get_map_age_seconds(),
        "map_version": metadata.get("version", 1),
        "obstacle_count": len(obstacles),
        "map_quality_score": metadata.get("map_quality_score", 1.0),
        "slam_status": slam_status,
        "splat_available": splat_available,
        "last_update": nav_map.get("last_updated", nav_map.get("generated_at")),
        "map_update_error": server_state.map_update_last_error,
        "map_update_error_at": server_state.map_update_last_error_at,
    }


async def get_map_preview() -> dict:
    """GET /api/navigation/map/preview - Get lightweight map preview.

    Returns a simplified preview suitable for dashboard rendering.
    """
    nav_map = _get_navigation_map()

    if not nav_map:
        return {
            "status": "no_map",
            "preview": None,
        }

    obstacles = nav_map.get("obstacles", [])
    tiles = nav_map.get("tiles") or []

    # Create simplified preview with just positions and radii
    preview_obstacles = []
    for obs in obstacles[:100]:  # Limit to 100 for preview
        preview_obstacles.append({
            "id": obs.get("obstacle_id") or obs.get("asset_id", "unknown"),
            "x": obs.get("x_ned") or 0.0,
            "y": obs.get("y_ned") or 0.0,
            "lat": obs.get("latitude"),
            "lon": obs.get("longitude"),
            "r": obs.get("radius_m", 5.0),
            "h": obs.get("height_m", 10.0),
            "type": obs.get("obstacle_type") or obs.get("asset_type", "unknown"),
        })

    metadata = nav_map.get("metadata", {})

    def _resolve_bounds() -> dict[str, float]:
        min_x = metadata.get("bounds_min_x", 0)
        max_x = metadata.get("bounds_max_x", 0)
        min_y = metadata.get("bounds_min_y", 0)
        max_y = metadata.get("bounds_max_y", 0)

        if min_x != max_x and min_y != max_y:
            return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}

        if not tiles:
            return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}

        tile_min_x = min((t.get("origin_x_ned", 0.0) for t in tiles), default=0.0)
        tile_min_y = min((t.get("origin_y_ned", 0.0) for t in tiles), default=0.0)
        tile_max_x = tile_min_x
        tile_max_y = tile_min_y

        for tile in tiles:
            resolution = float(tile.get("resolution_m", metadata.get("resolution_m", 1.0)))
            width_cells = int(tile.get("width_cells", 0))
            height_cells = int(tile.get("height_cells", 0))
            origin_x = float(tile.get("origin_x_ned", tile_min_x))
            origin_y = float(tile.get("origin_y_ned", tile_min_y))
            tile_max_x = max(tile_max_x, origin_x + width_cells * resolution)
            tile_max_y = max(tile_max_y, origin_y + height_cells * resolution)

        return {
            "min_x": tile_min_x,
            "max_x": tile_max_x,
            "min_y": tile_min_y,
            "max_y": tile_max_y,
        }

    def _build_heatmap(bounds: dict[str, float]) -> dict[str, Any] | None:
        if not tiles:
            return None

        span_x = bounds["max_x"] - bounds["min_x"]
        span_y = bounds["max_y"] - bounds["min_y"]
        if span_x <= 0 or span_y <= 0:
            return None

        grid_size = 64
        values = [0.0] * (grid_size * grid_size)
        counts = [0] * (grid_size * grid_size)

        for tile in tiles:
            occupancy = tile.get("occupancy") or []
            if not occupancy:
                continue
            width_cells = int(tile.get("width_cells", 0))
            height_cells = int(tile.get("height_cells", 0))
            if width_cells <= 0 or height_cells <= 0:
                continue
            resolution = float(tile.get("resolution_m", metadata.get("resolution_m", 1.0)))
            origin_x = float(tile.get("origin_x_ned", bounds["min_x"]))
            origin_y = float(tile.get("origin_y_ned", bounds["min_y"]))

            for idx, occ in enumerate(occupancy):
                if occ is None:
                    continue
                try:
                    occ_val = float(occ)
                except (TypeError, ValueError):
                    continue
                if occ_val <= 0:
                    continue
                cell_x = idx % width_cells
                cell_y = idx // width_cells
                if cell_y >= height_cells:
                    continue
                x_ned = origin_x + cell_x * resolution
                y_ned = origin_y + cell_y * resolution
                gx = int(((x_ned - bounds["min_x"]) / span_x) * grid_size)
                gy = int(((y_ned - bounds["min_y"]) / span_y) * grid_size)
                if gx < 0 or gy < 0 or gx >= grid_size or gy >= grid_size:
                    continue
                gidx = gy * grid_size + gx
                values[gidx] += occ_val
                counts[gidx] += 1

        normalized = []
        for value, count in zip(values, counts):
            if count == 0:
                normalized.append(0.0)
            else:
                normalized.append(min(1.0, (value / count) / 255.0))

        return {
            "width": grid_size,
            "height": grid_size,
            "values": normalized,
            "bounds": bounds,
        }

    bounds = _resolve_bounds()
    heatmap = _build_heatmap(bounds)

    return {
        "status": "ok",
        "preview": {
            "obstacle_count": len(obstacles),
            "obstacles": preview_obstacles,
            "bounds": bounds,
            "resolution_m": metadata.get("resolution_m", 1.0),
            "heatmap": heatmap,
        },
        "map_age_s": _get_map_age_seconds(),
    }


async def get_map_health() -> dict:
    """GET /api/navigation/map/health - Summarize map health and gate status.

    Extended in Agent B Phase 6 to include gate history and proxy health.
    """
    nav_map = _get_navigation_map()
    config = get_config_manager().config
    map_age_s = _get_map_age_seconds()

    # Get gate history from server state (populated by MapUpdateService)
    gate_history = getattr(server_state, "map_gate_history", [])
    proxy_health = _get_proxy_health()

    if not nav_map:
        return {
            "map_available": False,
            "map_age_s": map_age_s,
            "map_quality_score": 0.0,
            "map_valid": False,
            "gate_status": "no_map",
            "gate_reason": "No navigation map available",
            "gate_details": {},
            "last_update": None,
            "gate_history": gate_history[-10:],  # Last 10 gate results
            "proxy_health": proxy_health,
        }

    metadata = nav_map.get("metadata", {})
    map_context = MapContext.from_navigation_map(
        nav_map,
        stale_threshold_s=config.mapping.max_map_age_s,
        min_quality_score=config.mapping.min_quality_score,
    )
    gate = MapUpdateGate(
        SafetyGateConfig(
            min_map_confidence=config.mapping.min_quality_score,
            max_map_age_s=config.mapping.max_map_age_s,
        )
    )
    gate_result = gate.check_update(
        nav_map,
        previous_map=getattr(server_state, "last_valid_navigation_map", None),
    )

    return {
        "map_available": True,
        "map_age_s": map_age_s,
        "map_quality_score": metadata.get("map_quality_score", 0.0),
        "map_valid": map_context.map_valid,
        "gate_status": gate_result.result.value,
        "gate_reason": gate_result.reason,
        "gate_details": gate_result.details,
        "stale_threshold_s": config.mapping.max_map_age_s,
        "min_quality_score": config.mapping.min_quality_score,
        "last_update": nav_map.get("last_updated", nav_map.get("generated_at")),
        "gate_history": gate_history[-10:],  # Last 10 gate results
        "proxy_health": proxy_health,
    }


def _get_proxy_health() -> dict[str, Any]:
    """Get planning proxy health status."""
    splat_artifacts = getattr(server_state, "splat_artifacts", None)

    if not splat_artifacts:
        return {
            "available": False,
            "path": None,
            "obstacle_count": 0,
            "last_updated": None,
            "age_s": None,
        }

    proxy_path = splat_artifacts.get("planning_proxy")
    proxy_updated = splat_artifacts.get("planning_proxy_updated_at")
    proxy_obstacles = splat_artifacts.get("planning_proxy_obstacle_count", 0)

    # Calculate proxy age
    proxy_age_s = None
    if proxy_updated:
        try:
            updated_time = datetime.fromisoformat(proxy_updated.replace("Z", "+00:00"))
            proxy_age_s = (datetime.now(updated_time.tzinfo) - updated_time).total_seconds()
        except Exception:
            pass

    return {
        "available": proxy_path is not None,
        "path": proxy_path,
        "obstacle_count": proxy_obstacles,
        "last_updated": proxy_updated,
        "age_s": proxy_age_s,
    }


async def list_fused_maps() -> dict:
    """GET /api/navigation/map/fused - List stored fused maps."""
    base_dir = Path(get_config_manager().config.mapping.fused_map_dir)
    store = MapArtifactStore(MapArtifactStoreConfig(base_dir=base_dir))
    maps = store.list_maps()
    if not maps:
        return {
            "status": "no_maps",
            "maps": [],
            "count": 0,
            "base_dir": str(base_dir),
        }
    return {
        "status": "ok",
        "maps": maps,
        "count": len(maps),
        "base_dir": str(base_dir),
    }


async def get_splat_scenes() -> dict:
    """GET /api/navigation/splat/scenes - List available splat scenes.

    Returns list of stored Gaussian splat reconstructions.
    """
    from pathlib import Path

    splat_dir = Path(__file__).resolve().parents[2] / "data" / "splats"

    if not splat_dir.exists():
        return {
            "status": "no_splats",
            "scenes": [],
            "splat_dir": str(splat_dir),
        }

    scenes = []
    for scene_path in splat_dir.iterdir():
        if scene_path.is_dir() and scene_path.name.startswith("scene_"):
            run_id = scene_path.name[6:]  # Remove "scene_" prefix

            # Find latest version
            versions = []
            for v_path in scene_path.iterdir():
                if v_path.is_dir() and v_path.name.startswith("v"):
                    try:
                        versions.append(int(v_path.name[1:]))
                    except ValueError:
                        pass

            if versions:
                latest_v = max(versions)
                version_dir = scene_path / f"v{latest_v}"

                # Load metadata if exists
                metadata_path = version_dir / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                # Check for preview file
                preview_exists = (version_dir / "preview.ply").exists()
                model_exists = (version_dir / "model.ply").exists()
                planning_proxy_exists = (version_dir / "planning_proxy.json").exists()

                scenes.append({
                    "run_id": run_id,
                    "version": latest_v,
                    "versions_available": len(versions),
                    "preview_available": preview_exists,
                    "model_available": model_exists,
                    "planning_proxy_available": planning_proxy_exists,
                    "gaussian_count": metadata.get("gaussians", {}).get("count", 0),
                    "quality": {
                        "psnr": metadata.get("quality", {}).get("psnr", 0),
                        "ssim": metadata.get("quality", {}).get("ssim", 0),
                    },
                    "created_at": metadata.get("created_at"),
                })
            else:
                # Backward-compatible: look for non-versioned scene.json/preview.ply
                preview_exists = (scene_path / "preview.ply").exists()
                model_exists = (scene_path / "model.ply").exists()
                planning_proxy_exists = (scene_path / "planning_proxy.json").exists()
                metadata_path = scene_path / "scene.json"
                metadata = {}
                if metadata_path.exists():
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                if preview_exists or metadata:
                    scenes.append({
                        "run_id": run_id,
                        "version": 1,
                        "versions_available": 1,
                        "preview_available": preview_exists,
                        "model_available": model_exists,
                        "planning_proxy_available": planning_proxy_exists,
                        "gaussian_count": metadata.get("gaussian_count", 0),
                        "quality": {
                            "psnr": metadata.get("psnr", 0),
                            "ssim": metadata.get("ssim", 0),
                        },
                        "created_at": metadata.get("created_at"),
                    })

    return {
        "status": "ok",
        "scenes": scenes,
        "scene_count": len(scenes),
    }


async def get_splat_preview(run_id: str) -> dict:
    """GET /api/navigation/splat/preview/{run_id} - Get splat preview points.

    Returns point cloud data from the splat preview for visualization.
    """
    from pathlib import Path

    splat_dir = Path(__file__).resolve().parents[2] / "data" / "splats"
    scene_dir = splat_dir / f"scene_{run_id}"

    if not scene_dir.exists():
        return {
            "status": "not_found",
            "run_id": run_id,
            "points": [],
        }

    # Find latest version
    versions = []
    for v_path in scene_dir.iterdir():
        if v_path.is_dir() and v_path.name.startswith("v"):
            try:
                versions.append(int(v_path.name[1:]))
            except ValueError:
                pass

    if not versions:
        preview_path = scene_dir / "preview.ply"
        if not preview_path.exists():
            return {
                "status": "no_versions",
                "run_id": run_id,
                "points": [],
            }
        versions = [1]

    latest_v = max(versions)
    if latest_v == 1 and (scene_dir / "preview.ply").exists():
        preview_path = scene_dir / "preview.ply"
    else:
        preview_path = scene_dir / f"v{latest_v}" / "preview.ply"

    if not preview_path.exists():
        return {
            "status": "no_preview",
            "run_id": run_id,
            "version": latest_v,
            "points": [],
        }

    # Parse PLY file (simplified - assumes ASCII format)
    points = []
    try:
        with open(preview_path, "r") as f:
            in_header = True
            vertex_count = 0

            for line in f:
                line = line.strip()
                if in_header:
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    elif line == "end_header":
                        in_header = False
                else:
                    if len(points) >= min(vertex_count, 5000):  # Limit to 5000 points
                        break
                    parts = line.split()
                    if len(parts) >= 3:
                        points.append({
                            "x": float(parts[0]),
                            "y": float(parts[1]),
                            "z": float(parts[2]),
                            "r": int(parts[3]) if len(parts) > 5 else 128,
                            "g": int(parts[4]) if len(parts) > 5 else 128,
                            "b": int(parts[5]) if len(parts) > 5 else 128,
                        })
    except Exception as e:
        logger.warning("splat_preview_parse_error", run_id=run_id, error=str(e))
        return {
            "status": "parse_error",
            "run_id": run_id,
            "error": str(e),
            "points": [],
        }

    # Compute bounds
    if points:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        zs = [p["z"] for p in points]
        bounds = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "min_z": min(zs),
            "max_z": max(zs),
        }
    else:
        bounds = {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}

    return {
        "status": "ok",
        "run_id": run_id,
        "version": latest_v,
        "point_count": len(points),
        "points": points,
        "bounds": bounds,
    }


async def get_splat_planning_proxy(run_id: str) -> dict:
    """GET /api/navigation/splat/proxy/{run_id} - Get planning proxy map for a splat run."""
    from pathlib import Path

    splat_dir = Path(__file__).resolve().parents[2] / "data" / "splats"
    scene_dir = splat_dir / f"scene_{run_id}"

    if not scene_dir.exists():
        return {"status": "not_found", "run_id": run_id, "proxy": None}

    versions = []
    for v_path in scene_dir.iterdir():
        if v_path.is_dir() and v_path.name.startswith("v"):
            try:
                versions.append(int(v_path.name[1:]))
            except ValueError:
                pass

    if not versions:
        proxy_path = scene_dir / "planning_proxy.json"
        version = 1
    else:
        version = max(versions)
        proxy_path = scene_dir / f"v{version}" / "planning_proxy.json"

    if not proxy_path.exists():
        return {
            "status": "no_proxy",
            "run_id": run_id,
            "version": version,
            "proxy": None,
        }

    try:
        import json

        with open(proxy_path, encoding="utf-8") as f:
            proxy = json.load(f)
    except Exception as exc:
        logger.warning("splat_proxy_parse_error", run_id=run_id, error=str(exc))
        return {
            "status": "parse_error",
            "run_id": run_id,
            "version": version,
            "proxy": None,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "run_id": run_id,
        "version": version,
        "proxy": proxy,
    }


async def run_real_capture(payload: dict | None = None) -> dict:
    """POST /api/navigation/real_capture - Capture real sensor frames."""
    config_mgr = get_config_manager()
    mapping_cfg = config_mgr.config.mapping
    if not mapping_cfg.real_capture_enabled:
        raise HTTPException(
            status_code=409,
            detail="Real capture disabled. Enable mapping.real_capture_enabled in config.",
        )

    payload = payload or {}
    output_dir_value = payload.get("output_dir") or mapping_cfg.real_capture_output_dir
    output_dir = config_mgr.resolve_path(str(output_dir_value))
    camera_index = int(payload.get("camera_index", mapping_cfg.real_capture_camera_index))
    width = int(payload.get("width", mapping_cfg.real_capture_width))
    height = int(payload.get("height", mapping_cfg.real_capture_height))
    fps = int(payload.get("fps", mapping_cfg.real_capture_fps))
    frames = int(payload.get("frames", mapping_cfg.real_capture_frames))
    interval_s = float(payload.get("interval_s", mapping_cfg.real_capture_interval_s))

    calibration_value = payload.get("calibration_path", mapping_cfg.real_capture_calibration_path)
    calibration_path = None
    if calibration_value:
        calibration_path = config_mgr.resolve_path(str(calibration_value))
        if not calibration_path.exists():
            raise HTTPException(status_code=404, detail="Calibration file not found.")
        try:
            from mapping.calibration import load_calibration

            load_calibration(calibration_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Calibration invalid: {exc}") from exc

    capture_config = RealCaptureConfig(
        output_dir=output_dir,
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
        calibration_path=calibration_path,
    )

    capture = RealSensorCapture(capture_config)
    try:
        await asyncio.to_thread(
            capture.capture_sequence,
            frames=frames,
            interval_s=interval_s,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        capture.close()

    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "frames": frames,
        "interval_s": interval_s,
        "camera_index": camera_index,
    }


# -----------------------------------------------------------------------------
# Route Registration
# -----------------------------------------------------------------------------


def register_navigation_routes(app: FastAPI) -> None:
    """Register navigation map and SLAM routes."""
    # Phase 0: Core map endpoints
    app.get("/api/navigation/map/latest")(get_navigation_map_latest)
    app.get("/api/navigation/obstacles")(get_navigation_obstacles)
    app.get("/api/navigation/map/metadata")(get_navigation_map_metadata)

    # Phase 2: SLAM status
    app.get("/api/slam/status")(get_slam_status)

    # Phase 3: Splat endpoints
    app.get("/api/navigation/splat/scenes")(get_splat_scenes)
    app.get("/api/navigation/splat/preview/{run_id}")(get_splat_preview)
    app.get("/api/navigation/splat/proxy/{run_id}")(get_splat_planning_proxy)

    # Phase 6: Dashboard endpoints
    app.get("/api/navigation/map/status")(get_map_status)
    app.get("/api/navigation/map/preview")(get_map_preview)
    app.get("/api/navigation/map/health")(get_map_health)
    app.get("/api/navigation/map/fused")(list_fused_maps)
    app.post("/api/navigation/real_capture")(run_real_capture)

    logger.info("navigation_routes_registered")
