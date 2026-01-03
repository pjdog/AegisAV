"""Map fusion pipeline for SLAM + Gaussian splat outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from mapping.obstacle_extraction import (
    ExtractionConfig,
    MapMetadataResult,
    ObstacleExtractor,
    Point3D,
)

logger = structlog.get_logger(__name__)


@dataclass
class MapFusionConfig:
    """Configuration for map fusion and occupancy output."""

    resolution_m: float = 2.0
    tile_size_cells: int = 120
    voxel_size_m: float | None = None
    min_points: int = 50
    max_points: int = 200000
    occupancy_threshold: int = 1
    coordinate_frame: str = "NED"


@dataclass
class MapFusionResult:
    """Results for a fused navigation map."""

    navigation_map: dict[str, Any]
    points_used: int
    obstacle_count: int


class MapFusion:
    """Fuses SLAM geometry and splat previews into planning-grade maps."""

    def __init__(
        self,
        config: MapFusionConfig | None = None,
        extraction_config: ExtractionConfig | None = None,
    ) -> None:
        self.config = config or MapFusionConfig()
        self._base_extraction_config = extraction_config or ExtractionConfig()
        self.extractor = ObstacleExtractor(self._base_extraction_config)

    def _tune_extraction_config(self, points: list[Point3D]) -> tuple[ExtractionConfig, float]:
        if not points:
            return self._base_extraction_config, 0.0

        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        coverage_area = max((max_x - min_x) * (max_y - min_y), 0.0)
        if coverage_area <= 0.0:
            return self._base_extraction_config, 0.0

        point_density = len(points) / coverage_area
        config = self._base_extraction_config

        if point_density < 0.05 or len(points) < 500:
            tuned = replace(
                config,
                cluster_distance_m=max(3.5, config.cluster_distance_m),
                min_cluster_points=max(4, config.min_cluster_points // 2),
                min_height_m=max(0.3, config.min_height_m * 0.7),
                min_radius_m=max(0.3, config.min_radius_m * 0.7),
            )
        elif point_density > 1.5 or len(points) > 200000:
            tuned = replace(
                config,
                cluster_distance_m=max(1.0, config.cluster_distance_m * 0.8),
                min_cluster_points=max(config.min_cluster_points, 20),
            )
        else:
            tuned = config

        return tuned, point_density

    def _adaptive_occupancy_threshold(self, point_density: float) -> int:
        threshold = max(1, int(self.config.occupancy_threshold))
        if point_density <= 0:
            return threshold
        if point_density >= 2.5:
            return max(threshold, 3)
        if point_density >= 1.0:
            return max(threshold, 2)
        return threshold

    def build_navigation_map(
        self,
        point_cloud_path: Path,
        map_id: str | None = None,
        scenario_id: str | None = None,
        source: str = "fused_map",
        slam_confidence: float = 1.0,
        splat_quality: float = 1.0,
        geo_ref: Any | None = None,
    ) -> MapFusionResult:
        """Build a navigation map from a point cloud file."""
        points = self._load_point_cloud(point_cloud_path)
        if not points or len(points) < self.config.min_points:
            nav_map = {
                "scenario_id": scenario_id,
                "generated_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "source": source,
                "obstacles": [],
                "metadata": {
                    "map_id": map_id or "fused_map",
                    "version": 1,
                    "coordinate_frame": self.config.coordinate_frame,
                    "bounds_min_x": 0.0,
                    "bounds_max_x": 0.0,
                    "bounds_min_y": 0.0,
                    "bounds_max_y": 0.0,
                    "bounds_min_z": 0.0,
                    "bounds_max_z": 0.0,
                    "resolution_m": self.config.resolution_m,
                    "voxel_size_m": self.config.voxel_size_m,
                    "obstacle_count": 0,
                    "map_quality_score": 0.0,
                    "slam_confidence": slam_confidence,
                    "splat_quality": splat_quality,
                },
                "tiles": [],
            }
            return MapFusionResult(
                navigation_map=nav_map,
                points_used=len(points),
                obstacle_count=0,
            )

        tuned_config, point_density = self._tune_extraction_config(points)
        self.extractor.config = tuned_config
        obstacles = self.extractor.extract(points)
        metadata = self.extractor.get_map_metadata(
            points,
            obstacles,
            map_id=map_id or "fused_map",
            slam_confidence=slam_confidence,
            splat_quality=splat_quality,
        )
        metadata_flat = self._flatten_metadata(metadata)
        metadata_flat["coordinate_frame"] = self.config.coordinate_frame
        metadata_flat["resolution_m"] = self.config.resolution_m
        metadata_flat["voxel_size_m"] = self.config.voxel_size_m
        metadata_flat["point_density_m2"] = point_density
        if geo_ref:
            gps_bounds = self._compute_gps_bounds(metadata, geo_ref)
            metadata_flat.update(gps_bounds)

        occupancy_threshold = self._adaptive_occupancy_threshold(point_density)
        metadata_flat["occupancy_threshold"] = occupancy_threshold
        tiles = self._build_occupancy_tiles(points, metadata, occupancy_threshold)
        metadata_flat["tile_count"] = len(tiles)

        voxels = self._build_voxels(points, metadata)
        metadata_flat["voxel_count"] = len(voxels)

        nav_map = {
            "scenario_id": scenario_id,
            "generated_at": metadata.generated_at,
            "last_updated": metadata.generated_at,
            "source": source,
            "obstacles": [obs.to_dict() for obs in obstacles],
            "metadata": metadata_flat,
            "tiles": tiles,
            "voxels": voxels if voxels else None,
        }

        logger.info(
            "map_fused",
            source=source,
            points=len(points),
            obstacles=len(obstacles),
            tiles=len(tiles),
        )

        return MapFusionResult(
            navigation_map=nav_map,
            points_used=len(points),
            obstacle_count=len(obstacles),
        )

    def _load_point_cloud(self, path: Path) -> list[Point3D]:
        path = Path(path)
        if not path.exists():
            logger.warning("point_cloud_missing", path=str(path))
            return []

        if path.suffix.lower() == ".npy":
            data = np.load(path)
            return self._points_from_array(data)

        if path.suffix.lower() == ".npz":
            data = np.load(path)
            if "points" in data:
                return self._points_from_array(data["points"])
            return []

        if path.suffix.lower() == ".ply":
            return self._load_ply(path)

        logger.warning("unsupported_point_cloud", path=str(path))
        return []

    def _points_from_array(self, data: np.ndarray) -> list[Point3D]:
        if data.ndim != 2 or data.shape[1] < 3:
            return []

        if data.shape[0] > self.config.max_points:
            step = int(math.ceil(data.shape[0] / self.config.max_points))
            data = data[::step]

        points: list[Point3D] = []
        for row in data:
            points.append(Point3D(x=float(row[0]), y=float(row[1]), z=float(row[2])))
        return points

    def _load_ply(self, path: Path) -> list[Point3D]:
        with open(path, encoding="utf-8") as f:
            header = []
            line = f.readline()
            if not line.startswith("ply"):
                return []
            header.append(line)

            vertex_count = 0
            format_ascii = False
            while True:
                line = f.readline()
                if not line:
                    return []
                header.append(line)
                if line.startswith("format"):
                    format_ascii = "ascii" in line
                if line.startswith("element vertex"):
                    parts = line.split()
                    if len(parts) >= 3:
                        vertex_count = int(parts[2])
                if line.strip() == "end_header":
                    break

            if not format_ascii:
                logger.warning("ply_binary_unsupported", path=str(path))
                return []

            points: list[Point3D] = []
            for i in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                points.append(Point3D(x=float(parts[0]), y=float(parts[1]), z=float(parts[2])))

        if len(points) > self.config.max_points:
            step = int(math.ceil(len(points) / self.config.max_points))
            points = points[::step]

        return points

    def _build_occupancy_tiles(
        self,
        points: list[Point3D],
        metadata: MapMetadataResult,
        occupancy_threshold: int,
    ) -> list[dict[str, Any]]:
        if not points:
            return []

        res = self.config.resolution_m
        min_x = metadata.bounds_min_x
        max_x = metadata.bounds_max_x
        min_y = metadata.bounds_min_y
        max_y = metadata.bounds_max_y

        width = max(1, int(math.ceil((max_x - min_x) / res)))
        height = max(1, int(math.ceil((max_y - min_y) / res)))

        threshold = max(1, int(occupancy_threshold))
        grid_counts = np.zeros((height, width), dtype=np.uint16)
        for point in points:
            x_idx = int((point.x - min_x) / res)
            y_idx = int((point.y - min_y) / res)
            if 0 <= x_idx < width and 0 <= y_idx < height:
                grid_counts[y_idx, x_idx] += 1

        grid = np.where(grid_counts >= threshold, 255, 0).astype(np.uint8)

        tiles: list[dict[str, Any]] = []
        tile_size = max(10, int(self.config.tile_size_cells))
        tiles_x = max(1, int(math.ceil(width / tile_size)))
        tiles_y = max(1, int(math.ceil(height / tile_size)))

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x0 = tx * tile_size
                x1 = min(width, (tx + 1) * tile_size)
                y0 = ty * tile_size
                y1 = min(height, (ty + 1) * tile_size)
                tile_grid = grid[y0:y1, x0:x1]
                tiles.append({
                    "tile_id": f"tile_{tx}_{ty}",
                    "x_index": tx,
                    "y_index": ty,
                    "resolution_m": res,
                    "width_cells": x1 - x0,
                    "height_cells": y1 - y0,
                    "origin_x_ned": min_x + x0 * res,
                    "origin_y_ned": min_y + y0 * res,
                    "occupancy": tile_grid.flatten().tolist(),
                    "timestamp": metadata.generated_at,
                })

        return tiles

    def _build_voxels(
        self,
        points: list[Point3D],
        metadata: MapMetadataResult,
    ) -> list[dict[str, Any]]:
        if not self.config.voxel_size_m or not points:
            return []

        size = float(self.config.voxel_size_m)
        min_x = metadata.bounds_min_x
        min_y = metadata.bounds_min_y
        min_z = metadata.bounds_min_z

        voxel_counts: dict[tuple[int, int, int], int] = {}
        for point in points:
            ix = int((point.x - min_x) / size)
            iy = int((point.y - min_y) / size)
            iz = int((point.z - min_z) / size)
            key = (ix, iy, iz)
            voxel_counts[key] = voxel_counts.get(key, 0) + 1

        voxels: list[dict[str, Any]] = []
        for (ix, iy, iz), count in voxel_counts.items():
            voxels.append({
                "x": ix,
                "y": iy,
                "z": iz,
                "occupancy": min(1.0, count / 5.0),
                "confidence": 1.0,
                "color_rgb": None,
            })

        if len(voxels) > self.config.max_points:
            step = int(math.ceil(len(voxels) / self.config.max_points))
            voxels = voxels[::step]

        return voxels

    @staticmethod
    def _flatten_metadata(metadata: MapMetadataResult) -> dict[str, Any]:
        return {
            "map_id": metadata.map_id,
            "version": metadata.version,
            "generated_at": metadata.generated_at,
            "bounds_min_x": metadata.bounds_min_x,
            "bounds_max_x": metadata.bounds_max_x,
            "bounds_min_y": metadata.bounds_min_y,
            "bounds_max_y": metadata.bounds_max_y,
            "bounds_min_z": metadata.bounds_min_z,
            "bounds_max_z": metadata.bounds_max_z,
            "resolution_m": metadata.resolution_m,
            "voxel_size_m": metadata.voxel_size_m,
            "obstacle_count": metadata.obstacle_count,
            "tile_count": 1,
            "voxel_count": 0,
            "map_quality_score": metadata.map_quality_score,
            "slam_confidence": metadata.slam_confidence,
            "splat_quality": metadata.splat_quality,
        }

    @staticmethod
    def _compute_gps_bounds(metadata: MapMetadataResult, geo_ref: Any) -> dict[str, float]:
        try:
            corners = [
                geo_ref.ned_to_gps(metadata.bounds_min_x, metadata.bounds_min_y, 0.0),
                geo_ref.ned_to_gps(metadata.bounds_min_x, metadata.bounds_max_y, 0.0),
                geo_ref.ned_to_gps(metadata.bounds_max_x, metadata.bounds_min_y, 0.0),
                geo_ref.ned_to_gps(metadata.bounds_max_x, metadata.bounds_max_y, 0.0),
            ]
            lats = [c[0] for c in corners]
            lons = [c[1] for c in corners]
            return {
                "origin_latitude": geo_ref.latitude,
                "origin_longitude": geo_ref.longitude,
                "origin_altitude_m": geo_ref.altitude,
                "bounds_min_latitude": min(lats),
                "bounds_max_latitude": max(lats),
                "bounds_min_longitude": min(lons),
                "bounds_max_longitude": max(lons),
            }
        except Exception:
            return {}
