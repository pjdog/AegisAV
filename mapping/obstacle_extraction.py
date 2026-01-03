"""Obstacle extraction from fused SLAM/splat maps.

Phase 4 Worker B: Implement obstacle extraction from fused map.

This module provides:
- Point cloud clustering into obstacles
- Height and radius estimation from geometry
- Map metadata generation (resolution, bounds, timestamp)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Point3D:
    """A 3D point with optional attributes."""

    x: float  # North (NED)
    y: float  # East (NED)
    z: float  # Down (NED)
    confidence: float = 1.0
    color_rgb: tuple[int, int, int] | None = None

    def distance_to(self, other: Point3D) -> float:
        """Calculate 3D distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def horizontal_distance_to(self, other: Point3D) -> float:
        """Calculate horizontal (2D) distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx**2 + dy**2)


@dataclass
class ExtractedObstacle:
    """An obstacle extracted from the fused map."""

    obstacle_id: str
    centroid: Point3D

    # Geometry
    radius_m: float
    height_m: float
    min_z: float  # Top of obstacle (NED, so negative = higher)
    max_z: float  # Bottom of obstacle

    # Classification
    obstacle_type: str = "unknown"
    confidence: float = 1.0

    # Source tracking
    point_count: int = 0
    source: str = "fused_map"
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Bounding box (optional)
    bbox_min_x: float | None = None
    bbox_max_x: float | None = None
    bbox_min_y: float | None = None
    bbox_max_y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "obstacle_id": self.obstacle_id,
            "x_ned": self.centroid.x,
            "y_ned": self.centroid.y,
            "z_ned": self.centroid.z,
            "radius_m": self.radius_m,
            "height_m": self.height_m,
            "obstacle_type": self.obstacle_type,
            "confidence": self.confidence,
            "point_count": self.point_count,
            "source": self.source,
            "detected_at": self.detected_at,
        }

        if self.bbox_min_x is not None:
            result["bbox"] = {
                "min_x": self.bbox_min_x,
                "max_x": self.bbox_max_x,
                "min_y": self.bbox_min_y,
                "max_y": self.bbox_max_y,
                "min_z": self.min_z,
                "max_z": self.max_z,
            }

        return result


@dataclass
class MapMetadataResult:
    """Metadata about the fused navigation map."""

    map_id: str
    version: int = 1
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Bounds
    bounds_min_x: float = 0.0
    bounds_max_x: float = 0.0
    bounds_min_y: float = 0.0
    bounds_max_y: float = 0.0
    bounds_min_z: float = 0.0
    bounds_max_z: float = 0.0

    # Resolution
    resolution_m: float = 1.0
    voxel_size_m: float | None = None

    # Statistics
    total_points: int = 0
    obstacle_count: int = 0
    coverage_area_m2: float = 0.0

    # Quality
    map_quality_score: float = 1.0
    slam_confidence: float = 1.0
    splat_quality: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "map_id": self.map_id,
            "version": self.version,
            "generated_at": self.generated_at,
            "bounds": {
                "min_x": self.bounds_min_x,
                "max_x": self.bounds_max_x,
                "min_y": self.bounds_min_y,
                "max_y": self.bounds_max_y,
                "min_z": self.bounds_min_z,
                "max_z": self.bounds_max_z,
            },
            "resolution_m": self.resolution_m,
            "voxel_size_m": self.voxel_size_m,
            "statistics": {
                "total_points": self.total_points,
                "obstacle_count": self.obstacle_count,
                "coverage_area_m2": self.coverage_area_m2,
            },
            "quality": {
                "map_quality_score": self.map_quality_score,
                "slam_confidence": self.slam_confidence,
                "splat_quality": self.splat_quality,
            },
        }


@dataclass
class ExtractionConfig:
    """Configuration for obstacle extraction."""

    # Clustering parameters
    cluster_distance_m: float = 2.0  # Max distance between points in cluster
    min_cluster_points: int = 10  # Minimum points to form an obstacle
    max_cluster_points: int = 10000  # Maximum points per cluster

    # Height filtering
    min_height_m: float = 0.5  # Minimum obstacle height
    max_height_m: float = 200.0  # Maximum obstacle height
    ground_threshold_z: float = 0.5  # Points below this are ground (NED, positive = down)

    # Radius estimation
    min_radius_m: float = 0.5
    max_radius_m: float = 100.0
    radius_padding: float = 1.2  # Multiply radius by this for safety margin

    # Confidence thresholds
    min_point_confidence: float = 0.3
    min_obstacle_confidence: float = 0.5


class ObstacleExtractor:
    """Extracts obstacles from fused point cloud data.

    Implements a simple clustering approach:
    1. Filter ground points
    2. Cluster remaining points by proximity
    3. Compute bounding geometry for each cluster
    4. Classify obstacles by shape/size

    Usage:
        extractor = ObstacleExtractor(config)
        obstacles = extractor.extract(point_cloud)
        metadata = extractor.get_map_metadata(point_cloud, obstacles)
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        """Initialize the extractor.

        Args:
            config: Extraction configuration.
        """
        self.config = config or ExtractionConfig()
        self._obstacle_counter = 0

    def extract(self, points: list[Point3D]) -> list[ExtractedObstacle]:
        """Extract obstacles from a point cloud.

        Args:
            points: List of 3D points from fused map.

        Returns:
            List of extracted obstacles.
        """
        if not points:
            return []

        # Filter by confidence
        filtered = [p for p in points if p.confidence >= self.config.min_point_confidence]

        # Filter ground points (keep obstacles above ground)
        obstacle_points = [
            p
            for p in filtered
            if p.z < self.config.ground_threshold_z  # In NED, negative z is above ground
        ]

        if not obstacle_points:
            logger.debug("no_obstacle_points", total=len(points), filtered=len(filtered))
            return []

        # Cluster points
        clusters = self._cluster_points(obstacle_points)

        # Extract obstacles from clusters
        obstacles = []
        for cluster in clusters:
            if len(cluster) < self.config.min_cluster_points:
                continue
            if len(cluster) > self.config.max_cluster_points:
                # Large cluster - might need subdivision
                cluster = cluster[: self.config.max_cluster_points]

            obstacle = self._cluster_to_obstacle(cluster)
            if obstacle:
                obstacles.append(obstacle)

        logger.info(
            "obstacles_extracted",
            total_points=len(points),
            obstacle_points=len(obstacle_points),
            clusters=len(clusters),
            obstacles=len(obstacles),
        )

        return obstacles

    def _cluster_points(self, points: list[Point3D]) -> list[list[Point3D]]:
        """Simple clustering by proximity.

        Uses a greedy approach - not optimal but fast for moderate point counts.
        For production, consider using scipy.cluster or sklearn.
        """
        if not points:
            return []

        remaining = list(points)
        clusters: list[list[Point3D]] = []

        while remaining:
            # Start new cluster with first remaining point
            seed = remaining.pop(0)
            cluster = [seed]

            # Find all points within distance
            i = 0
            while i < len(remaining):
                point = remaining[i]

                # Check distance to any point in cluster
                for cp in cluster:
                    if point.distance_to(cp) <= self.config.cluster_distance_m:
                        cluster.append(remaining.pop(i))
                        break
                else:
                    i += 1

            clusters.append(cluster)

        return clusters

    def _cluster_to_obstacle(self, cluster: list[Point3D]) -> ExtractedObstacle | None:
        """Convert a point cluster to an obstacle."""
        if not cluster:
            return None

        # Compute bounding box
        min_x = min(p.x for p in cluster)
        max_x = max(p.x for p in cluster)
        min_y = min(p.y for p in cluster)
        max_y = max(p.y for p in cluster)
        min_z = min(p.z for p in cluster)  # Top (most negative in NED)
        max_z = max(p.z for p in cluster)  # Bottom

        # Compute centroid
        cx = sum(p.x for p in cluster) / len(cluster)
        cy = sum(p.y for p in cluster) / len(cluster)
        cz = sum(p.z for p in cluster) / len(cluster)

        # Compute height and radius
        height = max_z - min_z
        width_x = max_x - min_x
        width_y = max_y - min_y
        radius = max(width_x, width_y) / 2 * self.config.radius_padding

        # Filter by size constraints
        if height < self.config.min_height_m or height > self.config.max_height_m:
            return None
        if radius < self.config.min_radius_m or radius > self.config.max_radius_m:
            return None

        # Classify obstacle type
        obstacle_type = self._classify_obstacle(height, radius, len(cluster))

        # Compute confidence based on point density
        volume = math.pi * radius**2 * height
        point_density = len(cluster) / max(volume, 0.001)
        confidence = min(1.0, point_density / 10.0)  # Normalize

        if confidence < self.config.min_obstacle_confidence:
            return None

        # Generate ID
        self._obstacle_counter += 1
        obstacle_id = f"obs_{self._obstacle_counter:04d}"

        return ExtractedObstacle(
            obstacle_id=obstacle_id,
            centroid=Point3D(x=cx, y=cy, z=cz),
            radius_m=radius,
            height_m=height,
            min_z=min_z,
            max_z=max_z,
            obstacle_type=obstacle_type,
            confidence=confidence,
            point_count=len(cluster),
            bbox_min_x=min_x,
            bbox_max_x=max_x,
            bbox_min_y=min_y,
            bbox_max_y=max_y,
        )

    def _classify_obstacle(self, height: float, radius: float, point_count: int) -> str:
        """Classify obstacle type based on geometry."""
        aspect_ratio = height / max(radius, 0.1)

        # Simple heuristic classification
        if height > 50 and aspect_ratio > 3:
            return "tower"
        elif height > 20 and aspect_ratio > 2:
            return "building_tall"
        elif height > 5 and radius > 10:
            return "building"
        elif height > 3 and radius < 5:
            return "tree"
        elif height < 3 and radius > 5:
            return "structure_low"
        elif height < 2:
            return "ground_obstacle"
        else:
            return "unknown"

    def get_map_metadata(
        self,
        points: list[Point3D],
        obstacles: list[ExtractedObstacle],
        map_id: str = "fused_map",
        slam_confidence: float = 1.0,
        splat_quality: float = 1.0,
    ) -> MapMetadataResult:
        """Generate map metadata from points and obstacles.

        Args:
            points: All points in the map.
            obstacles: Extracted obstacles.
            map_id: Map identifier.
            slam_confidence: SLAM tracking confidence.
            splat_quality: Gaussian splat reconstruction quality.

        Returns:
            MapMetadataResult with computed metadata.
        """
        if not points:
            return MapMetadataResult(map_id=map_id)

        # Compute bounds
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)
        min_z = min(p.z for p in points)
        max_z = max(p.z for p in points)

        # Compute coverage area
        coverage_area = (max_x - min_x) * (max_y - min_y)

        # Estimate resolution from point density
        if coverage_area > 0:
            point_density = len(points) / coverage_area
            resolution = 1.0 / math.sqrt(max(point_density, 0.01))
        else:
            resolution = 1.0

        # Compute quality score
        avg_confidence = sum(p.confidence for p in points) / len(points)
        quality_score = avg_confidence * 0.4 + slam_confidence * 0.3 + splat_quality * 0.3

        return MapMetadataResult(
            map_id=map_id,
            bounds_min_x=min_x,
            bounds_max_x=max_x,
            bounds_min_y=min_y,
            bounds_max_y=max_y,
            bounds_min_z=min_z,
            bounds_max_z=max_z,
            resolution_m=resolution,
            total_points=len(points),
            obstacle_count=len(obstacles),
            coverage_area_m2=coverage_area,
            map_quality_score=quality_score,
            slam_confidence=slam_confidence,
            splat_quality=splat_quality,
        )

    def reset(self) -> None:
        """Reset the extractor state."""
        self._obstacle_counter = 0


def obstacles_to_navigation_map(
    obstacles: list[ExtractedObstacle],
    metadata: MapMetadataResult,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Convert extracted obstacles to navigation map format.

    This produces the format expected by server_state.navigation_map.

    Args:
        obstacles: Extracted obstacles.
        metadata: Map metadata.
        scenario_id: Associated scenario ID.

    Returns:
        Navigation map dictionary.
    """
    obstacle_list = []
    for obs in obstacles:
        obstacle_list.append({
            "obstacle_id": obs.obstacle_id,
            "asset_id": None,  # From extraction, not scenario
            "name": f"{obs.obstacle_type}_{obs.obstacle_id}",
            "obstacle_type": obs.obstacle_type,
            "x_ned": obs.centroid.x,
            "y_ned": obs.centroid.y,
            "z_ned": obs.centroid.z,
            "latitude": None,  # Would need geo conversion
            "longitude": None,
            "radius_m": obs.radius_m,
            "height_m": obs.height_m,
            "confidence": obs.confidence,
            "source": obs.source,
            "detected_at": obs.detected_at,
        })

    return {
        "scenario_id": scenario_id,
        "generated_at": metadata.generated_at,
        "last_updated": metadata.generated_at,
        "source": "fused_map",
        "obstacles": obstacle_list,
        "metadata": metadata.to_dict(),
    }
