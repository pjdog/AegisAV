"""Mission Planner.

Loads mission configuration and optimizes inspection routes.
Provides high-level mission management for autonomous operations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from autonomy.path_planner import PathPlanner, PathPlannerConfig
from autonomy.vehicle_state import Position
from mapping.decision_context import MapContext
from simulation.coordinate_utils import GeoReference

logger = logging.getLogger(__name__)


class InspectionTarget(BaseModel):
    """A target asset to be inspected."""

    target_id: str
    name: str
    target_type: str = "other"

    # Position
    latitude: float
    longitude: float
    altitude_m: float = 0.0

    # Inspection parameters
    inspection_altitude_agl: float = 20.0
    orbit_radius_m: float = 20.0
    dwell_time_s: float = 30.0

    # Priority (lower = higher priority)
    priority: int = 1

    # Status
    inspected: bool = False
    last_inspection: datetime | None = None


class MissionWaypoint(BaseModel):
    """A waypoint in the mission plan."""

    waypoint_id: str
    target_id: str | None = None  # If this is an inspection waypoint

    # Position
    latitude: float
    longitude: float
    altitude_msl: float

    # Actions
    action: str = "flyover"  # flyover, inspect, hover, land
    dwell_time_s: float = 0.0

    # Sequencing
    sequence: int = 0
    completed: bool = False


class MissionPlan(BaseModel):
    """Complete mission plan with ordered waypoints."""

    mission_id: str
    mission_name: str
    created_at: datetime

    # Home position
    home_latitude: float
    home_longitude: float
    home_altitude_msl: float

    # Waypoints in execution order
    waypoints: list[MissionWaypoint]

    # Targets for reference
    targets: list[InspectionTarget]

    # Metrics
    total_distance_m: float = 0.0
    estimated_flight_time_s: float = 0.0
    estimated_battery_percent: float = 0.0

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)

    @property
    def num_targets(self) -> int:
        return len(self.targets)

    @property
    def completed_targets(self) -> int:
        return sum(1 for t in self.targets if t.inspected)


@dataclass
class MissionConfig:
    """Mission configuration loaded from YAML."""

    mission_name: str = "default_mission"

    # Home/dock position
    home_latitude: float = 47.397742
    home_longitude: float = 8.545594
    home_altitude_m: float = 488.0

    # Targets
    targets: list[dict] = field(default_factory=list)

    # Obstacles
    obstacles: list[dict] = field(default_factory=list)

    # Flight parameters
    cruise_velocity_ms: float = 5.0
    inspection_velocity_ms: float = 2.0
    return_altitude_agl: float = 30.0

    # Battery parameters
    battery_reserve_percent: float = 20.0
    consumption_rate_percent_per_m: float = 0.01  # % per meter flown


@dataclass
class MissionPlannerConfig:
    """Configuration for mission planner."""

    # Route optimization
    optimize_route: bool = True
    optimization_method: str = "nearest_neighbor"  # nearest_neighbor, genetic, etc.

    # Path planning
    path_planner_config: PathPlannerConfig = field(default_factory=PathPlannerConfig)

    # Mission constraints
    max_flight_time_s: float = 1800.0  # 30 minutes
    max_distance_m: float = 5000.0

    # Navigation map usage
    use_navigation_map: bool = True
    map_stale_threshold_s: float = 60.0
    map_min_quality_score: float = 0.3


class MissionPlanner:
    """Plans and manages inspection missions.

    Loads mission configuration from YAML, optimizes inspection routes,
    and provides mission waypoints for the flight controller.

    Example:
        planner = MissionPlanner(config)
        planner.load_mission("configs/mission_config.yaml")

        plan = planner.create_plan()
        for wp in plan.waypoints:
            await flight_controller.execute_waypoint(wp)
    """

    def __init__(self, config: MissionPlannerConfig | None = None) -> None:
        """Initialize mission planner.

        Args:
            config: Planner configuration
        """
        self._config = config or MissionPlannerConfig()
        self._mission_config: MissionConfig | None = None
        self._geo_ref: GeoReference | None = None
        self._path_planner: PathPlanner | None = None
        self._current_plan: MissionPlan | None = None
        self._navigation_map: dict[str, Any] | None = None

    def set_navigation_map(self, nav_map: dict[str, Any] | None) -> None:
        """Update the navigation map used for obstacle-aware planning."""
        self._navigation_map = nav_map
        if self._path_planner:
            self._apply_navigation_map()

    def _apply_navigation_map(self) -> MapContext | None:
        """Apply navigation map obstacles to the path planner if valid."""
        if not self._path_planner or not self._navigation_map:
            return None

        map_context = MapContext.from_navigation_map(
            self._navigation_map,
            stale_threshold_s=self._config.map_stale_threshold_s,
            min_quality_score=self._config.map_min_quality_score,
        )

        self._path_planner.clear_obstacles()
        self._apply_mission_obstacles()

        if not map_context.map_valid:
            logger.info(
                "navigation_map_skipped",
                map_age_s=map_context.map_age_s,
                quality=map_context.map_quality_score,
            )
            return map_context

        loaded = self._path_planner.load_obstacles_from_map(self._navigation_map)
        logger.info("navigation_map_applied", obstacle_count=loaded)
        return map_context

    def refresh_navigation_map(self) -> MapContext | None:
        """Re-evaluate map validity and reapply obstacles."""
        if self._navigation_map:
            return self._apply_navigation_map()
        return None

    def load_mission(self, config_path: str | Path) -> bool:
        """Load mission configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            True if loaded successfully
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.error(f"Mission config not found: {path}")
                return False

            with open(path) as f:
                data = yaml.safe_load(f)

            self._mission_config = self._parse_config(data)

            # Set up geo reference
            self._geo_ref = GeoReference(
                latitude=self._mission_config.home_latitude,
                longitude=self._mission_config.home_longitude,
                altitude=self._mission_config.home_altitude_m,
            )

            # Set up path planner
            self._path_planner = PathPlanner(
                geo_reference=self._geo_ref,
                config=self._config.path_planner_config,
            )

            self._apply_mission_obstacles()

            logger.info(f"Loaded mission: {self._mission_config.mission_name}")
            logger.info(f"  Targets: {len(self._mission_config.targets)}")
            logger.info(f"  Obstacles: {len(self._mission_config.obstacles)}")

            return True

        except Exception as e:
            logger.error(f"Failed to load mission config: {e}")
            return False

    def load_mission_dict(self, config: dict) -> bool:
        """Load mission configuration from dictionary.

        Args:
            config: Mission configuration dictionary

        Returns:
            True if loaded successfully
        """
        try:
            self._mission_config = self._parse_config(config)

            # Set up geo reference
            self._geo_ref = GeoReference(
                latitude=self._mission_config.home_latitude,
                longitude=self._mission_config.home_longitude,
                altitude=self._mission_config.home_altitude_m,
            )

            # Set up path planner
            self._path_planner = PathPlanner(
                geo_reference=self._geo_ref,
                config=self._config.path_planner_config,
            )

            self._apply_mission_obstacles()

            logger.info(f"Loaded mission from dict: {self._mission_config.mission_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load mission config: {e}")
            return False

    def _parse_config(self, data: dict) -> MissionConfig:
        """Parse configuration dictionary into MissionConfig."""
        mission = data.get("mission", {})
        home = data.get("home", {})

        return MissionConfig(
            mission_name=mission.get("name", "default_mission"),
            home_latitude=home.get("latitude", 47.397742),
            home_longitude=home.get("longitude", 8.545594),
            home_altitude_m=home.get("altitude_m", 488.0),
            targets=data.get("assets", []),
            obstacles=data.get("obstacles", []),
            cruise_velocity_ms=data.get("flight", {}).get("cruise_velocity_ms", 5.0),
            inspection_velocity_ms=data.get("flight", {}).get("inspection_velocity_ms", 2.0),
            return_altitude_agl=data.get("flight", {}).get("return_altitude_agl", 30.0),
            battery_reserve_percent=data.get("battery", {}).get("reserve_percent", 20.0),
        )

    def _apply_mission_obstacles(self) -> None:
        """Apply mission-defined obstacles to the path planner."""
        if not self._path_planner or not self._mission_config:
            return
        for obs_data in self._mission_config.obstacles:
            self._path_planner.add_obstacle_gps(
                latitude=obs_data.get("latitude", 0),
                longitude=obs_data.get("longitude", 0),
                radius_m=obs_data.get("radius_m", 20),
                height_m=obs_data.get("height_m", 30),
                obstacle_id=obs_data.get("id", ""),
                name=obs_data.get("name", ""),
            )

    def create_plan(
        self,
        current_position: Position | None = None,
        battery_percent: float = 100.0,
        exclude_target_ids: set[str] | None = None,
    ) -> MissionPlan | None:
        """Create optimized mission plan.

        Args:
            current_position: Current vehicle position (uses home if None)
            battery_percent: Current battery level

        Returns:
            MissionPlan with ordered waypoints, or None if planning failed
        """
        if not self._mission_config:
            logger.error("No mission configuration loaded")
            return None

        config = self._mission_config

        if self._config.use_navigation_map:
            self._apply_navigation_map()

        # Parse targets
        targets = []
        for t in config.targets:
            pos = t.get("position", {})
            insp = t.get("inspection", {})

            target = InspectionTarget(
                target_id=t.get("id", f"target_{len(targets)}"),
                name=t.get("name", "Unknown"),
                target_type=t.get("type", "other"),
                latitude=pos.get("latitude", config.home_latitude),
                longitude=pos.get("longitude", config.home_longitude),
                altitude_m=pos.get("altitude_m", config.home_altitude_m),
                inspection_altitude_agl=insp.get("altitude_agl_m", 20),
                orbit_radius_m=insp.get("orbit_radius_m", 20),
                dwell_time_s=insp.get("dwell_time_s", 30),
                priority=t.get("priority", 1),
            )
            targets.append(target)

        if exclude_target_ids:
            targets = [t for t in targets if t.target_id not in exclude_target_ids]

        if not targets:
            logger.warning("No targets in mission")

        # Optimize route order
        if self._config.optimize_route and len(targets) > 1:
            targets = self._optimize_route(targets, current_position)

        # Generate waypoints
        waypoints = []
        total_distance = 0.0
        sequence = 0

        # Start position
        start_lat = current_position.latitude if current_position else config.home_latitude
        start_lon = current_position.longitude if current_position else config.home_longitude
        start_alt = (
            current_position.altitude_msl if current_position else config.home_altitude_m + 30
        )

        last_lat, last_lon, last_alt = start_lat, start_lon, start_alt

        # Add takeoff waypoint if at home
        if not current_position:
            waypoints.append(
                MissionWaypoint(
                    waypoint_id=f"wp_{sequence}",
                    latitude=config.home_latitude,
                    longitude=config.home_longitude,
                    altitude_msl=config.home_altitude_m + config.return_altitude_agl,
                    action="takeoff",
                    sequence=sequence,
                )
            )
            sequence += 1
            last_alt = config.home_altitude_m + config.return_altitude_agl

        # Add waypoints for each target
        for target in targets:
            # Inspection altitude (MSL)
            inspection_alt = target.altitude_m + target.inspection_altitude_agl

            # Plan path to target (using path planner if available)
            if self._path_planner and self._geo_ref:
                path = self._path_planner.plan_path_gps(
                    last_lat,
                    last_lon,
                    last_alt,
                    target.latitude,
                    target.longitude,
                    inspection_alt,
                    velocity=config.cruise_velocity_ms,
                )

                # Add intermediate waypoints if path has avoidance maneuvers
                if path.avoidance_maneuvers > 0:
                    for i, wp_ned in enumerate(path.waypoints[1:-1]):
                        wp_lat, wp_lon, wp_alt = self._geo_ref.ned_to_gps(*wp_ned)
                        waypoints.append(
                            MissionWaypoint(
                                waypoint_id=f"wp_{sequence}",
                                latitude=wp_lat,
                                longitude=wp_lon,
                                altitude_msl=wp_alt,
                                action="flyover",
                                sequence=sequence,
                            )
                        )
                        sequence += 1

                total_distance += path.total_distance_m
            else:
                # Simple distance calculation
                distance = self._haversine_distance(
                    last_lat, last_lon, target.latitude, target.longitude
                )
                total_distance += distance

            # Add inspection waypoint
            waypoints.append(
                MissionWaypoint(
                    waypoint_id=f"wp_{sequence}",
                    target_id=target.target_id,
                    latitude=target.latitude,
                    longitude=target.longitude,
                    altitude_msl=inspection_alt,
                    action="inspect",
                    dwell_time_s=target.dwell_time_s,
                    sequence=sequence,
                )
            )
            sequence += 1

            last_lat, last_lon, last_alt = target.latitude, target.longitude, inspection_alt

        # Add return to home
        return_alt = config.home_altitude_m + config.return_altitude_agl

        waypoints.append(
            MissionWaypoint(
                waypoint_id=f"wp_{sequence}",
                latitude=config.home_latitude,
                longitude=config.home_longitude,
                altitude_msl=return_alt,
                action="flyover",
                sequence=sequence,
            )
        )
        sequence += 1

        # Add distance for return
        total_distance += self._haversine_distance(
            last_lat, last_lon, config.home_latitude, config.home_longitude
        )

        # Add landing waypoint
        waypoints.append(
            MissionWaypoint(
                waypoint_id=f"wp_{sequence}",
                latitude=config.home_latitude,
                longitude=config.home_longitude,
                altitude_msl=config.home_altitude_m,
                action="land",
                sequence=sequence,
            )
        )

        # Calculate estimates
        flight_time = total_distance / config.cruise_velocity_ms
        battery_used = total_distance * config.consumption_rate_percent_per_m

        # Add dwell time
        for wp in waypoints:
            flight_time += wp.dwell_time_s

        plan = MissionPlan(
            mission_id=f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mission_name=config.mission_name,
            created_at=datetime.now(),
            home_latitude=config.home_latitude,
            home_longitude=config.home_longitude,
            home_altitude_msl=config.home_altitude_m,
            waypoints=waypoints,
            targets=targets,
            total_distance_m=total_distance,
            estimated_flight_time_s=flight_time,
            estimated_battery_percent=battery_used,
        )

        self._current_plan = plan
        logger.info(
            f"Created mission plan: {len(waypoints)} waypoints, "
            f"{total_distance:.0f}m, ~{flight_time / 60:.1f} min"
        )

        return plan

    def _optimize_route(
        self,
        targets: list[InspectionTarget],
        start_position: Position | None = None,
    ) -> list[InspectionTarget]:
        """Optimize target visit order using nearest neighbor heuristic.

        Args:
            targets: List of targets to optimize
            start_position: Starting position (uses home if None)

        Returns:
            Optimized list of targets
        """
        if not targets:
            return targets

        config = self._mission_config
        if not config:
            return targets

        # Start from current position or home
        if start_position:
            current_lat = start_position.latitude
            current_lon = start_position.longitude
        else:
            current_lat = config.home_latitude
            current_lon = config.home_longitude

        # Sort by priority first, then optimize within priority groups
        targets_by_priority: dict[int, list[InspectionTarget]] = {}
        for t in targets:
            if t.priority not in targets_by_priority:
                targets_by_priority[t.priority] = []
            targets_by_priority[t.priority].append(t)

        optimized = []
        for priority in sorted(targets_by_priority.keys()):
            remaining = list(targets_by_priority[priority])

            # Nearest neighbor within this priority group
            while remaining:
                nearest = min(
                    remaining,
                    key=lambda t: self._haversine_distance(
                        current_lat, current_lon, t.latitude, t.longitude
                    ),
                )
                optimized.append(nearest)
                remaining.remove(nearest)
                current_lat = nearest.latitude
                current_lon = nearest.longitude

        return optimized

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS points."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @property
    def current_plan(self) -> MissionPlan | None:
        """Get the current mission plan."""
        return self._current_plan

    @property
    def geo_reference(self) -> GeoReference | None:
        """Get the geo reference for this mission."""
        return self._geo_ref

    @property
    def path_planner(self) -> PathPlanner | None:
        """Get the path planner instance."""
        return self._path_planner


def load_mission_config(config_path: str | Path) -> MissionConfig | None:
    """Convenience function to load mission config from YAML.

    Args:
        config_path: Path to YAML file

    Returns:
        MissionConfig or None if loading failed
    """
    try:
        path = Path(config_path)
        with open(path) as f:
            data = yaml.safe_load(f)

        mission = data.get("mission", {})
        home = data.get("home", {})

        return MissionConfig(
            mission_name=mission.get("name", "default"),
            home_latitude=home.get("latitude", 47.397742),
            home_longitude=home.get("longitude", 8.545594),
            home_altitude_m=home.get("altitude_m", 488.0),
            targets=data.get("assets", []),
            obstacles=data.get("obstacles", []),
        )
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None
