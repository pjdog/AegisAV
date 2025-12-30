"""Path Planner with A* Obstacle Avoidance.

Generates obstacle-free flight paths using A* search algorithm.
Supports both GPS and NED coordinate systems.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

from simulation.coordinate_utils import GeoReference

logger = logging.getLogger(__name__)


class PathPlannerStatus(Enum):
    """Status of path planning operation."""

    SUCCESS = "success"
    NO_PATH_FOUND = "no_path_found"
    START_IN_OBSTACLE = "start_in_obstacle"
    GOAL_IN_OBSTACLE = "goal_in_obstacle"
    INVALID_INPUT = "invalid_input"


@dataclass
class Obstacle:
    """Cylindrical obstacle for collision checking.

    Obstacles are modeled as vertical cylinders with a position,
    horizontal radius, and height above ground.
    """

    # Position (NED coordinates)
    north: float
    east: float
    ground_level: float = 0.0  # Down coordinate of ground (usually 0)

    # Dimensions
    radius_m: float = 20.0  # Horizontal radius
    height_m: float = 30.0  # Height above ground

    # Metadata
    obstacle_id: str = ""
    name: str = ""

    @property
    def top_down(self) -> float:
        """Down coordinate of obstacle top (negative = altitude)."""
        return self.ground_level - self.height_m

    def contains_point(self, north: float, east: float, down: float, margin: float = 0.0) -> bool:
        """Check if a point is inside the obstacle cylinder.

        Args:
            north: Point north coordinate
            east: Point east coordinate
            down: Point down coordinate (negative = altitude)
            margin: Safety margin in meters

        Returns:
            True if point is inside obstacle (with margin)
        """
        # Check horizontal distance
        dx = north - self.north
        dy = east - self.east
        horizontal_dist = math.sqrt(dx * dx + dy * dy)

        if horizontal_dist > self.radius_m + margin:
            return False

        # Check vertical (point is inside if it's below obstacle top and above ground)
        # down coordinate: more negative = higher altitude
        if down < self.top_down - margin:
            return False  # Above obstacle
        if down > self.ground_level + margin:
            return False  # Below ground

        return True

    def distance_to_surface(self, north: float, east: float, down: float) -> float:
        """Calculate minimum distance from point to obstacle surface.

        Args:
            north, east, down: Point coordinates

        Returns:
            Distance in meters (negative if inside obstacle)
        """
        # Horizontal distance to cylinder axis
        dx = north - self.north
        dy = east - self.east
        horiz_dist = math.sqrt(dx * dx + dy * dy)

        # Distance to cylinder surface
        horiz_to_surface = horiz_dist - self.radius_m

        # Vertical distance (negative down means higher altitude)
        if down < self.top_down:
            # Above obstacle
            vert_dist = self.top_down - down
        elif down > self.ground_level:
            # Below ground (shouldn't happen)
            vert_dist = down - self.ground_level
        else:
            # At obstacle height
            vert_dist = 0.0

        if horiz_to_surface > 0 and vert_dist > 0:
            # Outside both horizontally and vertically
            return math.sqrt(horiz_to_surface**2 + vert_dist**2)
        elif horiz_to_surface > 0:
            return horiz_to_surface
        elif vert_dist > 0:
            return vert_dist
        else:
            # Inside obstacle
            return max(horiz_to_surface, -vert_dist)


@dataclass
class Waypoint:
    """A waypoint in NED coordinates."""

    north: float
    east: float
    down: float

    def distance_to(self, other: Waypoint) -> float:
        """Euclidean distance to another waypoint."""
        return math.sqrt(
            (self.north - other.north) ** 2
            + (self.east - other.east) ** 2
            + (self.down - other.down) ** 2
        )

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple."""
        return (self.north, self.east, self.down)


class FlightPath(BaseModel):
    """Result of path planning - a sequence of waypoints."""

    waypoints: list[tuple[float, float, float]]  # List of (north, east, down)
    total_distance_m: float
    estimated_time_s: float
    avoidance_maneuvers: int
    status: str = "success"

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)

    def as_waypoints(self) -> list[Waypoint]:
        """Convert to list of Waypoint objects."""
        return [Waypoint(n, e, d) for n, e, d in self.waypoints]


@dataclass
class Node:
    """A* search node."""

    position: tuple[float, float, float]
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    parent: Node | None = None

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: Node) -> bool:
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.position == other.position

    def __hash__(self) -> int:
        return hash(self.position)


@dataclass
class PathPlannerConfig:
    """Configuration for path planner."""

    # Grid resolution for A* search
    grid_resolution_m: float = 5.0  # meters between grid points

    # Safety margins
    obstacle_margin_m: float = 5.0  # Extra margin around obstacles
    min_altitude_agl_m: float = 5.0  # Minimum altitude above ground
    max_altitude_agl_m: float = 120.0  # Maximum altitude (regulatory)

    # A* parameters
    max_iterations: int = 10000
    vertical_cost_factor: float = 1.5  # Penalize vertical movement

    # Path optimization
    simplify_path: bool = True
    simplify_tolerance_m: float = 2.0

    # Default cruise velocity
    default_velocity_ms: float = 5.0


class PathPlanner:
    """A* path planner with obstacle avoidance.

    Uses 3D A* search to find obstacle-free paths between waypoints.
    Supports cylindrical obstacles modeled with position, radius, and height.

    Example:
        planner = PathPlanner(geo_ref, config)
        planner.add_obstacle(Obstacle(north=100, east=50, radius_m=20, height_m=30))

        path = planner.plan_path(
            start=(0, 0, -30),  # NED: home at 30m altitude
            goal=(200, 100, -30),  # 200m north, 100m east, 30m altitude
        )

        for wp in path.waypoints:
            await backend.goto_position_ned(*wp)
    """

    def __init__(
        self,
        geo_reference: GeoReference | None = None,
        config: PathPlannerConfig | None = None,
    ) -> None:
        """Initialize path planner.

        Args:
            geo_reference: Geographic reference for GPS<->NED conversion
            config: Planner configuration
        """
        self._geo_ref = geo_reference
        self._config = config or PathPlannerConfig()
        self._obstacles: list[Obstacle] = []

    def set_geo_reference(self, geo_ref: GeoReference) -> None:
        """Set geographic reference for GPS conversions."""
        self._geo_ref = geo_ref

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the map."""
        self._obstacles.append(obstacle)
        logger.debug(f"Added obstacle: {obstacle.name} at ({obstacle.north:.1f}, {obstacle.east:.1f})")

    def add_obstacle_gps(
        self,
        latitude: float,
        longitude: float,
        radius_m: float,
        height_m: float,
        obstacle_id: str = "",
        name: str = "",
    ) -> None:
        """Add an obstacle using GPS coordinates.

        Args:
            latitude: Obstacle center latitude
            longitude: Obstacle center longitude
            radius_m: Horizontal radius
            height_m: Height above ground
            obstacle_id: Optional ID
            name: Optional name
        """
        if not self._geo_ref:
            logger.warning("No geo reference set, cannot add GPS obstacle")
            return

        north, east, _ = self._geo_ref.gps_to_ned(latitude, longitude, self._geo_ref.altitude)

        self.add_obstacle(
            Obstacle(
                north=north,
                east=east,
                radius_m=radius_m,
                height_m=height_m,
                obstacle_id=obstacle_id,
                name=name,
            )
        )

    def load_obstacles_from_map(self, nav_map: dict[str, Any]) -> int:
        """Load obstacles from a navigation map dict.

        Args:
            nav_map: Navigation map from build_navigation_map()

        Returns:
            Number of obstacles loaded
        """
        obstacles = nav_map.get("obstacles", [])
        count = 0

        for obs in obstacles:
            lat = obs.get("latitude")
            lon = obs.get("longitude")
            radius = obs.get("radius_m", 20.0)
            height = obs.get("height_m", 30.0)

            if lat and lon:
                self.add_obstacle_gps(
                    latitude=lat,
                    longitude=lon,
                    radius_m=radius,
                    height_m=height,
                    obstacle_id=obs.get("asset_id", ""),
                    name=obs.get("name", ""),
                )
                count += 1

        logger.info(f"Loaded {count} obstacles from navigation map")
        return count

    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        self._obstacles.clear()

    def is_point_safe(self, north: float, east: float, down: float) -> bool:
        """Check if a point is collision-free.

        Args:
            north, east, down: Point in NED coordinates

        Returns:
            True if point is safe (not inside any obstacle)
        """
        margin = self._config.obstacle_margin_m

        for obs in self._obstacles:
            if obs.contains_point(north, east, down, margin):
                return False

        # Check altitude constraints
        altitude_agl = -down
        if altitude_agl < self._config.min_altitude_agl_m:
            return False
        if altitude_agl > self._config.max_altitude_agl_m:
            return False

        return True

    def is_path_segment_safe(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        check_interval: float = 1.0,
    ) -> bool:
        """Check if a straight path segment is collision-free.

        Args:
            start: Start point (north, east, down)
            end: End point (north, east, down)
            check_interval: Distance between collision checks in meters

        Returns:
            True if entire segment is safe
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 0.1:
            return self.is_point_safe(*start)

        num_checks = max(2, int(distance / check_interval) + 1)

        for i in range(num_checks):
            t = i / (num_checks - 1)
            n = start[0] + t * dx
            e = start[1] + t * dy
            d = start[2] + t * dz

            if not self.is_point_safe(n, e, d):
                return False

        return True

    def plan_path(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        velocity: float | None = None,
    ) -> FlightPath:
        """Plan an obstacle-free path from start to goal.

        Args:
            start: Start position (north, east, down) in NED
            goal: Goal position (north, east, down) in NED
            velocity: Optional cruise velocity for time estimation

        Returns:
            FlightPath with waypoints and metrics
        """
        velocity = velocity or self._config.default_velocity_ms

        # Check start and goal validity
        if not self.is_point_safe(*start):
            logger.warning("Start position is inside an obstacle")
            return FlightPath(
                waypoints=[start],
                total_distance_m=0,
                estimated_time_s=0,
                avoidance_maneuvers=0,
                status=PathPlannerStatus.START_IN_OBSTACLE.value,
            )

        if not self.is_point_safe(*goal):
            logger.warning("Goal position is inside an obstacle")
            return FlightPath(
                waypoints=[start],
                total_distance_m=0,
                estimated_time_s=0,
                avoidance_maneuvers=0,
                status=PathPlannerStatus.GOAL_IN_OBSTACLE.value,
            )

        # Try direct path first
        if self.is_path_segment_safe(start, goal):
            distance = math.sqrt(
                (goal[0] - start[0]) ** 2
                + (goal[1] - start[1]) ** 2
                + (goal[2] - start[2]) ** 2
            )
            return FlightPath(
                waypoints=[start, goal],
                total_distance_m=distance,
                estimated_time_s=distance / velocity,
                avoidance_maneuvers=0,
                status=PathPlannerStatus.SUCCESS.value,
            )

        # Use A* search
        logger.info(f"Direct path blocked, running A* search...")
        path = self._astar_search(start, goal)

        if not path:
            logger.warning("No path found")
            return FlightPath(
                waypoints=[start],
                total_distance_m=0,
                estimated_time_s=0,
                avoidance_maneuvers=0,
                status=PathPlannerStatus.NO_PATH_FOUND.value,
            )

        # Simplify path
        if self._config.simplify_path:
            path = self._simplify_path(path)

        # Calculate metrics
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += math.sqrt(
                (path[i + 1][0] - path[i][0]) ** 2
                + (path[i + 1][1] - path[i][1]) ** 2
                + (path[i + 1][2] - path[i][2]) ** 2
            )

        avoidance_maneuvers = len(path) - 2  # Intermediate waypoints

        return FlightPath(
            waypoints=path,
            total_distance_m=total_distance,
            estimated_time_s=total_distance / velocity,
            avoidance_maneuvers=avoidance_maneuvers,
            status=PathPlannerStatus.SUCCESS.value,
        )

    def plan_path_gps(
        self,
        start_lat: float,
        start_lon: float,
        start_alt_msl: float,
        goal_lat: float,
        goal_lon: float,
        goal_alt_msl: float,
        velocity: float | None = None,
    ) -> FlightPath:
        """Plan path using GPS coordinates.

        Args:
            start_lat, start_lon, start_alt_msl: Start position GPS
            goal_lat, goal_lon, goal_alt_msl: Goal position GPS
            velocity: Optional cruise velocity

        Returns:
            FlightPath with NED waypoints
        """
        if not self._geo_ref:
            return FlightPath(
                waypoints=[],
                total_distance_m=0,
                estimated_time_s=0,
                avoidance_maneuvers=0,
                status=PathPlannerStatus.INVALID_INPUT.value,
            )

        start_ned = self._geo_ref.gps_to_ned(start_lat, start_lon, start_alt_msl)
        goal_ned = self._geo_ref.gps_to_ned(goal_lat, goal_lon, goal_alt_msl)

        return self.plan_path(start_ned, goal_ned, velocity)

    def _astar_search(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
    ) -> list[tuple[float, float, float]] | None:
        """Perform A* search for path.

        Args:
            start: Start position (north, east, down)
            goal: Goal position (north, east, down)

        Returns:
            List of waypoints or None if no path found
        """
        resolution = self._config.grid_resolution_m

        # Discretize start and goal
        start_grid = self._to_grid(start, resolution)
        goal_grid = self._to_grid(goal, resolution)

        # Initialize A*
        start_node = Node(
            position=start_grid,
            g_cost=0,
            h_cost=self._heuristic(start_grid, goal_grid),
        )

        open_set: list[Node] = [start_node]
        heapq.heapify(open_set)
        closed_set: set[tuple[float, float, float]] = set()
        node_map: dict[tuple[float, float, float], Node] = {start_grid: start_node}

        iterations = 0
        max_iterations = self._config.max_iterations

        while open_set and iterations < max_iterations:
            iterations += 1

            current = heapq.heappop(open_set)

            if current.position == goal_grid:
                # Found path, reconstruct it
                return self._reconstruct_path(current)

            closed_set.add(current.position)

            # Explore neighbors
            for neighbor_pos in self._get_neighbors(current.position, resolution):
                if neighbor_pos in closed_set:
                    continue

                # Check if position is safe
                if not self.is_point_safe(*neighbor_pos):
                    continue

                # Check if path to neighbor is safe
                if not self.is_path_segment_safe(current.position, neighbor_pos):
                    continue

                # Calculate costs
                move_cost = self._movement_cost(current.position, neighbor_pos)
                g_cost = current.g_cost + move_cost
                h_cost = self._heuristic(neighbor_pos, goal_grid)

                if neighbor_pos in node_map:
                    existing = node_map[neighbor_pos]
                    if g_cost < existing.g_cost:
                        existing.g_cost = g_cost
                        existing.parent = current
                        heapq.heapify(open_set)
                else:
                    neighbor_node = Node(
                        position=neighbor_pos,
                        g_cost=g_cost,
                        h_cost=h_cost,
                        parent=current,
                    )
                    heapq.heappush(open_set, neighbor_node)
                    node_map[neighbor_pos] = neighbor_node

        logger.warning(f"A* search exhausted after {iterations} iterations")
        return None

    def _to_grid(
        self, pos: tuple[float, float, float], resolution: float
    ) -> tuple[float, float, float]:
        """Snap position to grid."""
        return (
            round(pos[0] / resolution) * resolution,
            round(pos[1] / resolution) * resolution,
            round(pos[2] / resolution) * resolution,
        )

    def _heuristic(
        self, pos: tuple[float, float, float], goal: tuple[float, float, float]
    ) -> float:
        """A* heuristic (Euclidean distance with vertical penalty)."""
        horiz_dist = math.sqrt((goal[0] - pos[0]) ** 2 + (goal[1] - pos[1]) ** 2)
        vert_dist = abs(goal[2] - pos[2])
        return horiz_dist + vert_dist * self._config.vertical_cost_factor

    def _movement_cost(
        self,
        from_pos: tuple[float, float, float],
        to_pos: tuple[float, float, float],
    ) -> float:
        """Cost to move between adjacent cells."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dz = to_pos[2] - from_pos[2]

        horiz_dist = math.sqrt(dx * dx + dy * dy)
        vert_dist = abs(dz)

        return horiz_dist + vert_dist * self._config.vertical_cost_factor

    def _get_neighbors(
        self, pos: tuple[float, float, float], resolution: float
    ) -> list[tuple[float, float, float]]:
        """Get valid neighbor positions."""
        neighbors = []

        # 26-connected neighbors (3D grid)
        for dn in [-resolution, 0, resolution]:
            for de in [-resolution, 0, resolution]:
                for dd in [-resolution, 0, resolution]:
                    if dn == 0 and de == 0 and dd == 0:
                        continue

                    neighbor = (pos[0] + dn, pos[1] + de, pos[2] + dd)
                    neighbors.append(neighbor)

        return neighbors

    def _reconstruct_path(self, node: Node) -> list[tuple[float, float, float]]:
        """Reconstruct path from goal node back to start."""
        path = []
        current: Node | None = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return path

    def _simplify_path(
        self, path: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        """Simplify path by removing redundant waypoints.

        Uses visibility-based simplification: keep only waypoints
        where the previous segment direction changes significantly.
        """
        if len(path) <= 2:
            return path

        simplified = [path[0]]

        i = 0
        while i < len(path) - 1:
            # Try to skip as many waypoints as possible
            j = len(path) - 1
            while j > i + 1:
                if self.is_path_segment_safe(path[i], path[j]):
                    break
                j -= 1

            simplified.append(path[j])
            i = j

        return simplified
