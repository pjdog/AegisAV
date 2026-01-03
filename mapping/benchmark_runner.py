"""Simulation benchmark runner for map-driven planning."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from autonomy.path_planner import PathPlanner, PathPlannerConfig
from mapping.capture_replay import CaptureReplay, ReplayConfig
from mapping.map_fusion import MapFusion, MapFusionConfig

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark metrics for a single run."""

    run_id: str
    map_source: str
    obstacle_count: int
    baseline_distance_m: float
    baseline_safe: bool
    baseline_time_s: float
    map_distance_m: float
    map_safe: bool
    map_time_s: float
    avoidance_maneuvers: int
    replan_count: int
    delta_distance_m: float
    delta_time_s: float
    baseline_collision_risk: bool
    map_collision_risk: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "map_source": self.map_source,
            "obstacle_count": self.obstacle_count,
            "baseline_distance_m": self.baseline_distance_m,
            "baseline_safe": self.baseline_safe,
            "baseline_time_s": self.baseline_time_s,
            "map_distance_m": self.map_distance_m,
            "map_safe": self.map_safe,
            "map_time_s": self.map_time_s,
            "avoidance_maneuvers": self.avoidance_maneuvers,
            "replan_count": self.replan_count,
            "delta_distance_m": self.delta_distance_m,
            "delta_time_s": self.delta_time_s,
            "baseline_collision_risk": self.baseline_collision_risk,
            "map_collision_risk": self.map_collision_risk,
            "status": self.status,
        }


def _parse_point(text: str) -> tuple[float, float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Point must be 'north,east,down'")
    return (parts[0], parts[1], parts[2])


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = _parse_point(args.start) if args.start else None
    goal = _parse_point(args.goal) if args.goal else None

    if args.sequence_dir:
        replay = CaptureReplay.from_directory(
            Path(args.sequence_dir), ReplayConfig(include_images=False)
        )
        if replay.sequence.frames:
            start = start or (
                replay.sequence.frames[0].pose.x,
                replay.sequence.frames[0].pose.y,
                replay.sequence.frames[0].pose.z,
            )
            goal = goal or (
                replay.sequence.frames[-1].pose.x,
                replay.sequence.frames[-1].pose.y,
                replay.sequence.frames[-1].pose.z,
            )

    if start is None or goal is None:
        logger.error("start_goal_missing")
        return 1

    map_fusion = MapFusion(MapFusionConfig(resolution_m=args.map_resolution_m))
    nav_map = None
    map_source = "unknown"
    point_cloud = Path(args.point_cloud) if args.point_cloud else None

    if point_cloud:
        fusion = map_fusion.build_navigation_map(point_cloud, map_id=args.map_id)
        nav_map = fusion.navigation_map
        map_source = "point_cloud"

    if not nav_map:
        logger.error("navigation_map_missing")
        return 1

    planner_config = PathPlannerConfig()
    planner = PathPlanner(config=planner_config)
    planner.clear_obstacles()
    planner.load_obstacles_from_map(nav_map)

    baseline_distance = _distance(start, goal)
    baseline_safe = planner.is_path_segment_safe(start, goal)

    path = planner.plan_path(start, goal)
    map_distance = path.total_distance_m
    map_safe = path.status == "success"
    velocity_ms = max(0.1, float(planner_config.default_velocity_ms))
    baseline_time_s = baseline_distance / velocity_ms
    map_time_s = map_distance / velocity_ms

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result = BenchmarkResult(
        run_id=run_id,
        map_source=map_source,
        obstacle_count=len(nav_map.get("obstacles", [])),
        baseline_distance_m=baseline_distance,
        baseline_safe=baseline_safe,
        baseline_time_s=baseline_time_s,
        map_distance_m=map_distance,
        map_safe=map_safe,
        map_time_s=map_time_s,
        avoidance_maneuvers=path.avoidance_maneuvers,
        replan_count=path.avoidance_maneuvers,
        delta_distance_m=map_distance - baseline_distance,
        delta_time_s=map_time_s - baseline_time_s,
        baseline_collision_risk=not baseline_safe,
        map_collision_risk=not map_safe,
        status=path.status,
    )

    json_path = output_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2))

    csv_path = output_dir / f"{run_id}.csv"
    csv_path.write_text(
        "run_id,map_source,obstacle_count,baseline_distance_m,baseline_safe,baseline_time_s,"
        "map_distance_m,map_safe,map_time_s,avoidance_maneuvers,replan_count,delta_distance_m,delta_time_s,"
        "baseline_collision_risk,map_collision_risk,status\n"
        f"{run_id},{map_source},{result.obstacle_count},{baseline_distance},"
        f"{baseline_safe},{baseline_time_s},{map_distance},{map_safe},{map_time_s},"
        f"{path.avoidance_maneuvers},{result.replan_count},{result.delta_distance_m},{result.delta_time_s},"
        f"{result.baseline_collision_risk},{result.map_collision_risk},{path.status}\n"
    )

    logger.info("benchmark_complete", output=str(json_path))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run map planning benchmarks.")
    parser.add_argument("--point-cloud", required=True, help="Path to point cloud (PLY/NPY)")
    parser.add_argument(
        "--sequence-dir", default=None, help="Capture sequence directory for start/goal"
    )
    parser.add_argument("--start", default=None, help="Start NED 'north,east,down'")
    parser.add_argument("--goal", default=None, help="Goal NED 'north,east,down'")
    parser.add_argument("--map-id", default=None, help="Map identifier")
    parser.add_argument("--map-resolution-m", type=float, default=2.0)
    parser.add_argument("--output-dir", default="metrics/map_benchmarks")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
