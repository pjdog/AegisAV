#!/usr/bin/env python3
"""Run map planning benchmarks over one or more point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from mapping.benchmark_runner import run as run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run map planning benchmarks.")
    parser.add_argument("--point-cloud", action="append", help="Path to a point cloud")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help="Glob for point clouds")
    parser.add_argument("--sequence-dir", default=None, help="Capture sequence for start/goal")
    parser.add_argument("--start", default=None, help="Start NED 'north,east,down'")
    parser.add_argument("--goal", default=None, help="Goal NED 'north,east,down'")
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--map-resolution-m", type=float, default=2.0)
    parser.add_argument("--output-dir", default="metrics/map_benchmarks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    point_clouds = []
    if args.point_cloud:
        point_clouds.extend([Path(p) for p in args.point_cloud])
    if args.glob_pattern:
        point_clouds.extend(sorted(Path(".").glob(args.glob_pattern)))

    if not point_clouds:
        print("No point clouds provided.")
        return 1

    for cloud in point_clouds:
        ns = SimpleNamespace(
            point_cloud=str(cloud),
            sequence_dir=args.sequence_dir,
            start=args.start,
            goal=args.goal,
            map_id=args.map_id,
            map_resolution_m=args.map_resolution_m,
            output_dir=args.output_dir,
            run_id=None,
        )
        run_benchmark(ns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
