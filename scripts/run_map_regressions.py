#!/usr/bin/env python3
"""Run lightweight map regressions on stored navigation maps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _evaluate_map(map_path: Path, criteria: dict[str, Any]) -> dict[str, Any]:
    if not map_path.exists():
        return {
            "map_path": str(map_path),
            "status": "missing",
            "quality": 0.0,
            "obstacle_count": 0,
            "tile_count": 0,
            "failures": ["map_missing"],
        }

    data = _load_json(map_path)
    metadata = data.get("metadata") or {}
    obstacles = data.get("obstacles") or []
    tiles = data.get("tiles") or []

    quality = float(metadata.get("map_quality_score", 0.0) or 0.0)
    obstacle_count = len(obstacles)
    tile_count = len(tiles)

    failures: list[str] = []
    min_quality = float(criteria.get("min_quality_score", 0.3))
    min_obstacles = int(criteria.get("min_obstacles", 0))
    require_tiles = bool(criteria.get("require_tiles", False))
    require_bounds = bool(criteria.get("require_bounds", True))

    if quality < min_quality:
        failures.append("quality_below_threshold")
    if obstacle_count < min_obstacles:
        failures.append("insufficient_obstacles")
    if require_tiles and tile_count == 0:
        failures.append("tiles_missing")
    if require_bounds:
        bounds = (
            metadata.get("bounds_min_x"),
            metadata.get("bounds_max_x"),
            metadata.get("bounds_min_y"),
            metadata.get("bounds_max_y"),
        )
        if any(b is None for b in bounds):
            failures.append("bounds_missing")

    status = "pass" if not failures else "fail"
    return {
        "map_path": str(map_path),
        "status": status,
        "quality": quality,
        "obstacle_count": obstacle_count,
        "tile_count": tile_count,
        "failures": failures,
    }


def _resolve_maps(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
    tests: list[tuple[Path, dict[str, Any]]] = []
    if args.manifest:
        manifest = _load_json(Path(args.manifest))
        for entry in manifest.get("tests", []):
            path = Path(entry.get("map_path", ""))
            if not path.is_absolute():
                path = Path.cwd() / path
            tests.append((path, entry))
        return tests

    criteria = {
        "min_quality_score": args.min_quality_score,
        "min_obstacles": args.min_obstacles,
        "require_tiles": args.require_tiles,
        "require_bounds": args.require_bounds,
    }

    for raw in args.map_path or []:
        path = Path(raw)
        tests.append((path, criteria))

    if args.glob_pattern:
        for path in sorted(Path(".").glob(args.glob_pattern)):
            tests.append((path, criteria))

    return tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run map regression checks.")
    parser.add_argument("--map", dest="map_path", action="append", help="Navigation map JSON path")
    parser.add_argument(
        "--glob", dest="glob_pattern", default=None, help="Glob for navigation_map.json files"
    )
    parser.add_argument("--manifest", default=None, help="Manifest JSON for regression tests")
    parser.add_argument("--min-quality-score", type=float, default=0.3)
    parser.add_argument("--min-obstacles", type=int, default=0)
    parser.add_argument("--require-tiles", action="store_true", default=False)
    parser.add_argument("--require-bounds", action="store_true", default=True)
    parser.add_argument("--output-dir", default="metrics/map_regressions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tests = _resolve_maps(args)
    if not tests:
        print("No maps provided.")
        return 1

    results = []
    for map_path, criteria in tests:
        results.append(_evaluate_map(map_path, criteria))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"regression_{stamp}.json"
    csv_path = output_dir / f"regression_{stamp}.csv"

    json_path.write_text(json.dumps({"results": results}, indent=2))
    csv_path.write_text(
        "map_path,status,quality,obstacle_count,tile_count,failures\n"
        + "\n".join(
            f"{r['map_path']},{r['status']},{r['quality']},"
            f"{r['obstacle_count']},{r['tile_count']},"
            f"{'|'.join(r['failures'])}"
            for r in results
        )
        + "\n"
    )

    failures = [r for r in results if r["status"] != "pass"]
    if failures:
        print(f"Regression failures: {len(failures)}")
        return 2

    print(f"Regression report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
