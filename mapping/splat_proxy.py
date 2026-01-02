"""CLI to convert Gaussian splat previews into planning-grade navigation maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import structlog

from mapping.map_fusion import MapFusion, MapFusionConfig

logger = structlog.get_logger(__name__)


def _resolve_preview_from_scene(scene_path: Path) -> tuple[Path | None, dict[str, Any]]:
    data: dict[str, Any] = {}
    try:
        with open(scene_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("scene_load_failed", path=str(scene_path), error=str(exc))
        return None, {}

    preview = data.get("preview_point_cloud") or data.get("preview")
    if not preview:
        files = data.get("files") or {}
        preview = files.get("preview")

    preview_path = None
    if preview:
        preview_path = Path(preview)
        if not preview_path.is_absolute():
            preview_path = scene_path.parent / preview_path
    return preview_path, data


def _resolve_preview_path(
    scene_path: Path | None,
    preview_path: Path | None,
) -> tuple[Path | None, dict[str, Any]]:
    if preview_path:
        return preview_path, {}
    if not scene_path:
        return None, {}
    return _resolve_preview_from_scene(scene_path)


def write_planning_proxy(
    nav_map: dict[str, Any],
    output_dir: Path,
    *,
    filename: str = "planning_proxy.json",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(nav_map, indent=2))
    return output_path


def build_planning_proxy(
    preview_path: Path,
    output_dir: Path,
    *,
    map_id: str,
    scenario_id: str | None,
    source: str,
    splat_quality: float,
    config: MapFusionConfig | None = None,
    output_name: str = "planning_proxy.json",
) -> tuple[dict[str, Any], Path]:
    fusion = MapFusion(config or MapFusionConfig())
    result = fusion.build_navigation_map(
        point_cloud_path=preview_path,
        map_id=map_id,
        scenario_id=scenario_id,
        source=source,
        slam_confidence=1.0,
        splat_quality=splat_quality,
    )
    output_path = write_planning_proxy(result.navigation_map, output_dir, filename=output_name)
    return result.navigation_map, output_path


def run(args: argparse.Namespace) -> int:
    scene_path = Path(args.scene) if args.scene else None
    preview_path = Path(args.preview) if args.preview else None

    preview_path, scene_data = _resolve_preview_path(scene_path, preview_path)
    if not preview_path or not preview_path.exists():
        logger.error("preview_missing", path=str(preview_path) if preview_path else None)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    if not output_dir:
        output_dir = scene_path.parent if scene_path else Path("data/navigation_maps")

    map_id = args.map_id or scene_data.get("run_id") or scene_data.get("scene_id") or "splat_proxy"

    nav_map, output_path = build_planning_proxy(
        preview_path,
        output_dir,
        map_id=map_id,
        scenario_id=args.scenario_id,
        source=args.source,
        splat_quality=args.splat_quality,
        config=MapFusionConfig(
            resolution_m=args.resolution_m,
            tile_size_cells=args.tile_size_cells,
            voxel_size_m=args.voxel_size_m,
            min_points=args.min_points,
            max_points=args.max_points,
        ),
        output_name="navigation_map.json",
    )
    logger.info(
        "splat_proxy_complete",
        output_path=str(output_path),
        obstacles=len(nav_map.get("obstacles", [])),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Gaussian splat preview into a navigation map proxy."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", help="Path to scene.json for a splat run")
    group.add_argument("--preview", help="Path to a preview point cloud (.ply/.npy/.npz)")

    parser.add_argument("--output-dir", default=None, help="Output directory for navigation_map.json")
    parser.add_argument("--map-id", default=None, help="Override map_id in metadata")
    parser.add_argument("--scenario-id", default=None, help="Scenario ID to attach to navigation map")
    parser.add_argument("--source", default="splat_proxy", help="Source label for map metadata")
    parser.add_argument("--splat-quality", type=float, default=1.0, help="Quality score override")

    parser.add_argument("--resolution-m", type=float, default=2.0, help="Map resolution in meters")
    parser.add_argument("--tile-size-cells", type=int, default=120, help="Occupancy tile size")
    parser.add_argument("--voxel-size-m", type=float, default=None, help="Voxel size in meters")
    parser.add_argument("--min-points", type=int, default=50, help="Minimum points required")
    parser.add_argument("--max-points", type=int, default=200000, help="Maximum points used")

    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
