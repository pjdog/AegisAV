"""Standalone SLAM runner for capture bundles."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mapping.capture_replay import CaptureReplay, ReplayConfig
from mapping.keyframe_selector import KeyframeSelectionConfig, KeyframeSelector
from mapping.point_cloud import apply_pose, depth_to_points, euler_deg_to_quaternion, write_ply

logger = logging.getLogger(__name__)


def _build_keyframe_config(args: argparse.Namespace) -> KeyframeSelectionConfig:
    base = KeyframeSelectionConfig()
    return KeyframeSelectionConfig(
        min_time_interval_s=args.min_time_interval_s or base.min_time_interval_s,
        max_time_interval_s=args.max_time_interval_s or base.max_time_interval_s,
        min_translation_m=args.min_translation_m or base.min_translation_m,
        min_rotation_deg=args.min_rotation_deg or base.min_rotation_deg,
        velocity_threshold_ms=args.velocity_threshold_ms or base.velocity_threshold_ms,
        blur_threshold=args.blur_threshold or base.blur_threshold,
        min_feature_count=args.min_feature_count or base.min_feature_count,
        overlap_target=args.overlap_target or base.overlap_target,
    )


def _select_keyframes(frames: list[Any], config: KeyframeSelectionConfig) -> dict[int, str]:
    selector = KeyframeSelector(config)
    keyframes: dict[int, str] = {}

    for idx, frame in enumerate(frames):
        result = selector.check_keyframe(idx, frame.pose)
        if result.is_keyframe:
            frame_id = frame.frame_id or f"{idx:06d}"
            keyframes[idx] = frame_id

    if not keyframes and frames:
        keyframes[0] = frames[0].frame_id or "000000"

    return keyframes


def _pose_graph_entry(frame: Any, is_keyframe: bool, reason: str | None) -> dict[str, Any]:
    entry = {
        "frame_id": frame.frame_id or "",
        "timestamp_s": frame.timestamp_s,
        "pose": {
            "x": frame.pose.x,
            "y": frame.pose.y,
            "z": frame.pose.z,
            "roll_deg": frame.pose.roll_deg,
            "pitch_deg": frame.pose.pitch_deg,
            "yaw_deg": frame.pose.yaw_deg,
            "vx": frame.pose.vx,
            "vy": frame.pose.vy,
            "vz": frame.pose.vz,
        },
        "camera_pose": frame.camera_pose,
        "intrinsics": {
            "fx": frame.fx,
            "fy": frame.fy,
            "cx": frame.cx,
            "cy": frame.cy,
            "width": frame.width,
            "height": frame.height,
        },
        "image_path": str(frame.image_path) if frame.image_path else None,
        "depth_path": str(frame.depth_path) if frame.depth_path else None,
        "is_keyframe": is_keyframe,
        "keyframe_reason": reason,
    }
    return entry


def _collect_point_cloud(
    frames: list[Any],
    keyframe_indices: list[int],
    max_points: int,
    subsample: int,
) -> np.ndarray:
    if not keyframe_indices:
        return np.empty((0, 3), dtype=np.float32)

    points: list[np.ndarray] = []
    max_per_frame = max(500, int(math.ceil(max_points / max(1, len(keyframe_indices)))))

    for idx in keyframe_indices:
        frame = frames[idx]
        if not frame.depth_path:
            continue

        try:
            depth = np.load(frame.depth_path)
        except Exception as exc:
            logger.warning("depth_load_failed: %s", exc)
            continue

        intrinsics = {
            "fx": frame.fx,
            "fy": frame.fy,
            "cx": frame.cx,
            "cy": frame.cy,
        }
        cam_points = depth_to_points(
            depth, intrinsics, subsample=subsample, max_points=max_per_frame
        )
        if cam_points.size == 0:
            continue

        camera_pose = frame.camera_pose or {}
        position = camera_pose.get("position")
        orientation = camera_pose.get("orientation")
        if not position or not orientation:
            orientation = euler_deg_to_quaternion(
                frame.pose.roll_deg,
                frame.pose.pitch_deg,
                frame.pose.yaw_deg,
            )
            position = {"x": frame.pose.x, "y": frame.pose.y, "z": frame.pose.z}

        world_points = apply_pose(cam_points, position, orientation)
        points.append(world_points)

        if points and sum(p.shape[0] for p in points) >= max_points:
            break

    if not points:
        return np.empty((0, 3), dtype=np.float32)

    combined = np.vstack(points)
    if combined.shape[0] > max_points:
        step = int(math.ceil(combined.shape[0] / max_points))
        combined = combined[::step]

    return combined.astype(np.float32)


def _run_external_backend(backend: str, input_dir: Path, output_dir: Path) -> bool:
    env_var = {
        "orb_slam3": "ORB_SLAM3_CMD",
        "vins_fusion": "VINS_FUSION_CMD",
    }.get(backend)

    if not env_var:
        return False

    cmd_template = os.environ.get(env_var)
    if not cmd_template:
        logger.error("SLAM backend command not set: %s", env_var)
        return False

    cmd_rendered = cmd_template.format(input_dir=str(input_dir), output_dir=str(output_dir))
    cmd = shlex.split(cmd_rendered)
    logger.info("SLAM backend command: %s", cmd)
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        logger.error("SLAM backend failed: %s", exc)
        return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _count_points_in_cloud(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    if path.suffix.lower() == ".npy":
        data = np.load(path)
        return int(data.shape[0]) if data.ndim >= 2 else 0
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        points = data.get("points")
        return int(points.shape[0]) if isinstance(points, np.ndarray) else 0
    if path.suffix.lower() == ".ply":
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("element vertex"):
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            return int(parts[2])
                    if line.strip() == "end_header":
                        break
        except Exception:
            return 0
    return 0


def _build_status_from_pose_graph(
    pose_graph_path: Path,
    backend: str,
    point_cloud_path: Path | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    data = _read_json(pose_graph_path)
    frames = data.get("frames", [])
    keyframes = data.get("keyframes", [])
    keyframe_count = (
        len(keyframes) if isinstance(keyframes, list) else int(data.get("keyframe_count", 0))
    )
    frame_count = int(data.get("frame_count", len(frames)))
    map_point_count = _count_points_in_cloud(point_cloud_path)

    status = {
        "enabled": True,
        "running": False,
        "backend": backend,
        "tracking_state": "complete",
        "keyframe_count": keyframe_count,
        "map_point_count": map_point_count,
        "loop_closure_count": int(metrics.get("loop_closure_count", 0)),
        "pose_confidence": float(metrics.get("pose_confidence", 1.0)),
        "reprojection_error": float(metrics.get("reprojection_error", 0.0)),
        "drift_estimate_m": float(metrics.get("drift_estimate_m", 0.0)),
        "last_frame_ms": float(metrics.get("last_frame_ms", 0.0)),
        "avg_frame_ms": float(metrics.get("avg_frame_ms", 0.0)),
        "last_update": datetime.now().isoformat(),
        "frame_count": frame_count,
    }
    return status


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    if not output_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/slam_runs") / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = CaptureReplay.from_directory(input_dir, ReplayConfig(include_images=False))
    frames = replay.sequence.frames
    if not frames:
        logger.error("No frames found in %s", input_dir)
        return 1

    if args.backend != "telemetry":
        ran = _run_external_backend(args.backend, input_dir, output_dir)
        external_pose_graph = output_dir / "pose_graph.json"
        external_status = output_dir / "slam_status.json"
        external_metrics = output_dir / "slam_metrics.json"
        allow_fallback = getattr(args, "allow_telemetry_fallback", True)

        if ran and external_pose_graph.exists():
            metrics = _read_json(external_metrics) if external_metrics.exists() else {}
            if not external_status.exists():
                point_cloud = None
                if isinstance(data := _read_json(external_pose_graph), dict):
                    candidate = data.get("point_cloud")
                    if candidate:
                        point_cloud = Path(candidate)
                        if not point_cloud.is_absolute():
                            point_cloud = external_pose_graph.parent / point_cloud
                status = _build_status_from_pose_graph(
                    external_pose_graph, args.backend, point_cloud, metrics
                )
                external_status.write_text(json.dumps(status, indent=2))
            logger.info("External SLAM output detected, skipping telemetry fallback.")
            return 0

        if not allow_fallback:
            logger.error("External SLAM output missing; telemetry fallback disabled.")
            return 2

        logger.warning("External SLAM output missing, falling back to telemetry.")

    keyframe_config = _build_keyframe_config(args)
    keyframes = _select_keyframes(frames, keyframe_config)
    keyframe_indices = sorted(keyframes.keys())

    frame_entries: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        is_keyframe = idx in keyframes
        frame_entries.append(
            _pose_graph_entry(frame, is_keyframe, "selected" if is_keyframe else None)
        )

    point_cloud_path = None
    point_count = 0
    if not args.no_pointcloud:
        points = _collect_point_cloud(
            frames, keyframe_indices, args.max_points, args.depth_subsample
        )
        point_count = points.shape[0]
        point_cloud_path = output_dir / "map_points.ply"
        write_ply(points, point_cloud_path)

    pose_graph = {
        "format_version": 1,
        "backend": args.backend,
        "generated_at": datetime.now().isoformat(),
        "sequence_id": replay.sequence.sequence_id,
        "frame_count": len(frames),
        "keyframe_count": len(keyframe_indices),
        "keyframes": [frames[idx].frame_id or f"{idx:06d}" for idx in keyframe_indices],
        "frames": frame_entries,
        "point_cloud": str(point_cloud_path) if point_cloud_path else None,
    }

    pose_graph_path = output_dir / "pose_graph.json"
    pose_graph_path.write_text(json.dumps(pose_graph, indent=2))

    status = {
        "enabled": True,
        "running": False,
        "backend": args.backend,
        "tracking_state": "complete",
        "keyframe_count": len(keyframe_indices),
        "map_point_count": point_count,
        "loop_closure_count": 0,
        "pose_confidence": 1.0,
        "reprojection_error": 0.0,
        "drift_estimate_m": 0.0,
        "last_frame_ms": 0.0,
        "avg_frame_ms": 0.0,
        "last_update": datetime.now().isoformat(),
    }
    status_path = output_dir / "slam_status.json"
    status_path.write_text(json.dumps(status, indent=2))

    logger.info("SLAM output written to %s", output_dir)
    logger.info("Pose graph: %s", pose_graph_path)
    if point_cloud_path:
        logger.info("Point cloud: %s (%s points)", point_cloud_path, point_count)
    logger.info("Status: %s", status_path)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SLAM backend on captured frames.")
    parser.add_argument("input_dir", help="Capture sequence directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for SLAM artifacts")
    parser.add_argument(
        "--backend",
        choices=["telemetry", "orb_slam3", "vins_fusion"],
        default="telemetry",
        help="SLAM backend selection (telemetry = pose from capture metadata)",
    )
    parser.add_argument(
        "--allow-telemetry-fallback",
        dest="allow_telemetry_fallback",
        action="store_true",
        default=True,
        help="Allow telemetry fallback when external backend output is missing",
    )
    parser.add_argument(
        "--no-telemetry-fallback",
        dest="allow_telemetry_fallback",
        action="store_false",
        help="Disable telemetry fallback for external backends",
    )
    parser.add_argument("--no-pointcloud", action="store_true", help="Disable sparse PLY output")
    parser.add_argument(
        "--max-points", type=int, default=200000, help="Maximum points in PLY output"
    )
    parser.add_argument("--depth-subsample", type=int, default=6, help="Depth image subsample step")

    parser.add_argument("--min-time-interval-s", type=float, default=None)
    parser.add_argument("--max-time-interval-s", type=float, default=None)
    parser.add_argument("--min-translation-m", type=float, default=None)
    parser.add_argument("--min-rotation-deg", type=float, default=None)
    parser.add_argument("--velocity-threshold-ms", type=float, default=None)
    parser.add_argument("--blur-threshold", type=float, default=None)
    parser.add_argument("--min-feature-count", type=int, default=None)
    parser.add_argument("--overlap-target", type=float, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
