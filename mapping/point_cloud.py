"""Point cloud helpers for SLAM and splat previews."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def euler_deg_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> dict[str, float]:
    """Convert roll/pitch/yaw (deg) to quaternion dict."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return {
        "w": cr * cp * cy + sr * sp * sy,
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
    }


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def depth_to_points(
    depth: np.ndarray,
    intrinsics: dict[str, float],
    subsample: int = 4,
    max_points: int | None = None,
    min_depth: float = 0.5,
    max_depth: float = 200.0,
) -> np.ndarray:
    """Project a depth map to camera-frame point cloud."""
    if depth.ndim != 2:
        raise ValueError("Depth must be a 2D array")

    height, width = depth.shape
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", width / 2.0))
    cy = float(intrinsics.get("cy", height / 2.0))
    if fx == 0.0 or fy == 0.0:
        return np.empty((0, 3), dtype=np.float32)

    ys = np.arange(0, height, subsample, dtype=np.int32)
    xs = np.arange(0, width, subsample, dtype=np.int32)
    xs_grid, ys_grid = np.meshgrid(xs, ys)

    z = depth[ys_grid, xs_grid].astype(np.float32)
    mask = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    xs_valid = xs_grid[mask].astype(np.float32)
    ys_valid = ys_grid[mask].astype(np.float32)
    z_valid = z[mask]

    x = (xs_valid - cx) / fx * z_valid
    y = (ys_valid - cy) / fy * z_valid

    points = np.stack([x, y, z_valid], axis=1)

    if max_points and points.shape[0] > max_points:
        step = int(math.ceil(points.shape[0] / max_points))
        points = points[::step]

    return points.astype(np.float32)


def apply_pose(
    points: np.ndarray, position: dict[str, float], orientation: dict[str, float]
) -> np.ndarray:
    """Transform points from camera frame into world frame."""
    if points.size == 0:
        return points

    rot = quaternion_to_rotation_matrix(
        float(orientation.get("w", 1.0)),
        float(orientation.get("x", 0.0)),
        float(orientation.get("y", 0.0)),
        float(orientation.get("z", 0.0)),
    )
    rotated = points @ rot.T
    translation = np.array(
        [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)],
        dtype=np.float32,
    )
    return rotated + translation


def write_ply(points: Iterable[Iterable[float]], path: Path) -> None:
    """Write a point cloud to an ASCII PLY file."""
    point_list = list(points)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(point_list)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in point_list:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
