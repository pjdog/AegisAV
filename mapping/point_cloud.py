"""Point cloud helpers for SLAM and splat previews."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import cv2
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
    distortion: list[float] | None = None,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Project a depth map to camera-frame point cloud.

    Args:
        depth: 2D depth map array (raw sensor values)
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
        subsample: Subsample factor for the depth map
        max_points: Maximum number of points to return
        min_depth: Minimum valid depth in meters (after scaling)
        max_depth: Maximum valid depth in meters (after scaling)
        distortion: Optional distortion coefficients [k1, k2, p1, p2, k3]
        depth_scale: Scale factor to convert raw depth to meters (e.g., 0.001 for mm)

    Returns:
        Nx3 array of 3D points in camera frame [X=right, Y=down, Z=forward]
    """
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

    # Apply depth scale to convert raw values to meters
    z = z * depth_scale

    mask = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    xs_valid = xs_grid[mask].astype(np.float32)
    ys_valid = ys_grid[mask].astype(np.float32)
    z_valid = z[mask]

    # Apply distortion correction if coefficients are provided and non-zero
    if distortion is not None and any(d != 0.0 for d in distortion):
        # Build camera matrix
        camera_matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Distortion coefficients [k1, k2, p1, p2, k3]
        dist_coeffs = np.array(distortion[:5], dtype=np.float64)

        # Stack pixel coordinates for undistortion
        # cv2.undistortPoints expects Nx1x2 array
        points_2d = np.stack([xs_valid, ys_valid], axis=1).reshape(-1, 1, 2)

        # Undistort points - returns normalized coordinates
        undistorted = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs)
        undistorted = undistorted.reshape(-1, 2)

        # undistortPoints returns normalized coords, so we just multiply by z
        x = undistorted[:, 0] * z_valid
        y = undistorted[:, 1] * z_valid
    else:
        # No distortion correction - use pinhole model directly
        x = (xs_valid - cx) / fx * z_valid
        y = (ys_valid - cy) / fy * z_valid

    points = np.stack([x, y, z_valid], axis=1)

    if max_points and points.shape[0] > max_points:
        step = int(math.ceil(points.shape[0] / max_points))
        points = points[::step]

    return points.astype(np.float32)


def quaternion_inverse(
    qw: float, qx: float, qy: float, qz: float
) -> tuple[float, float, float, float]:
    """Compute the inverse of a quaternion (conjugate for unit quaternions)."""
    # For unit quaternions, inverse = conjugate
    return (qw, -qx, -qy, -qz)


def quaternion_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def apply_pose(
    points: np.ndarray,
    position: dict[str, float],
    orientation: dict[str, float],
    home_offset: dict[str, float] | None = None,
    home_orientation: dict[str, float] | None = None,
) -> np.ndarray:
    """Transform points from camera frame into world frame.

    Camera optical frame: X=right, Y=down, Z=forward (depth)
    AirSim NED frame: X=North, Y=East, Z=Down

    We first convert camera frame to NED-aligned frame:
      NED X (forward) <- Camera Z (forward/depth)
      NED Y (right)   <- Camera X (right)
      NED Z (down)    <- Camera Y (down)

    Then apply the camera world orientation from AirSim.

    Args:
        points: Nx3 array of points in camera frame
        position: Camera position {x, y, z} in world frame
        orientation: Camera orientation quaternion {w, x, y, z}
        home_offset: Optional home/base position to subtract, making home = origin (0,0,0)
        home_orientation: Optional home/base orientation for frame calibration.
            When provided, the drone's starting orientation on its flat base
            becomes the reference frame (level plane with roll=0, pitch=0).

    Returns:
        Nx3 array of points in world frame (relative to home if provided)
    """
    if points.size == 0:
        return points

    # Convert camera optical frame [X=right, Y=down, Z=forward]
    # to NED-aligned frame [X=forward, Y=right, Z=down]
    # This is: new_x = z, new_y = x, new_z = y
    points_ned = np.stack([points[:, 2], points[:, 0], points[:, 1]], axis=1)

    # Get current orientation quaternion
    qw = float(orientation.get("w", 1.0))
    qx = float(orientation.get("x", 0.0))
    qy = float(orientation.get("y", 0.0))
    qz = float(orientation.get("z", 0.0))

    # Apply home orientation calibration if provided
    # This makes the drone's starting orientation the reference frame
    # (flat base surface becomes the level plane)
    if home_orientation is not None:
        home_qw = float(home_orientation.get("w", 1.0))
        home_qx = float(home_orientation.get("x", 0.0))
        home_qy = float(home_orientation.get("y", 0.0))
        home_qz = float(home_orientation.get("z", 0.0))

        # Compute relative orientation: q_rel = q_home_inv * q_current
        # This gives the rotation relative to the home frame
        home_inv = quaternion_inverse(home_qw, home_qx, home_qy, home_qz)
        q_rel = quaternion_multiply(home_inv, (qw, qx, qy, qz))
        qw, qx, qy, qz = q_rel

    rot = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    rotated = points_ned @ rot.T

    translation = np.array(
        [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)],
        dtype=np.float32,
    )

    # Apply home offset calibration if provided
    # This makes the drone's starting position the origin (0,0,0)
    if home_offset is not None:
        home = np.array(
            [home_offset.get("x", 0.0), home_offset.get("y", 0.0), home_offset.get("z", 0.0)],
            dtype=np.float32,
        )
        translation = translation - home

        # Also rotate the translation offset by the home orientation inverse
        # so positions are in the calibrated frame
        if home_orientation is not None:
            home_qw = float(home_orientation.get("w", 1.0))
            home_qx = float(home_orientation.get("x", 0.0))
            home_qy = float(home_orientation.get("y", 0.0))
            home_qz = float(home_orientation.get("z", 0.0))
            home_rot_inv = quaternion_to_rotation_matrix(
                *quaternion_inverse(home_qw, home_qx, home_qy, home_qz)
            )
            translation = (home_rot_inv @ translation).astype(np.float32)

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
