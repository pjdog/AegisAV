"""Camera calibration utilities for real sensor capture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from mapping.manifest import SensorCalibration

logger = structlog.get_logger(__name__)


@dataclass
class CalibrationResult:
    """Calibration output for a camera."""

    calibration: SensorCalibration
    reprojection_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.to_dict(),
            "reprojection_error": self.reprojection_error,
        }


def save_calibration(path: Path, result: CalibrationResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("calibration_saved", path=str(path))


def load_calibration(path: Path) -> SensorCalibration:
    data = json.loads(Path(path).read_text())
    cal = data.get("calibration", {}).get("intrinsics", {})
    depth = data.get("calibration", {}).get("depth", {})
    return SensorCalibration(
        fx=cal.get("fx", 0.0),
        fy=cal.get("fy", 0.0),
        cx=cal.get("cx", 0.0),
        cy=cal.get("cy", 0.0),
        width=cal.get("width", 0),
        height=cal.get("height", 0),
        depth_scale=depth.get("scale", 1.0),
        depth_min_m=depth.get("min_m", 0.1),
        depth_max_m=depth.get("max_m", 100.0),
    )


def calibrate_chessboard(
    image_dir: Path,
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> CalibrationResult:
    """Calibrate using a chessboard pattern and OpenCV."""
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required for calibration") from exc

    image_dir = Path(image_dir)
    images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No calibration images found in {image_dir}")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints = []
    img_shape = None

    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            objpoints.append(objp)
            imgpoints.append(corners)

    if not objpoints or img_shape is None:
        raise RuntimeError("No chessboard corners detected")

    ret, mtx, _dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_shape,
        None,
        None,
    )

    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])

    calibration = SensorCalibration(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=img_shape[0],
        height=img_shape[1],
    )

    return CalibrationResult(
        calibration=calibration,
        reprojection_error=float(ret),
    )
