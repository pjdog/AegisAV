"""Real sensor capture for RGB + optional IMU bundles."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from mapping.calibration import load_calibration

logger = structlog.get_logger(__name__)


@dataclass
class RealCaptureConfig:
    """Configuration for real sensor capture."""

    output_dir: Path
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 15
    calibration_path: Path | None = None


class RealSensorCapture:
    """Capture RGB frames (and optional IMU) from a real camera."""

    def __init__(self, config: RealCaptureConfig) -> None:
        self.config = config
        self._cap = None
        self._intrinsics = None

        if self.config.calibration_path:
            self._intrinsics = load_calibration(self.config.calibration_path)

    def open(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenCV is required for real camera capture") from exc

        self._cap = cv2.VideoCapture(self.config.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "frames").mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def capture_sequence(
        self,
        frames: int,
        interval_s: float,
        imu_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if self._cap is None:
            self.open()

        for _ in range(frames):
            success, frame = self._cap.read()
            if not success:
                logger.warning("frame_capture_failed")
                continue

            timestamp_ns = time.time_ns()
            frame_id = str(timestamp_ns)
            rgb_path = self.config.output_dir / "frames" / f"{frame_id}.png"

            # Save image
            import cv2  # type: ignore

            cv2.imwrite(str(rgb_path), frame)

            metadata = {
                "frame_id": frame_id,
                "timestamp": datetime.now().isoformat(),
                "timestamp_ns": timestamp_ns,
                "camera": {
                    "name": f"camera_{self.config.camera_index}",
                    "resolution": [int(frame.shape[1]), int(frame.shape[0])],
                    "fov_deg": None,
                    "intrinsics": {
                        "fx": self._intrinsics.fx if self._intrinsics else 0.0,
                        "fy": self._intrinsics.fy if self._intrinsics else 0.0,
                        "cx": self._intrinsics.cx if self._intrinsics else 0.0,
                        "cy": self._intrinsics.cy if self._intrinsics else 0.0,
                        "width": self._intrinsics.width
                        if self._intrinsics
                        else int(frame.shape[1]),
                        "height": self._intrinsics.height
                        if self._intrinsics
                        else int(frame.shape[0]),
                    },
                    "pose": None,
                },
                "telemetry": None,
                "imu": imu_provider() if imu_provider else None,
                "files": {"rgb": str(rgb_path), "depth": None},
            }

            meta_path = self.config.output_dir / "frames" / f"{frame_id}.json"
            meta_path.write_text(json.dumps(metadata, indent=2))

            time.sleep(interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture real camera frames for mapping.")
    parser.add_argument("--output-dir", default="data/maps/real_capture")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--calibration", default=None, help="Path to calibration JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RealCaptureConfig(
        output_dir=Path(args.output_dir),
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        calibration_path=Path(args.calibration) if args.calibration else None,
    )
    capture = RealSensorCapture(config)
    capture.capture_sequence(frames=args.frames, interval_s=args.interval_s)
    capture.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
