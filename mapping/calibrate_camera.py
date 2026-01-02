"""CLI for camera calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from mapping.calibration import calibrate_chessboard, save_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a camera using chessboard images.")
    parser.add_argument("image_dir", help="Directory containing chessboard images")
    parser.add_argument("--pattern-cols", type=int, default=9, help="Chessboard columns")
    parser.add_argument("--pattern-rows", type=int, default=6, help="Chessboard rows")
    parser.add_argument("--square-size-m", type=float, default=0.025, help="Square size in meters")
    parser.add_argument("--output", default="data/calibration/camera_calibration.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = calibrate_chessboard(
        Path(args.image_dir),
        (args.pattern_cols, args.pattern_rows),
        args.square_size_m,
    )
    save_calibration(Path(args.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
