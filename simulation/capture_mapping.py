#!/usr/bin/env python3
"""Capture synchronized mapping bundles from AirSim."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from simulation.realtime_bridge import RealtimeAirSimBridge, RealtimeBridgeConfig


async def run_capture(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) if args.output_dir else None

    config = RealtimeBridgeConfig(
        host=args.host,
        vehicle_name=args.vehicle,
    )
    bridge = RealtimeAirSimBridge(config)

    if not await bridge.connect():
        print("Failed to connect to AirSim.")
        return 1

    try:
        for idx in range(args.frames):
            result = await bridge.capture_mapping_bundle(
                output_dir=output_dir,
                include_depth=not args.no_depth,
                include_imu=not args.no_imu,
            )
            if not result.get("success"):
                print(f"[{idx + 1}/{args.frames}] capture failed: {result.get('error')}")
            else:
                print(f"[{idx + 1}/{args.frames}] captured {result.get('metadata_path')}")
            await asyncio.sleep(args.interval_s)
    finally:
        await bridge.disconnect()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture AirSim mapping bundles.")
    parser.add_argument("--host", default="127.0.0.1", help="AirSim host/IP")
    parser.add_argument("--vehicle", default="Drone1", help="Vehicle name")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to capture")
    parser.add_argument("--interval-s", type=float, default=0.5, help="Seconds between captures")
    parser.add_argument("--output-dir", default=None, help="Output directory for capture bundles")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth capture")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU capture")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_capture(args))


if __name__ == "__main__":
    raise SystemExit(main())
