#!/usr/bin/env python3
"""Smoke test for async decision queues with configurable vehicles."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import httpx

from agent.server.main import app, server_state

logger = logging.getLogger(__name__)


def _sample_state(vehicle_id: str) -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "vehicle_id": vehicle_id,
        "position": {
            "latitude": 47.397742,
            "longitude": 8.545594,
            "altitude_msl": 488.0,
            "altitude_agl": 10.0,
        },
        "velocity": {"north": 0.0, "east": 0.0, "down": 0.0},
        "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "battery": {"voltage": 22.8, "current": 5.0, "remaining_percent": 80.0},
        "mode": "GUIDED",
        "armed": True,
        "in_air": True,
        "gps": {
            "fix_type": 3,
            "satellites_visible": 12,
            "hdop": 0.8,
            "vdop": 1.0,
        },
        "health": {
            "sensors_healthy": True,
            "gps_healthy": True,
            "battery_healthy": True,
            "motors_healthy": True,
            "ekf_healthy": True,
        },
        "home_position": {
            "latitude": 47.397742,
            "longitude": 8.545594,
            "altitude_msl": 488.0,
            "altitude_agl": 0.0,
        },
    }


def _parse_vehicle_ids(args: argparse.Namespace) -> list[str]:
    if args.vehicle_ids:
        return [item.strip() for item in args.vehicle_ids.split(",") if item.strip()]
    if args.count > 0:
        return [
            f"{args.prefix}-{index}"
            for index in range(args.start_index, args.start_index + args.count)
        ]
    return ["aegis-alpha", "aegis-bravo"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async queue smoke test")
    parser.add_argument("--vehicle-ids", default="", help="Comma-separated vehicle IDs")
    parser.add_argument("--count", type=int, default=0, help="Number of vehicles to simulate")
    parser.add_argument("--prefix", default="aegis", help="Prefix for generated vehicle IDs")
    parser.add_argument("--start-index", type=int, default=1, help="Start index for IDs")
    parser.add_argument(
        "--server-config",
        default="configs/agent_config.yaml",
        help="Server config to load",
    )
    parser.add_argument(
        "--mission-config",
        default="configs/mission_config.yaml",
        help="Mission config to load",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=2.0,
        help="Decision poll timeout in seconds",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(__file__).resolve().parents[1]
    server_state.load_config(root / args.server_config)
    server_state.load_mission(root / args.mission_config)
    vehicle_ids = _parse_vehicle_ids(args)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for vehicle_id in vehicle_ids:
            state = _sample_state(vehicle_id)
            response = await client.post("/state/async", json=state)
            response.raise_for_status()

        decisions = {}
        for vehicle_id in vehicle_ids:
            response = await client.get(
                "/decisions/next",
                params={"vehicle_id": vehicle_id, "timeout_s": args.poll_timeout},
            )
            response.raise_for_status()
            decisions[vehicle_id] = response.json()

    for vehicle_id, decision in decisions.items():
        logger.info(
            "%s: %s (%s)",
            vehicle_id,
            decision.get("action"),
            decision.get("decision_id"),
        )


if __name__ == "__main__":
    asyncio.run(main())
