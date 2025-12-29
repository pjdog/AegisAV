#!/usr/bin/env python3
"""Generate per-drone config files from a base template."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate drone config files")
    parser.add_argument(
        "--base-config",
        default="configs/agent_config.yaml",
        help="Base config to copy",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/generated",
        help="Output directory for generated configs",
    )
    parser.add_argument("--vehicle-ids", default="", help="Comma-separated vehicle IDs")
    parser.add_argument("--count", type=int, default=0, help="Number of configs to generate")
    parser.add_argument("--prefix", default="aegis", help="Prefix for generated IDs")
    parser.add_argument("--start-index", type=int, default=1, help="Start index for IDs")
    parser.add_argument("--base-port", type=int, default=14550, help="Base MAVLink port")
    parser.add_argument("--port-step", type=int, default=1, help="Port increment per drone")
    parser.add_argument("--source-system-base", type=int, default=240, help="Base source_system")
    parser.add_argument("--source-system-step", type=int, default=1, help="Increment per drone")
    parser.add_argument(
        "--connection-template",
        default="udp:127.0.0.1:{port}",
        help="Template for MAVLink connection string",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def _vehicle_ids(args: argparse.Namespace) -> list[str]:
    if args.vehicle_ids:
        return [item.strip() for item in args.vehicle_ids.split(",") if item.strip()]
    if args.count > 0:
        return [
            f"{args.prefix}-{index}"
            for index in range(args.start_index, args.start_index + args.count)
        ]
    raise ValueError("Provide --vehicle-ids or --count")


def _apply_overrides(
    base: dict,
    *,
    vehicle_id: str,
    connection: str,
    source_system: int,
) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("agent", {})["name"] = vehicle_id
    client = config.setdefault("client", {})
    client["vehicle_id"] = vehicle_id
    mavlink = config.setdefault("mavlink", {})
    mavlink["connection"] = connection
    mavlink["source_system"] = source_system
    return config


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(__file__).resolve().parents[1]
    base_path = root / args.base_config
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        logger.error("Base config not found: %s", base_path)
        return 1

    with open(base_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    vehicle_ids = _vehicle_ids(args)
    generated = []
    for index, vehicle_id in enumerate(vehicle_ids):
        port = args.base_port + (index * args.port_step)
        source_system = args.source_system_base + (index * args.source_system_step)
        if not (1 <= source_system <= 255):
            logger.error(
                "source_system %s out of range for %s",
                source_system,
                vehicle_id,
            )
            return 1

        connection = args.connection_template.format(port=port, vehicle_id=vehicle_id)
        config = _apply_overrides(
            base_config,
            vehicle_id=vehicle_id,
            connection=connection,
            source_system=source_system,
        )

        filename = f"agent_config.{vehicle_id.replace(' ', '_')}.yaml"
        target = output_dir / filename
        if target.exists() and not args.overwrite:
            logger.info("Skipping existing file: %s", target)
            continue

        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        generated.append(target)

    if generated:
        logger.info("Generated configs:")
        for path in generated:
            logger.info("  %s", path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
