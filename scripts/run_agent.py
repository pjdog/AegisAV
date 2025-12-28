#!/usr/bin/env python3
"""
AegisAV Main Entry Point

Runs both agent server and client for development/demo purposes.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from asyncio.subprocess import Process
from datetime import datetime
from pathlib import Path

from agent.client.main import main as client_main
from agent.server.goal_selector import GoalSelector
from agent.server.main import main as server_main
from agent.server.risk_evaluator import RiskEvaluator
from agent.server.world_model import (
    Asset,
    AssetType,
    DockStatus,
    WorldModel,
)
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSState,
    Position,
    VehicleHealth,
    VehicleState,
    Velocity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def start_server() -> Process:
    """Start the agent server in a subprocess."""
    logger.info("Starting agent server...")
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "agent.server.main",
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        cwd=str(Path(__file__).parent.parent),
    )


async def start_client(config_path: str, connection: str) -> Process:
    """Start the agent client in a subprocess."""
    logger.info("Starting agent client...")

    cmd = [
        sys.executable,
        "-m",
        "agent.client.main",
        "--config",
        config_path,
    ]

    if connection:
        cmd.extend(["--connection", connection])

    return await asyncio.create_subprocess_exec(
        *cmd,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        cwd=str(Path(__file__).parent.parent),
    )


async def run_demo(scenario: str) -> None:
    """
    Run a demo scenario.

    This is a basic demo that simulates agent behavior
    without requiring ArduPilot SITL.
    """
    logger.info(f"Running demo scenario: {scenario}")

    # Initialize components
    world_model = WorldModel()
    goal_selector = GoalSelector()
    risk_evaluator = RiskEvaluator()

    # Set up demo world
    dock_pos = Position(47.397742, 8.545594, 488.0)
    world_model.set_dock(dock_pos, DockStatus.AVAILABLE)

    # Add demo assets
    world_model.add_asset(
        Asset(
            asset_id="demo-001",
            name="Demo Asset 1",
            asset_type=AssetType.SOLAR_PANEL,
            position=Position(47.398000, 8.546000, 490.0),
            priority=1,
        )
    )

    world_model.add_asset(
        Asset(
            asset_id="demo-002",
            name="Demo Asset 2",
            asset_type=AssetType.WIND_TURBINE,
            position=Position(47.397500, 8.544000, 510.0),
            priority=2,
        )
    )

    # Start mission
    world_model.start_mission("demo-mission", "Demo Inspection Mission")

    # Simulate a few decision cycles
    battery_percent = 80.0
    current_pos = dock_pos

    for i in range(10):
        # Simulate vehicle state
        vehicle_state = VehicleState(
            timestamp=datetime.now(),
            position=current_pos,
            velocity=Velocity(0, 0, 0),
            attitude=Attitude(0, 0, 0),
            battery=BatteryState(22.8, 5.0, battery_percent),
            mode=FlightMode.GUIDED,
            armed=True,
            in_air=True,
            gps=GPSState(3, 12, 0.9, 0.9),
            health=VehicleHealth(True, True, True, True, True),
            home_position=dock_pos,
        )

        world_model.update_vehicle(vehicle_state)

        # Get snapshot and make decision
        snapshot = world_model.get_snapshot()
        if snapshot:
            risk = risk_evaluator.evaluate(snapshot)
            goal = await goal_selector.select_goal(snapshot)

            logger.info(f"Cycle {i + 1}:")
            logger.info(f"  Battery: {battery_percent:.1f}%")
            logger.info(f"  Risk Level: {risk.overall_level.value}")
            logger.info(f"  Goal: {goal.goal_type.value} - {goal.reason}")

            # Simulate battery drain
            battery_percent -= 5.0

            # Simulate movement
            if goal.target_asset:
                current_pos = goal.target_asset.position
                world_model.record_inspection(goal.target_asset.asset_id)

        await asyncio.sleep(1)

    logger.info("Demo complete!")


async def run_full(config_path: str, connection: str) -> None:
    """Run both server and client processes."""
    server_proc = await start_server()
    await asyncio.sleep(2)
    client_proc = await start_client(config_path, connection)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        await stop_event.wait()
    finally:
        client_proc.terminate()
        server_proc.terminate()
        await client_proc.wait()
        await server_proc.wait()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AegisAV Agent Runner")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run agent server only")
    server_parser.add_argument("--config", default="configs/agent_config.yaml")

    # Client command
    client_parser = subparsers.add_parser("client", help="Run agent client only")
    client_parser.add_argument("--config", default="configs/agent_config.yaml")
    client_parser.add_argument("--connection", help="MAVLink connection string")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo without SITL")
    demo_parser.add_argument(
        "--scenario",
        choices=["basic", "anomaly", "battery"],
        default="basic",
        help="Demo scenario to run",
    )

    # Full command (server + client)
    full_parser = subparsers.add_parser("full", help="Run server and client")
    full_parser.add_argument("--config", default="configs/agent_config.yaml")
    full_parser.add_argument("--connection", default="udp:127.0.0.1:14550")

    args = parser.parse_args()

    if args.command == "server":
        # Run server directly
        server_main()

    elif args.command == "client":
        # Run client directly
        sys.argv = ["", "--config", args.config]
        if args.connection:
            sys.argv.extend(["--connection", args.connection])
        client_main()

    elif args.command == "demo":
        asyncio.run(run_demo(args.scenario))

    elif args.command == "full":
        asyncio.run(run_full(args.config, args.connection))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
