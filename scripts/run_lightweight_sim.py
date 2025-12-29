#!/usr/bin/env python3
"""Run the Lightweight Drone Simulator.

A standalone physics-based drone simulation that runs without Unreal Engine.
Includes a web-based 3D visualizer accessible at http://localhost:8081/viz/

Usage:
    python scripts/run_lightweight_sim.py [OPTIONS]

Options:
    --port PORT         Server port (default: 8081)
    --drones N          Number of drones (default: 2)
    --wind SPEED        Wind speed in m/s (default: 3)
    --scenario NAME     Scenario: farm, industrial, simple (default: farm)
    --speed FACTOR      Simulation speed factor (default: 1.0)

Examples:
    # Basic run with 2 drones
    python scripts/run_lightweight_sim.py

    # 4 drones with strong wind
    python scripts/run_lightweight_sim.py --drones 4 --wind 10

    # Industrial scenario at 2x speed
    python scripts/run_lightweight_sim.py --scenario industrial --speed 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from pathlib import Path

import numpy as np
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.lightweight.physics import DroneConfig, EnvironmentConfig
from simulation.lightweight.server import create_app
from simulation.lightweight.simulator import LightweightSim

logger = logging.getLogger(__name__)


def create_farm_scenario(sim: LightweightSim, num_drones: int) -> None:
    """Create a farm inspection scenario with solar panels."""
    # Solar panel array (4x3 grid)
    for row in range(4):
        for col in range(3):
            x = 40 + row * 25
            y = -30 + col * 30
            asset_id = f"solar_{row:02d}_{col:02d}"

            # Random anomaly (10% chance)
            has_anomaly = np.random.random() < 0.1
            severity = np.random.uniform(0.4, 0.9) if has_anomaly else 0.0

            sim.add_asset(
                asset_id,
                np.array([float(x), float(y), 0.0]),
                "solar_panel",
                has_anomaly=has_anomaly,
                anomaly_severity=severity,
            )

    # Add drones at dock
    for i in range(num_drones):
        drone_id = f"drone_{i + 1:03d}"
        start_pos = np.array([float(i * 3), 0.0, 0.0])
        sim.add_drone(drone_id, DroneConfig(), start_pos)


def create_industrial_scenario(sim: LightweightSim, num_drones: int) -> None:
    """Create an industrial facility scenario with tanks and pipes."""
    # Storage tanks
    tank_positions = [
        (-50, -30),
        (-50, 0),
        (-50, 30),
        (-80, -15),
        (-80, 15),
    ]
    for i, (x, y) in enumerate(tank_positions):
        sim.add_asset(
            f"tank_{i:03d}",
            np.array([float(x), float(y), 0.0]),
            "tank",
            has_anomaly=(i == 2),  # One tank has issue
            anomaly_severity=0.6 if i == 2 else 0.0,
        )

    # Solar panels on roof
    for i in range(4):
        x = 30 + i * 15
        sim.add_asset(
            f"solar_{i:03d}",
            np.array([float(x), 40.0, 5.0]),
            "solar_panel",
        )

    # Add drones
    for i in range(num_drones):
        drone_id = f"drone_{i + 1:03d}"
        start_pos = np.array([float(i * 3), 0.0, 0.0])
        sim.add_drone(drone_id, DroneConfig(), start_pos)


def create_simple_scenario(sim: LightweightSim, num_drones: int) -> None:
    """Create a simple test scenario."""
    # Just a few assets in a line
    for i in range(5):
        sim.add_asset(
            f"asset_{i:03d}",
            np.array([30.0 + i * 20, 0.0, 0.0]),
            "solar_panel",
            has_anomaly=(i == 2),
            anomaly_severity=0.5 if i == 2 else 0.0,
        )

    # Single drone
    for i in range(num_drones):
        sim.add_drone(f"drone_{i + 1:03d}", DroneConfig())


async def run_demo_mission(sim: LightweightSim) -> None:
    """Run a simple demo mission."""
    await asyncio.sleep(2)  # Wait for sim to start

    # Get first drone
    drone_ids = list(sim.drones.keys())
    if not drone_ids:
        return

    drone_id = drone_ids[0]
    logger.info("🚁 Starting demo mission with %s", drone_id)

    # Takeoff
    logger.info("  → Taking off...")
    sim.takeoff(drone_id, 15.0)
    await asyncio.sleep(5)

    # Visit some assets
    assets = list(sim.world.assets.values())[:3]
    for asset in assets:
        logger.info("  → Flying to %s...", asset.asset_id)
        target = asset.position.copy()
        target[2] = -20  # 20m altitude
        sim.goto(drone_id, target)
        await asyncio.sleep(8)

        # Hover and "inspect"
        logger.info("  → Inspecting %s...", asset.asset_id)
        if asset.has_anomaly:
            logger.info("    ⚠️  Anomaly detected! Severity: %.1f%%", asset.anomaly_severity * 100)
        await asyncio.sleep(3)

    # Return to launch
    logger.info("  → Returning to dock...")
    sim.rtl(drone_id)
    await asyncio.sleep(10)

    # Land
    logger.info("  → Landing...")
    sim.land(drone_id)
    await asyncio.sleep(5)

    logger.info("✅ Demo mission complete!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AegisAV Lightweight Drone Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument("--drones", type=int, default=2, help="Number of drones")
    parser.add_argument("--wind", type=float, default=3.0, help="Wind speed (m/s)")
    parser.add_argument("--scenario", choices=["farm", "industrial", "simple"], default="farm")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed factor")
    parser.add_argument("--demo", action="store_true", help="Run demo mission")
    parser.add_argument("--no-server", action="store_true", help="Run without web server")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("  AegisAV Lightweight Drone Simulator")
    logger.info("=" * 60)
    logger.info("  Scenario: %s", args.scenario)
    logger.info("  Drones: %s", args.drones)
    logger.info("  Wind: %.1f m/s", args.wind)
    logger.info("  Speed: %.1fx", args.speed)
    logger.info("=" * 60)

    # Create environment
    env = EnvironmentConfig(
        wind_speed_ms=args.wind,
        wind_direction_rad=math.radians(45),  # NE wind
        wind_gust_intensity=0.3,
        wind_turbulence=0.15,
    )

    # Create simulator
    sim = LightweightSim(env_config=env)
    sim.set_real_time_factor(args.speed)

    # Set up scenario
    if args.scenario == "farm":
        create_farm_scenario(sim, args.drones)
    elif args.scenario == "industrial":
        create_industrial_scenario(sim, args.drones)
    else:
        create_simple_scenario(sim, args.drones)

    logger.info("📍 Created %s assets", len(sim.world.assets))
    logger.info("🚁 Created %s drones", len(sim.drones))

    if args.no_server:
        # Run simulation without web server
        async def run_headless() -> None:
            await sim.start()
            logger.info("⚡ Simulation running (Ctrl+C to stop)")

            if args.demo:
                await run_demo_mission(sim)
            else:
                # Run indefinitely
                try:
                    while True:
                        await asyncio.sleep(1)
                        # Print status every 10 seconds
                        if int(sim.get_sim_time()) % 10 == 0:
                            for drone_id in sim.drones:
                                state = sim.get_vehicle_state(drone_id)
                                if state:
                                    logger.info(
                                        "  %s: Alt=%.1fm Bat=%.0f%% Mode=%s",
                                        drone_id,
                                        state.position.altitude_msl,
                                        state.battery.remaining_percent,
                                        state.mode.value,
                                    )
                except KeyboardInterrupt:
                    pass

            await sim.stop()
            logger.info("✅ Simulation stopped")

        asyncio.run(run_headless())
    else:
        # Run with web server
        logger.info("🌐 Web visualizer: http://localhost:%s/viz/", args.port)
        logger.info("📡 API docs: http://localhost:%s/docs", args.port)
        logger.info("Press Ctrl+C to stop")

        app = create_app(sim)

        if args.demo:
            # Start demo mission in background
            @app.on_event("startup")
            async def start_demo() -> None:
                app.state.demo_task = asyncio.create_task(run_demo_mission(sim))
                await asyncio.sleep(0)

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
