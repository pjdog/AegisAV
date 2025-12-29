#!/usr/bin/env python3
"""End-to-end simulation harness.

Runs a full simulation loop: server + simulated drones + scenario runner.
Can run standalone or connect to an existing server.

Usage:
    python scripts/run_simulation.py --scenario normal_ops_001 --duration 30
    python scripts/run_simulation.py --scenario battery_cascade_001 --seed 42 --verbose
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.edge_config import EdgeComputeProfile  # noqa: E402
from agent.server.scenario_runner import ScenarioRunner  # noqa: E402
from agent.server.scenarios import get_all_scenarios, get_scenario  # noqa: E402

logger = logging.getLogger("simulation")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from httpx and other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def list_scenarios() -> None:
    """Print available scenarios."""
    scenarios = get_all_scenarios()
    logger.info("Available scenarios:")
    logger.info("-" * 60)
    for s in scenarios:
        logger.info("  %s %s", f"{s.scenario_id:<25}", s.name)
        logger.info("    Category: %s, Difficulty: %s", s.category.value, s.difficulty)
        logger.info("    Drones: %s, Duration: %s min", len(s.drones), s.duration_minutes)
        logger.info("")


async def run_simulation(
    scenario_id: str,
    duration_s: float = 30.0,
    time_scale: float = 10.0,
    seed: int | None = None,
    profile: str | None = None,
    log_dir: Path | None = None,
) -> dict:
    """Run a simulation.

    Args:
        scenario_id: ID of scenario to run
        duration_s: Maximum real-time duration in seconds
        time_scale: Simulation speed multiplier
        seed: Random seed for determinism
        profile: Edge compute profile name
        log_dir: Directory for decision logs

    Returns:
        Summary dict with run results
    """
    # Validate scenario
    scenario = get_scenario(scenario_id)
    if not scenario:
        available = [s.scenario_id for s in get_all_scenarios()]
        logger.error(f"Scenario not found: {scenario_id}")
        logger.error(f"Available: {available}")
        return {"error": f"Scenario not found: {scenario_id}"}

    # Set up log directory
    if log_dir is None:
        log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create runner with optional seed
    runner = ScenarioRunner(
        tick_interval_s=1.0,
        decision_interval_s=5.0,
        log_dir=log_dir,
        seed=seed,
    )

    # Load scenario
    if not await runner.load_scenario(scenario_id):
        return {"error": f"Failed to load scenario: {scenario_id}"}

    # Log configuration
    logger.info("=" * 60)
    logger.info(f"Simulation: {scenario.name}")
    logger.info(f"  Run ID: {runner.run_id}")
    logger.info(f"  Scenario: {scenario_id}")
    logger.info(f"  Drones: {len(scenario.drones)}")
    logger.info(f"  Duration: {scenario.duration_minutes} min (simulated)")
    logger.info(f"  Time scale: {time_scale}x")
    logger.info(f"  Max real duration: {duration_s}s")
    if seed is not None:
        logger.info(f"  Seed: {seed}")
    if profile:
        logger.info(f"  Edge profile: {profile}")
    logger.info("=" * 60)

    # Run simulation
    try:
        completed = await runner.run(
            time_scale=time_scale,
            max_duration_s=duration_s,
        )

        # Save decision log
        log_path = runner.save_decision_log(log_dir)

        # Get summary
        summary = runner.get_summary()
        summary["log_path"] = str(log_path)
        summary["completed"] = completed

        # Print summary
        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Run ID: {summary.get('run_id')}")
        logger.info(f"  Duration: {summary.get('duration_s', 0):.1f}s (simulated)")
        logger.info(f"  Completed: {summary.get('is_complete', False)}")
        logger.info(f"  Total decisions: {summary.get('total_decisions', 0)}")
        logger.info(f"  Battery consumed: {summary.get('total_battery_consumed', 0):.1f}%")
        logger.info(f"  Anomalies: {summary.get('anomalies_triggered', 0)}")
        logger.info(f"  Events fired: {summary.get('events_fired', 0)}")
        logger.info("")
        logger.info("  Drone Status:")
        for drone in summary.get("drones", []):
            logger.info(
                f"    {drone['name']}: {drone['final_state']} "
                f"(battery: {drone['final_battery']:.1f}%, "
                f"decisions: {drone['decisions_made']})"
            )
        logger.info("")
        logger.info(f"  Log saved to: {log_path}")

        return summary

    except asyncio.CancelledError:
        logger.info("Simulation cancelled")
        return {"error": "cancelled", "run_id": runner.run_id}

    except Exception as e:
        logger.exception(f"Simulation error: {e}")
        return {"error": str(e), "run_id": runner.run_id}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end AegisAV simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scenario normal_ops_001 --duration 30
  %(prog)s --scenario battery_cascade_001 --seed 42 --verbose
  %(prog)s --list
        """,
    )

    parser.add_argument(
        "--scenario",
        "-s",
        help="Scenario ID to run",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Maximum real-time duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--time-scale",
        "-t",
        type=float,
        default=10.0,
        help="Simulation speed multiplier (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic simulation",
    )
    parser.add_argument(
        "--profile",
        "-p",
        choices=[p.value for p in EdgeComputeProfile],
        help="Edge compute profile to use",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for decision logs (default: logs/)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.list:
        list_scenarios()
        return 0

    if not args.scenario:
        parser.error("--scenario is required (or use --list to see options)")

    result = asyncio.run(
        run_simulation(
            scenario_id=args.scenario,
            duration_s=args.duration,
            time_scale=args.time_scale,
            seed=args.seed,
            profile=args.profile,
            log_dir=args.log_dir,
        )
    )

    if result.get("error"):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
