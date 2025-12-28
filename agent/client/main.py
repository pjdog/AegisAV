"""
Agent Client Main Entry Point

Connects to the vehicle via MAVLink and to the agent server via HTTP.
Runs the main control loop: collect state -> send to server -> execute decisions.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml

from agent.client.action_executor import ActionExecutor, ExecutionState
from agent.client.state_collector import CollectorConfig, StateCollector
from autonomy.mavlink_interface import MAVLinkConfig, MAVLinkInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class AgentClient:
    """
    Main agent client orchestrating state collection and action execution.

    The client connects to:
    - The vehicle via MAVLink (receives telemetry, sends commands)
    - The agent server via HTTP (sends state, receives decisions)

    It runs a continuous loop:
    1. Collect vehicle state
    2. Send state to agent server
    3. Receive decision
    4. Execute decision
    5. Repeat
    """

    def __init__(
        self,
        mavlink_config: MAVLinkConfig,
        collector_config: CollectorConfig,
    ):
        self.mavlink = MAVLinkInterface(mavlink_config)
        self.state_collector = StateCollector(self.mavlink, collector_config)
        self.action_executor = ActionExecutor(self.mavlink)

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to vehicle and verify server."""
        # Connect to MAVLink
        logger.info("Connecting to vehicle...")
        if not await self.mavlink.connect():
            logger.error("Failed to connect to vehicle")
            return False

        # Check server health
        logger.info("Checking agent server...")
        if not await self.state_collector.check_server_health():
            logger.warning("Agent server not responding, will retry during operation")

        return True

    async def disconnect(self) -> None:
        """Disconnect from vehicle."""
        await self.mavlink.disconnect()

    async def run(self) -> None:
        """Run the main agent client loop."""
        if not await self.connect():
            return

        self._running = True
        logger.info("Agent client running")

        try:
            async for decision in self.state_collector.run():
                if not self._running:
                    break

                # Check if decision requires action
                action = decision.get("action", "none")
                if action not in ("none", "wait"):
                    # Execute the decision
                    result = await self.action_executor.execute(decision)

                    if result.state == ExecutionState.FAILED:
                        logger.error(f"Action failed: {result.message}")
                    elif result.state == ExecutionState.COMPLETED:
                        logger.info(f"Action completed in {result.duration_s:.1f}s")

                # Check for shutdown
                if self._shutdown_event.is_set():
                    break

        except asyncio.CancelledError:
            logger.info("Agent client cancelled")
        finally:
            self._running = False
            await self.disconnect()

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()


def load_config(config_path: Path) -> tuple[MAVLinkConfig, CollectorConfig]:
    """Load configuration from YAML file."""
    mavlink_config = MAVLinkConfig()
    collector_config = CollectorConfig()

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # MAVLink config
        mavlink_section = config.get("mavlink", {})
        mavlink_config = MAVLinkConfig(
            connection_string=mavlink_section.get("connection", mavlink_config.connection_string),
            source_system=mavlink_section.get("source_system", mavlink_config.source_system),
            timeout_ms=mavlink_section.get("timeout_ms", mavlink_config.timeout_ms),
        )

        # Client/collector config
        client_section = config.get("client", {})
        server_section = config.get("server", {})
        collector_config = CollectorConfig(
            server_url=client_section.get(
                "server_url",
                f"http://localhost:{server_section.get('port', 8080)}",
            ),
            update_interval_s=client_section.get("update_interval_ms", 100) / 1000,
        )

    return mavlink_config, collector_config


async def async_main(args: argparse.Namespace) -> None:
    """Async main entry point."""
    # Load config
    config_path = Path(args.config)
    mavlink_config, collector_config = load_config(config_path)

    # Override with CLI args
    if args.connection:
        mavlink_config.connection_string = args.connection
    if args.server:
        collector_config.server_url = args.server

    # Create client
    client = AgentClient(mavlink_config, collector_config)

    # Set up signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, client.stop)

    # Run
    await client.run()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AegisAV Agent Client")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/agent_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--connection",
        type=str,
        help="MAVLink connection string (overrides config)",
    )
    parser.add_argument(
        "--server",
        type=str,
        help="Agent server URL (overrides config)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
