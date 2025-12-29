"""State Collector.

Aggregates vehicle telemetry from MAVLink into structured state
updates for the agent server.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import httpx

from autonomy.mavlink_interface import MAVLinkInterface
from autonomy.vehicle_state import VehicleState

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """State collector configuration."""

    server_url: str = "http://localhost:8080"
    update_interval_s: float = 0.1
    http_timeout_s: float = 5.0
    retry_delay_s: float = 1.0
    async_mode: bool = False
    decision_poll_timeout_s: float = 10.0
    vehicle_id: str = "drone_001"


class StateCollector:
    """Collects vehicle state from MAVLink and sends to agent server.

    The state collector is the bridge between the vehicle telemetry
    and the agent server's decision-making. It:

    - Receives telemetry from MAVLinkInterface callbacks
    - Aggregates data into VehicleState objects
    - Sends state updates to the agent server via HTTP
    - Receives decisions from the server

    Example:
        collector = StateCollector(mavlink_interface, config)

        async for decision in collector.run():
            await action_executor.execute(decision)
    """

    def __init__(
        self,
        mavlink: MAVLinkInterface,
        config: CollectorConfig | None = None,
    ) -> None:
        """Initialize the StateCollector.

        Args:
            mavlink: MAVLink interface for receiving telemetry.
            config: Collector configuration settings.
        """
        self.mavlink = mavlink
        self.config = config or CollectorConfig()

        self._running = False
        self._last_state: VehicleState | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._async_tasks: list[asyncio.Task] = []

        # Register for state updates
        self.mavlink.on_state_update(self._on_state_update)

    def _on_state_update(self, state: VehicleState) -> None:
        """Callback for vehicle state updates."""
        self._last_state = state

    def _build_state_payload(self, state: VehicleState) -> dict:
        """Build state payload for server requests."""
        payload = state.to_dict()
        if self.config.vehicle_id:
            payload["vehicle_id"] = self.config.vehicle_id
        return payload

    async def start(self) -> None:
        """Start the state collector."""
        self._running = True
        self._http_client = httpx.AsyncClient(timeout=self.config.http_timeout_s)
        logger.info("State collector started")

    async def stop(self) -> None:
        """Stop the state collector."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
        logger.info("State collector stopped")

    async def send_state(self, state: VehicleState) -> dict | None:
        """Send state to agent server and receive decision.

        Args:
            state: Current vehicle state

        Returns:
            Decision dict from server, or None if failed
        """
        if not self._http_client:
            return None

        try:
            response = await self._http_client.post(
                f"{self.config.server_url}/state",
                json=self._build_state_payload(state),
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            logger.warning("Server request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Server error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Failed to send state: {e}")
            return None

    async def send_state_async(self, state: VehicleState) -> dict | None:
        """Send state to agent server without waiting for decision."""
        if not self._http_client:
            return None

        try:
            response = await self._http_client.post(
                f"{self.config.server_url}/state/async",
                json=self._build_state_payload(state),
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            logger.warning("Async state request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("Async state server error: %s", e.response.status_code)
            return None
        except Exception as e:
            logger.error("Failed to send async state: %s", e)
            return None

    async def fetch_decision(self) -> dict | None:
        """Fetch next decision from server (long-polling)."""
        if not self._http_client:
            return None

        try:
            response = await self._http_client.get(
                f"{self.config.server_url}/decisions/next",
                params={
                    "vehicle_id": self.config.vehicle_id,
                    "timeout_s": self.config.decision_poll_timeout_s,
                },
            )
            if response.status_code == 204:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.warning("Decision poll timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("Decision poll server error: %s", e.response.status_code)
            return None
        except Exception as e:
            logger.error("Failed to poll decision: %s", e)
            return None

    async def run(self) -> AsyncGenerator[dict, None]:
        """Run the state collection loop.

        Yields decisions received from the agent server.
        """
        await self.start()

        try:
            if self.config.async_mode:
                decision_queue: asyncio.Queue[dict] = asyncio.Queue()
                sender_task = asyncio.create_task(self._async_send_loop())
                receiver_task = asyncio.create_task(self._async_receive_loop(decision_queue))
                self._async_tasks = [sender_task, receiver_task]

                while self._running:
                    try:
                        decision = await asyncio.wait_for(decision_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if decision:
                        yield decision
            else:
                while self._running:
                    # Get current state
                    state = self._last_state or self.mavlink.get_current_state()

                    if state:
                        # Send to server
                        decision = await self.send_state(state)

                        if decision:
                            yield decision

                    await asyncio.sleep(self.config.update_interval_s)

        finally:
            for task in self._async_tasks:
                task.cancel()
            if self._async_tasks:
                await asyncio.gather(*self._async_tasks, return_exceptions=True)
            self._async_tasks.clear()
            await self.stop()

    async def _async_send_loop(self) -> None:
        """Send state updates without waiting for decisions."""
        while self._running:
            state = self._last_state or self.mavlink.get_current_state()
            if state:
                await self.send_state_async(state)
            await asyncio.sleep(self.config.update_interval_s)

    async def _async_receive_loop(self, decision_queue: asyncio.Queue[dict]) -> None:
        """Poll for decisions and enqueue them."""
        while self._running:
            decision = await self.fetch_decision()
            if decision:
                await decision_queue.put(decision)
            else:
                await asyncio.sleep(self.config.retry_delay_s)

    async def check_server_health(self) -> bool:
        """Check if agent server is healthy.

        Returns:
            True if server is responding
        """
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=self.config.http_timeout_s)

        try:
            response = await self._http_client.get(f"{self.config.server_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_edge_config(self) -> dict | None:
        """Fetch the edge compute simulation configuration from the server.

        Returns:
            Response dict from `/api/config/edge`, or None if failed.
        """
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=self.config.http_timeout_s)

        try:
            response = await self._http_client.get(f"{self.config.server_url}/api/config/edge")
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.warning("Edge config request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("Edge config server error: %s", e.response.status_code)
            return None
        except Exception as e:
            logger.error("Failed to fetch edge config: %s", e)
            return None

    async def send_feedback(self, feedback: dict[str, Any]) -> dict | None:
        """Send decision execution feedback to the agent server.

        Args:
            feedback: Feedback payload matching server DecisionFeedback model.

        Returns:
            Server response dict, or None if failed.
        """
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=self.config.http_timeout_s)

        try:
            response = await self._http_client.post(
                f"{self.config.server_url}/feedback",
                json=feedback,
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            logger.warning("Feedback request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("Feedback server error: %s", e.response.status_code)
            return None
        except Exception as e:
            logger.error("Failed to send feedback: %s", e)
            return None
