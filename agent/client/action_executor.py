"""Action Executor.

Translates high-level decisions from the agent server into
MAVLink commands for the flight controller.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from autonomy.mavlink_interface import MAVLinkInterface
from autonomy.mission_primitives import MissionPrimitives, OrbitPlan, PrimitiveResult
from autonomy.vehicle_state import Position

if TYPE_CHECKING:
    from agent.client.vision_client import InspectionVisionResults, VisionClient

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """Current execution state."""

    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ExecutionResult:
    """Result of executing a decision."""

    decision_id: str
    action: str
    state: ExecutionState
    message: str = ""
    duration_s: float = 0.0


class ActionExecutor:
    """Executes decisions by translating them to flight commands.

    The action executor receives decisions from the agent server
    and uses MissionPrimitives to execute them on the vehicle.

    It handles:
    - Translating decision actions to primitive calls
    - Monitoring execution progress
    - Reporting execution results
    - Handling abort requests

    Example:
        executor = ActionExecutor(mavlink_interface)

        async for decision in state_collector.run():
            result = await executor.execute(decision)
            if result.state == ExecutionState.FAILED:
                logger.error(f"Execution failed: {result.message}")
    """

    def __init__(self, mavlink: MAVLinkInterface, vision_client: "VisionClient | None" = None) -> None:
        """Initialize the ActionExecutor.

        Args:
            mavlink: MAVLink interface for vehicle communication.
            vision_client: Optional vision client for inspection captures.
        """
        self.mavlink = mavlink
        self.primitives = MissionPrimitives(mavlink)
        self.vision_client = vision_client

        self._current_decision: str | None = None
        self._state = ExecutionState.IDLE
        self._last_inspection_results: InspectionVisionResults | None = None

    @property
    def is_executing(self) -> bool:
        """Check if currently executing a decision."""
        return self._state == ExecutionState.EXECUTING

    def abort(self) -> None:
        """Abort current execution."""
        self.primitives.request_abort()
        self._state = ExecutionState.ABORTED

    async def execute(self, decision: dict) -> ExecutionResult:
        """Execute a decision from the agent server.

        Args:
            decision: Decision dict from server

        Returns:
            ExecutionResult indicating success/failure
        """
        start_time = time.time()

        decision_id = decision.get("decision_id", "unknown")
        action = decision.get("action", "none")
        parameters = decision.get("parameters", {})

        self._current_decision = decision_id
        self._state = ExecutionState.EXECUTING
        self.primitives.clear_abort()

        logger.info(f"Executing decision {decision_id}: {action}")

        try:
            result = await self._execute_action(action, parameters)

            duration = time.time() - start_time

            if result == PrimitiveResult.SUCCESS:
                self._state = ExecutionState.COMPLETED
                return ExecutionResult(
                    decision_id=decision_id,
                    action=action,
                    state=ExecutionState.COMPLETED,
                    message="Success",
                    duration_s=duration,
                )
            if result == PrimitiveResult.ABORTED:
                self._state = ExecutionState.ABORTED
                return ExecutionResult(
                    decision_id=decision_id,
                    action=action,
                    state=ExecutionState.ABORTED,
                    message="Aborted",
                    duration_s=duration,
                )
            self._state = ExecutionState.FAILED
            return ExecutionResult(
                decision_id=decision_id,
                action=action,
                state=ExecutionState.FAILED,
                message=f"Failed: {result.value}",
                duration_s=duration,
            )

        except Exception as e:
            self._state = ExecutionState.FAILED
            logger.exception(f"Execution error: {e}")
            return ExecutionResult(
                decision_id=decision_id,
                action=action,
                state=ExecutionState.FAILED,
                message=str(e),
                duration_s=time.time() - start_time,
            )

    async def _execute_action(self, action: str, parameters: dict) -> PrimitiveResult:
        """Execute a specific action type."""
        handlers = {
            "none": self._handle_wait,
            "wait": self._handle_wait,
            "abort": self._handle_abort,
            "takeoff": self._handle_takeoff,
            "goto": self._handle_goto,
            "inspect": self._handle_inspect,
            "dock": self._handle_dock,
            "rtl": self._handle_rtl,
            "land": self._handle_land,
        }
        handler = handlers.get(action)
        if not handler:
            logger.warning("Unknown action type: %s", action)
            return PrimitiveResult.FAILED
        return await handler(parameters)

    async def _handle_wait(self, parameters: dict) -> PrimitiveResult:
        """Handle wait action by sleeping for the specified duration.

        Args:
            parameters: Action parameters containing optional duration_s.

        Returns:
            PrimitiveResult indicating success.
        """
        duration = parameters.get("duration_s", 0)
        if duration > 0:
            await asyncio.sleep(duration)
        return PrimitiveResult.SUCCESS

    async def _handle_abort(self, _parameters: dict) -> PrimitiveResult:
        """Handle abort action by commanding return to launch.

        Args:
            _parameters: Action parameters (unused).

        Returns:
            PrimitiveResult indicating success or failure.
        """
        if await self.mavlink.return_to_launch():
            return PrimitiveResult.SUCCESS
        return PrimitiveResult.FAILED

    async def _handle_takeoff(self, parameters: dict) -> PrimitiveResult:
        """Handle takeoff action by arming and taking off to specified altitude.

        Args:
            parameters: Action parameters containing optional altitude_m.

        Returns:
            PrimitiveResult indicating success or failure.
        """
        altitude = parameters.get("altitude_m", 10.0)
        return await self.primitives.arm_and_takeoff(altitude)

    async def _handle_goto(self, parameters: dict) -> PrimitiveResult:
        """Handle goto action by flying to the specified position.

        Args:
            parameters: Action parameters containing position and optional speed_ms.

        Returns:
            PrimitiveResult indicating success or failure.
        """
        pos = parameters.get("position", {})
        if not pos:
            return PrimitiveResult.FAILED

        target = Position(
            latitude=pos["latitude"],
            longitude=pos["longitude"],
            altitude_msl=pos.get("altitude_msl", 0),
        )
        speed = parameters.get("speed_ms")
        return await self.primitives.goto(target, speed)

    async def _handle_inspect(self, parameters: dict) -> PrimitiveResult:
        """Handle inspect action by flying to position and performing inspection maneuver.

        Args:
            parameters: Action parameters containing position, orbit_radius_m,
                dwell_time_s, and asset_id.

        Returns:
            PrimitiveResult indicating success or failure.
        """
        pos = parameters.get("position", {})
        if not pos:
            return PrimitiveResult.FAILED

        target = Position(
            latitude=pos["latitude"],
            longitude=pos["longitude"],
            altitude_msl=pos.get("altitude_msl", 0),
        )

        result = await self.primitives.goto(target)
        if result != PrimitiveResult.SUCCESS:
            return result

        orbit_radius = parameters.get("orbit_radius_m", 0)
        dwell_time = parameters.get("dwell_time_s", 30)
        asset_id = parameters.get("asset_id", "unknown")

        # Clear previous inspection results to avoid stale feedback payloads
        self._last_inspection_results = None

        # Start vision capture if enabled
        vision_task = None
        if self.vision_client and self.vision_client.enabled:
            inspection_duration = dwell_time  # Use dwell time as duration
            vision_task = asyncio.create_task(
                self.vision_client.capture_during_inspection(
                    asset_id=asset_id,
                    duration_s=inspection_duration,
                    vehicle_state_fn=self._get_vehicle_state_dict,
                )
            )

        # Execute inspection maneuver
        if orbit_radius > 0:
            plan = OrbitPlan(radius=orbit_radius, altitude_agl=20, orbits=1)
            result = await self.primitives.orbit(target, plan)
        else:
            await asyncio.sleep(dwell_time)
            result = PrimitiveResult.SUCCESS

        # Wait for vision capture to complete
        if vision_task:
            try:
                self._last_inspection_results = await vision_task
                logger.info(
                    f"Vision capture complete: {len(self._last_inspection_results.captures)} images"
                )
            except Exception as e:
                logger.error(f"Vision capture failed: {e}")
                self._last_inspection_results = None
                # Continue anyway - vision failure shouldn't fail the inspection

        return result

    async def _handle_dock(self, parameters: dict) -> PrimitiveResult:
        """Handle dock action by flying to dock position or returning to launch.

        Args:
            parameters: Action parameters containing optional dock_position.

        Returns:
            PrimitiveResult indicating success or failure.
        """
        dock_pos = parameters.get("dock_position")
        if dock_pos:
            target = Position(
                latitude=dock_pos["latitude"],
                longitude=dock_pos["longitude"],
                altitude_msl=dock_pos.get("altitude_msl", 0),
            )
            return await self.primitives.dock(target)
        return await self.primitives.return_to_launch()

    async def _handle_rtl(self, _parameters: dict) -> PrimitiveResult:
        """Handle return-to-launch action.

        Args:
            _parameters: Action parameters (unused).

        Returns:
            PrimitiveResult indicating success or failure.
        """
        return await self.primitives.return_to_launch()

    async def _handle_land(self, _parameters: dict) -> PrimitiveResult:
        """Handle land action by landing at current position.

        Args:
            _parameters: Action parameters (unused).

        Returns:
            PrimitiveResult indicating success or failure.
        """
        return await self.primitives.land()

    def _get_vehicle_state_dict(self) -> dict:
        """Get current vehicle state as dictionary for vision metadata."""
        state = self.mavlink.get_vehicle_state()
        if not state:
            return {}

        return {
            "position": {
                "latitude": state.position.latitude,
                "longitude": state.position.longitude,
                "altitude_msl": state.position.altitude_msl,
            },
            "heading_deg": state.heading_deg,
            "altitude_agl": state.altitude_agl,
            "battery_percent": state.battery.remaining_percent if state.battery else None,
        }

    def get_last_inspection_results(self) -> "InspectionVisionResults | None":
        """Get vision results from the last inspection."""
        return self._last_inspection_results
