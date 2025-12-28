"""
Action Executor

Translates high-level decisions from the agent server into
MAVLink commands for the flight controller.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from autonomy.mavlink_interface import MAVLinkInterface
from autonomy.mission_primitives import MissionPrimitives, PrimitiveResult
from autonomy.vehicle_state import Position

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
    """
    Executes decisions by translating them to flight commands.

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

    def __init__(self, mavlink: MAVLinkInterface):
        self.mavlink = mavlink
        self.primitives = MissionPrimitives(mavlink)

        self._current_decision: str | None = None
        self._state = ExecutionState.IDLE

    @property
    def is_executing(self) -> bool:
        """Check if currently executing a decision."""
        return self._state == ExecutionState.EXECUTING

    def abort(self) -> None:
        """Abort current execution."""
        self.primitives.request_abort()
        self._state = ExecutionState.ABORTED

    async def execute(self, decision: dict) -> ExecutionResult:
        """
        Execute a decision from the agent server.

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
            elif result == PrimitiveResult.ABORTED:
                self._state = ExecutionState.ABORTED
                return ExecutionResult(
                    decision_id=decision_id,
                    action=action,
                    state=ExecutionState.ABORTED,
                    message="Aborted",
                    duration_s=duration,
                )
            else:
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
        duration = parameters.get("duration_s", 0)
        if duration > 0:
            await asyncio.sleep(duration)
        return PrimitiveResult.SUCCESS

    async def _handle_abort(self, _parameters: dict) -> PrimitiveResult:
        if await self.mavlink.return_to_launch():
            return PrimitiveResult.SUCCESS
        return PrimitiveResult.FAILED

    async def _handle_takeoff(self, parameters: dict) -> PrimitiveResult:
        altitude = parameters.get("altitude_m", 10.0)
        return await self.primitives.arm_and_takeoff(altitude)

    async def _handle_goto(self, parameters: dict) -> PrimitiveResult:
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

        if orbit_radius > 0:
            return await self.primitives.orbit(
                center=target,
                radius=orbit_radius,
                altitude_agl=20,
                orbits=1,
            )
        await asyncio.sleep(dwell_time)
        return PrimitiveResult.SUCCESS

    async def _handle_dock(self, parameters: dict) -> PrimitiveResult:
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
        return await self.primitives.return_to_launch()

    async def _handle_land(self, _parameters: dict) -> PrimitiveResult:
        return await self.primitives.land()
