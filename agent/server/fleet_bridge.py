"""Agent Fleet Bridge - Connects Agent Decisions to Multi-Drone Control.

This module bridges the gap between:
- The agent's decision-making system (single-drone focused)
- The multi-vehicle coordinator (manages multiple AirSim drones)

It provides:
- Per-drone action executors
- Decision routing to correct drone
- Fleet-wide coordination for multi-drone scenarios
- Telemetry aggregation from all drones
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Agent imports
from agent.api_models import ActionType
from agent.server.decision import Decision
from agent.server.scenarios import DOCK_ALTITUDE, DOCK_LATITUDE, DOCK_LONGITUDE

# Simulation imports
try:
    from simulation.airsim_action_executor import (
        AirSimActionExecutor,
        ExecutionResult,
        ExecutionStatus,
        FlightConfig,
    )
    from simulation.coordinate_utils import GeoReference
    from simulation.drone_coordinator import (
        CoordinatorState,
        DroneAssignment,
        DroneCoordinator,
        get_drone_coordinator,
    )
    from simulation.multi_vehicle_manager import (
        ManagedVehicle,
        MultiVehicleManager,
        VehicleState,
        get_multi_vehicle_manager,
    )
    from simulation.realtime_bridge import (
        RealtimeAirSimBridge,
        RealtimeBridgeConfig,
        connect_all_bridges,
        create_multi_vehicle_bridges,
    )

    MULTI_DRONE_AVAILABLE = True
except ImportError as e:
    MULTI_DRONE_AVAILABLE = False
    logging.warning(f"Multi-drone support not available: {e}")

logger = logging.getLogger(__name__)


class FleetState(str, Enum):
    """State of the fleet bridge."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass
class DroneExecutionContext:
    """Context for executing actions on a specific drone."""

    drone_id: str
    vehicle_name: str
    bridge: RealtimeAirSimBridge | None = None
    executor: AirSimActionExecutor | None = None
    current_action: ActionType | None = None
    current_decision: Decision | None = None
    last_result: ExecutionResult | None = None
    is_busy: bool = False


@dataclass
class FleetStatus:
    """Status summary of the entire fleet."""

    state: FleetState
    drone_count: int
    active_drones: int
    busy_drones: int
    drones: list[dict[str, Any]] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "drone_count": self.drone_count,
            "active_drones": self.active_drones,
            "busy_drones": self.busy_drones,
            "drones": self.drones,
            "last_update": self.last_update.isoformat(),
        }


class AgentFleetBridge:
    """Bridges agent decisions to multi-drone execution.

    This class manages:
    1. Multiple drone executors (one per AirSim vehicle)
    2. Decision routing based on drone_id
    3. Fleet-wide status aggregation
    4. Coordinated multi-drone actions

    Example:
        bridge = AgentFleetBridge(host="127.0.0.1")
        await bridge.connect()

        # Load scenario drones
        await bridge.setup_scenario(scenario)

        # Execute decision on specific drone
        result = await bridge.execute_decision("alpha", decision)

        # Or execute on all available
        results = await bridge.execute_fleet_action(ActionType.TAKEOFF)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        vehicle_mapping: dict[str, str] | None = None,
        flight_config: FlightConfig | None = None,
        geo_ref: GeoReference | None = None,
    ) -> None:
        """Initialize the fleet bridge.

        Args:
            host: AirSim host address
            vehicle_mapping: Pre-defined scenario drone_id -> AirSim vehicle mapping
            flight_config: Flight configuration for all executors
        """
        self.host = host
        self.vehicle_mapping = vehicle_mapping or {}
        self.flight_config = flight_config or FlightConfig()
        self._geo_ref = geo_ref or GeoReference(DOCK_LATITUDE, DOCK_LONGITUDE, DOCK_ALTITUDE)
        self.state = FleetState.DISCONNECTED

        # Core components
        self._coordinator: DroneCoordinator | None = None
        self._vehicle_manager: MultiVehicleManager | None = None

        # Per-drone execution contexts
        self._contexts: dict[str, DroneExecutionContext] = {}

        # Bridges (per vehicle)
        self._bridges: dict[str, RealtimeAirSimBridge] = {}

        # Callbacks
        self._on_execution_complete: list[Callable[[str, ExecutionResult], Awaitable[None]]] = []
        self._on_state_change: list[Callable[[FleetState], Awaitable[None]]] = []

        logger.info(f"AgentFleetBridge initialized (host={host})")

    async def connect(self) -> bool:
        """Connect to AirSim and initialize fleet management.

        Returns:
            True if connected and at least one drone is available.
        """
        if not MULTI_DRONE_AVAILABLE:
            logger.error("Multi-drone support not available")
            return False

        self.state = FleetState.CONNECTING
        await self._notify_state_change()

        try:
            # Get the coordinator
            self._coordinator = get_drone_coordinator(
                host=self.host,
                vehicle_mapping=self.vehicle_mapping,
            )

            # Connect coordinator (which connects vehicle manager)
            success = await self._coordinator.connect()
            if not success:
                self.state = FleetState.ERROR
                return False

            # Get vehicle manager reference
            self._vehicle_manager = self._coordinator._vehicle_manager

            # Discover vehicles
            vehicles = await self._vehicle_manager.discover_vehicles()
            logger.info(f"Discovered {len(vehicles)} AirSim vehicles")

            if not vehicles:
                logger.warning("No vehicles found")
                self.state = FleetState.ERROR
                return False

            # Create bridges for each vehicle
            self._bridges = create_multi_vehicle_bridges(
                vehicle_names=vehicles,
                host=self.host,
            )

            # Connect all bridges
            connection_results = await connect_all_bridges(self._bridges)

            # Create execution contexts for connected bridges
            for vehicle_name, connected in connection_results.items():
                if connected:
                    bridge = self._bridges[vehicle_name]

                    # Create executor for this drone
                    executor = AirSimActionExecutor(
                        bridge=bridge,
                        geo_ref=self._geo_ref,
                        config=self.flight_config,
                    )

                    # Get drone_id if mapped
                    drone_id = self._vehicle_manager.get_drone_id(vehicle_name)
                    if not drone_id:
                        drone_id = vehicle_name  # Use vehicle name as fallback

                    context = DroneExecutionContext(
                        drone_id=drone_id,
                        vehicle_name=vehicle_name,
                        bridge=bridge,
                        executor=executor,
                    )
                    self._contexts[drone_id] = context
                    logger.info(f"Created execution context: {drone_id} -> {vehicle_name}")

            if not self._contexts:
                logger.error("No execution contexts created")
                self.state = FleetState.ERROR
                return False

            self.state = FleetState.READY
            await self._notify_state_change()
            logger.info(f"Fleet bridge ready with {len(self._contexts)} drones")
            return True

        except Exception as e:
            logger.error(f"Failed to connect fleet bridge: {e}")
            self.state = FleetState.ERROR
            await self._notify_state_change()
            return False

    async def disconnect(self) -> None:
        """Disconnect all drones and cleanup."""
        # Stop all executors
        for context in self._contexts.values():
            if context.executor:
                try:
                    await context.executor.stop()
                except Exception as e:
                    logger.debug(f"Error stopping executor for {context.drone_id}: {e}")

        # Disconnect bridges
        for vehicle_name, bridge in self._bridges.items():
            try:
                await bridge.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting bridge {vehicle_name}: {e}")

        # Disconnect coordinator
        if self._coordinator:
            await self._coordinator.disconnect()

        self._contexts.clear()
        self._bridges.clear()
        self._coordinator = None
        self._vehicle_manager = None
        self.state = FleetState.DISCONNECTED
        await self._notify_state_change()

        logger.info("Fleet bridge disconnected")

    async def setup_scenario(self, scenario: Any) -> dict[str, bool]:
        """Setup drones for a scenario.

        Loads the scenario into the coordinator and assigns drones.

        Args:
            scenario: The scenario to load

        Returns:
            Dict of drone_id -> setup success
        """
        if not self._coordinator:
            logger.error("Coordinator not connected")
            return {}

        # Load scenario
        success = await self._coordinator.load_scenario(scenario)
        if not success:
            logger.error("Failed to load scenario")
            return {}

        # Get assignments
        assignments = self._coordinator.get_all_assignments()

        results = {}
        active_drone_ids: set[str] = set()
        for assignment in assignments:
            drone_id = assignment.scenario_drone.drone_id
            vehicle_name = assignment.airsim_vehicle_name
            active_drone_ids.add(drone_id)
            bridge = self._bridges.get(vehicle_name)

            # Ensure we have a context for this drone
            if drone_id not in self._contexts:
                # Create context if we have a bridge for this vehicle
                if bridge:
                    executor = AirSimActionExecutor(
                        bridge=bridge,
                        geo_ref=self._geo_ref,
                        config=self.flight_config,
                    )
                    context = DroneExecutionContext(
                        drone_id=drone_id,
                        vehicle_name=vehicle_name,
                        bridge=bridge,
                        executor=executor,
                    )
                    self._contexts[drone_id] = context
                    results[drone_id] = True
                    logger.info(f"Created context for scenario drone: {drone_id}")
                else:
                    results[drone_id] = False
                    logger.warning(f"No bridge for vehicle {vehicle_name}")
            else:
                context = self._contexts[drone_id]
                if not bridge:
                    results[drone_id] = False
                    logger.warning(f"No bridge for vehicle {vehicle_name}")
                    continue
                if context.vehicle_name != vehicle_name:
                    executor = AirSimActionExecutor(
                        bridge=bridge,
                        geo_ref=self._geo_ref,
                        config=self.flight_config,
                    )
                    context.vehicle_name = vehicle_name
                    context.bridge = bridge
                    context.executor = executor
                    context.is_busy = False
                    context.current_action = None
                    logger.info(
                        "Updated context for scenario drone",
                        drone_id=drone_id,
                        vehicle_name=vehicle_name,
                    )
                results[drone_id] = True

        stale_ids = set(self._contexts.keys()) - active_drone_ids
        for stale_id in stale_ids:
            self._contexts.pop(stale_id, None)
            logger.info("Removed stale drone context", drone_id=stale_id)

        # Position drones at initial positions
        position_results = await self._coordinator.setup_initial_positions()
        for drone_id, positioned in position_results.items():
            if drone_id in results:
                results[drone_id] = results[drone_id] and positioned

        return results

    async def execute_decision(
        self,
        drone_id: str,
        decision: Decision,
    ) -> ExecutionResult | None:
        """Execute a decision on a specific drone.

        Args:
            drone_id: The drone to execute on
            decision: The decision to execute

        Returns:
            ExecutionResult or None if drone not found
        """
        context = self._contexts.get(drone_id)
        if not context:
            logger.warning(f"No context for drone: {drone_id}")
            return None

        if not context.executor:
            logger.warning(f"No executor for drone: {drone_id}")
            return None

        if context.is_busy:
            logger.warning(f"Drone {drone_id} is busy")
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                message="Drone is busy with another action",
            )

        try:
            context.is_busy = True
            context.current_decision = decision
            context.current_action = decision.action

            self.state = FleetState.EXECUTING
            await self._notify_state_change()

            # Execute on the drone's executor
            result = await context.executor.execute(decision.model_dump())

            context.last_result = result
            context.is_busy = False
            context.current_action = None

            # Update fleet state
            if not any(c.is_busy for c in self._contexts.values()):
                self.state = FleetState.READY
                await self._notify_state_change()

            # Fire callbacks
            for callback in self._on_execution_complete:
                try:
                    await callback(drone_id, result)
                except Exception as e:
                    logger.debug(f"Callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"Execution error on {drone_id}: {e}")
            context.is_busy = False
            context.current_action = None
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=str(e),
            )

    async def execute_fleet_action(
        self,
        action: ActionType,
        parameters: dict[str, Any] | None = None,
        drone_ids: list[str] | None = None,
    ) -> dict[str, ExecutionResult]:
        """Execute an action on multiple drones.

        Args:
            action: The action type to execute
            parameters: Optional action parameters
            drone_ids: Specific drones to execute on, or None for all

        Returns:
            Dict of drone_id -> ExecutionResult
        """
        if drone_ids is None:
            drone_ids = list(self._contexts.keys())

        results = {}
        tasks = []

        for drone_id in drone_ids:
            decision = Decision(
                action=action,
                parameters=parameters or {},
            )
            tasks.append((drone_id, self.execute_decision(drone_id, decision)))

        for drone_id, task in tasks:
            result = await task
            if result:
                results[drone_id] = result
            else:
                results[drone_id] = ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    message="Execution returned None",
                )

        return results

    async def takeoff_all(self, altitude: float = 10.0) -> dict[str, bool]:
        """Take off all drones.

        Args:
            altitude: Target altitude in meters

        Returns:
            Dict of drone_id -> success
        """
        if self._coordinator:
            return await self._coordinator.takeoff_all(altitude)
        return {}

    async def land_all(self) -> dict[str, bool]:
        """Land all drones."""
        if self._coordinator:
            return await self._coordinator.land_all()
        return {}

    async def return_all_to_dock(self) -> dict[str, bool]:
        """Return all drones to dock."""
        if self._coordinator:
            return await self._coordinator.return_all_to_dock()
        return {}

    def get_status(self) -> FleetStatus:
        """Get current fleet status."""
        drones = []
        active = 0
        busy = 0

        for drone_id, context in self._contexts.items():
            # Get vehicle state if available
            vehicle_state = None
            if self._vehicle_manager:
                managed = self._vehicle_manager.get_vehicle(context.vehicle_name)
                if managed:
                    vehicle_state = managed.to_dict()
                    if managed.state == VehicleState.ACTIVE:
                        active += 1

            if context.is_busy:
                busy += 1

            drone_info = {
                "drone_id": drone_id,
                "vehicle_name": context.vehicle_name,
                "is_busy": context.is_busy,
                "current_action": context.current_action.value if context.current_action else None,
                "last_result": context.last_result.status.value if context.last_result else None,
                "vehicle_state": vehicle_state,
            }
            drones.append(drone_info)

        return FleetStatus(
            state=self.state,
            drone_count=len(self._contexts),
            active_drones=active,
            busy_drones=busy,
            drones=drones,
        )

    def get_context(self, drone_id: str) -> DroneExecutionContext | None:
        """Get execution context for a drone."""
        return self._contexts.get(drone_id)

    def get_executor(self, drone_id: str) -> AirSimActionExecutor | None:
        """Get the action executor for a specific drone."""
        context = self._contexts.get(drone_id)
        return context.executor if context else None

    def get_bridge(self, drone_id: str) -> RealtimeAirSimBridge | None:
        """Get the AirSim bridge for a specific drone."""
        context = self._contexts.get(drone_id)
        return context.bridge if context else None

    def get_drone_ids(self) -> list[str]:
        """Get all registered drone IDs."""
        return list(self._contexts.keys())

    def set_geo_ref(self, geo_ref: GeoReference) -> None:
        """Update the geo reference used by all executors."""
        self._geo_ref = geo_ref
        for context in self._contexts.values():
            if context.executor:
                context.executor.geo_ref = geo_ref

    def on_execution_complete(
        self,
        callback: Callable[[str, ExecutionResult], Awaitable[None]],
    ) -> None:
        """Register callback for execution completion."""
        self._on_execution_complete.append(callback)

    def on_state_change(
        self,
        callback: Callable[[FleetState], Awaitable[None]],
    ) -> None:
        """Register callback for state changes."""
        self._on_state_change.append(callback)

    async def _notify_state_change(self) -> None:
        """Notify registered callbacks of state change."""
        for callback in self._on_state_change:
            try:
                await callback(self.state)
            except Exception as e:
                logger.debug(f"State change callback error: {e}")


# Singleton instance
_fleet_bridge: AgentFleetBridge | None = None


def get_fleet_bridge(
    host: str = "127.0.0.1",
    vehicle_mapping: dict[str, str] | None = None,
) -> AgentFleetBridge:
    """Get or create the global AgentFleetBridge instance."""
    global _fleet_bridge

    if _fleet_bridge is None:
        _fleet_bridge = AgentFleetBridge(host, vehicle_mapping)

    return _fleet_bridge


async def reset_fleet_bridge() -> None:
    """Reset the global fleet bridge."""
    global _fleet_bridge

    if _fleet_bridge is not None:
        await _fleet_bridge.disconnect()
        _fleet_bridge = None
