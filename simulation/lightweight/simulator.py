"""Lightweight Drone Simulator.

Main simulator class that manages multiple drones, provides the same API
as AirSim bridge, and includes WebSocket streaming for visualization.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    Position,
    VehicleState,
    Velocity,
)
from simulation.lightweight.physics import (
    DroneConfig,
    DronePhysics,
    EnvironmentConfig,
    MotorCommand,
    SimpleFlightController,
)

logger = logging.getLogger(__name__)


@dataclass
class Asset:
    """Simulated inspectable asset."""

    asset_id: str
    position: np.ndarray  # NED position
    asset_type: str = "solar_panel"
    inspection_altitude_m: float = 10.0
    has_anomaly: bool = False
    anomaly_severity: float = 0.0


@dataclass
class SimulatedWorld:
    """Simulated world with assets and environment."""

    assets: dict[str, Asset] = field(default_factory=dict)
    dock_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_min: np.ndarray = field(default_factory=lambda: np.array([-500, -500, -200]))
    bounds_max: np.ndarray = field(default_factory=lambda: np.array([500, 500, 0]))

    def add_asset(self, asset: Asset) -> None:
        """Add an asset to the world."""
        self.assets[asset.asset_id] = asset

    def get_nearest_asset(self, position: np.ndarray) -> tuple[Asset | None, float]:
        """Get nearest asset to position."""
        if not self.assets:
            return None, float("inf")

        nearest = None
        min_dist = float("inf")

        for asset in self.assets.values():
            dist = np.linalg.norm(position[:2] - asset.position[:2])
            if dist < min_dist:
                min_dist = dist
                nearest = asset

        return nearest, min_dist


@dataclass
class DroneInstance:
    """A single drone in the simulation."""

    drone_id: str
    physics: DronePhysics
    controller: SimpleFlightController
    target_position: np.ndarray | None = None
    target_yaw: float = 0.0
    mode: str = "IDLE"  # IDLE, GUIDED, RTL, LAND
    current_goal: str | None = None


class LightweightSim:
    """Lightweight drone simulation with physics and visualization.

    Provides the same API as AirSim bridge for easy integration.

    Example:
        sim = LightweightSim()
        sim.add_drone("drone_001")
        await sim.start()

        # Command drone
        sim.goto("drone_001", Position(lat=0, lon=10, alt=20))

        # Get state (same as AirSim)
        state = sim.get_vehicle_state("drone_001")
    """

    def __init__(
        self,
        env_config: EnvironmentConfig | None = None,
        physics_hz: float = 200.0,
        state_hz: float = 50.0,
    ) -> None:
        """Initialize the simulator.

        Args:
            env_config: Environment configuration
            physics_hz: Physics update rate
            state_hz: State broadcast rate
        """
        self.env_config = env_config or EnvironmentConfig()
        self.physics_hz = physics_hz
        self.state_hz = state_hz
        self.physics_dt = 1.0 / physics_hz

        self.drones: dict[str, DroneInstance] = {}
        self.world = SimulatedWorld()

        self._running = False
        self._sim_time = 0.0
        self._real_time_factor = 1.0
        self._physics_task: asyncio.Task | None = None
        self._broadcast_task: asyncio.Task | None = None

        # WebSocket connections for visualization
        self._ws_connections: list[Any] = []

        # State history for replay/analysis
        self._state_history: list[dict] = []
        self._max_history = 10000

        logger.info("LightweightSim initialized")

    def add_drone(
        self,
        drone_id: str,
        config: DroneConfig | None = None,
        initial_position: np.ndarray | None = None,
    ) -> None:
        """Add a drone to the simulation.

        Args:
            drone_id: Unique identifier for the drone
            config: Drone physical configuration
            initial_position: Starting position (NED, meters)
        """
        config = config or DroneConfig()
        initial_pos = (
            initial_position if initial_position is not None else self.world.dock_position.copy()
        )

        physics = DronePhysics(
            config=config,
            env_config=self.env_config,
            initial_position=initial_pos,
        )
        controller = SimpleFlightController(config)

        self.drones[drone_id] = DroneInstance(
            drone_id=drone_id,
            physics=physics,
            controller=controller,
            target_position=initial_pos.copy(),
        )

        logger.info(f"Added drone {drone_id} at position {initial_pos}")

    def remove_drone(self, drone_id: str) -> bool:
        """Remove a drone from the simulation."""
        if drone_id in self.drones:
            del self.drones[drone_id]
            logger.info(f"Removed drone {drone_id}")
            return True
        return False

    async def start(self) -> None:
        """Start the simulation loop."""
        if self._running:
            logger.warning("Simulation already running")
            return

        self._running = True
        self._physics_task = asyncio.create_task(self._physics_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        logger.info("Simulation started")

    async def stop(self) -> None:
        """Stop the simulation."""
        self._running = False

        if self._physics_task:
            self._physics_task.cancel()
            try:
                await self._physics_task
            except asyncio.CancelledError:
                pass

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        logger.info("Simulation stopped")

    async def _physics_loop(self) -> None:
        """Main physics update loop."""
        last_time = time.time()

        while self._running:
            current_time = time.time()
            wall_dt = current_time - last_time
            last_time = current_time

            # Compute simulation dt based on real-time factor
            sim_dt = wall_dt * self._real_time_factor

            # Run physics steps
            steps = max(1, int(sim_dt / self.physics_dt))
            for _ in range(steps):
                self._step_physics(self.physics_dt)
                self._sim_time += self.physics_dt

            # Sleep to maintain target rate
            target_dt = 1.0 / self.physics_hz
            sleep_time = target_dt - (time.time() - current_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def _step_physics(self, dt: float) -> None:
        """Step all drones physics."""
        for drone in self.drones.values():
            # Compute motor command based on mode
            if drone.mode == "IDLE" or not drone.physics.state.armed:
                command = MotorCommand(throttle=np.zeros(4), armed=False)
            elif drone.mode in ("GUIDED", "RTL", "LAND"):
                if drone.target_position is not None:
                    command = drone.controller.compute_command(
                        drone.physics.state,
                        drone.target_position,
                        drone.target_yaw,
                    )
                else:
                    command = MotorCommand(throttle=np.ones(4) * 0.5, armed=True)
            else:
                command = MotorCommand(throttle=np.zeros(4), armed=False)

            # Step physics
            drone.physics.step(command, dt)

    async def _broadcast_loop(self) -> None:
        """Broadcast state to connected clients."""
        while self._running:
            # Collect all drone states
            states = {}
            for drone_id, drone in self.drones.items():
                states[drone_id] = self._get_drone_state_dict(drone)

            # Record history
            if len(self._state_history) >= self._max_history:
                self._state_history.pop(0)
            self._state_history.append({
                "sim_time": self._sim_time,
                "wall_time": time.time(),
                "drones": states,
            })

            # Broadcast to WebSocket clients
            if self._ws_connections:
                message = json.dumps({
                    "type": "state_update",
                    "sim_time": self._sim_time,
                    "drones": states,
                    "assets": {
                        aid: {
                            "position": a.position.tolist(),
                            "type": a.asset_type,
                            "has_anomaly": a.has_anomaly,
                        }
                        for aid, a in self.world.assets.items()
                    },
                })
                for ws in self._ws_connections[:]:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        self._ws_connections.remove(ws)

            await asyncio.sleep(1.0 / self.state_hz)

    def _get_drone_state_dict(self, drone: DroneInstance) -> dict:
        """Get drone state as dictionary for broadcasting."""
        state = drone.physics.state
        roll, pitch, yaw = drone.physics.get_euler_angles()

        return {
            "drone_id": drone.drone_id,
            "position": state.position.tolist(),
            "velocity": state.velocity.tolist(),
            "attitude": {
                "roll": math.degrees(roll),
                "pitch": math.degrees(pitch),
                "yaw": math.degrees(yaw),
            },
            "battery_percent": drone.physics.get_battery_percent(),
            "battery_voltage": state.battery_voltage,
            "armed": state.armed,
            "in_air": state.in_air,
            "crashed": state.crashed,
            "mode": drone.mode,
            "current_goal": drone.current_goal,
        }

    # =========================================================================
    # Public API (AirSim-compatible)
    # =========================================================================

    def arm(self, drone_id: str) -> bool:
        """Arm a drone."""
        if drone_id not in self.drones:
            return False
        self.drones[drone_id].physics.state.armed = True
        self.drones[drone_id].mode = "GUIDED"
        logger.info(f"Armed drone {drone_id}")
        return True

    def disarm(self, drone_id: str) -> bool:
        """Disarm a drone."""
        if drone_id not in self.drones:
            return False
        self.drones[drone_id].physics.state.armed = False
        self.drones[drone_id].mode = "IDLE"
        logger.info(f"Disarmed drone {drone_id}")
        return True

    def takeoff(self, drone_id: str, altitude_m: float = 10.0) -> bool:
        """Command drone to take off."""
        if drone_id not in self.drones:
            return False

        drone = self.drones[drone_id]
        if not drone.physics.state.armed:
            self.arm(drone_id)

        current_pos = drone.physics.state.position.copy()
        drone.target_position = np.array([current_pos[0], current_pos[1], -altitude_m])
        drone.mode = "GUIDED"
        drone.current_goal = "TAKEOFF"

        logger.info(f"Drone {drone_id} taking off to {altitude_m}m")
        return True

    def land(self, drone_id: str) -> bool:
        """Command drone to land."""
        if drone_id not in self.drones:
            return False

        drone = self.drones[drone_id]
        current_pos = drone.physics.state.position.copy()
        drone.target_position = np.array([current_pos[0], current_pos[1], 0.0])
        drone.mode = "LAND"
        drone.current_goal = "LAND"

        logger.info(f"Drone {drone_id} landing")
        return True

    def goto(
        self,
        drone_id: str,
        position: Position | np.ndarray,
        yaw: float = 0.0,
    ) -> bool:
        """Command drone to go to position.

        Args:
            drone_id: Drone identifier
            position: Target position (Position object or NED array)
            yaw: Target yaw in degrees
        """
        if drone_id not in self.drones:
            return False

        drone = self.drones[drone_id]

        if isinstance(position, Position):
            # Convert Position to NED (simplified - assumes local tangent plane)
            target = np.array([
                position.latitude,  # Using lat as North offset
                position.longitude,  # Using lon as East offset
                -position.altitude_msl,  # NED: down is positive
            ])
        else:
            target = position

        drone.target_position = target
        drone.target_yaw = math.radians(yaw)
        drone.mode = "GUIDED"
        drone.current_goal = "GOTO"

        logger.debug(f"Drone {drone_id} going to {target}")
        return True

    def rtl(self, drone_id: str) -> bool:
        """Return to launch (dock)."""
        if drone_id not in self.drones:
            return False

        drone = self.drones[drone_id]
        drone.target_position = self.world.dock_position.copy()
        drone.target_position[2] = -10.0  # RTL altitude
        drone.mode = "RTL"
        drone.current_goal = "RTL"

        logger.info(f"Drone {drone_id} returning to launch")
        return True

    def get_vehicle_state(self, drone_id: str) -> VehicleState | None:
        """Get vehicle state (AirSim-compatible API).

        Returns:
            VehicleState or None if drone not found
        """
        if drone_id not in self.drones:
            return None

        drone = self.drones[drone_id]
        state = drone.physics.state
        roll, pitch, yaw = drone.physics.get_euler_angles()

        # Convert NED position to our Position format
        # In real use, you'd convert to lat/lon
        position = Position(
            latitude=state.position[0],  # North as "latitude"
            longitude=state.position[1],  # East as "longitude"
            altitude_msl=abs(state.position[2]),  # Convert to positive altitude
            altitude_agl=abs(state.position[2]),
        )

        velocity = Velocity(
            north=state.velocity[0],
            east=state.velocity[1],
            down=state.velocity[2],
        )

        attitude = Attitude(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

        battery = BatteryState(
            voltage=state.battery_voltage,
            current=15.0,  # Estimated
            remaining_percent=drone.physics.get_battery_percent(),
        )

        # Map mode
        mode_map = {
            "IDLE": FlightMode.LOITER,
            "GUIDED": FlightMode.GUIDED,
            "RTL": FlightMode.RTL,
            "LAND": FlightMode.LAND,
        }
        flight_mode = mode_map.get(drone.mode, FlightMode.GUIDED)

        return VehicleState(
            timestamp=datetime.now(),
            position=position,
            velocity=velocity,
            attitude=attitude,
            battery=battery,
            mode=flight_mode,
            armed=state.armed,
            in_air=state.in_air,
        )

    def set_wind(
        self,
        speed_ms: float,
        direction_deg: float,
        gust_intensity: float = 0.0,
        turbulence: float = 0.0,
    ) -> None:
        """Set wind conditions."""
        self.env_config.wind_speed_ms = speed_ms
        self.env_config.wind_direction_rad = math.radians(direction_deg)
        self.env_config.wind_gust_intensity = gust_intensity
        self.env_config.wind_turbulence = turbulence

        # Update all drone wind models
        for drone in self.drones.values():
            drone.physics.wind.base_speed_ms = speed_ms
            drone.physics.wind.base_direction_rad = math.radians(direction_deg)
            drone.physics.wind.gust_intensity = gust_intensity
            drone.physics.wind.turbulence = turbulence

        logger.info(f"Wind set: {speed_ms} m/s from {direction_deg}°")

    def add_asset(
        self,
        asset_id: str,
        position: np.ndarray | Position,
        asset_type: str = "solar_panel",
        has_anomaly: bool = False,
        anomaly_severity: float = 0.0,
    ) -> None:
        """Add an inspectable asset to the world."""
        if isinstance(position, Position):
            pos = np.array([position.latitude, position.longitude, -position.altitude_msl])
        else:
            pos = position

        asset = Asset(
            asset_id=asset_id,
            position=pos,
            asset_type=asset_type,
            has_anomaly=has_anomaly,
            anomaly_severity=anomaly_severity,
        )
        self.world.add_asset(asset)
        logger.info(f"Added asset {asset_id} at {pos}")

    def get_sim_time(self) -> float:
        """Get current simulation time in seconds."""
        return self._sim_time

    def set_real_time_factor(self, factor: float) -> None:
        """Set simulation speed relative to real time.

        Args:
            factor: 1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed
        """
        self._real_time_factor = max(0.1, min(10.0, factor))
        logger.info(f"Real-time factor set to {self._real_time_factor}x")

    def register_websocket(self, websocket: Any) -> None:
        """Register a WebSocket connection for state broadcasts."""
        self._ws_connections.append(websocket)
        logger.info(f"WebSocket registered, total: {len(self._ws_connections)}")

    def unregister_websocket(self, websocket: Any) -> None:
        """Unregister a WebSocket connection."""
        if websocket in self._ws_connections:
            self._ws_connections.remove(websocket)
            logger.info(f"WebSocket unregistered, total: {len(self._ws_connections)}")

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running

    def get_state_history(self, last_n: int = 100) -> list[dict]:
        """Get recent state history."""
        return self._state_history[-last_n:]
