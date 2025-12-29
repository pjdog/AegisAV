"""Lightweight Simulator Server.

FastAPI server that runs the lightweight simulation and provides:
- WebSocket endpoint for real-time state updates
- REST API for commands
- Scenario management API
- Static file serving for the 3D visualizer
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from simulation.lightweight.physics import DroneConfig, EnvironmentConfig
from simulation.lightweight.simulator import LightweightSim

logger = logging.getLogger(__name__)


# =========================================================================
# Scenario Definitions (embedded for self-contained operation)
# =========================================================================

@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""

    id: str
    name: str
    description: str
    category: str
    difficulty: str
    duration_minutes: float
    drone_count: int
    asset_count: int
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a serializable scenario payload."""
        return {
            "scenario_id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty,
            "duration_minutes": self.duration_minutes,
            "drone_count": self.drone_count,
            "asset_count": self.asset_count,
            "tags": self.tags,
        }


SCENARIOS = [
    ScenarioConfig(
        id="normal_ops_001",
        name="Normal Operations",
        description="Routine solar panel inspection with minimal anomalies. Good baseline scenario for testing basic operations.",
        category="normal_operations",
        difficulty="easy",
        duration_minutes=20,
        drone_count=3,
        asset_count=6,
        tags=["training", "baseline"],
    ),
    ScenarioConfig(
        id="battery_cascade_001",
        name="Battery Cascade",
        description="Multi-drone battery degradation cascade requiring coordinated RTL decisions and resource prioritization.",
        category="battery_critical",
        difficulty="hard",
        duration_minutes=15,
        drone_count=3,
        asset_count=4,
        tags=["emergency", "battery", "coordination"],
    ),
    ScenarioConfig(
        id="gps_degrade_001",
        name="GPS Degradation",
        description="Progressive GPS signal loss forcing position estimation fallbacks and safety protocol activation.",
        category="gps_degraded",
        difficulty="hard",
        duration_minutes=25,
        drone_count=3,
        asset_count=3,
        tags=["gps", "navigation", "safety"],
    ),
    ScenarioConfig(
        id="weather_001",
        name="Weather Emergency",
        description="Sudden weather deterioration mid-mission requiring coordinated fleet recall before conditions worsen.",
        category="weather_adverse",
        difficulty="normal",
        duration_minutes=20,
        drone_count=3,
        asset_count=4,
        tags=["weather", "emergency", "recall"],
    ),
    ScenarioConfig(
        id="sensor_cascade_001",
        name="Sensor Cascade",
        description="Progressive sensor failures across fleet testing redundancy systems and graceful degradation protocols.",
        category="sensor_failure",
        difficulty="extreme",
        duration_minutes=18,
        drone_count=3,
        asset_count=3,
        tags=["sensor", "failure", "safety"],
    ),
    ScenarioConfig(
        id="multi_anom_001",
        name="Multi-Anomaly",
        description="Multiple simultaneous asset anomalies detected requiring intelligent prioritization and resource allocation.",
        category="multi_anomaly",
        difficulty="normal",
        duration_minutes=35,
        drone_count=2,
        asset_count=8,
        tags=["anomaly", "prioritization", "vision"],
    ),
    ScenarioConfig(
        id="coord_001",
        name="Coordination Challenge",
        description="Four drones must coordinate inspections in close proximity testing collision avoidance and task deconfliction.",
        category="coordination",
        difficulty="hard",
        duration_minutes=25,
        drone_count=4,
        asset_count=6,
        tags=["coordination", "multi-drone", "proximity"],
    ),
]

EDGE_PROFILES = [
    {"id": "FC_ONLY", "name": "FC Only", "description": "Flight controller only - no vision processing"},
    {"id": "MCU_HEURISTIC", "name": "MCU Heuristic", "description": "Basic heuristics on microcontroller"},
    {"id": "MCU_TINY_CNN", "name": "MCU Tiny CNN", "description": "TinyML CNN for basic detection"},
    {"id": "SBC_CPU", "name": "SBC CPU", "description": "Raspberry Pi class CPU inference"},
    {"id": "SBC_ACCEL", "name": "SBC Accelerated", "description": "SBC with Coral/NPU accelerator"},
    {"id": "JETSON_FULL", "name": "Jetson Full", "description": "Full Jetson GPU inference pipeline"},
]


@dataclass
class SimulationRunState:
    """Track active simulation run."""

    running: bool = False
    scenario_id: str | None = None
    mode: str = "live"
    edge_profile: str = "SBC_CPU"
    time_scale: float = 1.0
    start_time: datetime | None = None


run_state = SimulationRunState()

# Global simulator instance
simulator: LightweightSim | None = None


def create_app(sim: LightweightSim | None = None) -> FastAPI:
    """Create FastAPI app with simulator endpoints.

    Args:
        sim: Optional simulator instance. If None, creates a default one.

    Returns:
        Configured FastAPI application
    """
    global simulator

    app = FastAPI(
        title="AegisAV Lightweight Simulator",
        description="Physics-based drone simulation without Unreal Engine",
        version="1.0.0",
    )

    # CORS for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve visualizer static files
    visualizer_dir = Path(__file__).parent / "visualizer"
    if visualizer_dir.exists():
        app.mount("/viz", StaticFiles(directory=str(visualizer_dir), html=True), name="visualizer")
        logger.info(f"Visualizer mounted at /viz from {visualizer_dir}")

    @app.on_event("startup")
    async def startup() -> None:
        global simulator
        simulator = sim or create_default_simulator()
        await simulator.start()
        logger.info("Simulator started")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if simulator:
            await simulator.stop()
            logger.info("Simulator stopped")

    # =========================================================================
    # REST API
    # =========================================================================

    @app.get("/")
    def root() -> dict:
        """Root endpoint with links."""
        return {
            "name": "AegisAV Lightweight Simulator",
            "visualizer": "/viz/",
            "api_docs": "/docs",
            "websocket": "/ws/sim",
        }

    @app.get("/api/status")
    def get_status() -> dict:
        """Get simulator status."""
        if not simulator:
            return {"running": False}

        return {
            "running": simulator.is_running,
            "sim_time": simulator.get_sim_time(),
            "drones": list(simulator.drones.keys()),
            "assets": list(simulator.world.assets.keys()),
        }

    @app.get("/api/drones")
    def get_drones() -> dict:
        """Get all drone states."""
        if not simulator:
            return {"drones": {}}

        states = {}
        for drone_id in simulator.drones:
            state = simulator.get_vehicle_state(drone_id)
            if state:
                states[drone_id] = {
                    "position": {
                        "lat": state.position.latitude,
                        "lon": state.position.longitude,
                        "alt": state.position.altitude_msl,
                    },
                    "velocity": {
                        "north": state.velocity.north,
                        "east": state.velocity.east,
                        "down": state.velocity.down,
                    },
                    "battery_percent": state.battery.remaining_percent,
                    "armed": state.armed,
                    "mode": state.mode.value,
                }

        return {"drones": states}

    @app.post("/api/drones/{drone_id}/arm")
    def arm_drone(drone_id: str) -> dict:
        """Arm a drone."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        success = simulator.arm(drone_id)
        return {"success": success}

    @app.post("/api/drones/{drone_id}/disarm")
    def disarm_drone(drone_id: str) -> dict:
        """Disarm a drone."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        success = simulator.disarm(drone_id)
        return {"success": success}

    @app.post("/api/drones/{drone_id}/takeoff")
    def takeoff_drone(drone_id: str, altitude: float = 10.0) -> dict:
        """Command drone to take off."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        success = simulator.takeoff(drone_id, altitude)
        return {"success": success}

    @app.post("/api/drones/{drone_id}/land")
    def land_drone(drone_id: str) -> dict:
        """Command drone to land."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        success = simulator.land(drone_id)
        return {"success": success}

    @app.post("/api/drones/{drone_id}/rtl")
    def rtl_drone(drone_id: str) -> dict:
        """Command drone to return to launch."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        success = simulator.rtl(drone_id)
        return {"success": success}

    @app.post("/api/drones/{drone_id}/goto")
    def goto_drone(drone_id: str, x: float, y: float, z: float, yaw: float = 0.0) -> dict:
        """Command drone to go to position."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        success = simulator.goto(drone_id, np.array([x, y, -z]), yaw)
        return {"success": success}

    @app.post("/api/environment/wind")
    def set_wind(speed: float = 0.0, direction: float = 0.0) -> dict:
        """Set wind conditions."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        simulator.set_wind(speed, direction)
        return {"success": True, "wind_speed": speed, "wind_direction": direction}

    @app.post("/api/simulation/speed")
    def set_speed(factor: float = 1.0) -> dict:
        """Set simulation speed."""
        if not simulator:
            return {"success": False, "error": "Simulator not running"}
        simulator.set_real_time_factor(factor)
        return {"success": True, "speed_factor": factor}

    # =========================================================================
    # Scenario API
    # =========================================================================

    @app.get("/api/scenarios")
    def list_scenarios() -> dict:
        """List all available scenarios."""
        return {
            "scenarios": [s.to_dict() for s in SCENARIOS],
            "count": len(SCENARIOS),
        }

    @app.get("/api/scenarios/status")
    def get_scenario_status() -> dict:
        """Get current scenario status."""
        if not run_state.running:
            return {"running": False}

        elapsed = None
        if run_state.start_time:
            elapsed = (datetime.now() - run_state.start_time).total_seconds()

        return {
            "running": True,
            "scenario_id": run_state.scenario_id,
            "mode": run_state.mode,
            "edge_profile": run_state.edge_profile,
            "time_scale": run_state.time_scale,
            "elapsed_s": elapsed,
        }

    @app.get("/api/scenarios/{scenario_id}")
    def get_scenario(scenario_id: str) -> dict:
        """Get scenario details."""
        scenario = next((s for s in SCENARIOS if s.id == scenario_id), None)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")
        return {"scenario": scenario.to_dict()}

    @app.post("/api/scenarios/{scenario_id}/start")
    def start_scenario(
        scenario_id: str,
        mode: str = "live",
        edge_profile: str = "SBC_CPU",
        time_scale: float = 1.0,
    ) -> dict:
        """Start a scenario."""
        global run_state

        scenario = next((s for s in SCENARIOS if s.id == scenario_id), None)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")

        if run_state.running:
            raise HTTPException(status_code=409, detail="Scenario already running")

        # Update run state
        run_state.running = True
        run_state.scenario_id = scenario_id
        run_state.mode = mode
        run_state.edge_profile = edge_profile
        run_state.time_scale = max(0.5, min(5.0, time_scale))
        run_state.start_time = datetime.now()

        # Set simulation speed
        if simulator:
            simulator.set_real_time_factor(run_state.time_scale)

        logger.info(f"Started scenario {scenario_id} in {mode} mode at {time_scale}x speed")

        return {
            "status": "started",
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "mode": mode,
            "edge_profile": edge_profile,
            "time_scale": run_state.time_scale,
        }

    @app.post("/api/scenarios/stop")
    def stop_scenario() -> dict:
        """Stop the current scenario."""
        global run_state

        if not run_state.running:
            return {"status": "not_running"}

        scenario_id = run_state.scenario_id
        elapsed = None
        if run_state.start_time:
            elapsed = (datetime.now() - run_state.start_time).total_seconds()

        # Reset state
        run_state.running = False
        run_state.scenario_id = None
        run_state.start_time = None

        logger.info(f"Stopped scenario {scenario_id} after {elapsed:.1f}s")

        return {
            "status": "stopped",
            "scenario_id": scenario_id,
            "elapsed_s": elapsed,
        }

    @app.get("/api/config/edge-profiles")
    def get_edge_profiles() -> dict:
        """Get available edge compute profiles."""
        return {"profiles": EDGE_PROFILES}

    # =========================================================================
    # WebSocket
    # =========================================================================

    @app.websocket("/ws/sim")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time state updates."""
        await websocket.accept()
        logger.info("WebSocket client connected")

        if simulator:
            simulator.register_websocket(websocket)

        try:
            while True:
                # Receive commands from client
                data = await websocket.receive_text()
                try:
                    cmd = json.loads(data)
                    await handle_websocket_command(cmd, websocket)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if simulator:
                simulator.unregister_websocket(websocket)

    return app


async def handle_websocket_command(cmd: dict[str, Any], websocket: WebSocket) -> None:
    """Handle a WebSocket command."""
    if not simulator:
        await websocket.send_json({"error": "Simulator not running"})
        return

    command = cmd.get("command")
    drone_id = cmd.get("drone_id")

    response: dict[str, Any] = {"command": command, "success": False}

    if command == "takeoff" and drone_id:
        response["success"] = simulator.takeoff(drone_id)
    elif command == "land" and drone_id:
        response["success"] = simulator.land(drone_id)
    elif command == "rtl" and drone_id:
        response["success"] = simulator.rtl(drone_id)
    elif command == "arm" and drone_id:
        response["success"] = simulator.arm(drone_id)
    elif command == "disarm" and drone_id:
        response["success"] = simulator.disarm(drone_id)
    elif command == "goto" and drone_id:
        x, y, z = cmd.get("x", 0), cmd.get("y", 0), cmd.get("z", 10)
        response["success"] = simulator.goto(drone_id, np.array([x, y, -z]))
    elif command == "set_speed":
        factor = cmd.get("factor", 1.0)
        simulator.set_real_time_factor(factor)
        response["success"] = True
    elif command == "set_wind":
        speed = cmd.get("speed", 0.0)
        direction = cmd.get("direction", 0.0)
        simulator.set_wind(speed, direction)
        response["success"] = True
    else:
        response["error"] = f"Unknown command: {command}"

    await websocket.send_json(response)


def create_default_simulator() -> LightweightSim:
    """Create a default simulator with sample scenario."""
    # Environment with light wind
    env = EnvironmentConfig(
        wind_speed_ms=3.0,
        wind_direction_rad=0.0,
        wind_gust_intensity=0.2,
        wind_turbulence=0.1,
    )

    sim = LightweightSim(env_config=env)

    # Add drones
    sim.add_drone("drone_001", DroneConfig(), np.array([0.0, 0.0, 0.0]))
    sim.add_drone("drone_002", DroneConfig(), np.array([5.0, 0.0, 0.0]))

    # Add assets to inspect
    sim.add_asset("solar_001", np.array([50.0, 30.0, 0.0]), "solar_panel")
    sim.add_asset("solar_002", np.array([50.0, 50.0, 0.0]), "solar_panel")
    sim.add_asset("solar_003", np.array([70.0, 30.0, 0.0]), "solar_panel")
    sim.add_asset(
        "solar_004",
        np.array([70.0, 50.0, 0.0]),
        "solar_panel",
        has_anomaly=True,
        anomaly_severity=0.7,
    )
    sim.add_asset("tank_001", np.array([-40.0, 20.0, 0.0]), "tank")
    sim.add_asset("tank_002", np.array([-40.0, -20.0, 0.0]), "tank")

    return sim


def run_server(host: str = "127.0.0.1", port: int = 8081) -> None:
    """Run the simulator server.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting simulator server on {host}:{port}")
    logger.info(f"Visualizer available at http://localhost:{port}/viz/")

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
