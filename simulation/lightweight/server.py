"""Lightweight Simulator Server.

FastAPI server that runs the lightweight simulation and provides:
- WebSocket endpoint for real-time state updates
- REST API for commands
- Scenario management API
- Static file serving for the 3D visualizer
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent.server.config_manager import get_default_websocket_url
from simulation.lightweight.demo_recorder import DemoPlayer, DemoRecorder, list_demos
from simulation.lightweight.physics import DroneConfig, EnvironmentConfig
from simulation.lightweight.simulator import LightweightSim

logger = logging.getLogger(__name__)
LOG_ROOT = Path(__file__).resolve().parents[2] / "logs"
SYSTEM_LOG_FILE = LOG_ROOT / "lightweight_system.log"
AGENT_LOG_SOURCES = ("agent_decisions", "agent_telemetry", "agent_outcomes")


def _ensure_system_log_handler() -> None:
    """Ensure lightweight simulator logs are mirrored to a file."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == SYSTEM_LOG_FILE:
            return

    handler = logging.FileHandler(SYSTEM_LOG_FILE)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def _latest_log_file(paths: list[Path]) -> list[Path]:
    """Pick the most recent log file from a list."""
    existing = [path for path in paths if path.exists()]
    if not existing:
        return []
    return [max(existing, key=lambda path: path.stat().st_mtime)]


def _resolve_log_paths(source: str, run_id: str | None = None) -> list[Path]:
    """Resolve log files for a given source."""
    if source == "system":
        _ensure_system_log_handler()
        return [SYSTEM_LOG_FILE] if SYSTEM_LOG_FILE.exists() else []

    if source == "agent_decisions":
        if run_id:
            path = LOG_ROOT / f"decisions_{run_id}.jsonl"
            return [path] if path.exists() else []
        return _latest_log_file(list(LOG_ROOT.glob("decisions_*.jsonl")))

    if source == "agent_telemetry":
        if run_id:
            path = LOG_ROOT / f"telemetry_{run_id}.jsonl"
            return [path] if path.exists() else []
        return _latest_log_file(list(LOG_ROOT.glob("telemetry_*.jsonl")))

    if source == "agent_outcomes":
        outcomes_dir = LOG_ROOT / "outcomes"
        return _latest_log_file(list(outcomes_dir.glob("outcomes_*.jsonl")))

    return []


def _tail_lines(path: Path, limit: int) -> list[str]:
    """Read the last N lines from a file."""
    if limit <= 0 or not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=limit)]


def _parse_timestamp(value: str) -> datetime | None:
    """Parse a timestamp string into a datetime."""
    if not value:
        return None
    sanitized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(sanitized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _extract_timestamp(payload: dict[str, Any]) -> tuple[str | None, datetime | None]:
    """Extract a timestamp string and parsed datetime from a payload."""
    for key in ("timestamp", "updated_at", "started_at", "completed_at"):
        value = payload.get(key)
        if isinstance(value, str):
            parsed = _parse_timestamp(value)
            if parsed:
                return parsed.isoformat(), parsed
            return value, None
    return None, None


def _extract_line_timestamp(line: str) -> tuple[str | None, datetime | None]:
    """Extract a timestamp from a plain log line."""
    prefix = line.split(" | ", 1)[0]
    parsed = _parse_timestamp(prefix)
    if parsed:
        return parsed.isoformat(), parsed
    return None, None


def _normalize_sources(source: str | None, sources: str | None) -> list[str]:
    """Normalize requested log sources into a unique list."""
    raw_sources: list[str] = []
    if sources:
        raw_sources.extend(item.strip().lower() for item in sources.split(",") if item.strip())
    elif source:
        raw_sources.append(source.strip().lower())
    else:
        raw_sources.append("all")

    expanded: list[str] = []
    for item in raw_sources:
        if item in {"all", "*"}:
            expanded.extend(["system", *AGENT_LOG_SOURCES])
        elif item in {"agent", "agent_logs"}:
            expanded.extend(AGENT_LOG_SOURCES)
        elif item in {"sim", "lightweight", "system"}:
            expanded.append("system")
        else:
            expanded.append(item)

    return list(dict.fromkeys(expanded))


def _collect_log_entries(
    sources: list[str],
    limit: int,
    run_id: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect log entries across sources and merge them chronologically."""
    entries: list[tuple[datetime | None, int, dict[str, Any]]] = []
    files: list[str] = []
    order = 0

    for source in sources:
        for path in _resolve_log_paths(source, run_id):
            files.append(str(path))
            for line in _tail_lines(path, limit):
                if not line:
                    continue
                payload: dict[str, Any] | None = None
                timestamp: str | None = None
                sort_key: datetime | None = None
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = None

                if payload is not None:
                    timestamp, sort_key = _extract_timestamp(payload)
                    entry: dict[str, Any] = {
                        "source": source,
                        "path": str(path),
                        "timestamp": timestamp,
                        "payload": payload,
                    }
                else:
                    timestamp, sort_key = _extract_line_timestamp(line)
                    entry = {
                        "source": source,
                        "path": str(path),
                        "timestamp": timestamp,
                        "message": line,
                    }

                entries.append((sort_key, order, entry))
                order += 1

    entries.sort(key=lambda item: (item[0] or datetime.min, item[1]))
    if limit > 0:
        entries = entries[-limit:]
    entries.reverse()

    return [entry for _sort_key, _order, entry in entries], files


def _describe_log_sources() -> dict[str, dict[str, Any]]:
    """Return metadata about available log sources."""
    return {
        "system": {
            "description": "Lightweight simulator/system logs",
            "files": [str(path) for path in _resolve_log_paths("system")],
        },
        "agent_decisions": {
            "description": "Agent decision JSONL logs",
            "files": [str(path) for path in _resolve_log_paths("agent_decisions")],
        },
        "agent_telemetry": {
            "description": "Agent telemetry JSONL logs",
            "files": [str(path) for path in _resolve_log_paths("agent_telemetry")],
        },
        "agent_outcomes": {
            "description": "Agent outcome JSONL logs",
            "files": [str(path) for path in _resolve_log_paths("agent_outcomes")],
        },
    }


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


# =========================================================================
# Agent Server Bridge
# =========================================================================

class AgentBridge:
    """Bridge between lightweight sim and agent server for live reasoning."""

    def __init__(self, agent_url: str | None = None) -> None:
        if agent_url is None:
            agent_url = get_default_websocket_url()
        self.agent_url = agent_url
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.connected = False
        self._running = False
        self._telemetry_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self._visualizer_connections: list[WebSocket] = []
        self._event_callbacks: list[callable] = []

    async def connect(self) -> bool:
        """Connect to agent server WebSocket."""
        try:
            self.ws = await websockets.connect(
                self.agent_url,
                ping_interval=20,
                ping_timeout=10,
            )
            self.connected = True
            logger.info(f"Connected to agent server at {self.agent_url}")

            # Subscribe to all drones
            if simulator:
                drone_ids = list(simulator.drones.keys())
                await self.ws.send(json.dumps({"subscribe": drone_ids}))
                logger.info(f"Subscribed to drones: {drone_ids}")

            return True
        except Exception as e:
            logger.warning(f"Failed to connect to agent server: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from agent server."""
        self._running = False
        if self._telemetry_task:
            self._telemetry_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()
        if self.ws:
            await self.ws.close()
        self.connected = False
        logger.info("Disconnected from agent server")

    async def start(self) -> None:
        """Start telemetry sending and event receiving."""
        if not self.connected:
            if not await self.connect():
                return

        self._running = True
        self._telemetry_task = asyncio.create_task(self._telemetry_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Agent bridge started")

    async def stop(self) -> None:
        """Stop the bridge."""
        await self.disconnect()

    def register_visualizer(self, ws: WebSocket) -> None:
        """Register a visualizer WebSocket to receive agent events."""
        self._visualizer_connections.append(ws)

    def unregister_visualizer(self, ws: WebSocket) -> None:
        """Unregister a visualizer WebSocket."""
        if ws in self._visualizer_connections:
            self._visualizer_connections.remove(ws)

    async def _telemetry_loop(self) -> None:
        """Send drone telemetry to agent server periodically."""
        while self._running and self.connected:
            try:
                if simulator and run_state.running:
                    for drone_id, drone in simulator.drones.items():
                        state = drone.physics.state
                        telemetry = {
                            "type": "telemetry",
                            "drone_id": drone_id,
                            "timestamp_ms": simulator.sim_time * 1000,
                            "position": state.position.tolist(),
                            "velocity": state.velocity.tolist(),
                            "battery_percent": state.battery_percent,
                            "armed": state.armed,
                            "flight_mode": state.flight_mode.value if hasattr(state.flight_mode, 'value') else str(state.flight_mode),
                        }
                        if self.ws:
                            await self.ws.send(json.dumps(telemetry))
                await asyncio.sleep(1.0)  # Send telemetry every second
            except websockets.ConnectionClosed:
                logger.warning("Agent connection closed, attempting reconnect...")
                self.connected = False
                await asyncio.sleep(5)
                await self.connect()
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
                await asyncio.sleep(1)

    async def _receive_loop(self) -> None:
        """Receive events from agent server and forward to visualizers."""
        while self._running:
            try:
                if not self.connected or not self.ws:
                    await asyncio.sleep(1)
                    continue

                message = await self.ws.recv()
                data = json.loads(message)

                # Forward agent events to visualizers
                await self._broadcast_to_visualizers(data)

                # Handle decisions - execute on simulator
                if data.get("type") == "decision":
                    await self._handle_decision(data)

            except websockets.ConnectionClosed:
                logger.warning("Agent connection closed in receive loop")
                self.connected = False
                await asyncio.sleep(5)
                await self.connect()
            except Exception as e:
                logger.error(f"Receive loop error: {e}")
                await asyncio.sleep(0.1)

    async def _broadcast_to_visualizers(self, data: dict) -> None:
        """Broadcast agent event to all connected visualizers."""
        if not self._visualizer_connections:
            return

        message = json.dumps({"type": "agent_event", "event": data})
        for ws in self._visualizer_connections[:]:
            try:
                await ws.send_text(message)
            except Exception:
                self._visualizer_connections.remove(ws)

    async def _handle_decision(self, data: dict) -> None:
        """Execute a decision from the agent server."""
        if not simulator:
            return

        drone_id = data.get("drone_id")
        action = data.get("action")
        params = data.get("parameters", {})

        logger.info(f"Executing decision: {action} for {drone_id}")

        if action == "INSPECT":
            asset_id = params.get("asset_id")
            if asset_id:
                # Use the proper inspect_asset method
                simulator.inspect_asset(drone_id, asset_id)
        elif action == "RTL":
            simulator.rtl(drone_id)
        elif action == "DOCK" or action == "LAND":
            simulator.land(drone_id)
        elif action == "TAKEOFF":
            simulator.arm(drone_id)
            simulator.takeoff(drone_id)
        elif action == "ABORT":
            simulator.rtl(drone_id)
        elif action == "WAIT" or action == "HOVER":
            # Drone stays in place
            pass
        elif action == "GOTO":
            # Direct goto command
            pos = params.get("position")
            if pos and len(pos) >= 3:
                simulator.goto(drone_id, np.array(pos))


# Global agent bridge instance
agent_bridge: AgentBridge | None = None

# Demo recording/playback
demo_recorder: DemoRecorder | None = None
demo_player: DemoPlayer | None = None
_demo_playback_task: asyncio.Task | None = None


def create_app(sim: LightweightSim | None = None) -> FastAPI:
    """Create FastAPI app with simulator endpoints.

    Args:
        sim: Optional simulator instance. If None, creates a default one.

    Returns:
        Configured FastAPI application
    """
    global simulator

    _ensure_system_log_handler()

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
        global simulator, agent_bridge, demo_recorder, demo_player
        simulator = sim or create_default_simulator()
        await simulator.start()
        logger.info("Simulator started")

        # Initialize agent bridge (connects lazily when scenario starts)
        agent_bridge = AgentBridge()

        # Initialize demo recorder/player
        demo_recorder = DemoRecorder()
        demo_player = DemoPlayer()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        global agent_bridge
        if agent_bridge:
            await agent_bridge.stop()
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
            "agent_connected": agent_bridge.connected if agent_bridge else False,
        }

    @app.get("/api/logs/sources")
    def get_log_sources() -> dict:
        """List available log sources."""
        return {"sources": _describe_log_sources()}

    @app.get("/api/logs")
    def get_logs(
        source: str | None = Query(default=None),
        sources: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=2000),
        run_id: str | None = Query(default=None),
    ) -> dict:
        """Return recent log entries for requested sources."""
        requested = _normalize_sources(source, sources)
        invalid = [item for item in requested if item not in {"system", *AGENT_LOG_SOURCES}]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unknown_log_source",
                    "invalid": invalid,
                    "available": ["system", "agent", "all", *AGENT_LOG_SOURCES],
                },
            )

        entries, files = _collect_log_entries(requested, limit, run_id)
        return {
            "sources": requested,
            "files": files,
            "limit": limit,
            "entries": entries,
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
    async def start_scenario(
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

        # Set simulation speed and scenario info
        if simulator:
            simulator.set_real_time_factor(run_state.time_scale)
            simulator.set_scenario_info({
                "id": scenario_id,
                "name": scenario.name,
                "mode": mode,
                "edge_profile": edge_profile,
                "time_scale": run_state.time_scale,
                "duration_minutes": scenario.duration_minutes,
                "started_at": run_state.start_time.isoformat() if run_state.start_time else None,
            })

            # Arm and takeoff all drones when scenario starts
            for drone_id in simulator.drones.keys():
                simulator.arm(drone_id)
                simulator.takeoff(drone_id)

        # Connect to agent server in live mode
        if mode == "live" and agent_bridge:
            await agent_bridge.start()
            logger.info("Agent bridge started for live mode")

        logger.info(f"Started scenario {scenario_id} in {mode} mode at {time_scale}x speed")

        return {
            "status": "started",
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "mode": mode,
            "edge_profile": edge_profile,
            "time_scale": run_state.time_scale,
            "agent_connected": agent_bridge.connected if agent_bridge and mode == "live" else False,
        }

    @app.post("/api/scenarios/stop")
    async def stop_scenario() -> dict:
        """Stop the current scenario."""
        global run_state

        if not run_state.running:
            return {"status": "not_running"}

        scenario_id = run_state.scenario_id
        elapsed = None
        if run_state.start_time:
            elapsed = (datetime.now() - run_state.start_time).total_seconds()

        # Stop agent bridge
        if agent_bridge and agent_bridge.connected:
            await agent_bridge.stop()
            logger.info("Agent bridge stopped")

        # Reset state
        run_state.running = False
        run_state.scenario_id = None
        run_state.start_time = None

        # Clear scenario info from simulator and land drones
        if simulator:
            simulator.set_scenario_info(None)
            for drone_id in simulator.drones.keys():
                simulator.land(drone_id)

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
    # Demo Recording/Playback API
    # =========================================================================

    @app.get("/api/demos")
    def get_demos() -> dict:
        """List available demo recordings."""
        demos = list_demos()
        return {"demos": demos, "count": len(demos)}

    @app.post("/api/demos/record/start")
    def start_demo_recording() -> dict:
        """Start recording the current scenario."""
        if not demo_recorder:
            raise HTTPException(status_code=500, detail="Demo recorder not initialized")
        if not run_state.running:
            raise HTTPException(status_code=400, detail="No scenario running to record")
        if demo_recorder.is_recording:
            raise HTTPException(status_code=409, detail="Already recording")

        scenario = next((s for s in SCENARIOS if s.id == run_state.scenario_id), None)
        scenario_name = scenario.name if scenario else run_state.scenario_id

        demo_recorder.start(
            scenario_id=run_state.scenario_id or "unknown",
            scenario_name=scenario_name or "Unknown",
            edge_profile=run_state.edge_profile,
            time_scale=run_state.time_scale,
        )

        return {"status": "recording", "scenario_id": run_state.scenario_id}

    @app.post("/api/demos/record/stop")
    def stop_demo_recording() -> dict:
        """Stop recording and save the demo."""
        if not demo_recorder:
            raise HTTPException(status_code=500, detail="Demo recorder not initialized")
        if not demo_recorder.is_recording:
            raise HTTPException(status_code=400, detail="Not recording")

        path = demo_recorder.stop()
        if path:
            return {
                "status": "saved",
                "path": str(path),
                "filename": path.name,
            }
        return {"status": "no_events"}

    @app.get("/api/demos/record/status")
    def get_recording_status() -> dict:
        """Get demo recording status."""
        if not demo_recorder:
            return {"recording": False}
        return demo_recorder.get_stats()

    @app.post("/api/demos/{demo_id}/play")
    async def start_demo_playback(demo_id: str, speed: float = 1.0) -> dict:
        """Start playing a demo."""
        global _demo_playback_task

        if not demo_player:
            raise HTTPException(status_code=500, detail="Demo player not initialized")
        if demo_player.is_playing:
            raise HTTPException(status_code=409, detail="Playback already in progress")

        # Find the demo file
        demos = list_demos()
        demo = next((d for d in demos if d["id"] == demo_id), None)
        if not demo:
            raise HTTPException(status_code=404, detail=f"Demo not found: {demo_id}")

        try:
            metadata = demo_player.load(demo["path"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # Start playback in background
        async def playback_loop() -> None:
            async for event in demo_player.play(speed=speed):
                # Broadcast event to visualizers
                if simulator and simulator._ws_connections:
                    message = json.dumps(event.get("data", event))
                    for ws in simulator._ws_connections[:]:
                        try:
                            await ws.send_text(message)
                        except Exception as exc:
                            logger.debug("Playback send failed: %s", exc)

        _demo_playback_task = asyncio.create_task(playback_loop())

        return {
            "status": "playing",
            "demo_id": demo_id,
            "speed": speed,
            "metadata": metadata,
        }

    @app.post("/api/demos/stop")
    def stop_demo_playback() -> dict:
        """Stop demo playback."""
        global _demo_playback_task

        if not demo_player:
            raise HTTPException(status_code=500, detail="Demo player not initialized")

        demo_player.stop()

        if _demo_playback_task:
            _demo_playback_task.cancel()
            _demo_playback_task = None

        return {"status": "stopped"}

    @app.get("/api/demos/playback/status")
    def get_playback_status() -> dict:
        """Get demo playback status."""
        if not demo_player:
            return {"loaded": False}
        return demo_player.get_progress()

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

        # Register with agent bridge to receive agent events
        if agent_bridge:
            agent_bridge.register_visualizer(websocket)

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
            if agent_bridge:
                agent_bridge.unregister_visualizer(websocket)

    # Hook to record state updates during demo recording
    original_broadcast = None

    async def recording_broadcast_wrapper() -> None:
        """Wrapper that records state updates during demo recording."""
        nonlocal original_broadcast

        if simulator and original_broadcast:
            # Get the original broadcast behavior
            await original_broadcast()

            # Record if recording is active
            if demo_recorder and demo_recorder.is_recording:
                states = {}
                for drone_id, drone in simulator.drones.items():
                    states[drone_id] = simulator._get_drone_state_dict(drone)

                demo_recorder.record_event("state_update", {
                    "type": "state_update",
                    "sim_time": simulator._sim_time,
                    "drones": states,
                })

    return app


async def handle_websocket_command(cmd: dict[str, Any], websocket: WebSocket) -> None:
    """Handle a WebSocket command."""
    if not simulator:
        await websocket.send_json({"error": "Simulator not running"})
        return

    # Support new visualizer command format (type: 'command', command_type: '...')
    msg_type = cmd.get("type")
    if msg_type == "command":
        await handle_visualizer_command(cmd, websocket)
        return
    elif msg_type == "override_response":
        await handle_override_response(cmd, websocket)
        return
    elif msg_type == "chat_message":
        await handle_chat_message(cmd, websocket)
        return

    # Legacy command format
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


async def handle_visualizer_command(cmd: dict[str, Any], websocket: WebSocket) -> None:
    """Handle commands from the visualizer UI.

    Supports commands like:
    - inspect_asset: Send drone to inspect a specific asset
    - return_home: Return drone to home/dock position
    - abort: Emergency stop
    - resume: Resume after abort
    - goto: Fly to specific position
    """
    command_id = cmd.get("command_id", "")
    command_type = cmd.get("command_type", "")
    drone_id = cmd.get("drone_id", "")
    parameters = cmd.get("parameters", {})

    response: dict[str, Any] = {
        "type": "command_result",
        "command_id": command_id,
        "drone_id": drone_id,
        "success": False,
        "message": "",
    }

    if not simulator:
        response["message"] = "Simulator not running"
        await websocket.send_json(response)
        return

    if not drone_id:
        response["message"] = "No drone specified"
        await websocket.send_json(response)
        return

    # Check if drone exists
    if drone_id not in simulator.drones:
        response["message"] = f"Drone {drone_id} not found"
        await websocket.send_json(response)
        return

    drone = simulator.drones[drone_id]

    if command_type == "inspect_asset":
        asset_id = parameters.get("asset_id", "")
        if not asset_id:
            response["message"] = "No asset specified"
        elif asset_id not in simulator.world.assets:
            response["message"] = f"Asset {asset_id} not found"
        else:
            # Get asset position and send drone there
            success = simulator.inspect_asset(drone_id, asset_id)
            if success:
                response["success"] = True
                response["message"] = f"Dispatching {drone_id} to inspect {asset_id}"
            else:
                response["message"] = f"Failed to dispatch {drone_id} - drone may not be ready"

    elif command_type == "return_home":
        success = simulator.rtl(drone_id)
        if success:
            response["success"] = True
            response["message"] = f"{drone_id} returning to home"
        else:
            response["message"] = f"Failed to send {drone_id} home"

    elif command_type == "abort":
        # Land immediately
        success = simulator.land(drone_id)
        if success:
            response["success"] = True
            response["message"] = f"{drone_id} aborting - landing"
        else:
            response["message"] = f"Failed to abort {drone_id}"

    elif command_type == "resume":
        # Take off again after abort (mode is IDLE when landed)
        if drone.mode == "IDLE" or drone.mode == "LAND":
            simulator.arm(drone_id)
            success = simulator.takeoff(drone_id)
            if success:
                response["success"] = True
                response["message"] = f"{drone_id} resuming operations"
            else:
                response["message"] = f"Failed to resume {drone_id}"
        else:
            response["message"] = f"{drone_id} is already airborne"
            response["success"] = True

    elif command_type == "goto":
        x = parameters.get("x", 0)
        y = parameters.get("y", 0)
        z = parameters.get("z", 10)
        success = simulator.goto(drone_id, np.array([x, y, -z]))  # -z for altitude
        if success:
            response["success"] = True
            response["message"] = f"{drone_id} heading to ({x}, {y}, {z}m)"
        else:
            response["message"] = f"Failed to navigate {drone_id}"

    else:
        response["message"] = f"Unknown command type: {command_type}"

    await websocket.send_json(response)

    # Also broadcast to other connected clients so they see the result
    if simulator:
        await simulator.broadcast_message(response)


async def handle_override_response(cmd: dict[str, Any], websocket: WebSocket) -> None:
    """Handle override response from the visualizer UI.

    When user approves or denies an override request, this logs the decision
    and could integrate with the agent server for decision pipeline.
    """
    override_id = cmd.get("override_id", "")
    approved = cmd.get("approved", False)
    timeout = cmd.get("timeout", False)

    logger.info(
        f"Override response: id={override_id}, approved={approved}, timeout={timeout}"
    )

    # Acknowledge the response
    ack = {
        "type": "override_acknowledged",
        "override_id": override_id,
        "approved": approved,
        "timeout": timeout,
    }
    await websocket.send_json(ack)

    # Broadcast to all clients
    if simulator:
        await simulator.broadcast_message(ack)

    # TODO: Integrate with agent server to continue/abort decision pipeline
    # This would involve:
    # 1. Looking up the pending decision by override_id
    # 2. Either executing or aborting based on approved flag
    # 3. Notifying the agent server of the outcome


async def handle_chat_message(cmd: dict[str, Any], websocket: WebSocket) -> None:
    """Handle chat messages from the visualizer UI.

    Parses natural language commands and executes them.
    When agent server is connected, forwards for LLM processing.
    """
    import re

    message = cmd.get("message", "").strip().lower()

    response: dict[str, Any] = {
        "type": "chat_response",
        "message": "",
        "command_executed": False,
    }

    if not simulator:
        response["message"] = "Simulator is not running."
        await websocket.send_json(response)
        return

    # Try to parse drone references
    drone_match = re.search(r"drone\s*(\d+|one|two|1|2)", message, re.IGNORECASE)
    drone_id = "drone_001"  # default
    if drone_match:
        num = drone_match.group(1).replace("one", "1").replace("two", "2")
        drone_id = f"drone_00{num}"

    # Parse commands
    executed = False

    # Inspect command
    if any(word in message for word in ["inspect", "check", "scan", "look at"]):
        asset_match = re.search(r"(solar|tank)(?:\s*_?\s*(\d+))?", message, re.IGNORECASE)
        if asset_match:
            asset_type = asset_match.group(1).lower()
            asset_num = asset_match.group(2) or "001"
            asset_id = f"{asset_type}_{asset_num.zfill(3)}"

            if asset_id in simulator.world.assets:
                success = simulator.inspect_asset(drone_id, asset_id)
                if success:
                    response["message"] = f"Sending {drone_id} to inspect {asset_id}"
                    response["command_executed"] = True
                    response["drone_id"] = drone_id
                    executed = True
                else:
                    response["message"] = f"Cannot dispatch {drone_id} right now - it may be busy or not ready."
            else:
                response["message"] = f"Asset '{asset_id}' not found. Available: {', '.join(simulator.world.assets.keys())}"
        else:
            response["message"] = "Which asset should I inspect? Try: 'inspect solar_001' or 'check tank_001'"

    # Return home / RTL command
    elif any(word in message for word in ["home", "return", "rtl", "come back", "dock"]):
        success = simulator.rtl(drone_id)
        if success:
            response["message"] = f"{drone_id} is returning home"
            response["command_executed"] = True
            response["drone_id"] = drone_id
            executed = True
        else:
            response["message"] = f"Cannot send {drone_id} home right now."

    # Abort / Stop command
    elif any(word in message for word in ["abort", "stop", "halt", "emergency", "land"]):
        success = simulator.land(drone_id)
        if success:
            response["message"] = f"{drone_id} is landing now"
            response["command_executed"] = True
            response["drone_id"] = drone_id
            executed = True
        else:
            response["message"] = f"Cannot abort {drone_id} right now."

    # Status query
    elif any(word in message for word in ["status", "where", "position", "location"]):
        if drone_id in simulator.drones:
            drone = simulator.drones[drone_id]
            pos = drone.physics.state.position
            response["message"] = f"{drone_id} is at position ({pos[0]:.1f}, {pos[1]:.1f}, {-pos[2]:.1f}m), mode: {drone.mode}"
        else:
            response["message"] = f"Drone {drone_id} not found."

    # Help
    elif any(word in message for word in ["help", "commands", "what can"]):
        response["message"] = (
            "I understand these commands:\n"
            "• 'Send drone 1 to inspect solar_001'\n"
            "• 'Return drone 2 home'\n"
            "• 'Abort drone 1'\n"
            "• 'Status of drone 1'"
        )

    # Default response
    else:
        response["message"] = (
            "I'm not sure what you mean. Try:\n"
            "• 'Inspect solar panel'\n"
            "• 'Return home'\n"
            "• 'Abort'\n"
            "• 'Status'"
        )

    await websocket.send_json(response)

    # Log to timeline
    logger.info(f"Chat message: '{message}' -> {response['message'][:50]}...")


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
