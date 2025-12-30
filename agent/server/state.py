"""Shared server state and connection management."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import structlog
import yaml
from fastapi import WebSocket

from agent.api_models import DecisionResponse
from agent.edge_config import EdgeComputeProfile, default_edge_compute_config
from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.decision import Decision
from agent.server.goal_selector import GoalSelector
from agent.server.monitoring import OutcomeTracker
from agent.server.risk_evaluator import RiskEvaluator
from agent.server.world_model import DockStatus, WorldModel
from autonomy.vehicle_state import Position
from metrics.logger import DecisionLogger
from metrics.telemetry_logger import TelemetryLogger

logger = structlog.get_logger(__name__)


class DecisionQueueManager:
    """Async decision queues keyed by vehicle ID."""

    def __init__(self, maxsize: int = 100) -> None:
        """Initialize the queue manager."""
        self._queues: dict[str, asyncio.Queue[DecisionResponse]] = {}
        self._maxsize = maxsize

    def _get_queue(self, vehicle_id: str) -> asyncio.Queue[DecisionResponse]:
        if vehicle_id not in self._queues:
            self._queues[vehicle_id] = asyncio.Queue(maxsize=self._maxsize)
        return self._queues[vehicle_id]

    async def put(self, vehicle_id: str, decision: DecisionResponse) -> None:
        """Enqueue a decision for a vehicle, dropping oldest if full."""
        queue = self._get_queue(vehicle_id)
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await queue.put(decision)

    async def get(self, vehicle_id: str, timeout_s: float) -> DecisionResponse | None:
        """Await next decision for a vehicle."""
        queue = self._get_queue(vehicle_id)
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None


class ServerState:
    """Container for server-wide state and shared services."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self) -> None:
        """Initialize the ServerState."""
        self.world_model = WorldModel()
        self.goal_selector = GoalSelector()
        self.risk_evaluator = RiskEvaluator()
        self.start_time = datetime.now()
        self.decisions_made = 0
        self.last_decision: Decision | None = None
        self.config: dict = {}
        self.log_dir = Path(__file__).resolve().parents[2] / "logs"
        self.decision_logger = DecisionLogger(self.log_dir)
        self.telemetry_logger = TelemetryLogger(self.log_dir)
        self.current_run_id: str | None = None

        # Phase 2: Multi-agent critics
        self.critic_orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)
        self.outcome_tracker = OutcomeTracker(log_dir=self.log_dir / "outcomes")

        # Phase 3: Vision system
        self.vision_service = None  # Initialized in lifespan if enabled
        self.vision_enabled = False

        # Edge compute simulation profile (client reads this config)
        self.edge_config = default_edge_compute_config(EdgeComputeProfile.SBC_CPU)

        # Phase 4: Persistence (Redis)
        self.store = None  # Initialized in lifespan
        self.persistence_enabled = False

        # Async decision queue for long-polling clients
        self.decision_queue = DecisionQueueManager()

        # Telemetry cache for dashboard access
        self.known_vehicles: set[str] = set()
        self.latest_telemetry: dict[str, dict] = {}

        # AirSim realtime bridge (optional)
        self.airsim_bridge = None
        self.airsim_broadcaster = None
        self.airsim_connect_task: asyncio.Task | None = None
        self.airsim_last_error: str | None = None
        self.airsim_env_last: dict[str, object] = {}

        # AirSim flight control executor (initialized when bridge connects)
        self.airsim_action_executor = None
        self.airsim_geo_ref = None  # GeoReference for coordinate conversion

        # Camera streaming state
        self.camera_streaming: dict[str, asyncio.Task] = {}  # drone_id -> streaming task
        self.camera_stream_sequence: dict[str, int] = {}  # drone_id -> frame sequence

        # Pending AirSim actions queue (for when bridge not connected)
        self.airsim_pending_actions: list[dict] = []
        self.airsim_pending_env: dict[str, object] | None = None
        self.navigation_map: dict[str, object] | None = None
        self.last_depth_capture: dict[str, object] | None = None

    def load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info("config_loaded", path=str(config_path))

    def load_mission(self, mission_path: Path) -> None:
        """Load mission configuration.

        Args:
            mission_path: Path to the mission configuration file.
        """
        if mission_path.exists():
            with open(mission_path, encoding="utf-8") as f:
                mission_config = yaml.safe_load(f)

            # Load assets
            self.world_model.load_assets_from_config(mission_config)

            # Set dock position
            dock_config = mission_config.get("dock", {})
            pos = dock_config.get("position", {})
            if pos:
                self.world_model.set_dock(
                    position=Position(
                        latitude=pos.get("latitude", 0),
                        longitude=pos.get("longitude", 0),
                        altitude_msl=pos.get("altitude_m", 0),
                    ),
                    status=DockStatus.AVAILABLE,
                )

            logger.info("mission_loaded", path=str(mission_path))


server_state = ServerState()


class ConnectionManager:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self) -> None:
        """Initialize the ConnectionManager."""
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket) -> None:
        """Accept and track a new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept.
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("websocket_connected", total_connections=len(self.active_connections))

    def disconnect(self, websocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove.
        """
        self.active_connections.discard(websocket)
        logger.info("websocket_disconnected", total_connections=len(self.active_connections))

    async def broadcast(self, event) -> None:
        """Broadcast event to all connected clients.

        Args:
            event: Event to broadcast to all connections.
        """
        message = event.model_dump(mode="json")
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as exc:
                logger.warning("websocket_send_failed", error=str(exc))
                disconnected.add(connection)

        for conn in disconnected:
            self.disconnect(conn)


connection_manager = ConnectionManager()


class ScenarioRunState:
    """Track running scenario state."""

    def __init__(self) -> None:
        self.running: bool = False
        self.scenario_id: str | None = None
        self.mode: str = "live"  # 'live' or 'demo'
        self.edge_profile: str = "SBC_CPU"
        self.time_scale: float = 1.0
        self.start_time: datetime | None = None


scenario_run_state = ScenarioRunState()


class ScenarioRunnerState:
    """Track active scenario runner task."""

    def __init__(self) -> None:
        self.runner = None
        self.run_task: asyncio.Task | None = None
        self.is_running: bool = False
        self.last_error: str | None = None


scenario_runner_state = ScenarioRunnerState()
