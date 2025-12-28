"""
Agent Server Main Entry Point

FastAPI-based HTTP server providing the decision-making API.
The agent client sends vehicle state and receives decisions.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import structlog
import uvicorn
import yaml

try:
    import logfire
except ImportError:  # pragma: no cover - optional dependency
    logfire = None
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent.api_models import ActionType, DecisionResponse, HealthResponse, VehicleStateRequest
from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.dashboard import add_dashboard_routes
from agent.server.decision import Decision
from agent.server.events import Event, EventSeverity, EventType
from agent.server.goal_selector import GoalSelector
from agent.server.goals import Goal, GoalType
from agent.server.models import DecisionFeedback
from agent.server.monitoring import OutcomeTracker
from agent.server.risk_evaluator import RiskAssessment, RiskEvaluator, RiskThresholds
from agent.server.vision.vision_service import VisionService, VisionServiceConfig
from agent.server.world_model import DockStatus, WorldModel
from autonomy.vehicle_state import (
    Position,
)
from metrics.logger import DecisionLogContext, DecisionLogger
from vision.data_models import DetectionResult

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# Global state
class ServerState:
    """Container for server-wide state and shared services."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.world_model = WorldModel()
        self.goal_selector = GoalSelector()
        self.risk_evaluator = RiskEvaluator()
        self.start_time = datetime.now()
        self.decisions_made = 0
        self.last_decision: Decision | None = None
        self.config: dict = {}
        self.log_dir = Path(__file__).resolve().parents[2] / "logs"
        self.decision_logger = DecisionLogger(self.log_dir)
        self.current_run_id: str | None = None

        # Phase 2: Multi-agent critics
        self.critic_orchestrator = CriticOrchestrator(authority_model=AuthorityModel.ESCALATION)
        self.outcome_tracker = OutcomeTracker(log_dir=self.log_dir / "outcomes")

        # Phase 3: Vision system
        self.vision_service = None  # Initialized in lifespan if enabled
        self.vision_enabled = False

    def load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info("config_loaded", path=str(config_path))

    def load_mission(self, mission_path: Path) -> None:
        """Load mission configuration."""
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


# WebSocket connection manager for real-time event broadcasting
class ConnectionManager:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("websocket_connected", total_connections=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info("websocket_disconnected", total_connections=len(self.active_connections))

    async def broadcast(self, event: Event):
        """Broadcast event to all connected clients."""
        message = event.model_dump(mode="json")
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning("websocket_send_failed", error=str(e))
                disconnected.add(connection)

        # Clean up failed connections
        for conn in disconnected:
            self.disconnect(conn)


connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler."""
    await asyncio.sleep(0)
    # Startup
    config_dir = Path(__file__).parent.parent.parent.parent / "configs"

    server_state.load_config(config_dir / "agent_config.yaml")
    server_state.load_mission(config_dir / "mission_config.yaml")

    # Load risk thresholds
    risk_path = config_dir / "risk_thresholds.yaml"
    if risk_path.exists():
        with open(risk_path, encoding="utf-8") as f:
            risk_config = yaml.safe_load(f)

        thresholds = RiskThresholds(
            battery_warning_percent=risk_config.get("battery", {}).get("warning_percent", 30),
            battery_critical_percent=risk_config.get("battery", {}).get("abort_percent", 15),
            wind_warning_ms=risk_config.get("wind", {}).get("warning_ms", 8),
            wind_abort_ms=risk_config.get("wind", {}).get("abort_ms", 12),
        )
        server_state.risk_evaluator = RiskEvaluator(thresholds)

    # Initialize vision service if enabled
    vision_config_path = config_dir / "vision_config.yaml"
    if vision_config_path.exists():
        with open(vision_config_path, encoding="utf-8") as f:
            vision_config = yaml.safe_load(f)

        vision_enabled = vision_config.get("vision", {}).get("enabled", False)
        if vision_enabled:
            try:
                vision_service_config = VisionServiceConfig(
                    confidence_threshold=vision_config.get("vision", {})
                    .get("server", {})
                    .get("detection", {})
                    .get("confidence_threshold", 0.7),
                    severity_threshold=vision_config.get("vision", {})
                    .get("server", {})
                    .get("detection", {})
                    .get("severity_threshold", 0.4),
                )
                server_state.vision_service = VisionService(
                    world_model=server_state.world_model,
                    config=vision_service_config,
                )

                # Initialize vision service
                await server_state.vision_service.initialize()
                server_state.vision_enabled = True
                logger.info("vision_service_initialized")

            except Exception as e:
                logger.error("vision_service_init_failed", error=str(e))
                server_state.vision_enabled = False
    else:
        logger.info("vision_config_not_found", message="Vision system disabled")

    server_state.current_run_id = server_state.decision_logger.start_run()
    logger.info("server_started")

    yield

    # Shutdown
    if server_state.vision_service:
        await server_state.vision_service.shutdown()
        logger.info("vision_service_shutdown")

    server_state.decision_logger.end_run()
    logger.info("server_stopped")


app = FastAPI(
    title="AegisAV Agent Server",
    description="Agentic decision-making server for autonomous aerial monitoring",
    version="0.1.0",
    lifespan=lifespan,
)

# Dashboard routes
add_dashboard_routes(app, server_state.log_dir)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time event broadcasting to dashboard."""
    await connection_manager.connect(websocket)
    try:
        # Keep connection alive and receive ping/pong messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        connection_manager.disconnect(websocket)


@app.get("/api/config/agent")
async def get_agent_config():
    """Return the current agent orchestration configuration."""
    return {
        "use_advanced_engine": server_state.goal_selector.use_advanced_engine,
        "is_initialized": server_state.goal_selector.advanced_engine is not None,
    }


@app.post("/api/config/agent")
async def update_agent_config(config: dict):
    """Update agent orchestration configuration."""
    enabled = config.get("use_advanced_engine", True)
    await server_state.goal_selector.orchestrate(enabled)
    return {"status": "success", "use_advanced_engine": enabled}


if logfire:
    try:
        logfire.configure(send_to_logfire=False)  # Local only for now
        logfire.instrument_fastapi(app)
    except (ImportError, RuntimeError) as e:
        # Missing optional opentelemetry-instrumentation-fastapi
        logging.getLogger(__name__).warning(f"Logfire instrumentation disabled: {e}")


# In-memory log buffer for frontend access
class LogBufferHandler(logging.Handler):
    """Ring buffer log handler for dashboard access."""

    def __init__(self, capacity: int = 50):
        super().__init__()
        self.capacity = capacity
        self.buffer: list[dict] = []

    def emit(self, record: logging.LogRecord):
        """Store formatted log records in a bounded in-memory buffer."""
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "message": self.format(record),
            }
            self.buffer.append(log_entry)
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)
        except Exception:
            self.handleError(record)


log_buffer = LogBufferHandler()
logging.getLogger().addHandler(log_buffer)


@app.get("/api/logs")
async def get_logs():
    """Return the recent log buffer for the dashboard."""
    return {"logs": log_buffer.buffer}


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check server health status."""
    uptime = (datetime.now() - server_state.start_time).total_seconds()

    last_update = None
    update_time = server_state.world_model.time_since_update()
    if update_time:
        last_update = (datetime.now() - update_time).isoformat()

    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        last_state_update=last_update,
        decisions_made=server_state.decisions_made,
    )


@app.post("/api/execution_event")
async def receive_execution_event(event_data: dict) -> dict:
    """
    Receive execution event from client for WebSocket broadcast.

    Args:
        event_data: Execution event data from client

    Returns:
        Confirmation of receipt
    """
    # Broadcast client execution event
    await connection_manager.broadcast(
        Event(
            event_type=EventType.CLIENT_EXECUTION,
            timestamp=datetime.now(),
            data=event_data,
            severity=EventSeverity.INFO,
        )
    )

    return {"status": "received"}


@app.post("/feedback")
async def receive_feedback(feedback: DecisionFeedback) -> dict:
    """
    Receive feedback from client about decision execution outcomes.

    This endpoint allows the client to report back on how decisions played out,
    enabling the outcome tracker to close the feedback loop for learning.

    Args:
        feedback: Decision execution feedback from client

    Returns:
        Confirmation of feedback receipt
    """
    logger.info(
        "feedback_received",
        decision_id=feedback.decision_id,
        status=feedback.status.value,
        battery_consumed=feedback.battery_consumed,
    )

    # Update outcome tracker with execution results
    outcome = await server_state.outcome_tracker.process_feedback(feedback)

    # Process vision data if available and vision service is enabled
    vision_observation = None
    if feedback.anomaly_detected and server_state.vision_enabled and server_state.vision_service:
        try:
            client_detection: DetectionResult | None = None
            image_path: Path | None = None
            vehicle_state: dict | None = None
            asset_id = feedback.asset_inspected or "unknown"

            inspection_data = feedback.inspection_data or {}
            if isinstance(inspection_data, dict):
                asset_id_from_payload = inspection_data.get("asset_id")
                if isinstance(asset_id_from_payload, str) and asset_id_from_payload:
                    asset_id = asset_id_from_payload

                vehicle_state = inspection_data.get("vehicle_state")
                if vehicle_state is not None and not isinstance(vehicle_state, dict):
                    vehicle_state = None

                vision_payload = inspection_data.get("vision", {})
                if isinstance(vision_payload, dict):
                    vehicle_state = vehicle_state or vision_payload.get("vehicle_state")
                    if vehicle_state is not None and not isinstance(vehicle_state, dict):
                        vehicle_state = None

                    image_path_str = (
                        vision_payload.get("best_image_path")
                        or vision_payload.get("best_detection_image")
                        or vision_payload.get("image_path")
                    )
                    if isinstance(image_path_str, str) and image_path_str:
                        image_path = Path(image_path_str)

                    detection_payload = (
                        vision_payload.get("best_detection")
                        or vision_payload.get("client_detection")
                        or vision_payload.get("detection_result")
                    )
                    if isinstance(detection_payload, dict):
                        try:
                            client_detection = DetectionResult.model_validate(detection_payload)
                        except Exception as e:
                            logger.warning(
                                "feedback_client_detection_parse_failed",
                                error=str(e),
                            )

            # Process inspection result
            vision_observation = await server_state.vision_service.process_inspection_result(
                asset_id=asset_id,
                client_detection=client_detection,
                image_path=image_path,
                vehicle_state=vehicle_state,
            )

            # Broadcast vision detection event
            await connection_manager.broadcast(
                Event(
                    event_type=EventType.VISION_DETECTION,
                    timestamp=datetime.now(),
                    data={
                        "observation_id": vision_observation.observation_id,
                        "asset_id": vision_observation.asset_id,
                        "defect_detected": vision_observation.defect_detected,
                        "max_confidence": vision_observation.max_confidence,
                        "max_severity": vision_observation.max_severity,
                    },
                    severity=(
                        EventSeverity.WARNING
                        if vision_observation.defect_detected
                        else EventSeverity.INFO
                    ),
                )
            )

            if vision_observation.anomaly_created:
                logger.info(
                    "vision_anomaly_created",
                    asset_id=vision_observation.asset_id,
                    anomaly_id=vision_observation.anomaly_id,
                    severity=vision_observation.max_severity,
                    confidence=vision_observation.max_confidence,
                )

                # Broadcast anomaly created event
                await connection_manager.broadcast(
                    Event(
                        event_type=EventType.ANOMALY_CREATED,
                        timestamp=datetime.now(),
                        data={
                            "anomaly_id": vision_observation.anomaly_id or "",
                            "asset_id": vision_observation.asset_id,
                            "severity": vision_observation.max_severity,
                            "confidence": vision_observation.max_confidence,
                            "description": f"Vision anomaly detected on {vision_observation.asset_id}",
                        },
                        severity=EventSeverity.CRITICAL,
                    )
                )

        except Exception as e:
            logger.error("vision_processing_failed", error=str(e))

    if outcome:
        response = {
            "status": "feedback_received",
            "decision_id": feedback.decision_id,
            "outcome_status": outcome.execution_status.value,
        }

        if vision_observation:
            response["vision"] = {
                "observation_id": vision_observation.observation_id,
                "defect_detected": vision_observation.defect_detected,
                "anomaly_created": vision_observation.anomaly_created,
                "max_confidence": vision_observation.max_confidence,
                "max_severity": vision_observation.max_severity,
            }

        return response

    logger.warning("feedback_for_unknown_decision", decision_id=feedback.decision_id)
    return {
        "status": "feedback_received_but_unknown_decision",
        "decision_id": feedback.decision_id,
    }


@app.post("/state", response_model=DecisionResponse)
async def receive_state(state: VehicleStateRequest) -> DecisionResponse:
    """
    Receive vehicle state and return decision.

    This is the main endpoint called by the agent client.
    """
    logger.debug("state_received", armed=state.armed, mode=state.mode)

    # Convert to internal VehicleState using the helper method
    vehicle_state = (
        state.to_dataclass()
    )  # Note: despite the name, it returns a Pydantic-based VehicleState

    # Update world model
    server_state.world_model.update_vehicle(vehicle_state)

    # Get world snapshot
    snapshot = server_state.world_model.get_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=500, detail="World model not initialized")

    # Evaluate risk
    risk = server_state.risk_evaluator.evaluate(snapshot)

    # Broadcast risk update event
    await connection_manager.broadcast(
        Event(
            event_type=EventType.RISK_UPDATE,
            timestamp=datetime.now(),
            data={
                "risk_level": risk.overall_level.value,
                "risk_score": risk.overall_score,
                "factors": {name: f.value for name, f in risk.factors.items()},
                "abort_recommended": risk.abort_recommended,
                "warnings": risk.warnings,
            },
            severity=EventSeverity.WARNING if risk.abort_recommended else EventSeverity.INFO,
        )
    )

    # Log warnings
    for warning in risk.warnings:
        logger.warning("risk_warning", warning=warning)

    # Make decision
    decision, goal = await _make_decision(snapshot, risk)

    # Broadcast goal selection event if goal was selected
    if goal:
        await connection_manager.broadcast(
            Event(
                event_type=EventType.GOAL_SELECTED,
                timestamp=datetime.now(),
                data={
                    "goal_type": goal.goal_type.value,
                    "priority": goal.priority,
                    "target": goal.parameters,
                    "confidence": goal.confidence,
                    "reasoning": goal.reasoning,
                },
                severity=EventSeverity.INFO,
            )
        )

    # Phase 2: Validate decision with multi-agent critics
    approved, escalation = await server_state.critic_orchestrator.validate_decision(
        decision, snapshot, risk
    )

    # Broadcast critic validation event
    await connection_manager.broadcast(
        Event(
            event_type=EventType.CRITIC_VALIDATION,
            timestamp=datetime.now(),
            data={
                "approved": approved,
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "escalation": (
                    {
                        "reason": escalation.reason,
                        "severity": escalation.severity.value,
                    }
                    if escalation
                    else None
                ),
            },
            severity=EventSeverity.WARNING if not approved else EventSeverity.INFO,
        )
    )

    # If decision was blocked, override with abort
    if not approved:
        logger.warning(
            "decision_blocked_by_critics",
            decision_id=decision.decision_id,
            original_action=decision.action.value,
            reason=escalation.reason if escalation else "Unknown",
        )
        reason = (
            f"Decision blocked by critics: {escalation.reason if escalation else 'Safety concerns'}"
        )
        decision = Decision.abort(reason=reason)

    # Create outcome tracking for this decision
    server_state.outcome_tracker.create_outcome(decision)

    # Track decision
    server_state.last_decision = decision
    server_state.decisions_made += 1

    server_state.decision_logger.log_decision(
        DecisionLogContext(
            decision=decision,
            risk=risk,
            world=snapshot,
            goal=goal,
            escalation=escalation,
        )
    )

    # Log decision
    logger.info(
        "decision_made",
        decision_id=decision.decision_id,
        action=decision.action.value,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        risk_level=risk.overall_level.value,
    )

    # Broadcast server decision event
    await connection_manager.broadcast(
        Event(
            event_type=EventType.SERVER_DECISION,
            timestamp=datetime.now(),
            data={
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "parameters": decision.parameters,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            },
            severity=EventSeverity.INFO,
        )
    )

    # Map RiskLevel to numerical or string value expected by response
    # The new DecisionResponse uses risk_assessment: Dict[str, float]
    # Older one used risk_level: str

    return DecisionResponse(
        decision_id=decision.decision_id,
        action=decision.action.value,
        parameters=decision.parameters,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        risk_assessment={name: f.value for name, f in risk.factors.items()},
        timestamp=decision.timestamp,
    )


async def _make_decision(snapshot, risk: RiskAssessment) -> tuple[Decision, Goal | None]:
    # pylint: disable=too-many-return-statements
    """
    Core decision-making logic.

    Combines goal selection with risk assessment to produce a decision.
    """
    # Check if abort recommended
    if server_state.risk_evaluator.should_abort(risk):
        return Decision.abort(
            reason=risk.abort_reason or "Risk threshold exceeded",
            risk_factors={name: f.value for name, f in risk.factors.items()},
        ), None

    # Select goal
    goal: Goal = await server_state.goal_selector.select_goal(snapshot)

    # Convert goal to decision
    if goal.goal_type == GoalType.ABORT:
        return Decision.abort(goal.reason), goal

    if goal.goal_type in (
        GoalType.RETURN_LOW_BATTERY,
        GoalType.RETURN_MISSION_COMPLETE,
        GoalType.RETURN_WEATHER,
    ):
        return Decision.return_to_dock(
            reason=goal.reason,
            confidence=goal.confidence,
        ), goal

    if goal.goal_type == GoalType.INSPECT_ASSET and goal.target_asset:
        return Decision.inspect(
            asset_id=goal.target_asset.asset_id,
            position=Position(
                latitude=goal.target_asset.position.latitude,
                longitude=goal.target_asset.position.longitude,
                altitude_msl=goal.target_asset.position.altitude_msl
                + goal.target_asset.inspection_altitude_agl,
            ),
            reason=goal.reason,
            inspection={
                "orbit_radius_m": goal.target_asset.orbit_radius_m,
                "dwell_time_s": goal.target_asset.dwell_time_s,
            },
        ), goal

    if goal.goal_type == GoalType.INSPECT_ANOMALY and goal.target_asset:
        return Decision.inspect(
            asset_id=goal.target_asset.asset_id,
            position=Position(
                latitude=goal.target_asset.position.latitude,
                longitude=goal.target_asset.position.longitude,
                altitude_msl=goal.target_asset.position.altitude_msl
                + goal.target_asset.inspection_altitude_agl,
            ),
            reason=goal.reason,
            inspection={
                "orbit_radius_m": goal.target_asset.orbit_radius_m * 0.75,
                "dwell_time_s": goal.target_asset.dwell_time_s * 2,
            },
        ), goal

    if goal.goal_type == GoalType.WAIT:
        return Decision.wait(goal.reason), goal

    return Decision(
        action=ActionType.NONE,
        reasoning="No actionable goal",
    ), None


# Vision API Endpoints
@app.get("/api/vision/statistics")
async def get_vision_statistics():
    """Get vision system statistics."""
    if not server_state.vision_enabled or not server_state.vision_service:
        raise HTTPException(status_code=503, detail="Vision service not available")

    stats = server_state.vision_service.get_statistics()
    return {
        "enabled": True,
        "statistics": stats,
    }


@app.get("/api/vision/observations")
async def get_vision_observations(limit: int = 100):
    """Get recent vision observations."""
    if not server_state.vision_enabled or not server_state.vision_service:
        raise HTTPException(status_code=503, detail="Vision service not available")

    observations = server_state.vision_service.get_recent_observations(limit=limit)
    return {
        "observations": [
            {
                "observation_id": obs.observation_id,
                "asset_id": obs.asset_id,
                "timestamp": obs.timestamp.isoformat(),
                "defect_detected": obs.defect_detected,
                "max_confidence": obs.max_confidence,
                "max_severity": obs.max_severity,
                "anomaly_created": obs.anomaly_created,
                "anomaly_id": obs.anomaly_id,
            }
            for obs in observations
        ],
        "total": len(observations),
    }


@app.get("/api/vision/observations/{asset_id}")
async def get_asset_observations(asset_id: str):
    """Get vision observations for a specific asset."""
    if not server_state.vision_enabled or not server_state.vision_service:
        raise HTTPException(status_code=503, detail="Vision service not available")

    observations = server_state.vision_service.get_observations_for_asset(asset_id)
    return {
        "asset_id": asset_id,
        "observations": [
            {
                "observation_id": obs.observation_id,
                "timestamp": obs.timestamp.isoformat(),
                "defect_detected": obs.defect_detected,
                "max_confidence": obs.max_confidence,
                "max_severity": obs.max_severity,
                "anomaly_created": obs.anomaly_created,
            }
            for obs in observations
        ],
        "total": len(observations),
    }


def main() -> None:
    """Run the agent server."""
    logging.basicConfig(level=logging.INFO)

    uvicorn.run(
        "agent.server.main:app",
        host="0.0.0.0",  # noqa: S104
        port=8080,
        reload=True,
    )


if __name__ == "__main__":
    main()
