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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.api_models import ActionType, DecisionResponse, HealthResponse, VehicleStateRequest
from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.dashboard import add_dashboard_routes
from agent.server.decision import Decision
from agent.server.goal_selector import Goal, GoalSelector, GoalType
from agent.server.models import DecisionFeedback
from agent.server.monitoring import OutcomeTracker
from agent.server.risk_evaluator import RiskAssessment, RiskEvaluator, RiskThresholds
from agent.server.world_model import DockStatus, WorldModel
from autonomy.vehicle_state import (
    Position,
)
from metrics.logger import DecisionLogContext, DecisionLogger

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

    server_state.current_run_id = server_state.decision_logger.start_run()
    logger.info("server_started")

    yield

    # Shutdown
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
    logfire.configure(send_to_logfire=False)  # Local only for now
    logfire.instrument_fastapi(app)


# In-memory log buffer for frontend access
class LogBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 50):
        super().__init__()
        self.capacity = capacity
        self.buffer: list[dict] = []

    def emit(self, record: logging.LogRecord):
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

    if outcome:
        return {
            "status": "feedback_received",
            "decision_id": feedback.decision_id,
            "outcome_status": outcome.execution_status.value,
        }

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

    # Log warnings
    for warning in risk.warnings:
        logger.warning("risk_warning", warning=warning)

    # Make decision
    decision, goal = await _make_decision(snapshot, risk)

    # Phase 2: Validate decision with multi-agent critics
    approved, escalation = await server_state.critic_orchestrator.validate_decision(
        decision, snapshot, risk
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

    # Map RiskLevel to numerical or string value expected by response
    # The new DecisionResponse uses risk_assessment: Dict[str, float]
    # Older one used risk_level: str

    return DecisionResponse(
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

    elif goal.goal_type == GoalType.WAIT:
        return Decision.wait(goal.reason), goal

    else:
        return Decision(
            action=ActionType.NONE,
            reasoning="No actionable goal",
        ), None


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
