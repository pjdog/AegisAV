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
from typing import Optional

import structlog
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.server.decision import ActionType, Decision
from agent.server.goal_selector import Goal, GoalSelector, GoalType
from agent.server.risk_evaluator import RiskAssessment, RiskEvaluator, RiskThresholds
from agent.server.world_model import DockStatus, WorldModel
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSState,
    Position,
    VehicleHealth,
    VehicleState,
    Velocity,
)

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


# Pydantic models for API
class PositionModel(BaseModel):
    latitude: float
    longitude: float
    altitude_msl: float
    altitude_agl: float = 0.0


class VelocityModel(BaseModel):
    north: float
    east: float
    down: float


class AttitudeModel(BaseModel):
    roll: float
    pitch: float
    yaw: float


class BatteryModel(BaseModel):
    voltage: float
    current: float
    remaining_percent: float


class GPSModel(BaseModel):
    fix_type: int
    satellites_visible: int
    hdop: float
    vdop: float = 99.9


class HealthModel(BaseModel):
    sensors_healthy: bool = True
    gps_healthy: bool = True
    battery_healthy: bool = True
    motors_healthy: bool = True
    ekf_healthy: bool = True


class VehicleStateRequest(BaseModel):
    """Vehicle state sent by agent client."""
    
    timestamp: str
    position: PositionModel
    velocity: VelocityModel
    attitude: AttitudeModel
    battery: BatteryModel
    mode: str
    armed: bool
    in_air: bool
    gps: GPSModel
    health: HealthModel


class DecisionResponse(BaseModel):
    """Decision response to agent client."""
    
    decision_id: str
    action: str
    parameters: dict
    confidence: float
    reasoning: str
    risk_level: str
    timestamp: str


class HealthResponse(BaseModel):
    """Server health status."""
    
    status: str
    uptime_seconds: float
    last_state_update: Optional[str]
    decisions_made: int


# Global state
class ServerState:
    def __init__(self):
        self.world_model = WorldModel()
        self.goal_selector = GoalSelector()
        self.risk_evaluator = RiskEvaluator()
        self.start_time = datetime.now()
        self.decisions_made = 0
        self.last_decision: Optional[Decision] = None
        self.config: dict = {}
    
    def load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
            logger.info("config_loaded", path=str(config_path))
    
    def load_mission(self, mission_path: Path) -> None:
        """Load mission configuration."""
        if mission_path.exists():
            with open(mission_path) as f:
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
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    config_dir = Path(__file__).parent.parent.parent.parent / "configs"
    
    server_state.load_config(config_dir / "agent_config.yaml")
    server_state.load_mission(config_dir / "mission_config.yaml")
    
    # Load risk thresholds
    risk_path = config_dir / "risk_thresholds.yaml"
    if risk_path.exists():
        with open(risk_path) as f:
            risk_config = yaml.safe_load(f)
        
        thresholds = RiskThresholds(
            battery_warning_percent=risk_config.get("battery", {}).get("warning_percent", 30),
            battery_critical_percent=risk_config.get("battery", {}).get("abort_percent", 15),
            wind_warning_ms=risk_config.get("wind", {}).get("warning_ms", 8),
            wind_abort_ms=risk_config.get("wind", {}).get("abort_ms", 12),
        )
        server_state.risk_evaluator = RiskEvaluator(thresholds)
    
    logger.info("server_started")
    
    yield
    
    # Shutdown
    logger.info("server_stopped")


app = FastAPI(
    title="AegisAV Agent Server",
    description="Agentic decision-making server for autonomous aerial monitoring",
    version="0.1.0",
    lifespan=lifespan,
)

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


@app.post("/state", response_model=DecisionResponse)
async def receive_state(state: VehicleStateRequest) -> DecisionResponse:
    """
    Receive vehicle state and return decision.
    
    This is the main endpoint called by the agent client.
    """
    logger.debug("state_received", armed=state.armed, mode=state.mode)
    
    # Convert to internal VehicleState
    vehicle_state = VehicleState(
        timestamp=datetime.fromisoformat(state.timestamp),
        position=Position(
            latitude=state.position.latitude,
            longitude=state.position.longitude,
            altitude_msl=state.position.altitude_msl,
            altitude_agl=state.position.altitude_agl,
        ),
        velocity=Velocity(
            north=state.velocity.north,
            east=state.velocity.east,
            down=state.velocity.down,
        ),
        attitude=Attitude(
            roll=state.attitude.roll,
            pitch=state.attitude.pitch,
            yaw=state.attitude.yaw,
        ),
        battery=BatteryState(
            voltage=state.battery.voltage,
            current=state.battery.current,
            remaining_percent=state.battery.remaining_percent,
        ),
        mode=FlightMode.from_string(state.mode),
        armed=state.armed,
        in_air=state.in_air,
        gps=GPSState(
            fix_type=state.gps.fix_type,
            satellites_visible=state.gps.satellites_visible,
            hdop=state.gps.hdop,
            vdop=state.gps.vdop,
        ),
        health=VehicleHealth(
            sensors_healthy=state.health.sensors_healthy,
            gps_healthy=state.health.gps_healthy,
            battery_healthy=state.health.battery_healthy,
            motors_healthy=state.health.motors_healthy,
            ekf_healthy=state.health.ekf_healthy,
        ),
    )
    
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
    decision = _make_decision(snapshot, risk)
    
    # Track decision
    server_state.last_decision = decision
    server_state.decisions_made += 1
    
    # Log decision
    logger.info(
        "decision_made",
        decision_id=decision.decision_id,
        action=decision.action.value,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        risk_level=risk.overall_level.value,
    )
    
    return DecisionResponse(
        decision_id=decision.decision_id,
        action=decision.action.value,
        parameters=decision.parameters,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        risk_level=risk.overall_level.value,
        timestamp=decision.timestamp.isoformat(),
    )


def _make_decision(snapshot, risk: RiskAssessment) -> Decision:
    """
    Core decision-making logic.
    
    Combines goal selection with risk assessment to produce a decision.
    """
    # Check if abort recommended
    if server_state.risk_evaluator.should_abort(risk):
        return Decision.abort(
            reason=risk.abort_reason or "Risk threshold exceeded",
            risk_factors={name: f.value for name, f in risk.factors.items()},
        )
    
    # Select goal
    goal: Goal = server_state.goal_selector.select_goal(snapshot)
    
    # Convert goal to decision
    if goal.goal_type == GoalType.ABORT:
        return Decision.abort(goal.reason)
    
    elif goal.goal_type in (
        GoalType.RETURN_LOW_BATTERY,
        GoalType.RETURN_MISSION_COMPLETE,
        GoalType.RETURN_WEATHER,
    ):
        return Decision.return_to_dock(
            reason=goal.reason,
            confidence=goal.confidence,
        )
    
    elif goal.goal_type == GoalType.INSPECT_ASSET and goal.target_asset:
        return Decision.inspect(
            asset_id=goal.target_asset.asset_id,
            position=Position(
                latitude=goal.target_asset.position.latitude,
                longitude=goal.target_asset.position.longitude,
                altitude_msl=goal.target_asset.position.altitude_msl + goal.target_asset.inspection_altitude_agl,
            ),
            reason=goal.reason,
            orbit_radius=goal.target_asset.orbit_radius_m,
            dwell_time_s=goal.target_asset.dwell_time_s,
        )
    
    elif goal.goal_type == GoalType.INSPECT_ANOMALY and goal.target_asset:
        return Decision.inspect(
            asset_id=goal.target_asset.asset_id,
            position=Position(
                latitude=goal.target_asset.position.latitude,
                longitude=goal.target_asset.position.longitude,
                altitude_msl=goal.target_asset.position.altitude_msl + goal.target_asset.inspection_altitude_agl,
            ),
            reason=goal.reason,
            orbit_radius=goal.target_asset.orbit_radius_m * 0.75,  # Closer for anomaly
            dwell_time_s=goal.target_asset.dwell_time_s * 2,  # Longer for anomaly
        )
    
    elif goal.goal_type == GoalType.WAIT:
        return Decision.wait(goal.reason)
    
    else:
        return Decision(
            action=ActionType.NONE,
            reasoning="No actionable goal",
        )


def main() -> None:
    """Run the agent server."""
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    uvicorn.run(
        "agent.server.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )


if __name__ == "__main__":
    main()
