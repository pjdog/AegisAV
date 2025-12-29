"""Unreal Engine Real-Time Streaming Endpoint.

Provides high-frequency WebSocket streams for Unreal visualization:
- Telemetry stream: Position, velocity, IMU at 30-50 Hz
- Thinking stream: Agent reasoning, critic evaluations, decisions
- Combined stream: Both telemetry + thinking for thought bubble visualization

The protocol is designed for sub-100ms latency and flawless synchronization.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Message Types for Unreal Visualization
# =============================================================================


class UnrealMessageType(str, Enum):
    """Message types for Unreal WebSocket protocol."""

    # Telemetry (high frequency)
    TELEMETRY = "telemetry"  # Position, velocity, attitude
    IMU = "imu"  # Raw IMU data

    # Agent thinking (event-driven)
    THINKING_START = "thinking_start"  # Agent begins deliberation
    THINKING_UPDATE = "thinking_update"  # Intermediate reasoning
    THINKING_COMPLETE = "thinking_complete"  # Decision made

    # Critics
    CRITIC_START = "critic_start"  # Critic evaluation begins
    CRITIC_RESULT = "critic_result"  # Individual critic verdict

    # Risk & Goals
    RISK_UPDATE = "risk_update"  # Risk assessment changed
    GOAL_CHANGED = "goal_changed"  # Mission goal updated

    # Decisions
    DECISION = "decision"  # Final decision output
    DECISION_EXECUTING = "decision_executing"  # Client executing
    DECISION_COMPLETE = "decision_complete"  # Execution finished

    # Anomalies
    ANOMALY_DETECTED = "anomaly_detected"  # Vision detection
    ANOMALY_RESOLVED = "anomaly_resolved"  # Anomaly handled

    # System
    SYNC = "sync"  # Full state synchronization
    HEARTBEAT = "heartbeat"  # Keep-alive
    ERROR = "error"  # Error message


class CognitiveLevel(str, Enum):
    """Agent cognitive processing level."""
    REACTIVE = "reactive"  # Fast, rule-based
    DELIBERATIVE = "deliberative"  # Planning, reasoning
    REFLECTIVE = "reflective"  # Self-evaluation
    PREDICTIVE = "predictive"  # Future modeling


class CriticVerdict(str, Enum):
    """Critic evaluation verdict."""
    APPROVE = "approve"
    APPROVE_WITH_CONCERNS = "approve_with_concerns"
    ESCALATE = "escalate"
    REJECT = "reject"


# =============================================================================
# Message Schemas
# =============================================================================


class ThinkingState(BaseModel):
    """Current agent thinking state for visualization."""

    drone_id: str
    sequence: int
    timestamp_ms: float

    # Cognitive context
    cognitive_level: CognitiveLevel = CognitiveLevel.REACTIVE
    urgency: str = "normal"  # low, normal, high, critical

    # Current goal
    current_goal: str | None = None
    target_asset: str | None = None

    # Reasoning (for thought bubble text)
    situation: str = ""  # Brief situation assessment
    considerations: list[str] = Field(default_factory=list)  # Bullet points
    options: list[dict[str, Any]] = Field(default_factory=list)  # Evaluated options

    # Critics summary
    critics: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Risk summary
    risk_score: float = 0.0
    risk_level: str = "low"
    risk_factors: dict[str, float] = Field(default_factory=dict)

    # Decision (if made)
    decision_action: str | None = None
    decision_confidence: float = 0.0
    decision_reasoning: str = ""


class CriticEvaluation(BaseModel):
    """Individual critic evaluation result."""

    drone_id: str
    sequence: int
    timestamp_ms: float

    critic_name: str  # safety, efficiency, goal_alignment
    verdict: CriticVerdict
    confidence: float
    concerns: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    used_llm: bool = False


class DecisionMessage(BaseModel):
    """Decision output message."""

    drone_id: str
    sequence: int
    timestamp_ms: float

    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    reasoning: str

    # Context
    risk_level: str
    risk_score: float
    battery_percent: float

    # Critics
    critic_approved: bool
    critic_concerns: list[str] = Field(default_factory=list)
    escalation_level: str | None = None


class RiskUpdate(BaseModel):
    """Risk assessment update."""

    drone_id: str
    sequence: int
    timestamp_ms: float

    overall_score: float
    level: str  # low, moderate, high, critical
    factors: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    abort_recommended: bool = False


# =============================================================================
# Unreal Connection Manager
# =============================================================================


class UnrealConnectionManager:
    """Manages WebSocket connections from Unreal clients.

    Supports multiple concurrent Unreal instances with independent state.
    Provides high-frequency broadcast and per-drone filtering.
    """

    def __init__(self) -> None:
        self.connections: dict[str, WebSocket] = {}  # connection_id -> websocket
        self.drone_subscriptions: dict[str, set[str]] = {}  # connection_id -> drone_ids
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        drone_ids: list[str] | None = None
    ) -> None:
        """Accept a new Unreal connection."""
        await websocket.accept()
        self.connections[connection_id] = websocket
        self.drone_subscriptions[connection_id] = set(drone_ids) if drone_ids else set()
        logger.info(f"Unreal client connected: {connection_id}")

    def disconnect(self, connection_id: str) -> None:
        """Remove a connection."""
        self.connections.pop(connection_id, None)
        self.drone_subscriptions.pop(connection_id, None)
        logger.info(f"Unreal client disconnected: {connection_id}")

    async def next_sequence(self) -> int:
        """Get next sequence number."""
        async with self._lock:
            self._sequence += 1
            return self._sequence

    async def broadcast(
        self,
        message_type: UnrealMessageType,
        data: dict[str, Any],
        drone_id: str | None = None
    ) -> None:
        """Broadcast message to all connected Unreal clients.

        Args:
            message_type: Type of message
            data: Message payload
            drone_id: If set, only send to clients subscribed to this drone
        """
        message = {
            "type": message_type.value,
            "sequence": await self.next_sequence(),
            "timestamp_ms": time.time() * 1000,
            **data
        }

        disconnected = []

        for conn_id, websocket in self.connections.items():
            # Filter by drone subscription
            if drone_id:
                subs = self.drone_subscriptions.get(conn_id, set())
                if subs and drone_id not in subs:
                    continue

            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to {conn_id}: {e}")
                disconnected.append(conn_id)

        # Cleanup disconnected
        for conn_id in disconnected:
            self.disconnect(conn_id)

    async def send_to(
        self,
        connection_id: str,
        message_type: UnrealMessageType,
        data: dict[str, Any]
    ) -> bool:
        """Send message to specific connection."""
        websocket = self.connections.get(connection_id)
        if not websocket:
            return False

        try:
            message = {
                "type": message_type.value,
                "sequence": await self.next_sequence(),
                "timestamp_ms": time.time() * 1000,
                **data
            }
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False

    @property
    def active_connections(self) -> int:
        """Number of active connections."""
        return len(self.connections)


# Global manager instance
unreal_manager = UnrealConnectionManager()


# =============================================================================
# Thinking State Tracker
# =============================================================================


class ThinkingStateTracker:
    """Tracks and broadcasts agent thinking state in real-time.

    Integrates with the decision engine to emit thinking updates
    as the agent reasons through decisions.
    """

    def __init__(self, manager: UnrealConnectionManager) -> None:
        self.manager = manager
        self._states: dict[str, ThinkingState] = {}  # drone_id -> current state
        self._sequence = 0

    async def _next_seq(self) -> int:
        self._sequence += 1
        return self._sequence

    async def start_thinking(
        self,
        drone_id: str,
        goal: str | None = None,
        target: str | None = None,
        cognitive_level: CognitiveLevel = CognitiveLevel.DELIBERATIVE
    ) -> None:
        """Signal that agent is beginning to think."""
        state = ThinkingState(
            drone_id=drone_id,
            sequence=await self._next_seq(),
            timestamp_ms=time.time() * 1000,
            cognitive_level=cognitive_level,
            current_goal=goal,
            target_asset=target,
            situation="Evaluating current situation..."
        )
        self._states[drone_id] = state

        await self.manager.broadcast(
            UnrealMessageType.THINKING_START,
            state.model_dump(),
            drone_id=drone_id
        )

    async def update_thinking(
        self,
        drone_id: str,
        situation: str | None = None,
        considerations: list[str] | None = None,
        options: list[dict] | None = None,
        risk_score: float | None = None,
        risk_level: str | None = None,
        risk_factors: dict[str, float] | None = None
    ) -> None:
        """Update current thinking state."""
        state = self._states.get(drone_id)
        if not state:
            return

        if situation:
            state.situation = situation
        if considerations:
            state.considerations = considerations
        if options:
            state.options = options
        if risk_score is not None:
            state.risk_score = risk_score
        if risk_level:
            state.risk_level = risk_level
        if risk_factors:
            state.risk_factors = risk_factors

        state.sequence = await self._next_seq()
        state.timestamp_ms = time.time() * 1000

        await self.manager.broadcast(
            UnrealMessageType.THINKING_UPDATE,
            state.model_dump(),
            drone_id=drone_id
        )

    async def report_critic(
        self,
        drone_id: str,
        critic_name: str,
        verdict: CriticVerdict,
        confidence: float,
        concerns: list[str] | None = None,
        processing_time_ms: float = 0.0,
        used_llm: bool = False
    ) -> None:
        """Report a critic evaluation result."""
        eval_msg = CriticEvaluation(
            drone_id=drone_id,
            sequence=await self._next_seq(),
            timestamp_ms=time.time() * 1000,
            critic_name=critic_name,
            verdict=verdict,
            confidence=confidence,
            concerns=concerns or [],
            processing_time_ms=processing_time_ms,
            used_llm=used_llm
        )

        # Update state
        state = self._states.get(drone_id)
        if state:
            state.critics[critic_name] = {
                "verdict": verdict.value,
                "confidence": confidence,
                "concerns": concerns or []
            }

        await self.manager.broadcast(
            UnrealMessageType.CRITIC_RESULT,
            eval_msg.model_dump(),
            drone_id=drone_id
        )

    async def complete_thinking(
        self,
        drone_id: str,
        action: str,
        confidence: float,
        reasoning: str,
        _parameters: dict[str, Any] | None = None,
    ) -> None:
        """Signal that thinking is complete with a decision."""
        state = self._states.get(drone_id)
        if state:
            state.decision_action = action
            state.decision_confidence = confidence
            state.decision_reasoning = reasoning
            state.sequence = await self._next_seq()
            state.timestamp_ms = time.time() * 1000

            await self.manager.broadcast(
                UnrealMessageType.THINKING_COMPLETE,
                state.model_dump(),
                drone_id=drone_id
            )

    def get_state(self, drone_id: str) -> ThinkingState | None:
        """Get current thinking state for a drone."""
        return self._states.get(drone_id)


# Global tracker instance
thinking_tracker = ThinkingStateTracker(unreal_manager)


# =============================================================================
# WebSocket Route Handler
# =============================================================================


async def handle_unreal_websocket(
    websocket: WebSocket,
    connection_id: str | None = None
) -> None:
    """Handle WebSocket connection from Unreal client.

    Protocol:
    1. Client connects
    2. Client sends subscription message: {"subscribe": ["drone_001", "drone_002"]}
    3. Server streams telemetry + thinking events
    4. Client can send commands (future)

    Args:
        websocket: FastAPI WebSocket
        connection_id: Optional connection ID (generated if not provided)
    """
    if not connection_id:
        connection_id = f"unreal_{int(time.time() * 1000)}"

    try:
        await unreal_manager.connect(websocket, connection_id)

        # Send initial sync
        await unreal_manager.send_to(
            connection_id,
            UnrealMessageType.SYNC,
            {
                "connection_id": connection_id,
                "protocol_version": "1.0",
                "capabilities": [
                    "telemetry",
                    "thinking",
                    "critics",
                    "decisions"
                ]
            }
        )

        # Message handling loop
        while True:
            try:
                # Receive with timeout for heartbeat
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )

                # Handle subscription
                if "subscribe" in message:
                    drone_ids = message["subscribe"]
                    unreal_manager.drone_subscriptions[connection_id] = set(drone_ids)
                    logger.info(f"{connection_id} subscribed to: {drone_ids}")

                # Handle ping
                if message.get("type") == "ping":
                    await unreal_manager.send_to(
                        connection_id,
                        UnrealMessageType.HEARTBEAT,
                        {"pong": True}
                    )

            except asyncio.TimeoutError:
                # Send heartbeat
                await unreal_manager.send_to(
                    connection_id,
                    UnrealMessageType.HEARTBEAT,
                    {}
                )

    except WebSocketDisconnect:
        logger.info(f"Unreal client {connection_id} disconnected")
    except Exception as e:
        logger.error(f"Unreal WebSocket error: {e}")
    finally:
        unreal_manager.disconnect(connection_id)


# =============================================================================
# FastAPI Route Registration
# =============================================================================


def add_unreal_routes(app: Any) -> None:
    """Add Unreal streaming routes to FastAPI app.

    Args:
        app: FastAPI application instance
    """

    @app.websocket("/ws/unreal")
    async def websocket_unreal(websocket: WebSocket) -> None:
        await handle_unreal_websocket(websocket)

    @app.websocket("/ws/unreal/{connection_id}")
    async def websocket_unreal_with_id(websocket: WebSocket, connection_id: str) -> None:
        await handle_unreal_websocket(websocket, connection_id)

    @app.get("/api/unreal/status")
    def unreal_status() -> dict[str, Any]:
        """Get Unreal streaming status."""
        return {
            "active_connections": unreal_manager.active_connections,
            "subscriptions": {
                k: list(v) for k, v in unreal_manager.drone_subscriptions.items()
            }
        }

    logger.info("Unreal streaming routes registered")
