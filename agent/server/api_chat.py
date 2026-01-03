"""Chat API for discussing decisions during scenario runs.

Provides endpoints for sending and retrieving chat messages, with real-time
updates via WebSocket broadcasting. Includes AI-powered responses to discuss
decisions made during scenario runs.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None  # type: ignore[misc, assignment]
    TestModel = None  # type: ignore[misc, assignment]

from agent.server.llm_config import get_default_llm_model, llm_credentials_present
from agent.server.state import connection_manager, scenario_runner_state, server_state

# Background tasks storage to prevent garbage collection
_background_tasks: set[asyncio.Task[None]] = set()

logger = structlog.get_logger(__name__)


class ChatMessageType(str, Enum):
    """Types of chat messages."""

    USER = "user"  # Message from a human user
    SYSTEM = "system"  # System-generated message (decision, alert, etc.)
    AGENT = "agent"  # Message from the AI agent explaining decisions


class ChatMessage(BaseModel):
    """A chat message in the decision discussion."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    message_type: ChatMessageType = ChatMessageType.USER
    sender: str = "User"  # Display name of sender
    content: str  # The message text
    run_id: str | None = None  # Associated scenario run
    decision_id: str | None = None  # Reference to a specific decision
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""

    content: str = Field(..., min_length=1, max_length=2000)
    sender: str = Field(default="User", max_length=50)
    decision_id: str | None = None  # Optional reference to a decision
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatStore:
    """In-memory store for chat messages with run-based organization."""

    def __init__(self, max_messages_per_run: int = 500) -> None:
        """Initialize the chat store.

        Args:
            max_messages_per_run: Maximum messages to keep per run.
        """
        self.max_messages = max_messages_per_run
        # Messages organized by run_id
        self._messages: dict[str, list[ChatMessage]] = {}
        # Global messages (no run context)
        self._global_messages: list[ChatMessage] = []

    def add_message(self, message: ChatMessage) -> ChatMessage:
        """Add a message to the store.

        Args:
            message: The message to add.

        Returns:
            The added message with generated ID.
        """
        if message.run_id:
            if message.run_id not in self._messages:
                self._messages[message.run_id] = []
            messages = self._messages[message.run_id]
            messages.append(message)
            # Trim if over limit
            if len(messages) > self.max_messages:
                self._messages[message.run_id] = messages[-self.max_messages :]
        else:
            self._global_messages.append(message)
            if len(self._global_messages) > self.max_messages:
                self._global_messages = self._global_messages[-self.max_messages :]

        return message

    def get_messages(
        self,
        run_id: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[ChatMessage]:
        """Get messages, optionally filtered by run and time.

        Args:
            run_id: Filter to specific run (None for global).
            limit: Maximum messages to return.
            since: Only return messages after this time.

        Returns:
            List of matching messages.
        """
        messages = self._messages.get(run_id, []) if run_id else self._global_messages

        if since:
            messages = [m for m in messages if m.timestamp > since]

        return messages[-limit:]

    def get_all_runs(self) -> list[str]:
        """Get all run IDs with messages.

        Returns:
            List of run IDs.
        """
        return list(self._messages.keys())

    def clear_run(self, run_id: str) -> int:
        """Clear messages for a specific run.

        Args:
            run_id: The run to clear.

        Returns:
            Number of messages cleared.
        """
        if run_id in self._messages:
            count = len(self._messages[run_id])
            del self._messages[run_id]
            return count
        return 0


# Global chat store instance
chat_store = ChatStore()


async def generate_ai_response(
    user_message: str,
    run_id: str | None,
    recent_messages: list[ChatMessage],
) -> str | None:
    """Generate an AI response to a user message about the scenario.

    Args:
        user_message: The user's message/question.
        run_id: Current run ID for context.
        recent_messages: Recent chat history for context.

    Returns:
        AI response text, or None if AI is not available.
    """
    if not PYDANTIC_AI_AVAILABLE:
        logger.warning("pydantic_ai not available for chat responses")
        return None

    if not llm_credentials_present():
        return None

    # Build context from scenario state
    context_parts = []

    # Add scenario info if available
    if scenario_runner_state.runner:
        runner = scenario_runner_state.runner
        run_state = getattr(runner, "run_state", None)
        if run_state:
            context_parts.append(f"Scenario is currently running (run_id: {run_id})")
            assets_inspected = getattr(run_state, "assets_inspected", set())
            context_parts.append(f"Assets inspected so far: {len(assets_inspected)}")

    # Add recent decisions from server state
    if server_state.last_decision:
        decision = server_state.last_decision
        context_parts.append(f"Last decision: {decision.action} - {decision.reasoning}")

    # Add recent chat context
    if recent_messages:
        chat_context = "\n".join([
            f"{m.sender}: {m.content}"
            for m in recent_messages[-5:]  # Last 5 messages
        ])
        context_parts.append(f"Recent discussion:\n{chat_context}")

    system_prompt = """You are an AI assistant helping operators understand and discuss
decisions made during autonomous drone inspection scenarios. You have access to the
current scenario state and can explain decisions, answer questions about the mission,
and provide insights about the autonomous behavior.

Keep responses concise and focused on the mission context. Use technical but accessible language."""

    if context_parts:
        system_prompt += "\n\nCurrent context:\n" + "\n".join(context_parts)

    try:
        model = get_default_llm_model()
        # Use TestModel for development if no real credentials
        if not llm_credentials_present():
            agent: Agent[None, str] = Agent(
                TestModel(),
                system_prompt=system_prompt,
            )
        else:
            agent = Agent(
                model,
                system_prompt=system_prompt,
            )

        result = await agent.run(user_message)
        return result.data
    except Exception as exc:
        logger.warning("ai_chat_response_failed", error=str(exc))
        return None


async def broadcast_chat_message(message: ChatMessage) -> None:
    """Broadcast a chat message to all connected WebSocket clients.

    Args:
        message: The message to broadcast.
    """
    # Create a custom event for chat messages
    event_data = {
        "event_type": "chat_message",
        "message": message.model_dump(mode="json"),
    }

    # Use the connection manager's broadcast with a simple dict
    # We'll send it as a raw JSON message
    disconnected = set()
    for connection in connection_manager.active_connections:
        try:
            await connection.send_json(event_data)
        except Exception as exc:
            logger.warning("chat_broadcast_failed", error=str(exc))
            disconnected.add(connection)

    for conn in disconnected:
        connection_manager.disconnect(conn)


def register_chat_routes(app: FastAPI) -> None:
    """Register chat-related API routes.

    Args:
        app: FastAPI application instance.
    """

    @app.post("/api/chat/messages")
    async def send_message(request: ChatMessageRequest) -> dict[str, object]:
        """Send a chat message and get an AI response.

        Args:
            request: The message request.

        Returns:
            The created message and optional AI response.
        """
        # Get current run ID if available
        run_id = None
        if scenario_runner_state.runner:
            run_id = getattr(scenario_runner_state.runner, "run_id", None)
        if not run_id:
            run_id = server_state.current_run_id

        message = ChatMessage(
            message_type=ChatMessageType.USER,
            sender=request.sender,
            content=request.content,
            run_id=run_id,
            decision_id=request.decision_id,
            metadata=request.metadata,
        )

        stored_message = chat_store.add_message(message)

        # Broadcast user message to all connected clients
        await broadcast_chat_message(stored_message)

        logger.info(
            "chat_message_sent",
            message_id=stored_message.id,
            sender=stored_message.sender,
            run_id=run_id,
        )

        # Generate AI response in the background
        async def send_ai_response() -> None:
            try:
                recent_messages = chat_store.get_messages(run_id=run_id, limit=10)
                ai_response = await generate_ai_response(
                    request.content,
                    run_id,
                    recent_messages,
                )
                if ai_response:
                    ai_message = ChatMessage(
                        message_type=ChatMessageType.AGENT,
                        sender="AegisAV Assistant",
                        content=ai_response,
                        run_id=run_id,
                    )
                    chat_store.add_message(ai_message)
                    await broadcast_chat_message(ai_message)
                    logger.info(
                        "ai_chat_response_sent",
                        message_id=ai_message.id,
                        run_id=run_id,
                    )
            except Exception as exc:
                logger.warning("ai_response_generation_failed", error=str(exc))

        # Start AI response generation without blocking the user message return
        task = asyncio.create_task(send_ai_response())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        return {"message": stored_message.model_dump(mode="json")}

    @app.get("/api/chat/messages")
    def get_messages(
        run_id: str | None = None,
        limit: int = 100,
        current_run: bool = False,
    ) -> dict[str, object]:
        """Get chat messages.

        Args:
            run_id: Filter to specific run ID.
            limit: Maximum messages to return.
            current_run: If True, use current active run.

        Returns:
            List of messages.
        """
        # If current_run is True, get the active run ID
        if current_run:
            if scenario_runner_state.runner:
                run_id = getattr(scenario_runner_state.runner, "run_id", None)
            if not run_id:
                run_id = server_state.current_run_id

        messages = chat_store.get_messages(run_id=run_id, limit=limit)
        return {
            "messages": [m.model_dump(mode="json") for m in messages],
            "count": len(messages),
            "run_id": run_id,
        }

    @app.post("/api/chat/system")
    async def send_system_message(
        content: str,
        decision_id: str | None = None,
    ) -> dict:
        """Send a system message (for internal use / agent decisions).

        Args:
            content: The message content.
            decision_id: Optional decision reference.

        Returns:
            The created message.
        """
        run_id = None
        if scenario_runner_state.runner:
            run_id = getattr(scenario_runner_state.runner, "run_id", None)
        if not run_id:
            run_id = server_state.current_run_id

        message = ChatMessage(
            message_type=ChatMessageType.SYSTEM,
            sender="System",
            content=content,
            run_id=run_id,
            decision_id=decision_id,
        )

        stored_message = chat_store.add_message(message)
        await broadcast_chat_message(stored_message)

        return {"message": stored_message.model_dump(mode="json")}

    @app.get("/api/chat/runs")
    def get_chat_runs() -> dict[str, object]:
        """Get all runs that have chat messages.

        Returns:
            List of run IDs with message counts.
        """
        runs = chat_store.get_all_runs()
        run_info = []
        for rid in runs:
            messages = chat_store.get_messages(run_id=rid)
            run_info.append({
                "run_id": rid,
                "message_count": len(messages),
                "last_message": messages[-1].timestamp.isoformat() if messages else None,
            })
        return {"runs": run_info}

    @app.delete("/api/chat/runs/{run_id}")
    def clear_run_messages(run_id: str) -> dict[str, object]:
        """Clear all messages for a specific run.

        Args:
            run_id: The run to clear.

        Returns:
            Number of messages cleared.
        """
        count = chat_store.clear_run(run_id)
        return {"cleared": count, "run_id": run_id}


def add_decision_chat_message(
    decision_action: str,
    decision_reason: str,
    decision_id: str | None = None,
    run_id: str | None = None,
) -> ChatMessage:
    """Helper to add a system message when a decision is made.

    Args:
        decision_action: The action taken.
        decision_reason: Why the action was chosen.
        decision_id: The decision ID.
        run_id: The run ID.

    Returns:
        The created message.
    """
    content = f"**Decision:** {decision_action}\n**Reason:** {decision_reason}"

    message = ChatMessage(
        message_type=ChatMessageType.AGENT,
        sender="Agent",
        content=content,
        run_id=run_id,
        decision_id=decision_id,
        metadata={"action": decision_action, "reason": decision_reason},
    )

    return chat_store.add_message(message)
