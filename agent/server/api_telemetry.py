"""Telemetry, logs, storage, and websocket API routes."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from agent.api_models import HealthResponse
from agent.server.deps import auth_handler
from agent.server.feedback_store import (
    get_feedback_for_decision,
    get_outcome_for_decision,
    get_recent_feedback,
    get_recent_outcomes,
)
from agent.server.state import connection_manager, server_state


class LogBufferHandler(logging.Handler):
    """Ring buffer log handler for dashboard access."""

    def __init__(self, capacity: int = 50) -> None:
        """Initialize the LogBufferHandler.

        Args:
            capacity: Maximum number of log entries to retain.
        """
        super().__init__()
        self.capacity = capacity
        self.buffer: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store formatted log records in a bounded in-memory buffer.

        Args:
            record: Log record to store.
        """
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


def register_telemetry_routes(app: FastAPI) -> None:
    """Register telemetry, log, storage, and websocket routes."""

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time event broadcasting to dashboard.

        Args:
            websocket: WebSocket connection from client.
        """
        await connection_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket)
        except Exception as exc:
            logging.getLogger(__name__).error("websocket_error", exc_info=exc)
            connection_manager.disconnect(websocket)

    @app.get("/api/logs")
    async def get_logs() -> dict:
        """Return the recent log buffer for the dashboard.

        Returns:
            Dictionary containing list of recent log entries.
        """
        return {"logs": log_buffer.buffer}

    @app.get("/api/telemetry/latest")
    async def get_latest_telemetry(vehicle_id: str | None = None) -> dict:
        """Get the latest telemetry for a vehicle or all known vehicles."""
        if vehicle_id:
            payload = server_state.latest_telemetry.get(vehicle_id)
            if payload is None and server_state.store:
                payload = await server_state.store.get_latest_telemetry(vehicle_id)
            if payload is None:
                raise HTTPException(status_code=404, detail="Telemetry not found")
            return {"vehicle_id": vehicle_id, "telemetry": payload}

        vehicle_ids = sorted(server_state.known_vehicles)
        items: list[dict] = []
        for vid in vehicle_ids:
            payload = server_state.latest_telemetry.get(vid)
            if payload is None and server_state.store:
                payload = await server_state.store.get_latest_telemetry(vid)
            if payload is not None:
                items.append({"vehicle_id": vid, "telemetry": payload})

        return {"vehicles": items, "count": len(items)}

    @app.get("/api/dashboard/feedback")
    async def get_dashboard_feedback(limit: int = 50) -> dict:
        """Get the most recent decision feedback entries."""
        feedback = await get_recent_feedback(server_state.store, limit=limit)
        return {"feedback": feedback, "count": len(feedback)}

    @app.get("/api/dashboard/feedback/{decision_id}")
    async def get_dashboard_feedback_for_decision(decision_id: str) -> dict:
        """Get the latest feedback for a decision."""
        feedback = await get_feedback_for_decision(server_state.store, decision_id)
        if feedback is None:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return {"feedback": feedback}

    @app.get("/api/dashboard/outcomes")
    async def get_dashboard_outcomes(limit: int = 50) -> dict:
        """Get the most recent decision outcomes."""
        outcomes = await get_recent_outcomes(server_state.store, limit=limit)
        return {"outcomes": outcomes, "count": len(outcomes)}

    @app.get("/api/dashboard/outcomes/{decision_id}")
    async def get_dashboard_outcome_for_decision(decision_id: str) -> dict:
        """Get the latest outcome for a decision."""
        outcome = await get_outcome_for_decision(server_state.store, decision_id)
        if outcome is None:
            raise HTTPException(status_code=404, detail="Outcome not found")
        return {"outcome": outcome}

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
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            last_state_update=last_update,
            decisions_made=server_state.decisions_made,
        )

    @app.get("/api/storage/stats")
    async def get_storage_stats(_auth: dict = Depends(auth_handler)) -> dict:
        """Get storage statistics (requires API key)."""
        if server_state.store:
            stats = await server_state.store.get_stats()
            stats["persistence_enabled"] = server_state.persistence_enabled
            return stats
        return {"connected": False, "persistence_enabled": False}

    @app.get("/api/storage/anomalies")
    async def get_stored_anomalies(limit: int = 50, _auth: dict = Depends(auth_handler)) -> dict:
        """Get recent anomalies from storage."""
        if server_state.store:
            anomalies = await server_state.store.get_recent_anomalies(limit=limit)
            return {"anomalies": anomalies, "count": len(anomalies)}
        return {"anomalies": [], "count": 0}

    @app.get("/api/storage/missions")
    async def get_stored_missions(limit: int = 20, _auth: dict = Depends(auth_handler)) -> dict:
        """Get recent missions from storage."""
        if server_state.store:
            missions = await server_state.store.get_recent_missions(limit=limit)
            return {"missions": missions, "count": len(missions)}
        return {"missions": [], "count": 0}
