"""Dashboard routes for monitoring agent activity."""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.server.config_manager import get_config_manager
from agent.server.feedback_store import (
    get_anomalies_for_run,
    get_feedback_for_run,
    get_run_summary,
)
from agent.server.scenario_runner import ScenarioRunner
from agent.server.scenarios import (
    ScenarioCategory,
    get_all_scenarios,
    get_scenario,
    get_scenarios_by_category,
    get_scenarios_by_difficulty,
)
from agent.server.state import scenario_runner_state, server_state


class RunnerStartRequest(BaseModel):
    """Request to start scenario runner."""

    scenario_id: str
    time_scale: float = 1.0
    max_duration_s: float | None = None


class RunnerState:
    """Global state for the scenario runner."""

    def __init__(self) -> None:
        """Initialize the RunnerState."""
        self.runner: Any = None
        self.run_task: asyncio.Task | None = None
        self.is_running: bool = False
        self.last_error: str | None = None


# Global runner state
_runner_state = RunnerState()

# Global store reference (set by add_dashboard_routes)
_store: Any = None


def _load_entries(run_file: Path) -> list[dict[str, Any]]:
    """Load JSONL decision entries from a run file."""
    entries: list[dict[str, Any]] = []
    if not run_file.exists():
        return entries
    with open(run_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _calculate_relative_pos(
    origin_lat: float, origin_lon: float, target_lat: float, target_lon: float
) -> dict[str, float]:
    """Calculate relative X, Y in meters using flat earth approximation.

    Args:
        origin_lat: Origin latitude in degrees.
        origin_lon: Origin longitude in degrees.
        target_lat: Target latitude in degrees.
        target_lon: Target longitude in degrees.

    Returns:
        Dictionary with 'x' and 'y' keys representing relative position in meters.
    """
    # 1 deg lat ~ 111,111 meters
    # 1 deg lon ~ 111,111 * cos(lat) meters
    lat_m = (target_lat - origin_lat) * 111111
    lon_m = (target_lon - origin_lon) * 111111 * math.cos(math.radians(origin_lat))
    return {"x": lon_m, "y": lat_m}


def _list_runs(log_dir: Path) -> list[str]:
    """List run IDs sorted by timestamp string.

    Args:
        log_dir: Directory containing decision log files.

    Returns:
        List of run IDs sorted alphabetically.
    """
    runs = []
    for path in log_dir.glob("decisions_*.jsonl"):
        run_id = path.stem.replace("decisions_", "")
        runs.append(run_id)
    return sorted(runs)


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary metrics for a run.

    Args:
        entries: List of decision log entries.

    Returns:
        Dictionary containing summary metrics including total count and risk scores.
    """
    if not entries:
        return {
            "total": 0,
            "avg_risk": 0.0,
            "max_risk": 0.0,
            "time_start": None,
            "time_end": None,
        }
    risk_scores = [e.get("risk_score", 0) for e in entries]
    return {
        "total": len(entries),
        "avg_risk": sum(risk_scores) / len(risk_scores),
        "max_risk": max(risk_scores),
        "time_start": entries[0].get("timestamp"),
        "time_end": entries[-1].get("timestamp"),
    }


def _series(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare chart series for risk and battery.

    Args:
        entries: List of decision log entries.

    Returns:
        Tuple of (risk_series, battery_series) for charting.
    """
    risk_series = []
    battery_series = []
    for entry in entries:
        timestamp = entry.get("timestamp")
        risk_series.append({"t": timestamp, "value": entry.get("risk_score", 0)})
        battery_series.append({"t": timestamp, "value": entry.get("battery_percent", 0)})
    return risk_series, battery_series


def _action_counts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count decisions by action.

    Args:
        entries: List of decision log entries.

    Returns:
        List of dictionaries with action names and their counts.
    """
    counts: dict[str, int] = {}
    for entry in entries:
        action = entry.get("action", "unknown")
        counts[action] = counts.get(action, 0) + 1
    return [{"action": action, "count": count} for action, count in counts.items()]


def _recent(entries: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    """Return the most recent decision entries with full context.

    Args:
        entries: List of decision log entries.
        limit: Maximum number of entries to return.

    Returns:
        List of recent decision entries with reasoning context.
    """
    recent_entries = entries[-limit:]
    items = []
    for entry in recent_entries[::-1]:
        critic_validation = entry.get("critic_validation")
        if not critic_validation and "critic_approved" in entry:
            critic_validation = {
                "approved": entry.get("critic_approved", True),
                "escalation": {
                    "reason": entry.get("escalation_reason"),
                    "required_action": entry.get("escalation_level"),
                }
                if entry.get("escalation_reason")
                else None,
            }

        item = {
            "timestamp": entry.get("timestamp"),
            "elapsed_s": entry.get("elapsed_s"),
            "drone_id": entry.get("drone_id") or entry.get("vehicle_id"),
            "drone_name": entry.get("drone_name") or entry.get("vehicle_id"),
            "agent_label": entry.get("agent_label"),
            "action": entry.get("action"),
            "reason": entry.get("reason") or entry.get("reasoning"),
            "confidence": entry.get("confidence"),
            "battery_percent": entry.get("battery_percent"),
            "risk_level": entry.get("risk_level"),
            "vehicle_state": {
                "armed": entry.get("armed", False),
                "mode": entry.get("mode", "UNKNOWN"),
            },
            # Rich reasoning context
            "reasoning_context": entry.get("reasoning_context", {}),
            "alternatives": entry.get("alternatives", []),
            "critic_validation": critic_validation,
            "explanation": entry.get("explanation"),
            # Target info if present
            "target_asset": entry.get("target_asset"),
        }

        # Calculate spatial context if available
        vehicle_pos = entry.get("vehicle_position")
        assets = entry.get("assets", [])
        spatial_context = []

        if vehicle_pos and assets:
            v_lat = vehicle_pos.get("lat")
            v_lon = vehicle_pos.get("lon")
            for asset in assets:
                rel = _calculate_relative_pos(
                    v_lat, v_lon, asset.get("lat", 0), asset.get("lon", 0)
                )
                spatial_context.append({
                    "id": asset.get("id"),
                    "type": asset.get("type"),
                    "x": rel["x"],
                    "y": rel["y"],
                })

        item["spatial_context"] = spatial_context
        items.append(item)
    return items


def add_dashboard_routes(app: FastAPI, log_dir: Path, store: Any = None) -> None:
    """Register dashboard routes and static assets.

    Args:
        app: FastAPI application instance.
        log_dir: Directory containing decision log files.
        store: Optional state store for feedback persistence.
    """
    global _store
    _store = store
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = repo_root / "frontend" / "dist"
    assets_dir = dist_dir / "assets"
    public_favicon = repo_root / "frontend" / "public" / "aegis_logo.svg"
    if assets_dir.exists():
        app.mount("/dashboard/assets", StaticFiles(directory=assets_dir), name="dashboard-assets")

    def _resolve_favicon() -> Path | None:
        favicon = dist_dir / "aegis_logo.svg"
        if favicon.exists():
            return favicon
        if public_favicon.exists():
            return public_favicon
        return None

    def _favicon_response() -> Response:
        favicon = _resolve_favicon()
        if favicon:
            return Response(content=favicon.read_bytes(), media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="Favicon not found")

    @app.get("/dashboard/aegis_logo.svg")
    async def dashboard_favicon() -> Response:
        """Serve dashboard favicon."""
        return _favicon_response()

    @app.get("/aegis_logo.svg")
    async def root_favicon() -> Response:
        """Serve root favicon for non-dashboard pages."""
        return _favicon_response()

    @app.get("/favicon.ico")
    async def legacy_favicon() -> Response:
        """Serve a legacy favicon path (serves SVG, browsers handle it)."""
        return _favicon_response()

    def _serve_dashboard_html() -> HTMLResponse:
        """Serve the dashboard index.html."""
        dist_index = dist_dir / "index.html"
        if dist_index.exists():
            return HTMLResponse(dist_index.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<html><body><h1>Dashboard build missing.</h1>"
            "<p>Run the frontend build to generate /frontend/dist.</p></body></html>"
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        """Serve the dashboard HTML page."""
        return _serve_dashboard_html()

    @app.get("/dashboard/", response_class=HTMLResponse)
    async def dashboard_trailing() -> HTMLResponse:
        """Serve the dashboard HTML page (with trailing slash)."""
        return _serve_dashboard_html()

    @app.get("/dashboard/maps", response_class=HTMLResponse)
    async def dashboard_maps() -> HTMLResponse:
        """Serve the dashboard map page (SPA entry)."""
        return _serve_dashboard_html()

    @app.get("/dashboard/maps/", response_class=HTMLResponse)
    async def dashboard_maps_trailing() -> HTMLResponse:
        """Serve the dashboard map page (with trailing slash)."""
        return _serve_dashboard_html()

    @app.get("/api/dashboard/runs", response_class=JSONResponse)
    async def list_runs() -> JSONResponse:
        """List all available run IDs."""
        runs = _list_runs(log_dir)
        return JSONResponse({"runs": runs, "latest": runs[-1] if runs else None})

    @app.get("/api/dashboard/run/{run_id}", response_class=JSONResponse)
    async def run_data(run_id: str) -> JSONResponse:
        """Get detailed data for a specific run."""
        run_file = log_dir / f"decisions_{run_id}.jsonl"
        if not run_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        entries = _load_entries(run_file)
        summary = _summarize(entries)
        risk_series, battery_series = _series(entries)
        actions = _action_counts(entries)
        recent = _recent(entries)

        return JSONResponse({
            "run_id": run_id,
            "summary": summary,
            "actions": actions,
            "risk_series": risk_series,
            "battery_series": battery_series,
            "recent": recent,
        })

    # Scenario API endpoints

    @app.get("/api/dashboard/scenarios", response_class=JSONResponse)
    async def list_scenarios(
        category: str | None = None,
        difficulty: str | None = None,
    ) -> JSONResponse:
        """List all available scenarios with optional filtering."""
        if category:
            try:
                cat_enum = ScenarioCategory(category)
                scenarios = get_scenarios_by_category(cat_enum)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}") from e
        elif difficulty:
            scenarios = get_scenarios_by_difficulty(difficulty)
        else:
            scenarios = get_all_scenarios()

        return JSONResponse({
            "scenarios": [s.to_dict() for s in scenarios],
            "total": len(scenarios),
            "categories": [c.value for c in ScenarioCategory],
            "difficulties": ["easy", "normal", "hard", "extreme"],
        })

    @app.get("/api/dashboard/scenarios/{scenario_id}", response_class=JSONResponse)
    async def scenario_detail(scenario_id: str) -> JSONResponse:
        """Get detailed information about a specific scenario."""
        scenario = get_scenario(scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")

        return JSONResponse({
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "description": scenario.description,
            "category": scenario.category.value,
            "duration_minutes": scenario.duration_minutes,
            "difficulty": scenario.difficulty,
            "tags": scenario.tags,
            "drones": [
                {
                    "drone_id": d.drone_id,
                    "name": d.name,
                    "battery_percent": d.battery_percent,
                    "state": d.state.value,
                    "latitude": d.latitude,
                    "longitude": d.longitude,
                    "gps_fix_type": d.gps_fix_type,
                    "sensors_healthy": d.sensors_healthy,
                    "gps_healthy": d.gps_healthy,
                    "motors_healthy": d.motors_healthy,
                }
                for d in scenario.drones
            ],
            "assets": [
                {
                    "asset_id": a.asset_id,
                    "name": a.name,
                    "asset_type": a.asset_type,
                    "latitude": a.latitude,
                    "longitude": a.longitude,
                    "has_anomaly": a.has_anomaly,
                    "anomaly_severity": a.anomaly_severity,
                    "priority": a.priority,
                    "altitude_m": a.altitude_m,
                    "inspection_altitude_agl": a.inspection_altitude_agl,
                    "orbit_radius_m": a.orbit_radius_m,
                    "dwell_time_s": a.dwell_time_s,
                    "scale": a.scale,
                    "rotation_deg": a.rotation_deg,
                }
                for a in scenario.assets
            ],
            "environment": {
                "wind_speed_ms": scenario.environment.wind_speed_ms,
                "visibility_m": scenario.environment.visibility_m,
                "precipitation": scenario.environment.precipitation,
                "is_daylight": scenario.environment.is_daylight,
            },
            "events": [
                {
                    "timestamp_offset_s": e.timestamp_offset_s,
                    "event_type": e.event_type,
                    "description": e.description,
                }
                for e in scenario.events
            ],
        })

    # Scenario Runner API endpoints

    @app.post("/api/dashboard/runner/start", response_class=JSONResponse)
    async def start_runner(request: RunnerStartRequest) -> JSONResponse:
        """Start running a scenario simulation."""
        global _runner_state

        if _runner_state.is_running:
            raise HTTPException(status_code=409, detail="Runner already active")

        # Create and load scenario
        _runner_state.runner = ScenarioRunner(log_dir=log_dir)
        loaded = await _runner_state.runner.load_scenario(request.scenario_id)
        if not loaded:
            raise HTTPException(
                status_code=404, detail=f"Scenario not found: {request.scenario_id}"
            )

        _runner_state.is_running = True
        _runner_state.last_error = None

        # Run in background task
        async def run_scenario() -> None:
            global _runner_state
            try:
                await _runner_state.runner.run(
                    time_scale=request.time_scale,
                    max_duration_s=request.max_duration_s,
                )
                # Save log when complete
                if _runner_state.runner.run_state:
                    _runner_state.runner.save_decision_log()
            except Exception as e:
                _runner_state.last_error = str(e)
            finally:
                _runner_state.is_running = False

        _runner_state.run_task = asyncio.create_task(run_scenario())

        return JSONResponse({
            "status": "started",
            "scenario_id": request.scenario_id,
            "run_id": _runner_state.runner.run_id,
            "time_scale": request.time_scale,
        })

    @app.post("/api/dashboard/runner/stop", response_class=JSONResponse)
    async def stop_runner() -> JSONResponse:
        """Stop the currently running scenario.

        Returns:
            JSON response with stop status and summary.
        """
        global _runner_state

        if not _runner_state.is_running or not _runner_state.runner:
            raise HTTPException(status_code=409, detail="No runner active")

        _runner_state.runner.stop()
        _runner_state.is_running = False

        # Wait briefly for task to complete
        if _runner_state.run_task:
            try:
                if not _runner_state.run_task.done():
                    _runner_state.run_task.cancel()
                await asyncio.wait_for(_runner_state.run_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                pass
            finally:
                _runner_state.run_task = None

        # Save decision log
        log_path = None
        if _runner_state.runner.run_state:
            log_path = _runner_state.runner.save_decision_log()

        summary = _runner_state.runner.get_summary()

        return JSONResponse({
            "status": "stopped",
            "summary": summary,
            "log_path": str(log_path) if log_path else None,
        })

    @app.get("/api/dashboard/runner/status", response_class=JSONResponse)
    async def runner_status() -> JSONResponse:
        """Get current runner status.

        Returns:
            JSON response with current runner state and drone information.
        """
        global _runner_state

        # Check the shared scenario_runner_state first (used by /api/scenarios API)
        shared_runner = scenario_runner_state.runner
        if shared_runner and shared_runner.run_state:
            run_state = shared_runner.run_state
            # Build drone status list
            drones = []
            for drone_id, ds in run_state.drone_states.items():
                drones.append({
                    "drone_id": drone_id,
                    "name": ds.drone.name,
                    "battery_percent": ds.drone.battery_percent,
                    "state": ds.drone.state.value,
                    "in_air": ds.drone.in_air,
                    "latitude": ds.drone.latitude,
                    "longitude": ds.drone.longitude,
                    "current_goal": ds.current_goal.goal_type.value if ds.current_goal else None,
                    "decisions_made": ds.decisions_made,
                    "inspections_completed": ds.inspections_completed,
                    "sensors_healthy": ds.drone.sensors_healthy,
                    "gps_healthy": ds.drone.gps_healthy,
                    "motors_healthy": ds.drone.motors_healthy,
                })
            return JSONResponse({
                "is_running": scenario_runner_state.is_running,
                "run_id": run_state.run_id,
                "scenario_id": run_state.scenario.scenario_id,
                "scenario_name": run_state.scenario.name,
                "elapsed_seconds": run_state.elapsed_seconds,
                "is_complete": run_state.is_complete,
                "decision_count": len(run_state.decision_log),
                "drones": drones,
                "environment": {
                    "wind_speed_ms": run_state.environment.wind_speed_ms
                    if run_state.environment
                    else 0,
                    "visibility_m": run_state.environment.visibility_m
                    if run_state.environment
                    else 10000,
                },
                "last_error": scenario_runner_state.last_error,
                "preflight_status": server_state.preflight_status,
            })

        # Fall back to local _runner_state (used by /api/dashboard/runner API)
        if not _runner_state.runner or not _runner_state.runner.run_state:
            return JSONResponse({
                "is_running": _runner_state.is_running,
                "run_id": None,
                "scenario_id": None,
                "elapsed_seconds": 0,
                "drones": [],
                "last_error": _runner_state.last_error,
                "preflight_status": server_state.preflight_status,
            })

        run_state = _runner_state.runner.run_state

        # Build drone status list
        drones = []
        for drone_id, ds in run_state.drone_states.items():
            drones.append({
                "drone_id": drone_id,
                "name": ds.drone.name,
                "battery_percent": ds.drone.battery_percent,
                "state": ds.drone.state.value,
                "in_air": ds.drone.in_air,
                "latitude": ds.drone.latitude,
                "longitude": ds.drone.longitude,
                "current_goal": ds.current_goal.goal_type.value if ds.current_goal else None,
                "decisions_made": ds.decisions_made,
                "inspections_completed": ds.inspections_completed,
                "sensors_healthy": ds.drone.sensors_healthy,
                "gps_healthy": ds.drone.gps_healthy,
                "motors_healthy": ds.drone.motors_healthy,
            })

        return JSONResponse({
            "is_running": _runner_state.is_running,
            "run_id": run_state.run_id,
            "scenario_id": run_state.scenario.scenario_id,
            "scenario_name": run_state.scenario.name,
            "elapsed_seconds": run_state.elapsed_seconds,
            "is_complete": run_state.is_complete,
            "decision_count": len(run_state.decision_log),
            "drones": drones,
            "environment": {
                "wind_speed_ms": run_state.environment.wind_speed_ms
                if run_state.environment
                else 0,
                "visibility_m": run_state.environment.visibility_m
                if run_state.environment
                else 10000,
            },
            "last_error": _runner_state.last_error,
            "preflight_status": server_state.preflight_status,
        })

    @app.get("/api/dashboard/runner/decisions", response_class=JSONResponse)
    async def runner_decisions(limit: int = 20, offset: int = 0) -> JSONResponse:
        """Get recent decisions from the running scenario.

        Args:
            limit: Maximum number of decisions to return.
            offset: Number of decisions to skip from the end.

        Returns:
            JSON response with paginated decision list.
        """
        global _runner_state

        if not _runner_state.runner or not _runner_state.runner.run_state:
            return JSONResponse({"decisions": [], "total": 0})

        log = _runner_state.runner.run_state.decision_log
        total = len(log)

        # Get slice of decisions
        start = max(0, total - offset - limit)
        end = total - offset
        decisions = log[start:end][::-1]  # Reverse for most recent first

        return JSONResponse({
            "decisions": decisions,
            "total": total,
            "offset": offset,
            "limit": limit,
        })

    @app.get("/api/dashboard/runner/summary", response_class=JSONResponse)
    async def runner_summary() -> JSONResponse:
        """Get full summary of the current or most recent run.

        Returns comprehensive statistics including:
        - Run identification (run_id, scenario_id)
        - Completion status
        - Total decisions and battery consumption
        - Per-drone breakdowns

        Returns:
            JSON response with run summary or 404 if no run available.
        """
        global _runner_state

        if not _runner_state.runner or not _runner_state.runner.run_state:
            raise HTTPException(status_code=404, detail="No run available")

        summary = _runner_state.runner.get_summary()
        return JSONResponse(summary)

    # ========================================================================
    # Run-based feedback APIs
    # ========================================================================

    @app.get("/api/dashboard/run/{run_id}/feedback", response_class=JSONResponse)
    async def run_feedback(run_id: str) -> JSONResponse:
        """Get all feedback entries for a specific run.

        Args:
            run_id: The run ID to query.

        Returns:
            JSON response with feedback list for this run.
        """
        global _store

        feedback_list = await get_feedback_for_run(_store, run_id)
        return JSONResponse({
            "run_id": run_id,
            "feedback": feedback_list,
            "total": len(feedback_list),
        })

    @app.get("/api/dashboard/run/{run_id}/anomalies", response_class=JSONResponse)
    async def run_anomalies(run_id: str) -> JSONResponse:
        """Get all anomaly-related feedback for a specific run.

        Returns entries where anomalies were detected or resolved.

        Args:
            run_id: The run ID to query.

        Returns:
            JSON response with anomaly list.
        """
        global _store

        anomalies = await get_anomalies_for_run(_store, run_id)
        return JSONResponse({
            "run_id": run_id,
            "anomalies": anomalies,
            "total": len(anomalies),
        })

    @app.get("/api/dashboard/run/{run_id}/summary", response_class=JSONResponse)
    async def run_feedback_summary(run_id: str) -> JSONResponse:
        """Get summary statistics for a run's feedback.

        Returns aggregated metrics including success/fail counts,
        anomaly counts, battery consumption, and timing.

        Args:
            run_id: The run ID to summarize.

        Returns:
            JSON response with run summary or 404 if no feedback found.
        """
        global _store

        summary = await get_run_summary(_store, run_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No feedback found for run: {run_id}")

        return JSONResponse(summary.to_dict())

    # ========================================================================
    # Real-time spatial context API
    # ========================================================================

    @app.get("/api/dashboard/spatial", response_class=JSONResponse)
    async def get_spatial_context() -> JSONResponse:
        """Get real-time spatial context from running scenario or world model.

        Returns assets and drone positions relative to the primary drone for
        the Spatial Awareness radar widget.

        Returns:
            JSON response with assets in relative coordinates.
        """
        from agent.server.scenarios import DOCK_LATITUDE, DOCK_LONGITUDE

        spatial_assets = []
        drone_position = None

        # Try to get from running scenario first
        shared_runner = scenario_runner_state.runner
        if shared_runner and shared_runner.run_state and shared_runner.run_state.is_running:
            run_state = shared_runner.run_state

            # Get first drone as reference point
            if run_state.drone_states:
                first_drone = list(run_state.drone_states.values())[0]
                drone_position = {
                    "lat": first_drone.drone.latitude,
                    "lon": first_drone.drone.longitude,
                    "alt": first_drone.drone.altitude_agl,
                }

                # Get assets from scenario
                for asset in run_state.scenario.assets:
                    rel = _calculate_relative_pos(
                        drone_position["lat"],
                        drone_position["lon"],
                        asset.latitude,
                        asset.longitude,
                    )
                    spatial_assets.append({
                        "id": asset.asset_id,
                        "type": asset.asset_type,
                        "x": rel["x"],
                        "y": rel["y"],
                        "name": asset.name,
                        "has_anomaly": asset.has_anomaly,
                    })

        # Fallback to world model
        elif server_state.world_model:
            world_snap = server_state.world_model.get_snapshot()
            if world_snap is None:
                return JSONResponse({"assets": []})

            # Use vehicle position if available, otherwise dock
            if world_snap.vehicle and world_snap.vehicle.position:
                drone_position = {
                    "lat": world_snap.vehicle.position.latitude,
                    "lon": world_snap.vehicle.position.longitude,
                    "alt": world_snap.vehicle.position.altitude_agl or 0,
                }
            else:
                drone_position = {
                    "lat": DOCK_LATITUDE,
                    "lon": DOCK_LONGITUDE,
                    "alt": 0,
                }

            # Get assets from world model
            for asset in world_snap.assets:
                rel = _calculate_relative_pos(
                    drone_position["lat"],
                    drone_position["lon"],
                    asset.position.latitude,
                    asset.position.longitude,
                )
                spatial_assets.append({
                    "id": asset.asset_id,
                    "type": asset.asset_type.value
                    if hasattr(asset.asset_type, "value")
                    else str(asset.asset_type),
                    "x": rel["x"],
                    "y": rel["y"],
                    "name": asset.name,
                    "has_anomaly": asset.asset_id
                    in [a.asset_id for a in world_snap.get_anomaly_assets()],
                })

        return JSONResponse({
            "assets": spatial_assets,
            "drone_position": drone_position,
            "asset_count": len(spatial_assets),
        })

    # ========================================================================
    # Mission Metrics API - Success tracking for dashboard
    # ========================================================================

    @app.get("/api/dashboard/metrics", response_class=JSONResponse)
    async def get_mission_metrics() -> JSONResponse:
        """Get aggregated mission success metrics.

        Returns comprehensive metrics including:
        - Inspection coverage (assets inspected / total)
        - Anomaly detection counts by type
        - Decision success rate
        - Mission completion percentage

        Returns:
            JSON response with mission metrics.
        """
        metrics: dict[str, Any] = {
            "total_assets": 0,
            "assets_inspected": 0,
            "inspection_coverage_percent": 0.0,
            "assets_in_progress": [],
            "total_anomalies_detected": 0,
            "total_anomalies_expected": 0,
            "anomalies_resolved": 0,
            "anomalies_pending": 0,
            "defects_by_type": {},
            "decisions_total": 0,
            "decisions_successful": 0,
            "success_rate_percent": 0.0,
            "decision_quality_percent": 0.0,
            "execution_success_percent": 0.0,
            "anomaly_handling_percent": 0.0,
            "resource_use_percent": 0.0,
            "resource_use_available": False,
            "battery_consumed": 0.0,
            "battery_budget": 0.0,
            "mission_success_score": 0.0,
            "mission_success_grade": "UNKNOWN",
            "mission_success_components": {},
            "mission_completion_percent": 0.0,
            "recent_captures_count": 0,
            "recent_defects_count": 0,
        }

        # Try to get from running scenario first
        shared_runner = scenario_runner_state.runner
        if shared_runner and shared_runner.run_state:
            run_state = shared_runner.run_state

            # Inspection coverage from scenario
            total_assets = len(run_state.scenario.assets)
            assets_inspected = len(run_state.assets_inspected)
            metrics["total_assets"] = total_assets
            metrics["assets_inspected"] = assets_inspected
            metrics["inspection_coverage_percent"] = (
                (assets_inspected / total_assets * 100) if total_assets > 0 else 0.0
            )
            metrics["assets_in_progress"] = list(run_state.assets_in_progress.keys())

            # Decision counts from drone states
            total_decisions = 0
            total_inspections = 0
            for drone_state in run_state.drone_states.values():
                total_decisions += drone_state.decisions_made
                total_inspections += drone_state.inspections_completed

            metrics["decisions_total"] = total_decisions

            # Estimate decision success for simulated runs
            if run_state.decision_log:
                successful = 0
                for entry in run_state.decision_log:
                    action = str(entry.get("action", "")).lower()
                    critic = entry.get("critic_validation") or {}
                    approved = critic.get("approved", True)
                    if action != "abort" and approved:
                        successful += 1
                metrics["decisions_successful"] = successful
                if total_decisions > 0:
                    metrics["success_rate_percent"] = successful / total_decisions * 100
            elif total_decisions > 0:
                metrics["decisions_successful"] = total_decisions
                metrics["success_rate_percent"] = 100.0

            # Count anomalies from scenario assets
            anomalies_detected = 0
            anomalies_expected = 0
            defects_by_type: dict[str, int] = {}
            for asset in run_state.scenario.assets:
                if asset.has_anomaly:
                    anomalies_expected += 1
                if asset.has_anomaly and asset.asset_id in run_state.assets_inspected:
                    anomalies_detected += 1
                    # Use asset type as defect type for now
                    defect_type = asset.asset_type or "unknown"
                    defects_by_type[defect_type] = defects_by_type.get(defect_type, 0) + 1

            metrics["total_anomalies_detected"] = anomalies_detected
            metrics["total_anomalies_expected"] = anomalies_expected
            metrics["defects_by_type"] = defects_by_type
            metrics["anomalies_pending"] = (
                anomalies_detected  # All detected are pending until resolved
            )

            # Mission completion based on inspection progress
            metrics["mission_completion_percent"] = metrics["inspection_coverage_percent"]

            # Resource use from scenario battery consumption
            drone_count = len(run_state.drone_states)
            metrics["battery_consumed"] = float(getattr(run_state, "total_battery_consumed", 0.0))
            metrics["battery_budget"] = float(drone_count * 100.0)
            if metrics["battery_budget"] > 0:
                efficiency = 1.0 - (metrics["battery_consumed"] / metrics["battery_budget"])
                metrics["resource_use_percent"] = max(0.0, min(100.0, efficiency * 100.0))
                metrics["resource_use_available"] = True

        # Fallback to world model if no scenario running
        elif server_state.world_model:
            world_snap = server_state.world_model.get_snapshot()
            if world_snap is None:
                return JSONResponse(metrics)

            if world_snap.mission:
                metrics["total_assets"] = world_snap.mission.assets_total
                metrics["assets_inspected"] = world_snap.mission.assets_inspected
                metrics["inspection_coverage_percent"] = world_snap.mission.progress_percent

            # Count anomalies from world model
            anomalies = world_snap.anomalies
            metrics["total_anomalies_detected"] = len(anomalies)
            metrics["total_anomalies_expected"] = len(anomalies)

            defects_by_type: dict[str, int] = {}
            anomalies_resolved = 0
            for anomaly in anomalies:
                if anomaly.resolved:
                    anomalies_resolved += 1
                defect_type = anomaly.anomaly_type or "unknown"
                defects_by_type[defect_type] = defects_by_type.get(defect_type, 0) + 1

            metrics["defects_by_type"] = defects_by_type
            metrics["anomalies_resolved"] = anomalies_resolved
            metrics["anomalies_pending"] = len(anomalies) - anomalies_resolved
            metrics["mission_completion_percent"] = metrics["inspection_coverage_percent"]

        # Get vision service statistics if available
        if server_state.vision_service:
            observations = server_state.vision_service.observations
            metrics["recent_captures_count"] = len(observations)

            # Count defects from observations
            defects_count = 0
            for obs in observations.values():
                if obs.defect_detected:
                    defects_count += 1
                    # Update defects_by_type from detections
                    for det in obs.detections:
                        det_class = det.get("class", det.get("detection_class", "unknown"))
                        if det_class:
                            metrics["defects_by_type"][det_class] = (
                                metrics["defects_by_type"].get(det_class, 0) + 1
                            )
            metrics["recent_defects_count"] = defects_count

        # Only use outcome tracker as fallback if no scenario decision data
        # This prevents mixing scenario counts with stale tracker state
        if metrics["decisions_total"] == 0 and server_state.outcome_tracker:
            tracker = server_state.outcome_tracker
            if tracker.total_outcomes_tracked > 0:
                metrics["decisions_total"] = tracker.total_outcomes_tracked
                metrics["decisions_successful"] = tracker.successful_outcomes
                metrics["success_rate_percent"] = (
                    tracker.successful_outcomes / tracker.total_outcomes_tracked * 100
                )

        # Derive decision quality and execution success
        if metrics["decisions_total"] > 0:
            metrics["execution_success_percent"] = metrics["success_rate_percent"]

        # Scenario decision quality based on critic approvals
        if shared_runner and shared_runner.run_state and shared_runner.run_state.decision_log:
            approvals = 0
            total = 0
            for entry in shared_runner.run_state.decision_log:
                if entry.get("type") != "decision":
                    continue
                total += 1
                critic = entry.get("critic_validation") or {}
                if critic.get("approved", True):
                    approvals += 1
            if total > 0:
                metrics["decision_quality_percent"] = approvals / total * 100

        if metrics["decision_quality_percent"] == 0.0 and metrics["decisions_total"] > 0:
            metrics["decision_quality_percent"] = metrics["success_rate_percent"]

        # Anomaly handling score
        if metrics["total_anomalies_expected"] > 0:
            if shared_runner and shared_runner.run_state:
                metrics["anomaly_handling_percent"] = (
                    metrics["total_anomalies_detected"] / metrics["total_anomalies_expected"] * 100
                )
            else:
                metrics["anomaly_handling_percent"] = (
                    metrics["anomalies_resolved"] / metrics["total_anomalies_detected"] * 100
                    if metrics["total_anomalies_detected"] > 0
                    else 100.0
                )
        else:
            metrics["anomaly_handling_percent"] = 100.0

        # Mission success score: weighted blend of key components
        config = get_config_manager().config
        dashboard_cfg = getattr(config, "dashboard", None)
        weights_cfg = getattr(dashboard_cfg, "mission_success_weights", None)
        thresholds_cfg = getattr(dashboard_cfg, "mission_success_thresholds", None)

        weights = {
            "coverage": float(getattr(weights_cfg, "coverage", 0.30)),
            "anomaly": float(getattr(weights_cfg, "anomaly", 0.25)),
            "decision_quality": float(getattr(weights_cfg, "decision_quality", 0.20)),
            "execution": float(getattr(weights_cfg, "execution", 0.15)),
            "resource_use": float(getattr(weights_cfg, "resource_use", 0.10)),
        }
        components: dict[str, float] = {}

        if metrics["total_assets"] > 0:
            components["coverage"] = metrics["inspection_coverage_percent"]
        if metrics["decisions_total"] > 0:
            components["decision_quality"] = metrics["decision_quality_percent"]
            components["execution"] = metrics["execution_success_percent"]
        if metrics["anomaly_handling_percent"] > 0 or metrics["total_anomalies_expected"] == 0:
            components["anomaly"] = metrics["anomaly_handling_percent"]
        if metrics["resource_use_available"]:
            components["resource_use"] = metrics["resource_use_percent"]

        if components:
            total_weight = sum(weights[key] for key in components)
            mission_score = 0.0
            for key, value in components.items():
                mission_score += value * weights[key]
            if total_weight > 0:
                mission_score = mission_score / total_weight
            metrics["mission_success_score"] = max(0.0, min(100.0, mission_score))
            metrics["mission_success_components"] = components

        score = metrics["mission_success_score"]
        excellent_threshold = float(getattr(thresholds_cfg, "excellent", 85.0))
        good_threshold = float(getattr(thresholds_cfg, "good", 70.0))
        fair_threshold = float(getattr(thresholds_cfg, "fair", 55.0))

        if score >= excellent_threshold:
            metrics["mission_success_grade"] = "EXCELLENT"
        elif score >= good_threshold:
            metrics["mission_success_grade"] = "GOOD"
        elif score >= fair_threshold:
            metrics["mission_success_grade"] = "FAIR"
        elif score > 0:
            metrics["mission_success_grade"] = "POOR"

        # Clamp success rate to valid range [0, 100]
        metrics["success_rate_percent"] = max(0.0, min(100.0, metrics["success_rate_percent"]))

        return JSONResponse(metrics)
