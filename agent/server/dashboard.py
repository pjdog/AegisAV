"""
Dashboard routes for monitoring agent activity.
"""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.server.scenario_runner import ScenarioRunner
from agent.server.scenarios import (
    ScenarioCategory,
    get_all_scenarios,
    get_scenario,
    get_scenarios_by_category,
    get_scenarios_by_difficulty,
)


class RunnerStartRequest(BaseModel):
    """Request to start scenario runner."""

    scenario_id: str
    time_scale: float = 1.0
    max_duration_s: float | None = None


class RunnerState:
    """Global state for the scenario runner."""

    def __init__(self) -> None:
        self.runner: Any = None
        self.run_task: asyncio.Task | None = None
        self.is_running: bool = False
        self.last_error: str | None = None


# Global runner state
_runner_state = RunnerState()


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
    """Calculate relative X, Y in meters (flat earth approximation)."""
    # 1 deg lat ~ 111,111 meters
    # 1 deg lon ~ 111,111 * cos(lat) meters
    lat_m = (target_lat - origin_lat) * 111111
    lon_m = (target_lon - origin_lon) * 111111 * math.cos(math.radians(origin_lat))
    return {"x": lon_m, "y": lat_m}


def _list_runs(log_dir: Path) -> list[str]:
    """List run IDs sorted by timestamp string."""
    runs = []
    for path in log_dir.glob("decisions_*.jsonl"):
        run_id = path.stem.replace("decisions_", "")
        runs.append(run_id)
    return sorted(runs)


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary metrics for a run."""
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
    """Prepare chart series for risk and battery."""
    risk_series = []
    battery_series = []
    for entry in entries:
        timestamp = entry.get("timestamp")
        risk_series.append({"t": timestamp, "value": entry.get("risk_score", 0)})
        battery_series.append({"t": timestamp, "value": entry.get("battery_percent", 0)})
    return risk_series, battery_series


def _action_counts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count decisions by action."""
    counts: dict[str, int] = {}
    for entry in entries:
        action = entry.get("action", "unknown")
        counts[action] = counts.get(action, 0) + 1
    return [{"action": action, "count": count} for action, count in counts.items()]


def _recent(entries: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    """Return the most recent decision entries for the table."""
    recent_entries = entries[-limit:]
    items = []
    for entry in recent_entries[::-1]:
        item = {
            "timestamp": entry.get("timestamp"),
            "action": entry.get("action"),
            "confidence": entry.get("confidence"),
            "risk_level": entry.get("risk_level"),
            "battery_percent": entry.get("battery_percent"),
            "reason": entry.get("reason"),
            "vehicle_state": {
                "armed": entry.get("armed", False),
                "mode": entry.get("mode", "UNKNOWN"),
            },
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


def add_dashboard_routes(app: FastAPI, log_dir: Path) -> None:
    """Register dashboard routes and static assets."""
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = repo_root / "frontend" / "dist"
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/dashboard/assets", StaticFiles(directory=assets_dir), name="dashboard-assets")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        dist_index = dist_dir / "index.html"
        if dist_index.exists():
            return FileResponse(dist_index)
        return HTMLResponse(
            "<html><body><h1>Dashboard build missing.</h1>"
            "<p>Run the frontend build to generate /frontend/dist.</p></body></html>"
        )

    @app.get("/api/dashboard/runs", response_class=JSONResponse)
    def list_runs() -> JSONResponse:
        runs = _list_runs(log_dir)
        return JSONResponse({"runs": runs, "latest": runs[-1] if runs else None})

    @app.get("/api/dashboard/run/{run_id}", response_class=JSONResponse)
    def run_data(run_id: str) -> JSONResponse:
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
    def list_scenarios(
        category: str | None = None,
        difficulty: str | None = None,
    ) -> JSONResponse:
        """List all available scenarios with optional filtering."""
        if category:
            try:
                cat_enum = ScenarioCategory(category)
                scenarios = get_scenarios_by_category(cat_enum)
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid category: {category}"
                ) from e
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
    def scenario_detail(scenario_id: str) -> JSONResponse:
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
            raise HTTPException(status_code=404, detail=f"Scenario not found: {request.scenario_id}")

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
            "time_scale": request.time_scale,
        })

    @app.post("/api/dashboard/runner/stop", response_class=JSONResponse)
    async def stop_runner() -> JSONResponse:
        """Stop the currently running scenario."""
        global _runner_state

        if not _runner_state.is_running or not _runner_state.runner:
            raise HTTPException(status_code=409, detail="No runner active")

        _runner_state.runner.stop()

        # Wait briefly for task to complete
        if _runner_state.run_task:
            try:
                await asyncio.wait_for(_runner_state.run_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass

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
    def runner_status() -> JSONResponse:
        """Get current runner status."""
        global _runner_state

        if not _runner_state.runner or not _runner_state.runner.run_state:
            return JSONResponse({
                "is_running": _runner_state.is_running,
                "scenario_id": None,
                "elapsed_seconds": 0,
                "drones": [],
                "last_error": _runner_state.last_error,
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
            "scenario_id": run_state.scenario.scenario_id,
            "scenario_name": run_state.scenario.name,
            "elapsed_seconds": run_state.elapsed_seconds,
            "is_complete": run_state.is_complete,
            "decision_count": len(run_state.decision_log),
            "drones": drones,
            "environment": {
                "wind_speed_ms": run_state.environment.wind_speed_ms if run_state.environment else 0,
                "visibility_m": run_state.environment.visibility_m if run_state.environment else 10000,
            },
            "last_error": _runner_state.last_error,
        })

    @app.get("/api/dashboard/runner/decisions", response_class=JSONResponse)
    def runner_decisions(limit: int = 20, offset: int = 0) -> JSONResponse:
        """Get recent decisions from the running scenario."""
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
