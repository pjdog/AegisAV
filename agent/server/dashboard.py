"""
Dashboard routes for monitoring agent activity.
"""

import json
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


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
