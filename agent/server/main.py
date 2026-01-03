"""Agent Server Main Entry Point.

FastAPI-based HTTP server providing the decision-making API.
The agent client sends vehicle state and receives decisions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    import logfire
except ImportError:  # pragma: no cover - optional dependency
    logfire = None

from agent.server.airsim_support import (
    _airsim_bridge_connected,
    _airsim_launch_supported,
    _launch_airsim_process,
    _schedule_airsim_connect,
    broadcast_scenario_scene,
)
from agent.server.airsim_support import (
    _sync_airsim_scene as _sync_airsim_scene_impl,
)
from agent.server.api_airsim import register_airsim_routes
from agent.server.api_chat import register_chat_routes
from agent.server.api_config import register_config_routes
from agent.server.api_decision import register_decision_routes
from agent.server.api_navigation import register_navigation_routes
from agent.server.api_scenarios import register_scenario_routes
from agent.server.api_telemetry import register_telemetry_routes
from agent.server.api_vision import register_vision_routes
from agent.server.config_manager import get_config_manager
from agent.server.dashboard import add_dashboard_routes
from agent.server.lifecycle import lifespan
from agent.server.scenarios import get_scenario
from agent.server.state import scenario_run_state, server_state
from agent.server.unreal_stream import add_unreal_routes, unreal_manager

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


async def get_airsim_status() -> dict:
    config = get_config_manager().config
    connect_task = server_state.airsim_connect_task
    bridge = server_state.airsim_bridge
    return {
        "enabled": config.simulation.airsim_enabled,
        "host": config.simulation.airsim_host,
        "vehicle_name": config.simulation.airsim_vehicle_name,
        "bridge_connected": _airsim_bridge_connected(),
        "connecting": bool(connect_task and not connect_task.done()),
        "launch_supported": _airsim_launch_supported(),
        "last_error": server_state.airsim_last_error,
        "vehicles": getattr(bridge, "vehicle_names", []),
    }


async def start_airsim() -> dict:
    config = get_config_manager().config
    if not config.simulation.airsim_enabled:
        raise HTTPException(
            status_code=409,
            detail="AirSim integration is disabled. Enable it in Settings -> Simulation.",
        )

    launch_supported, launch_started, launch_message = _launch_airsim_process()
    if not launch_started:
        server_state.airsim_last_error = launch_message
    _schedule_airsim_connect()

    return {
        "launch_supported": launch_supported,
        "launch_started": launch_started,
        "launch_message": launch_message,
        "bridge_connected": _airsim_bridge_connected(),
        "connecting": True,
    }


async def _sync_airsim_scene(scenario, wait_for_connect: bool = False) -> dict:
    config = get_config_manager().config
    return await _sync_airsim_scene_impl(
        scenario,
        wait_for_connect=wait_for_connect,
        config_override=config,
    )


app = FastAPI(
    title="AegisAV Agent Server",
    description="Agentic decision-making server for autonomous aerial monitoring",
    version="0.1.0",
    lifespan=lifespan,
)

# Dashboard routes - use scenarios subdirectory where scenario decision logs are saved
add_dashboard_routes(app, server_state.log_dir / "scenarios")

# Unreal real-time streaming routes
add_unreal_routes(app)

register_airsim_routes(app)
register_chat_routes(app)
register_config_routes(app)
register_decision_routes(app)
register_navigation_routes(app)
register_scenario_routes(app)
register_telemetry_routes(app)
register_vision_routes(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if logfire:
    try:
        logfire.configure(send_to_logfire=False)
        logfire.instrument_fastapi(app)
    except (ImportError, RuntimeError) as exc:
        logging.getLogger(__name__).warning(f"Logfire instrumentation disabled: {exc}")


async def _on_unreal_connect(connection_id: str) -> None:
    """Called when a new Unreal client connects.

    Sends the current scene state if a scenario is running.
    """
    if not scenario_run_state.running or not scenario_run_state.scenario_id:
        logger.debug("unreal_connect_no_scenario", connection_id=connection_id)
        return

    scenario = get_scenario(scenario_run_state.scenario_id)
    if not scenario:
        return

    logger.info(
        "unreal_connect_syncing_scene",
        connection_id=connection_id,
        scenario_id=scenario.scenario_id,
    )

    await asyncio.sleep(0.1)
    await broadcast_scenario_scene(scenario, include_defects=True)


unreal_manager.set_on_connect(_on_unreal_connect)

# Serve overlay static files for OBS Browser Source
overlay_dir = Path(__file__).resolve().parents[2] / "overlay"
if overlay_dir.exists():
    app.mount("/overlay", StaticFiles(directory=str(overlay_dir), html=True), name="overlay")
    logger.info("overlay_mounted", path=str(overlay_dir))


def main() -> None:
    """Run the agent server.

    Starts the uvicorn server with hot-reload enabled.
    """
    logging.basicConfig(level=logging.INFO)
    config = get_config_manager().load()
    host = os.environ.get("AEGIS_HOST", config.server.host)
    port = int(os.environ.get("AEGIS_PORT", config.server.port))
    reload_env = os.environ.get("AEGIS_RELOAD")
    if reload_env is None:
        reload_enabled = platform.system() != "Windows"
    else:
        reload_enabled = reload_env.strip().lower() in {"1", "true", "yes", "on"}

    uvicorn.run(
        "agent.server.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
