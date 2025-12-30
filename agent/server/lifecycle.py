"""FastAPI lifespan management for server startup/shutdown."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import yaml
from fastapi import FastAPI

from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.llm_config import apply_llm_settings, resolve_llm_model
from agent.server.persistence import RedisConfig, create_store
from agent.server.risk_evaluator import RiskEvaluator, RiskThresholds
from agent.server.state import server_state
from agent.server.unreal_stream import unreal_manager
from agent.server.vision.vision_service import VisionService, VisionServiceConfig
from agent.server.config_manager import get_config_manager
from agent.server.airsim_support import _start_airsim_bridge, _stop_airsim_bridge

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler.

    Args:
        _app: FastAPI application instance.

    Yields:
        None after startup, resumes for shutdown.
    """
    await asyncio.sleep(0)
    # Startup
    config_dir = Path(__file__).resolve().parents[3] / "configs"

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
                    unreal_manager=unreal_manager,
                )

                await server_state.vision_service.initialize()
                server_state.vision_enabled = True
                logger.info("vision_service_initialized")

            except Exception as exc:
                raise RuntimeError(
                    f"Vision system initialization failed: {exc}. "
                    "Set vision.enabled=false in config to disable vision."
                ) from exc
        else:
            logger.info("vision_config_disabled", message="Vision system disabled in config")
    else:
        logger.info(
            "vision_config_not_found", message="Vision config file not found, vision disabled"
        )

    config = get_config_manager().config
    apply_llm_settings(config)
    server_state.critic_orchestrator = CriticOrchestrator(
        authority_model=AuthorityModel.ESCALATION,
        enable_llm=config.agent.use_llm,
        llm_model=resolve_llm_model(config.agent.llm_model, config.agent.llm_provider),
    )

    # Initialize persistence based on config
    if config.redis.enabled:
        try:
            telemetry_ttl = max(0, int(config.redis.telemetry_ttl_hours) * 3600)
            detection_ttl = max(0, int(config.redis.detection_ttl_days) * 86400)
            anomaly_ttl = max(0, int(config.redis.anomaly_ttl_days) * 86400)
            mission_ttl = max(0, int(config.redis.mission_ttl_days) * 86400)
            redis_config = RedisConfig(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
                telemetry_ttl=telemetry_ttl,
                detection_ttl=detection_ttl,
                anomaly_ttl=anomaly_ttl,
                mission_ttl=mission_ttl,
            )
            server_state.store = create_store(redis_config)
            connected = await server_state.store.connect()
            if connected:
                server_state.persistence_enabled = True
                logger.info("persistence_initialized", type="redis", host=config.redis.host)
            else:
                raise RuntimeError(
                    f"Redis connection failed to {config.redis.host}:{config.redis.port}. "
                    "Set redis.enabled=false in config to use in-memory storage."
                )
        except Exception as exc:
            raise RuntimeError(
                f"Redis persistence failed: {exc}. "
                "Set redis.enabled=false in config to use in-memory storage."
            ) from exc
    else:
        from agent.server.persistence import InMemoryStore  # noqa: PLC0415

        server_state.store = InMemoryStore()
        await server_state.store.connect()
        server_state.persistence_enabled = False
        logger.info("persistence_initialized", type="in_memory", reason="redis_disabled")

    server_state.current_run_id = server_state.decision_logger.start_run()
    server_state.telemetry_logger.start_run(server_state.current_run_id)
    logger.info("server_started")

    await _start_airsim_bridge()

    yield

    # Shutdown
    await _stop_airsim_bridge()
    if server_state.airsim_connect_task and not server_state.airsim_connect_task.done():
        server_state.airsim_connect_task.cancel()
        try:
            await server_state.airsim_connect_task
        except asyncio.CancelledError:
            pass
    if server_state.vision_service:
        await server_state.vision_service.shutdown()
        logger.info("vision_service_shutdown")

    if server_state.store:
        await server_state.store.disconnect()
        logger.info("persistence_shutdown")

    server_state.decision_logger.end_run()
    server_state.telemetry_logger.end_run()
    logger.info("server_stopped")
