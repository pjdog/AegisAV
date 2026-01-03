"""FastAPI lifespan management for server startup/shutdown."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import yaml
from fastapi import FastAPI

from agent.server.critics import AuthorityModel, CriticOrchestrator
from agent.server.llm_config import apply_llm_settings, resolve_llm_model
from agent.server.persistence import RedisConfig, create_store
from agent.server.risk_evaluator import RiskEvaluator, RiskThresholds
from agent.server.state import scenario_run_state, scenario_runner_state, server_state
from agent.server.unreal_stream import unreal_manager
from agent.server.vision.vision_service import VisionService, VisionServiceConfig
from agent.server.config_manager import get_config_manager
from agent.server.airsim_support import (
    _schedule_airsim_connect,
    _start_airsim_bridge,
    _stop_airsim_bridge,
    _sync_airsim_scene,
)
from agent.server.scenarios import get_scenario
from mapping.map_update import MapUpdateConfig, MapUpdateService
from mapping.safety_gates import PlannerSafetyGate, SafetyGateConfig

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
    config_manager = get_config_manager()
    config_mgr = config_manager
    try:
        os.chdir(config_manager.project_root)
        logger.info("cwd_set", path=str(config_manager.project_root))
    except Exception as exc:
        logger.warning("cwd_set_failed", path=str(config_manager.project_root), error=str(exc))

    autorun_task: asyncio.Task | None = None

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
                server_cfg = vision_config.get("vision", {}).get("server", {})
                splat_change_cfg = server_cfg.get("splat_change_detection", {})
                vision_service_config = VisionServiceConfig(
                    confidence_threshold=server_cfg.get("detection", {}).get(
                        "confidence_threshold", 0.7
                    ),
                    severity_threshold=server_cfg.get("detection", {}).get(
                        "severity_threshold", 0.4
                    ),
                    enable_splat_change_detection=splat_change_cfg.get("enabled", False),
                    splat_change_threshold=splat_change_cfg.get("threshold", 0.85),
                    splat_change_min_age_s=splat_change_cfg.get("min_age_s", 600.0),
                    splat_change_max_age_s=splat_change_cfg.get("max_age_s", 3600.0),
                )

                def _resolve_splat_scene() -> Path | None:
                    splat_artifacts = getattr(server_state, "splat_artifacts", None) or {}
                    scene_path = splat_artifacts.get("scene_path") or splat_artifacts.get("scene")
                    if not scene_path:
                        return None
                    return config_manager.resolve_path(str(scene_path))
                server_state.vision_service = VisionService(
                    world_model=server_state.world_model,
                    config=vision_service_config,
                    unreal_manager=unreal_manager,
                    splat_scene_provider=_resolve_splat_scene,
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
    server_state.planner_safety_gate = PlannerSafetyGate(
        SafetyGateConfig(
            min_map_confidence=config.mapping.min_quality_score,
            max_map_age_s=config.mapping.max_map_age_s,
        )
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

    if config.simulation.airsim_enabled:
        _schedule_airsim_connect()
        logger.info("airsim_connect_scheduled")

    if config.mapping.enabled:
        map_config = MapUpdateConfig(
            enabled=True,
            update_interval_s=config.mapping.update_interval_s,
            prefer_splat=config.mapping.prefer_splat,
            max_map_age_s=config.mapping.max_map_age_s,
            min_quality_score=config.mapping.min_quality_score,
            slam_dir=config_mgr.resolve_path(config.mapping.slam_dir),
            splat_dir=config_mgr.resolve_path(config.mapping.splat_dir),
            map_resolution_m=config.mapping.map_resolution_m,
            tile_size_cells=getattr(config.mapping, "tile_size_cells", 120),
            voxel_size_m=config.mapping.voxel_size_m,
            max_points=config.mapping.max_points,
            min_points=config.mapping.min_points,
            fused_map_dir=config_mgr.resolve_path(config.mapping.fused_map_dir),
            fused_map_max_versions=config.mapping.fused_map_max_versions,
            fused_map_max_age_days=config.mapping.fused_map_max_age_days,
            fused_map_keep_last=config.mapping.fused_map_keep_last,
            proxy_regen_interval_s=config.mapping.proxy_regen_interval_s,
            proxy_max_points=config.mapping.proxy_max_points,
            splat_auto_train=config.mapping.splat_auto_train,
            splat_backend=config.mapping.splat_backend,
            splat_iterations=config.mapping.splat_iterations,
        )
        server_state.map_update_service = MapUpdateService(map_config, server_state)
        await server_state.map_update_service.start()

    async def _autorun_preflight_mapping() -> None:
        mapping_cfg = config.mapping
        if not (mapping_cfg.enabled and mapping_cfg.preflight_enabled and mapping_cfg.preflight_autorun):
            return

        scenario_id = mapping_cfg.preflight_autorun_scenario_id
        if not scenario_id:
            logger.warning("preflight_autorun_missing_scenario")
            return

        scenario = get_scenario(scenario_id)
        if not scenario:
            logger.warning("preflight_autorun_scenario_missing", scenario_id=scenario_id)
            return

        max_attempts = max(1, int(getattr(mapping_cfg, "preflight_autorun_max_attempts", 1)))
        retry_delay_s = max(2.0, float(getattr(mapping_cfg, "preflight_autorun_retry_delay_s", 15.0)))
        attempt = 0

        while attempt < max_attempts:
            if scenario_run_state.running or scenario_runner_state.is_running:
                logger.info("preflight_autorun_skipped", scenario_id=scenario_id, reason="scenario_running")
                return

            bridge = server_state.airsim_bridge
            if bridge and getattr(bridge, "connected", False) and server_state.airsim_action_executor:
                delay_s = max(0.0, float(mapping_cfg.preflight_autorun_delay_s))
                if delay_s:
                    await asyncio.sleep(delay_s)
                try:
                    await _sync_airsim_scene(
                        scenario,
                        wait_for_connect=True,
                        config_override=config,
                    )
                except Exception as exc:
                    logger.warning(
                        "preflight_autorun_scene_sync_failed",
                        scenario_id=scenario_id,
                        error=str(exc),
                    )
                try:
                    from agent.server.api_scenarios import _run_preflight_slam_mapping

                    await _run_preflight_slam_mapping(
                        scenario,
                        server_state.airsim_action_executor,
                    )
                except Exception as exc:
                    logger.warning(
                        "preflight_autorun_failed",
                        scenario_id=scenario_id,
                        error=str(exc),
                    )

                status = (server_state.preflight_status or {}).get("status")
                if status == "complete" and server_state.navigation_map:
                    logger.info("preflight_autorun_complete", scenario_id=scenario_id, attempt=attempt + 1)
                    return

                attempt += 1
                logger.warning(
                    "preflight_autorun_retry",
                    scenario_id=scenario_id,
                    attempt=attempt,
                    status=status,
                )
                await asyncio.sleep(retry_delay_s)
                continue

            if not bridge or not getattr(bridge, "connected", False):
                _schedule_airsim_connect()

            await asyncio.sleep(2.0)

        logger.warning("preflight_autorun_exhausted", scenario_id=scenario_id, attempts=max_attempts)

    if config.mapping.preflight_autorun:
        autorun_task = asyncio.create_task(_autorun_preflight_mapping())

    yield

    # Shutdown
    if autorun_task and not autorun_task.done():
        autorun_task.cancel()
        try:
            await autorun_task
        except asyncio.CancelledError:
            pass
    if server_state.map_update_service:
        await server_state.map_update_service.stop()
        server_state.map_update_service = None
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
