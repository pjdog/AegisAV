"""Scenario API routes and execution helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException

from agent.edge_config import EdgeComputeProfile, default_edge_compute_config
from agent.server.airsim_support import (
    _airsim_bridge_connected,
    _apply_airsim_environment,
    _execute_airsim_action,
    _map_scenario_environment,
    _schedule_airsim_connect,
    _schedule_airsim_environment,
    _sync_airsim_scene,
    _update_airsim_georef_for_scenario,
    broadcast_battery_update,
    broadcast_dock_state,
    broadcast_scenario_scene,
    seed_navigation_map_from_assets,
)
from agent.server.config_manager import get_config_manager
from agent.server.events import Event, EventSeverity, EventType
from agent.server.goals import Goal
from agent.server.scenario_runner import ScenarioRunner
from agent.server.scenarios import (
    DOCK_ALTITUDE,
    DOCK_LATITUDE,
    DOCK_LONGITUDE,
    ScenarioCategory,
    get_all_scenarios,
    get_scenario,
    get_scenarios_by_category,
    get_scenarios_by_difficulty,
)
from agent.server.state import (
    connection_manager,
    scenario_run_state,
    scenario_runner_state,
    server_state,
)
from agent.server.unreal_stream import (
    CognitiveLevel as UnrealCognitiveLevel,
    UnrealMessageType,
    thinking_tracker,
    unreal_manager,
)
from autonomy.vehicle_state import Position
from agent.server.world_model import DockStatus

logger = structlog.get_logger(__name__)


async def _broadcast_scenario_decision(drone_id: str, decision: dict) -> None:
    """Send scenario decisions to Unreal/overlay and dashboard clients."""
    action = str(decision.get("action", "UNKNOWN"))
    reason = str(decision.get("reason", ""))
    confidence = float(decision.get("confidence", 0.0))
    risk_score = float(decision.get("risk_score", 0.0))
    risk_level = str(decision.get("risk_level", "low")).lower()

    logger.info(
        "scenario_decision_received",
        drone_id=drone_id,
        action=action,
        confidence=confidence,
        reason=reason[:100] if reason else "",
        has_target_asset=bool(decision.get("target_asset")),
    )

    await thinking_tracker.start_thinking(
        drone_id=drone_id,
        goal=action,
        cognitive_level=UnrealCognitiveLevel.DELIBERATIVE,
    )
    await thinking_tracker.update_thinking(
        drone_id=drone_id,
        situation=reason or "Scenario decision executed.",
        considerations=[reason] if reason else [],
        risk_score=risk_score,
        risk_level=risk_level,
    )
    await thinking_tracker.complete_thinking(
        drone_id=drone_id,
        action=action,
        confidence=confidence,
        reasoning=reason,
    )

    await connection_manager.broadcast(
        Event(
            event_type=EventType.SERVER_DECISION,
            timestamp=datetime.now(),
            data={
                "decision_id": decision.get("decision_id"),
                "action": action,
                "confidence": confidence,
                "reasoning": reason,
                "risk_level": decision.get("risk_level"),
                "risk_score": decision.get("risk_score"),
                "battery_percent": decision.get("battery_percent"),
                "drone_id": drone_id,
            },
            severity=EventSeverity.INFO,
        )
    )

    executor = server_state.airsim_action_executor
    if action.lower() not in ("none", "wait"):
        exec_decision = dict(decision)
        exec_decision["drone_id"] = drone_id

        if not executor:
            config = get_config_manager().config
            if config.simulation.airsim_enabled:
                max_queued = 50
                if len(server_state.airsim_pending_actions) < max_queued:
                    server_state.airsim_pending_actions.append(exec_decision)
                    logger.info(
                        "airsim_action_queued",
                        action=action,
                        drone_id=drone_id,
                        queue_size=len(server_state.airsim_pending_actions),
                    )
                    _schedule_airsim_connect()
                else:
                    logger.warning(
                        "airsim_action_queue_full",
                        action=action,
                        drone_id=drone_id,
                        max_queued=max_queued,
                    )
            else:
                logger.debug(
                    "airsim_action_skipped_disabled",
                    action=action,
                    drone_id=drone_id,
                )
        else:
            try:
                asyncio.create_task(_execute_airsim_action(exec_decision))
            except Exception as exc:
                logger.warning(
                    "airsim_action_schedule_failed",
                    action=action,
                    error=str(exc),
                )


def register_scenario_routes(app: FastAPI) -> None:
    """Register scenario routes."""

    @app.get("/api/scenarios")
    async def list_scenarios(
        category: str | None = None,
        difficulty: str | None = None,
    ) -> dict:
        """List all available scenarios with optional filtering."""
        scenarios = []
        if category:
            try:
                cat_enum = ScenarioCategory(category)
                scenarios = get_scenarios_by_category(cat_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid category: {category}. "
                        f"Valid: {[c.value for c in ScenarioCategory]}"
                    ),
                ) from None
        elif difficulty:
            scenarios = get_scenarios_by_difficulty(difficulty)
        else:
            scenarios = get_all_scenarios()

        return {
            "scenarios": [s.to_dict() for s in scenarios],
            "count": len(scenarios),
            "categories": [c.value for c in ScenarioCategory],
        }

    @app.get("/api/scenarios/status")
    async def get_scenario_status() -> dict:
        """Get the status of the currently running scenario."""
        config = get_config_manager().config
        connect_task = server_state.airsim_connect_task

        airsim_status = {
            "enabled": config.simulation.airsim_enabled,
            "connected": _airsim_bridge_connected(),
            "connecting": bool(connect_task and not connect_task.done()),
            "executor_available": server_state.airsim_action_executor is not None,
            "pending_actions": len(server_state.airsim_pending_actions),
            "vehicle_name": config.simulation.airsim_vehicle_name,
            "host": config.simulation.airsim_host,
            "last_error": server_state.airsim_last_error,
        }

        if not scenario_run_state.running:
            return {
                "running": False,
                "unreal_connections": unreal_manager.active_connections,
                "airsim": airsim_status,
            }

        elapsed = None
        if scenario_run_state.start_time:
            elapsed = (datetime.now() - scenario_run_state.start_time).total_seconds()

        return {
            "running": True,
            "scenario_id": scenario_run_state.scenario_id,
            "mode": scenario_run_state.mode,
            "edge_profile": scenario_run_state.edge_profile,
            "time_scale": scenario_run_state.time_scale,
            "elapsed_s": elapsed,
            "start_time": (
                scenario_run_state.start_time.isoformat() if scenario_run_state.start_time else None
            ),
            "unreal_connections": unreal_manager.active_connections,
            "airsim": airsim_status,
            "environment_applied": server_state.airsim_env_last or None,
        }

    @app.get("/api/scenarios/{scenario_id}")
    async def get_scenario_detail(scenario_id: str) -> dict:
        """Get detailed information about a specific scenario."""
        scenario = get_scenario(scenario_id)
        if not scenario:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario not found: {scenario_id}",
            )

        return {
            "scenario": scenario.to_dict(),
            "drones": [
                {
                    "drone_id": d.drone_id,
                    "name": d.name,
                    "battery_percent": d.battery_percent,
                    "state": d.state.value,
                    "position": {
                        "latitude": d.latitude,
                        "longitude": d.longitude,
                        "altitude_agl": d.altitude_agl,
                    },
                }
                for d in scenario.drones
            ],
            "assets": [
                {
                    "asset_id": a.asset_id,
                    "name": a.name,
                    "asset_type": a.asset_type,
                    "priority": a.priority,
                    "has_anomaly": a.has_anomaly,
                    "anomaly_severity": a.anomaly_severity,
                    "position": {
                        "latitude": a.latitude,
                        "longitude": a.longitude,
                    },
                }
                for a in scenario.assets
            ],
            "environment": {
                "wind_speed_ms": scenario.environment.wind_speed_ms,
                "wind_direction_deg": scenario.environment.wind_direction_deg,
                "visibility_m": scenario.environment.visibility_m,
                "temperature_c": scenario.environment.temperature_c,
                "precipitation": scenario.environment.precipitation,
                "is_daylight": scenario.environment.is_daylight,
            },
            "events": [
                {
                    "timestamp_offset_s": e.timestamp_offset_s,
                    "event_type": e.event_type,
                    "description": e.description,
                    "data": e.data,
                }
                for e in scenario.events
            ],
        }

    @app.post("/api/scenarios/{scenario_id}/start")
    async def start_scenario(
        scenario_id: str,
        mode: str = "live",
        edge_profile: str = "SBC_CPU",
        time_scale: float = 1.0,
    ) -> dict:
        """Start a scenario execution."""
        try:
            config = get_config_manager().config
            logger.info(
                "scenario_start_requested",
                scenario_id=scenario_id,
                mode=mode,
                edge_profile=edge_profile,
                time_scale=time_scale,
                airsim_enabled=config.simulation.airsim_enabled,
                airsim_connected=_airsim_bridge_connected(),
                airsim_host=config.simulation.airsim_host,
                airsim_vehicle=config.simulation.airsim_vehicle_name,
            )

            scenario = get_scenario(scenario_id)
            if not scenario:
                raise HTTPException(
                    status_code=404,
                    detail=f"Scenario not found: {scenario_id}",
                )

            if scenario_run_state.running or scenario_runner_state.is_running:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Scenario already running: "
                        f"{scenario_run_state.scenario_id}. Stop it first."
                    ),
                )

            if mode not in ("live", "demo"):
                raise HTTPException(status_code=400, detail="Mode must be 'live' or 'demo'")

            valid_profiles = [
                "FC_ONLY",
                "MCU_HEURISTIC",
                "MCU_TINY_CNN",
                "SBC_CPU",
                "SBC_ACCEL",
                "JETSON_FULL",
            ]
            if edge_profile not in valid_profiles:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid edge profile: {edge_profile}. "
                        f"Valid: {valid_profiles}"
                    ),
                )

            time_scale = max(0.5, min(5.0, time_scale))

            scenario_log_dir = server_state.log_dir / "scenarios"
            runner = ScenarioRunner(log_dir=scenario_log_dir)
            loaded = await runner.load_scenario(scenario_id)
            if not loaded:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load scenario: {scenario_id}",
                )

            _update_airsim_georef_for_scenario(scenario)
            logger.info(
                "scenario_loaded",
                scenario_id=scenario_id,
                scenario_name=scenario.name,
                drones=len(scenario.drones),
                assets=len(scenario.assets),
                defects=len(scenario.defects),
            )

            def on_decision(drone_id: str, goal: Goal, decision: dict) -> None:
                logger.info(
                    "scenario_decision",
                    scenario_id=scenario_id,
                    drone_id=drone_id,
                    action=decision.get("action"),
                )
                del goal
                asyncio.create_task(_broadcast_scenario_decision(drone_id, decision))

            runner.on_decision = on_decision
            last_tick_log: dict[str, int] = {"elapsed": -10}

            def on_tick(run_state) -> None:
                _schedule_airsim_environment(run_state.environment)
                elapsed = int(run_state.elapsed_seconds)
                if elapsed - last_tick_log["elapsed"] >= 10:
                    last_tick_log["elapsed"] = elapsed
                    logger.info(
                        "scenario_tick",
                        scenario_id=scenario_id,
                        elapsed_s=elapsed,
                        decisions=run_state.decisions_count,
                        drones=len(run_state.drone_states),
                    )

            runner.on_tick = on_tick
            if runner.run_state:
                _schedule_airsim_environment(runner.run_state.environment)

            scenario_run_state.running = True
            scenario_run_state.scenario_id = scenario_id
            scenario_run_state.mode = mode
            scenario_run_state.edge_profile = edge_profile
            scenario_run_state.time_scale = time_scale
            scenario_run_state.start_time = datetime.now()

            try:
                profile = EdgeComputeProfile[edge_profile]
                server_state.edge_config = default_edge_compute_config(profile)
            except KeyError:
                pass

            logger.info(
                "scenario_started",
                scenario_id=scenario_id,
                mode=mode,
                edge_profile=edge_profile,
                time_scale=time_scale,
            )

            scenario_runner_state.runner = runner
            scenario_runner_state.is_running = True
            scenario_runner_state.last_error = None

            async def run_scenario() -> None:
                try:
                    logger.info(
                        "scenario_runner_begin",
                        scenario_id=scenario_id,
                        run_id=runner.run_id,
                    )
                    await runner.run(time_scale=time_scale)
                    if runner.run_state:
                        runner.save_decision_log(log_dir=scenario_log_dir)
                        logger.info(
                            "scenario_runner_complete",
                            scenario_id=scenario_id,
                            run_id=runner.run_id,
                            elapsed_s=runner.run_state.elapsed_seconds,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    scenario_runner_state.last_error = str(exc)
                    logger.exception(
                        "scenario_runner_failed",
                        scenario_id=scenario_id,
                        error=str(exc),
                    )
                finally:
                    scenario_runner_state.is_running = False
                    if scenario_runner_state.runner is runner:
                        scenario_runner_state.runner = None
                    scenario_run_state.running = False
                    scenario_run_state.scenario_id = None
                    scenario_run_state.start_time = None

            scenario_runner_state.run_task = asyncio.create_task(run_scenario())

            await connection_manager.broadcast(
                Event(
                    event_type=EventType.CLIENT_EXECUTION,
                    timestamp=datetime.now(),
                    data={
                        "event": "scenario_started",
                        "scenario_id": scenario_id,
                        "scenario_name": scenario.name,
                        "run_id": runner.run_id,
                        "mode": mode,
                        "edge_profile": edge_profile,
                        "time_scale": time_scale,
                        "drone_count": len(scenario.drones),
                        "asset_count": len(scenario.assets),
                        "defect_count": len(scenario.defects),
                    },
                    severity=EventSeverity.INFO,
                )
            )

            await broadcast_scenario_scene(scenario, include_defects=True)
            if unreal_manager.active_connections == 0:
                logger.warning(
                    "unreal_scene_not_visible",
                    scenario_id=scenario_id,
                    hint="No Unreal clients connected. Docks/assets will not appear.",
                )

            try:
                navigation_map = seed_navigation_map_from_assets(scenario)
                if navigation_map:
                    server_state.navigation_map = navigation_map
                    logger.info(
                        "navigation_map_ready",
                        scenario_id=scenario_id,
                        obstacle_count=len(navigation_map.get("obstacles", [])),
                        source=navigation_map.get("source"),
                    )
                    if server_state.store:
                        await server_state.store.set_state(
                            f"navigation:map:{scenario_id}",
                            navigation_map,
                        )
                        await server_state.store.set_state("navigation:map:latest", navigation_map)
            except Exception as exc:
                logger.warning(
                    "navigation_map_build_failed",
                    scenario_id=scenario_id,
                    error=str(exc),
                )

            if scenario.environment:
                mapped = _map_scenario_environment(scenario.environment)
                server_state.airsim_env_last = mapped
                await _apply_airsim_environment(
                    mapped,
                    scenario_id=scenario_id,
                    scenario_name=scenario.name,
                )

            server_state.world_model.set_dock(
                position=Position(
                    latitude=DOCK_LATITUDE,
                    longitude=DOCK_LONGITUDE,
                    altitude_msl=DOCK_ALTITUDE,
                ),
                status=DockStatus.AVAILABLE,
            )
            await broadcast_dock_state(
                dock_id="dock_main",
                status="available",
                latitude=DOCK_LATITUDE,
                longitude=DOCK_LONGITUDE,
                altitude_m=DOCK_ALTITUDE,
                beacon_active=True,
            )

            for drone in scenario.drones:
                await broadcast_battery_update(
                    drone_id=drone.drone_id,
                    percent=drone.battery_percent,
                    is_charging=False,
                )

            airsim_sync: dict[str, object] | None = None
            if config.simulation.airsim_enabled:
                if _airsim_bridge_connected():
                    airsim_sync = await _sync_airsim_scene(scenario)
                else:
                    asyncio.create_task(_sync_airsim_scene(scenario, wait_for_connect=True))
                    airsim_sync = {"scheduled": True, "reason": "airsim_not_connected"}
                    logger.warning(
                        "airsim_not_connected_at_start",
                        scenario_id=scenario_id,
                        host=config.simulation.airsim_host,
                        vehicle=config.simulation.airsim_vehicle_name,
                    )
                if server_state.airsim_action_executor is None:
                    logger.warning(
                        "airsim_executor_missing",
                        scenario_id=scenario_id,
                        reason="No AirSim action executor available yet.",
                    )
                elif server_state.navigation_map:
                    server_state.airsim_action_executor.set_avoid_zones(
                        server_state.navigation_map.get("obstacles", [])
                    )

            logger.info(
                "scenario_start_success",
                scenario_id=scenario_id,
                run_id=runner.run_id,
            )

            # Auto-fly mission: automatically takeoff and fly through all assets
            if mode == "live":

                async def auto_fly_mission() -> None:
                    """Automatically fly through all scenario assets."""
                    try:
                        # Wait for AirSim executor to become available (up to 30 seconds)
                        wait_attempts = 0
                        max_wait_attempts = 30
                        while server_state.airsim_action_executor is None and wait_attempts < max_wait_attempts:
                            wait_attempts += 1
                            logger.debug(
                                "auto_fly_waiting_for_executor",
                                attempt=wait_attempts,
                                max_attempts=max_wait_attempts,
                            )
                            await asyncio.sleep(1.0)

                        executor = server_state.airsim_action_executor
                        if not executor:
                            logger.warning(
                                "auto_fly_aborted_no_executor",
                                scenario_id=scenario_id,
                                hint="AirSim executor not available after waiting. Check AirSim connection.",
                            )
                            return

                        # Wait a moment for everything to initialize
                        await asyncio.sleep(2.0)

                        logger.info(
                            "auto_fly_mission_starting",
                            scenario_id=scenario_id,
                            asset_count=len(scenario.assets),
                        )

                        # Takeoff if not flying
                        if not executor.is_flying:
                            logger.info("auto_fly_takeoff")
                            takeoff_result = await executor.execute({
                                "action": "takeoff",
                                "parameters": {"altitude_agl": 30.0},
                                "reasoning": f"Auto-takeoff for scenario {scenario_id}",
                                "confidence": 1.0,
                            })
                            if takeoff_result.status.value != "completed":
                                logger.error(
                                    "auto_fly_takeoff_failed",
                                    error=takeoff_result.error,
                                )
                                return
                            await asyncio.sleep(1.0)

                        # Fly to each asset in sequence
                        logger.info(
                            "auto_fly_starting_asset_loop",
                            asset_count=len(scenario.assets),
                            scenario_running=scenario_runner_state.is_running,
                            executor_flying=executor.is_flying,
                        )

                        if len(scenario.assets) == 0:
                            logger.warning("auto_fly_no_assets", scenario_id=scenario_id)

                        for i, asset in enumerate(scenario.assets):
                            # Don't check is_running - we want to fly even if scenario runner stopped
                            # The scenario runner handles decision-making, but we want manual flight

                            logger.info(
                                "auto_fly_inspecting_asset",
                                asset_index=i + 1,
                                asset_count=len(scenario.assets),
                                asset_id=asset.asset_id,
                                asset_name=asset.name,
                            )

                            logger.info(
                                "auto_fly_executing_inspect",
                                asset_id=asset.asset_id,
                                lat=asset.latitude,
                                lon=asset.longitude,
                            )

                            inspect_result = await executor.execute({
                                "action": "inspect_asset",
                                "target_asset": {
                                    "asset_id": asset.asset_id,
                                    "name": asset.name,
                                    "latitude": asset.latitude,
                                    "longitude": asset.longitude,
                                    "inspection_altitude_agl": getattr(
                                        asset, "inspection_altitude_agl", 30.0
                                    ),
                                    "orbit_radius_m": getattr(asset, "orbit_radius_m", 20.0),
                                    "dwell_time_s": getattr(asset, "dwell_time_s", 15.0),
                                },
                                "reasoning": f"Inspecting asset {i + 1}/{len(scenario.assets)}: {asset.name}",
                                "confidence": 1.0,
                            })

                            logger.info(
                                "auto_fly_inspect_result",
                                asset_id=asset.asset_id,
                                status=inspect_result.status.value,
                                duration=inspect_result.duration_s,
                            )

                            if inspect_result.status.value == "failed":
                                logger.warning(
                                    "auto_fly_inspect_failed",
                                    asset_id=asset.asset_id,
                                    error=inspect_result.error,
                                )

                        # Return to home after all assets
                        logger.info("auto_fly_returning_home")
                        await executor.execute({
                            "action": "return_low_battery",
                            "reasoning": "Mission complete, returning to home",
                            "confidence": 1.0,
                        })

                        logger.info("auto_fly_mission_complete", scenario_id=scenario_id)

                    except asyncio.CancelledError:
                        logger.info("auto_fly_cancelled")
                        raise
                    except Exception as exc:
                        logger.exception(
                            "auto_fly_mission_error",
                            scenario_id=scenario_id,
                            error=str(exc),
                        )

                fly_task = asyncio.create_task(auto_fly_mission())

                def _on_fly_done(task: asyncio.Task) -> None:
                    try:
                        exc = task.exception()
                        if exc:
                            logger.error(
                                "auto_fly_task_failed",
                                scenario_id=scenario_id,
                                error=str(exc),
                                error_type=type(exc).__name__,
                            )
                    except asyncio.CancelledError:
                        logger.info("auto_fly_task_cancelled", scenario_id=scenario_id)
                    except asyncio.InvalidStateError:
                        pass

                fly_task.add_done_callback(_on_fly_done)
                logger.info("auto_fly_mission_scheduled", scenario_id=scenario_id)

            return {
                "status": "started",
                "scenario_id": scenario_id,
                "scenario_name": scenario.name,
                "run_id": runner.run_id,
                "mode": mode,
                "edge_profile": edge_profile,
                "time_scale": time_scale,
                "start_time": (
                    scenario_run_state.start_time.isoformat()
                    if scenario_run_state.start_time
                    else None
                ),
                "defects_spawned": len(scenario.defects) if scenario.defects else 0,
                "assets_spawned": len(scenario.assets),
                "environment_applied": bool(scenario.environment),
                "airsim_sync": airsim_sync,
                "unreal_connections": unreal_manager.active_connections,
                "airsim_connected": _airsim_bridge_connected(),
                "airsim_executor_available": server_state.airsim_action_executor is not None,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "scenario_start_failed",
                scenario_id=scenario_id,
                error=str(exc),
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start scenario: {exc}",
            ) from exc

    @app.post("/api/scenarios/stop")
    async def stop_scenario() -> dict:
        """Stop the currently running scenario."""
        logger.info("scenario_stop_requested")
        try:
            if not scenario_run_state.running and not scenario_runner_state.is_running:
                return {"status": "not_running"}

            scenario_id = scenario_run_state.scenario_id
            duration = None
            if scenario_run_state.start_time:
                duration = (datetime.now() - scenario_run_state.start_time).total_seconds()

            runner = scenario_runner_state.runner
            if runner:
                runner.stop()
            scenario_runner_state.is_running = False
            if scenario_runner_state.run_task:
                try:
                    if not scenario_runner_state.run_task.done():
                        scenario_runner_state.run_task.cancel()
                    await asyncio.wait_for(scenario_runner_state.run_task, timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    pass
                finally:
                    scenario_runner_state.run_task = None
                    scenario_runner_state.runner = None

            scenario_run_state.running = False
            scenario_run_state.scenario_id = None
            scenario_run_state.start_time = None

            logger.info("scenario_stopped", scenario_id=scenario_id, duration_s=duration)

            await connection_manager.broadcast(
                Event(
                    event_type=EventType.CLIENT_EXECUTION,
                    timestamp=datetime.now(),
                    data={
                        "event": "scenario_stopped",
                        "scenario_id": scenario_id,
                        "duration_s": duration,
                    },
                    severity=EventSeverity.INFO,
                )
            )

            if unreal_manager.active_connections > 0:
                await unreal_manager.broadcast(
                    UnrealMessageType.CLEAR_DEFECTS,
                    {"scenario_id": scenario_id},
                )

            return {
                "status": "stopped",
                "scenario_id": scenario_id,
                "duration_s": duration,
            }

        except Exception as exc:
            logger.exception("scenario_stop_failed", error=str(exc))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stop scenario: {exc}",
            ) from exc
