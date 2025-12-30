"""AirSim and Unreal bridge helpers shared across route modules."""

from __future__ import annotations

import asyncio
import math
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import structlog

from agent.server.config_manager import get_config_manager
from agent.server.navigation_map import build_navigation_map
from agent.server.scenarios import DOCK_ALTITUDE, DOCK_LATITUDE, DOCK_LONGITUDE, Scenario
from agent.server.state import server_state
from agent.server.unreal_stream import (
    BatteryUpdateMessage,
    DockUpdateMessage,
    EnvironmentUpdateMessage,
    SpawnAnomalyMarkerMessage,
    SpawnAssetMessage,
    SpawnDefectMessage,
    UnrealMessageType,
    unreal_manager,
)

logger = structlog.get_logger(__name__)

# Precipitation type to AirSim weather parameter mapping
PRECIPITATION_MAP: dict[str, dict[str, float]] = {
    "none": {"rain": 0.0, "snow": 0.0, "fog": 0.0, "dust": 0.0},
    "mist": {"fog": 0.15},
    "light_fog": {"fog": 0.3},
    "fog": {"fog": 0.6},
    "heavy_fog": {"fog": 0.85},
    "haze": {"fog": 0.2, "dust": 0.1},
    "overcast": {"fog": 0.1},
    "light_rain": {"rain": 0.3},
    "rain": {"rain": 0.6},
    "rainy": {"rain": 0.6},
    "heavy_rain": {"rain": 0.85, "fog": 0.2},
    "storm": {"rain": 0.9, "fog": 0.4},
    "snow": {"snow": 0.5},
    "snowy": {"snow": 0.5},
    "heavy_snow": {"snow": 0.8, "fog": 0.3},
    "blizzard": {"snow": 0.95, "fog": 0.5},
    "dust": {"dust": 0.4},
    "sand": {"dust": 0.5},
    "sandstorm": {"dust": 0.8, "fog": 0.3},
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _map_scenario_environment(env) -> dict[str, object]:
    """Map scenario environment conditions to AirSim weather parameters."""
    precipitation = (env.precipitation or "none").strip().lower()

    precip_effects = PRECIPITATION_MAP.get(precipitation, {})
    rain = precip_effects.get("rain", 0.0)
    snow = precip_effects.get("snow", 0.0)
    fog = precip_effects.get("fog", 0.0)
    dust = precip_effects.get("dust", 0.0)

    visibility = max(0.0, float(env.visibility_m or 10000.0))
    if visibility < 10000.0:
        vis_fog = 1.0 - (visibility / 10000.0) ** 0.5
        fog = max(fog, vis_fog)

    hour = getattr(env, "hour", 12 if env.is_daylight else 22)

    return {
        "rain": _clamp(rain),
        "snow": _clamp(snow),
        "fog": _clamp(fog),
        "dust": _clamp(dust),
        "hour": hour,
        "is_daylight": bool(env.is_daylight),
        "wind_speed_ms": float(env.wind_speed_ms),
        "wind_direction_deg": float(env.wind_direction_deg),
    }


def _env_changed(current: dict[str, object], updated: dict[str, object]) -> bool:
    if not current:
        return True

    thresholds = {
        "rain": 0.05,
        "snow": 0.05,
        "fog": 0.05,
        "dust": 0.05,
        "wind_speed_ms": 0.2,
        "wind_direction_deg": 2.0,
    }
    for key, value in updated.items():
        if key not in current:
            return True
        if isinstance(value, (int, float)):
            if abs(float(current[key]) - float(value)) > thresholds.get(key, 0.0):
                return True
        else:
            if current[key] != value:
                return True
    return False


def _update_airsim_georef_for_scenario(scenario: Scenario) -> bool:
    if not scenario.drones:
        return False
    try:
        from simulation.coordinate_utils import GeoReference

        first_drone = scenario.drones[0]
        ref_lat = first_drone.latitude
        ref_lon = first_drone.longitude
        ref_alt = 0.0

        new_geo_ref = GeoReference(ref_lat, ref_lon, ref_alt)
        server_state.airsim_geo_ref = new_geo_ref
        if server_state.airsim_action_executor:
            server_state.airsim_action_executor.geo_ref = new_geo_ref

        logger.info(
            "airsim_georef_updated",
            scenario_id=scenario.scenario_id,
            reference=f"({ref_lat:.6f}, {ref_lon:.6f})",
        )
        return True
    except Exception as exc:
        logger.warning("airsim_georef_update_failed", error=str(exc))
        return False


async def _apply_airsim_environment(
    mapped: dict[str, object],
    scenario_id: str | None = None,
    scenario_name: str | None = None,
) -> None:
    bridge = server_state.airsim_bridge
    if not bridge or not getattr(bridge, "connected", False):
        return

    try:
        await bridge.set_weather(
            rain=float(mapped["rain"]),
            snow=float(mapped["snow"]),
            fog=float(mapped["fog"]),
            dust=float(mapped["dust"]),
        )
        time_ok = await bridge.set_time_of_day(
            hour=int(mapped["hour"]),
            is_enabled=True,
            celestial_clock_speed=1.0,
        )
        await bridge.set_wind(
            speed_ms=float(mapped["wind_speed_ms"]),
            direction_deg=float(mapped["wind_direction_deg"]),
        )
        if not time_ok:
            logger.warning(
                "airsim_time_of_day_unavailable",
                hint="BP_Sky_Sphere missing or time-of-day not supported in this level.",
            )
    except Exception as exc:
        logger.warning("airsim_environment_apply_failed", error=str(exc))

    if unreal_manager.active_connections > 0:
        try:
            env_msg = EnvironmentUpdateMessage(
                timestamp_ms=time.time() * 1000,
                hour=int(mapped["hour"]),
                is_daylight=bool(mapped.get("is_daylight", True)),
                rain=float(mapped["rain"]),
                snow=float(mapped["snow"]),
                fog=float(mapped["fog"]),
                dust=float(mapped["dust"]),
                wind_speed_ms=float(mapped["wind_speed_ms"]),
                wind_direction_deg=float(mapped["wind_direction_deg"]),
                visibility_m=float(mapped.get("visibility_m", 10000.0)),
                scenario_id=scenario_id,
                scenario_name=scenario_name,
            )
            await unreal_manager.broadcast({
                "type": UnrealMessageType.ENVIRONMENT_UPDATE.value,
                **env_msg.model_dump(),
            })
        except Exception as exc:
            logger.warning("unreal_environment_broadcast_failed", error=str(exc))


async def _sync_airsim_scene(
    scenario: Scenario,
    wait_for_connect: bool = False,
    config_override=None,
) -> dict[str, object]:
    config = config_override or get_config_manager().config
    if not config.simulation.airsim_enabled:
        logger.info("airsim_scene_sync_skipped", reason="airsim_disabled")
        return {"synced": False, "reason": "airsim_disabled"}

    if wait_for_connect and not _airsim_bridge_connected():
        logger.info("airsim_scene_sync_wait", scenario_id=scenario.scenario_id)
        await _ensure_airsim_bridge()

    bridge = server_state.airsim_bridge
    logger.info(
        "airsim_scene_sync_start",
        scenario_id=scenario.scenario_id,
        bridge_connected=bool(bridge and getattr(bridge, "connected", False)),
        vehicle_name=getattr(bridge, "config", None).vehicle_name if bridge else None,
        vehicles=getattr(bridge, "vehicle_names", []),
    )
    if not bridge or not getattr(bridge, "connected", False):
        logger.warning(
            "airsim_scene_sync_failed",
            scenario_id=scenario.scenario_id,
            reason="airsim_not_connected",
        )
        return {"synced": False, "reason": "airsim_not_connected"}

    if not scenario.drones:
        logger.warning(
            "airsim_scene_sync_failed",
            scenario_id=scenario.scenario_id,
            reason="no_drones",
        )
        return {"synced": False, "reason": "no_drones"}

    _update_airsim_georef_for_scenario(scenario)

    first_drone = scenario.drones[0]
    altitude = float(first_drone.altitude_agl or 0.0)
    pose_ok = await bridge.set_vehicle_pose(0.0, 0.0, -altitude)
    if not pose_ok:
        logger.warning(
            "airsim_scene_sync_failed",
            scenario_id=scenario.scenario_id,
            reason="pose_failed",
        )
        return {"synced": False, "reason": "pose_failed"}

    if scenario.environment:
        mapped = _map_scenario_environment(scenario.environment)
        if _env_changed(server_state.airsim_env_last, mapped):
            server_state.airsim_env_last = mapped
            await _apply_airsim_environment(
                mapped,
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
            )

    logger.info(
        "airsim_scene_sync_complete",
        scenario_id=scenario.scenario_id,
        vehicle_name=config.simulation.airsim_vehicle_name,
    )
    return {
        "synced": True,
        "scenario_id": scenario.scenario_id,
        "vehicle_name": config.simulation.airsim_vehicle_name,
    }


def _schedule_airsim_environment(env) -> None:
    """Schedule environment update to AirSim if connected, queue if not."""
    bridge = server_state.airsim_bridge
    mapped = _map_scenario_environment(env)

    if _env_changed(server_state.airsim_env_last, mapped):
        server_state.airsim_env_last = mapped
    else:
        return

    if not bridge or not getattr(bridge, "connected", False):
        config = get_config_manager().config
        if config.simulation.airsim_enabled:
            _schedule_airsim_connect()
        return

    asyncio.create_task(_apply_airsim_environment(mapped))


def _airsim_bridge_connected() -> bool:
    bridge = server_state.airsim_bridge
    return bool(bridge and getattr(bridge, "connected", False))


async def broadcast_dock_state(
    dock_id: str = "dock_main",
    status: str = "available",
    latitude: float = 0.0,
    longitude: float = 0.0,
    altitude_m: float = 0.0,
    docked_drone_id: str | None = None,
    charge_percent: float | None = None,
    landing_pad_active: bool = False,
    beacon_active: bool = True,
    charging_animation: bool = False,
) -> None:
    """Broadcast dock state to Unreal clients."""
    if unreal_manager.active_connections == 0:
        return

    msg = DockUpdateMessage(
        dock_id=dock_id,
        status=status,
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude_m,
        docked_drone_id=docked_drone_id,
        charge_percent=charge_percent,
        landing_pad_active=landing_pad_active,
        beacon_active=beacon_active,
        charging_animation=charging_animation,
    )
    await unreal_manager.broadcast({
        "type": UnrealMessageType.DOCK_UPDATE.value,
        **msg.model_dump(),
    })


async def broadcast_battery_update(
    drone_id: str,
    percent: float,
    voltage: float = 22.2,
    current: float = -5.0,
    is_charging: bool = False,
    time_remaining_s: float | None = None,
    low_threshold: float = 30.0,
    critical_threshold: float = 15.0,
) -> None:
    """Broadcast battery state to Unreal clients with warning/critical alerts."""
    if unreal_manager.active_connections == 0:
        return

    is_low = percent <= low_threshold
    is_critical = percent <= critical_threshold

    msg = BatteryUpdateMessage(
        drone_id=drone_id,
        timestamp_ms=time.time() * 1000,
        percent=percent,
        voltage=voltage,
        current=current,
        is_charging=is_charging,
        is_low=is_low,
        is_critical=is_critical,
        time_remaining_s=time_remaining_s,
        low_threshold=low_threshold,
        critical_threshold=critical_threshold,
    )

    await unreal_manager.broadcast({
        "type": UnrealMessageType.BATTERY_UPDATE.value,
        **msg.model_dump(),
    })

    if is_critical:
        await unreal_manager.broadcast({
            "type": UnrealMessageType.BATTERY_CRITICAL.value,
            "drone_id": drone_id,
            "percent": percent,
            "timestamp_ms": time.time() * 1000,
        })
    elif is_low:
        await unreal_manager.broadcast({
            "type": UnrealMessageType.BATTERY_WARNING.value,
            "drone_id": drone_id,
            "percent": percent,
            "timestamp_ms": time.time() * 1000,
        })


async def broadcast_spawn_asset(
    asset_id: str,
    asset_type: str,
    name: str,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
    priority: int = 1,
    has_anomaly: bool = False,
    anomaly_severity: float = 0.0,
) -> None:
    """Broadcast asset spawn to Unreal clients."""
    if unreal_manager.active_connections == 0:
        return

    msg = SpawnAssetMessage(
        asset_id=asset_id,
        asset_type=asset_type,
        name=name,
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude_m,
        priority=priority,
        has_anomaly=has_anomaly,
        anomaly_severity=anomaly_severity,
    )
    await unreal_manager.broadcast({
        "type": UnrealMessageType.SPAWN_ASSET.value,
        **msg.model_dump(),
    })


async def broadcast_spawn_anomaly_marker(
    anomaly_id: str,
    asset_id: str,
    severity: float,
    latitude: float,
    longitude: float,
    altitude_m: float,
    label: str = "",
) -> None:
    """Broadcast anomaly marker spawn to Unreal clients."""
    if unreal_manager.active_connections == 0:
        return

    if severity < 0.4:
        color_r, color_g, color_b = 0.2, 0.8, 0.2
    elif severity < 0.7:
        color_r, color_g, color_b = 1.0, 0.8, 0.0
    else:
        color_r, color_g, color_b = 1.0, 0.2, 0.0

    msg = SpawnAnomalyMarkerMessage(
        anomaly_id=anomaly_id,
        asset_id=asset_id,
        severity=severity,
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude_m,
        color_r=color_r,
        color_g=color_g,
        color_b=color_b,
        pulse=severity >= 0.7,
        label=label,
    )
    await unreal_manager.broadcast({
        "type": UnrealMessageType.SPAWN_ANOMALY_MARKER.value,
        **msg.model_dump(),
    })


async def broadcast_scenario_scene(scenario: Scenario, include_defects: bool = True) -> None:
    """Broadcast full scenario scene setup to Unreal clients."""
    if unreal_manager.active_connections == 0:
        logger.warning(
            "broadcast_scenario_scene_skipped",
            reason="no_unreal_connections",
            scenario_id=scenario.scenario_id,
        )
        return

    await unreal_manager.broadcast({
        "type": UnrealMessageType.CLEAR_ASSETS.value,
        "scenario_id": scenario.scenario_id,
    })
    await unreal_manager.broadcast({
        "type": UnrealMessageType.CLEAR_ANOMALY_MARKERS.value,
        "scenario_id": scenario.scenario_id,
    })

    dock = server_state.world_model.get_dock()
    if dock:
        await broadcast_dock_state(
            dock_id="dock_main",
            status=dock.status.value,
            latitude=dock.position.latitude,
            longitude=dock.position.longitude,
            altitude_m=dock.position.altitude_msl,
            beacon_active=True,
        )
    elif scenario.drones:
        await broadcast_dock_state(
            dock_id="dock_main",
            status="available",
            latitude=DOCK_LATITUDE,
            longitude=DOCK_LONGITUDE,
            altitude_m=DOCK_ALTITUDE,
            beacon_active=True,
        )

    for asset in scenario.assets:
        await broadcast_spawn_asset(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            name=asset.name,
            latitude=asset.latitude,
            longitude=asset.longitude,
            priority=asset.priority,
            has_anomaly=asset.has_anomaly,
            anomaly_severity=asset.anomaly_severity,
        )

    if include_defects and scenario.defects:
        await unreal_manager.broadcast({
            "type": UnrealMessageType.CLEAR_DEFECTS.value,
            "scenario_id": scenario.scenario_id,
        })
        for defect in scenario.defects:
            spawn_msg = SpawnDefectMessage(
                asset_id=defect.asset_id,
                defect_type=defect.defect_type.upper(),
                severity=defect.severity,
                uv_x=defect.uv_x,
                uv_y=defect.uv_y,
                size=defect.size,
                defect_id=defect.defect_id,
            )
            await unreal_manager.broadcast({
                "type": UnrealMessageType.SPAWN_DEFECT.value,
                **spawn_msg.model_dump(),
            })

    for asset in scenario.assets:
        if asset.has_anomaly and asset.anomaly_severity > 0:
            await broadcast_spawn_anomaly_marker(
                anomaly_id=f"anom_{asset.asset_id}",
                asset_id=asset.asset_id,
                severity=asset.anomaly_severity,
                latitude=asset.latitude,
                longitude=asset.longitude,
                altitude_m=20.0,
                label=f"{asset.name}: {asset.anomaly_severity:.0%}",
            )

    if server_state.airsim_env_last:
        await unreal_manager.broadcast({
            "type": UnrealMessageType.ENVIRONMENT_UPDATE.value,
            "timestamp_ms": time.time() * 1000,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            **server_state.airsim_env_last,
        })

    logger.info(
        "scenario_scene_broadcast",
        scenario_id=scenario.scenario_id,
        assets=len(scenario.assets),
        defects=len(scenario.defects),
        unreal_connections=unreal_manager.active_connections,
    )


def _airsim_launch_script() -> Path:
    return Path(__file__).resolve().parents[2] / "start_airsim.bat"


def _airsim_launch_supported() -> bool:
    script_path = _airsim_launch_script()
    return script_path.exists() and platform.system().lower() == "windows"


def _launch_airsim_process() -> tuple[bool, bool, str]:
    script_path = _airsim_launch_script()
    if not script_path.exists():
        return False, False, f"AirSim launch script not found: {script_path}"
    if platform.system().lower() != "windows":
        return False, False, "AirSim launch is only supported when running on Windows."
    try:
        subprocess.Popen(  # noqa: S603
            ["cmd", "/c", str(script_path)],
            cwd=str(script_path.parent),
        )
        return True, True, "AirSim launch initiated."
    except Exception as exc:
        return True, False, f"AirSim launch failed: {exc}"


async def _ensure_airsim_bridge(
    retry_seconds: float = 45.0,
    interval_seconds: float = 2.5,
) -> bool:
    start_time = time.monotonic()
    while time.monotonic() - start_time < retry_seconds:
        if await _start_airsim_bridge():
            return True
        await asyncio.sleep(interval_seconds)
    if not server_state.airsim_last_error:
        server_state.airsim_last_error = "AirSim connection timed out."
    return False


def _schedule_airsim_connect() -> bool:
    task = server_state.airsim_connect_task
    if task and not task.done():
        return True

    async def _connect() -> None:
        try:
            await _ensure_airsim_bridge()
        finally:
            server_state.airsim_connect_task = None

    server_state.airsim_connect_task = asyncio.create_task(_connect())
    return True


async def _start_airsim_bridge() -> bool:
    config = get_config_manager().config
    if not config.simulation.airsim_enabled:
        logger.info("airsim_bridge_disabled")
        server_state.airsim_last_error = "AirSim integration is disabled."
        return False

    if _airsim_bridge_connected():
        bridge = server_state.airsim_bridge
        if bridge and bridge.config.host == config.simulation.airsim_host:
            return True
        logger.info("airsim_host_changed", new_host=config.simulation.airsim_host)

    if server_state.airsim_bridge or server_state.airsim_broadcaster:
        await _stop_airsim_bridge()

    try:
        from simulation.realtime_bridge import (  # noqa: PLC0415
            RealtimeAirSimBridge,
            RealtimeBridgeConfig,
            TelemetryBroadcaster,
        )
    except Exception as exc:
        logger.warning("airsim_bridge_unavailable", error=str(exc))
        server_state.airsim_last_error = f"AirSim bridge unavailable: {exc}"
        return False

    bridge_config = RealtimeBridgeConfig(
        host=config.simulation.airsim_host,
        vehicle_name=config.simulation.airsim_vehicle_name or "Drone1",
        save_images=config.vision.save_images,
        output_dir=Path(config.vision.image_output_dir),
    )
    if bridge_config.save_images:
        bridge_config.output_dir.mkdir(parents=True, exist_ok=True)
    bridge = RealtimeAirSimBridge(bridge_config)
    if not await bridge.connect():
        logger.warning("airsim_bridge_connect_failed")
        server_state.airsim_last_error = "Failed to connect to AirSim."
        return False

    async def broadcast(payload: dict) -> None:
        drone_id = payload.get("drone_id") or bridge_config.vehicle_name
        message = dict(payload)
        message["drone_id"] = drone_id
        await unreal_manager.broadcast_raw(message, drone_id=drone_id)

    broadcaster = TelemetryBroadcaster(bridge, broadcast)
    await broadcaster.start()

    try:
        from simulation.airsim_action_executor import AirSimActionExecutor, FlightConfig
        from simulation.coordinate_utils import GeoReference

        default_lat = 47.641468
        default_lon = -122.140165
        default_alt = 0.0

        geo_ref = GeoReference(default_lat, default_lon, default_alt)
        flight_config = FlightConfig(
            default_altitude_agl=30.0,
            default_velocity=5.0,
            inspection_orbit_radius=20.0,
            inspection_dwell_time=30.0,
        )

        executor = AirSimActionExecutor(
            bridge=bridge,
            geo_ref=geo_ref,
            config=flight_config,
            drone_id=bridge_config.vehicle_name,
        )

        server_state.airsim_action_executor = executor
        server_state.airsim_geo_ref = geo_ref
        logger.info(
            "airsim_action_executor_initialized",
            drone_id=bridge_config.vehicle_name,
            reference=f"({default_lat}, {default_lon})",
        )
        if server_state.navigation_map:
            executor.set_avoid_zones(server_state.navigation_map.get("obstacles", []))
            logger.info(
                "airsim_avoid_zones_applied",
                obstacle_count=len(server_state.navigation_map.get("obstacles", [])),
            )
    except Exception as exc:
        logger.warning("airsim_action_executor_init_failed", error=str(exc))

    server_state.airsim_bridge = bridge
    server_state.airsim_broadcaster = broadcaster
    server_state.airsim_last_error = None
    logger.info("airsim_bridge_started", vehicle=bridge_config.vehicle_name)

    asyncio.create_task(_flush_pending_airsim())
    return True


async def _flush_pending_airsim() -> None:
    """Apply any pending environment and execute queued actions."""
    if server_state.airsim_env_last:
        logger.info(
            "airsim_applying_pending_environment",
            env_keys=list(server_state.airsim_env_last.keys()),
        )
        await _apply_airsim_environment(server_state.airsim_env_last)

    pending = server_state.airsim_pending_actions
    if pending:
        count = len(pending)
        logger.info("airsim_flushing_pending_actions", count=count)
        server_state.airsim_pending_actions = []

        if count > 0:
            latest = pending[-1]
            logger.info(
                "airsim_executing_latest_pending",
                action=latest.get("action"),
                drone_id=latest.get("drone_id"),
                skipped=count - 1,
            )
            await _execute_airsim_action(latest)


async def _stop_airsim_bridge() -> None:
    broadcaster = server_state.airsim_broadcaster
    bridge = server_state.airsim_bridge
    executor = server_state.airsim_action_executor

    server_state.airsim_broadcaster = None
    server_state.airsim_bridge = None
    server_state.airsim_action_executor = None
    server_state.airsim_geo_ref = None

    if executor:
        try:
            await executor.cancel()
        except Exception as exc:
            logger.warning("airsim_executor_cancel_failed", error=str(exc))

    if broadcaster:
        try:
            await broadcaster.stop()
        except Exception as exc:
            logger.warning("airsim_bridge_stop_failed", error=str(exc))

    if bridge:
        try:
            await bridge.disconnect()
        except Exception as exc:
            logger.warning("airsim_bridge_disconnect_failed", error=str(exc))


async def _execute_airsim_action(decision: dict) -> None:
    """Execute a flight action in AirSim."""
    executor = server_state.airsim_action_executor
    if not executor:
        logger.warning(
            "airsim_action_executor_missing",
            action=decision.get("action"),
        )
        return

    action = decision.get("action", "unknown")
    logger.info(
        "airsim_action_executing",
        action=action,
        drone_id=decision.get("drone_id"),
        has_target=bool(decision.get("target_asset")),
    )

    try:
        result = await executor.execute(decision)
        if result.status.value == "failed":
            logger.warning(
                "airsim_action_failed",
                action=result.action,
                error=result.error,
            )
        else:
            logger.info(
                "airsim_action_completed",
                action=result.action,
                status=result.status.value,
                duration_s=result.duration_s,
            )
            action_name = str(result.action).lower()
            if action_name in ("inspect_asset", "inspect_anomaly", "inspect"):
                asyncio.create_task(_capture_airsim_vision(decision, result))
        avoidance = result.details.get("avoidance") if result.details else None
        if avoidance:
            logger.info(
                "airsim_avoidance_applied",
                action=result.action,
                drone_id=result.drone_id,
                avoidance=avoidance,
            )
            if server_state.store:
                await server_state.store.set_state(
                    "navigation:last_avoidance",
                    {
                        "decision_id": decision.get("decision_id"),
                        "action": result.action,
                        "drone_id": result.drone_id,
                        "timestamp": datetime.now().isoformat(),
                        "avoidance": avoidance,
                    },
                )
    except Exception as exc:
        logger.exception(
            "airsim_action_error",
            action=action,
            error=str(exc),
        )


def _build_vehicle_state_from_capture(capture) -> dict | None:
    telemetry = getattr(capture, "telemetry", None)
    if not telemetry or not telemetry.pose:
        return None
    position = telemetry.pose.position
    geo_ref = server_state.airsim_geo_ref
    lat = lon = alt = None
    if geo_ref:
        lat, lon, alt = geo_ref.ned_to_gps(position.x, position.y, position.z)

    return {
        "position": {
            "latitude": lat,
            "longitude": lon,
            "altitude_msl": alt,
        },
        "altitude_agl": -position.z,
    }


def _merge_navigation_obstacles(
    base_map: dict[str, object] | None,
    obstacles: list[dict[str, object]],
    source: str,
    scenario_id: str | None = None,
) -> dict[str, object]:
    existing = list((base_map or {}).get("obstacles", []))
    known = {item.get("asset_id") for item in existing}
    for obstacle in obstacles:
        asset_id = obstacle.get("asset_id")
        if asset_id and asset_id in known:
            continue
        existing.append(obstacle)
        if asset_id:
            known.add(asset_id)

    return {
        "scenario_id": scenario_id or (base_map or {}).get("scenario_id"),
        "generated_at": datetime.now().isoformat(),
        "source": source,
        "obstacles": existing,
    }


async def _capture_airsim_depth_obstacles(
    bridge,
    geo_ref,
    max_depth_m: float = 120.0,
    window_ratio: float = 0.2,
) -> dict[str, object]:
    if not bridge or not getattr(bridge, "client", None):
        return {"success": False, "error": "no_bridge"}

    try:
        import numpy as np
        import airsim  # type: ignore
    except Exception as exc:
        return {"success": False, "error": f"depth_dependencies_unavailable: {exc}"}

    camera_name = bridge.config.camera_name
    vehicle_name = bridge.config.vehicle_name
    info = await asyncio.to_thread(
        bridge.client.simGetCameraInfo,
        camera_name,
        vehicle_name,
    )
    fov_deg = getattr(info, "fov", 90.0)

    responses = await asyncio.to_thread(
        bridge.client.simGetImages,
        [
            airsim.ImageRequest(
                camera_name,
                airsim.ImageType.DepthPerspective,
                pixels_as_float=True,
                compress=False,
            )
        ],
        vehicle_name,
    )
    if not responses:
        return {"success": False, "error": "no_depth_response"}

    response = responses[0]
    if response.width == 0 or response.height == 0:
        return {"success": False, "error": "empty_depth_frame"}

    depth = np.array(response.image_data_float, dtype=np.float32)
    depth = depth.reshape(response.height, response.width)

    h, w = depth.shape
    window_h = max(1, int(h * window_ratio))
    window_w = max(1, int(w * window_ratio))
    h0 = (h - window_h) // 2
    w0 = (w - window_w) // 2
    window = depth[h0:h0 + window_h, w0:w0 + window_w]

    valid = window[np.isfinite(window)]
    valid = valid[valid > 0.5]
    if valid.size == 0:
        return {"success": True, "min_depth_m": None, "obstacles": []}

    min_depth = float(np.min(valid))
    if min_depth > max_depth_m:
        return {"success": True, "min_depth_m": min_depth, "obstacles": []}

    telemetry = await bridge.get_synchronized_state()
    if not telemetry or not telemetry.pose:
        return {"success": False, "error": "no_pose"}

    q = telemetry.pose.orientation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )

    north = telemetry.pose.position.x + math.cos(yaw) * min_depth
    east = telemetry.pose.position.y + math.sin(yaw) * min_depth
    down = telemetry.pose.position.z

    lat, lon, alt = geo_ref.ned_to_gps(north, east, down)
    obstacle_id = f"depth_{int(time.time() * 1000)}"
    obstacle = {
        "asset_id": obstacle_id,
        "name": "DepthObstacle",
        "asset_type": "depth_obstacle",
        "latitude": lat,
        "longitude": lon,
        "radius_m": max(5.0, min_depth * 0.3),
        "height_m": max(3.0, min_depth * 0.2),
        "source": "airsim_depth",
    }

    return {
        "success": True,
        "min_depth_m": min_depth,
        "fov_deg": fov_deg,
        "obstacles": [obstacle],
        "timestamp": datetime.now().isoformat(),
    }


async def _capture_airsim_vision(decision: dict, result) -> None:
    if not server_state.vision_enabled or not server_state.vision_service:
        return
    bridge = server_state.airsim_bridge
    if not bridge or not getattr(bridge, "connected", False):
        logger.warning(
            "vision_capture_skipped",
            reason="airsim_not_connected",
        )
        return

    capture = await bridge.capture_frame_synchronized(include_image=True)
    if not capture.success or not capture.image_path:
        logger.warning(
            "vision_capture_failed",
            error=capture.error,
        )
        return

    target_asset = decision.get("target_asset", {})
    asset_id = (
        target_asset.get("asset_id")
        or target_asset.get("name")
        or decision.get("asset_id")
        or "unknown"
    )

    vehicle_state = _build_vehicle_state_from_capture(capture)
    logger.info(
        "vision_capture_saved",
        asset_id=asset_id,
        image_path=str(capture.image_path),
        sequence=getattr(capture, "sequence", None),
    )

    try:
        observation = await server_state.vision_service.process_inspection_result(
            asset_id=asset_id,
            image_path=capture.image_path,
            vehicle_state=vehicle_state,
        )
        logger.info(
            "vision_observation_created",
            asset_id=asset_id,
            observation_id=observation.observation_id,
            defect_detected=observation.defect_detected,
            anomaly_created=observation.anomaly_created,
        )
        if server_state.store:
            await server_state.store.add_detection(
                asset_id,
                observation.model_dump(mode="json"),
            )
            if observation.anomaly_created and observation.anomaly_id:
                await server_state.store.add_anomaly(
                    observation.anomaly_id,
                    {
                        "anomaly_id": observation.anomaly_id,
                        "asset_id": observation.asset_id,
                        "severity": observation.max_severity,
                        "confidence": observation.max_confidence,
                        "source": "airsim_camera",
                        "timestamp": observation.timestamp.isoformat(),
                    },
                )
            await server_state.store.set_state(
                "vision:last_capture",
                {
                    "asset_id": asset_id,
                    "observation_id": observation.observation_id,
                    "image_path": str(capture.image_path),
                    "timestamp": observation.timestamp.isoformat(),
                    "defect_detected": observation.defect_detected,
                    "anomaly_created": observation.anomaly_created,
                },
            )

        if server_state.airsim_geo_ref:
            depth_result = await _capture_airsim_depth_obstacles(
                bridge,
                server_state.airsim_geo_ref,
            )
            server_state.last_depth_capture = depth_result
            if depth_result.get("obstacles"):
                server_state.navigation_map = _merge_navigation_obstacles(
                    server_state.navigation_map,
                    depth_result["obstacles"],
                    source="airsim_depth",
                    scenario_id=decision.get("scenario_id"),
                )
                if server_state.airsim_action_executor:
                    server_state.airsim_action_executor.set_avoid_zones(
                        server_state.navigation_map.get("obstacles", [])
                    )
                logger.info(
                    "vision_depth_obstacles",
                    count=len(depth_result["obstacles"]),
                    min_depth_m=depth_result.get("min_depth_m"),
                )
                if server_state.store:
                    await server_state.store.set_state(
                        "navigation:depth_obstacles:latest",
                        depth_result,
                    )
                    await server_state.store.set_state(
                        "navigation:map:latest",
                        server_state.navigation_map,
                    )
    except Exception as exc:
        logger.warning(
            "vision_processing_failed",
            asset_id=asset_id,
            error=str(exc),
        )


def seed_navigation_map_from_assets(scenario: Scenario) -> dict[str, object] | None:
    if server_state.vision_enabled and server_state.vision_service:
        return server_state.vision_service.seed_navigation_map(scenario.assets, scenario.scenario_id)
    return build_navigation_map(
        scenario.assets,
        scenario.scenario_id,
        source="scenario_assets",
    )
