"""AirSim, Unreal, camera, dock, battery, and environment API routes."""

from __future__ import annotations

import asyncio
import base64
import time

import structlog
from fastapi import Depends, FastAPI, HTTPException

from agent.server.airsim_support import (
    PRECIPITATION_MAP,
    _airsim_bridge_connected,
    _airsim_launch_supported,
    _apply_airsim_environment,
    _launch_airsim_process,
    _schedule_airsim_connect,
    _start_airsim_bridge,
    _stop_airsim_bridge,
    _sync_airsim_scene,
    broadcast_battery_update,
    broadcast_dock_state,
    broadcast_scenario_scene,
    broadcast_spawn_anomaly_marker,
)
from agent.server.config_manager import get_config_manager
from agent.server.deps import auth_handler
from agent.server.scenarios import get_scenario
from agent.server.state import scenario_run_state, scenario_runner_state, server_state
from agent.server.unreal_stream import CameraFrame, UnrealMessageType, unreal_manager
from agent.server.world_model import DockStatus

logger = structlog.get_logger(__name__)


async def _camera_stream_loop(drone_id: str, fps: float = 10.0) -> None:
    """Background task that captures and broadcasts camera frames."""
    interval = 1.0 / fps
    bridge = server_state.airsim_bridge

    while drone_id in server_state.camera_streaming:
        try:
            start = time.perf_counter()

            if bridge and bridge.connected:
                frame_result = await bridge.capture_frame_synchronized(include_image=True)

                if frame_result.success and frame_result.image_path:
                    image_data = frame_result.image_path.read_bytes()
                    image_b64 = base64.b64encode(image_data).decode("ascii")

                    seq = server_state.camera_stream_sequence.get(drone_id, 0) + 1
                    server_state.camera_stream_sequence[drone_id] = seq

                    camera_frame = CameraFrame(
                        drone_id=drone_id,
                        sequence=seq,
                        timestamp_ms=frame_result.server_timestamp_ms,
                        image_base64=image_b64,
                        width=1280,
                        height=720,
                        camera_name=bridge.config.camera_name if bridge else "front_center",
                        fov_deg=90.0,
                    )

                    await unreal_manager.broadcast({
                        "type": UnrealMessageType.CAMERA_FRAME.value,
                        **camera_frame.model_dump(),
                    })

            elapsed = time.perf_counter() - start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("camera_stream_error", drone_id=drone_id, error=str(exc))
            await asyncio.sleep(interval)


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


async def get_airsim_environment() -> dict:
    """Get current environment state being applied to AirSim."""
    return {
        "current": server_state.airsim_env_last,
        "bridge_connected": _airsim_bridge_connected(),
        "precipitation_types": list(PRECIPITATION_MAP.keys()),
    }


async def sync_airsim_scene(scenario_id: str | None = None) -> dict:
    """Sync scenario state into AirSim for the active scenario (or a specific ID)."""
    config = get_config_manager().config
    if not config.simulation.airsim_enabled:
        raise HTTPException(status_code=409, detail="AirSim integration is disabled.")

    scenario = None
    if scenario_id:
        scenario = get_scenario(scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")
    elif scenario_run_state.scenario_id:
        scenario = get_scenario(scenario_run_state.scenario_id)

    if not scenario:
        return {"synced": False, "reason": "no_scenario"}

    return await _sync_airsim_scene(scenario)


async def sync_unreal_scene() -> dict:
    """Request full scene sync to Unreal Engine."""
    scenario = None
    if scenario_run_state.running and scenario_run_state.scenario_id:
        scenario = get_scenario(scenario_run_state.scenario_id)

    if not scenario:
        return {
            "status": "no_scenario",
            "message": "No scenario is currently running",
            "unreal_connections": unreal_manager.active_connections,
        }

    if unreal_manager.active_connections == 0:
        return {
            "status": "no_clients",
            "message": "No Unreal clients connected",
            "scenario_id": scenario.scenario_id,
        }

    await broadcast_scenario_scene(scenario, include_defects=True)

    if server_state.airsim_env_last:
        await _apply_airsim_environment(
            server_state.airsim_env_last,
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
        )

    return {
        "status": "synced",
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.name,
        "assets_broadcast": len(scenario.assets),
        "defects_broadcast": len(scenario.defects),
        "unreal_connections": unreal_manager.active_connections,
        "environment": server_state.airsim_env_last,
    }


async def get_unreal_scene_state() -> dict:
    """Get current scene state that would be broadcast to Unreal."""
    scenario = None
    if scenario_run_state.running and scenario_run_state.scenario_id:
        scenario = get_scenario(scenario_run_state.scenario_id)

    if not scenario:
        return {
            "scenario_running": False,
            "unreal_connections": unreal_manager.active_connections,
        }

    dock_snapshot = server_state.world_model.get_dock()
    dock_state = None
    if dock_snapshot:
        dock_state = {
            "status": dock_snapshot.status.value,
            "latitude": dock_snapshot.position.latitude,
            "longitude": dock_snapshot.position.longitude,
        }

    return {
        "scenario_running": True,
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.name,
        "unreal_connections": unreal_manager.active_connections,
        "assets": [
            {
                "asset_id": a.asset_id,
                "name": a.name,
                "type": a.asset_type,
                "latitude": a.latitude,
                "longitude": a.longitude,
                "has_anomaly": a.has_anomaly,
                "anomaly_severity": a.anomaly_severity,
            }
            for a in scenario.assets
        ],
        "defects": [
            {
                "defect_id": d.defect_id,
                "asset_id": d.asset_id,
                "type": d.defect_type,
                "severity": d.severity,
            }
            for d in scenario.defects
        ],
        "dock": dock_state,
        "environment": server_state.airsim_env_last,
        "drones": [
            {
                "drone_id": d.drone_id,
                "name": d.name,
                "state": d.state.value,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "battery_percent": d.battery_percent,
            }
            for d in scenario.drones
        ],
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


async def reconnect_airsim() -> dict:
    """Force reconnect to AirSim with current config."""
    config = get_config_manager().config
    if not config.simulation.airsim_enabled:
        raise HTTPException(
            status_code=409,
            detail="AirSim integration is disabled.",
        )

    await _stop_airsim_bridge()
    success = await _start_airsim_bridge()

    return {
        "status": "connected" if success else "failed",
        "host": config.simulation.airsim_host,
        "vehicle_name": config.simulation.airsim_vehicle_name,
        "bridge_connected": _airsim_bridge_connected(),
        "executor_available": server_state.airsim_action_executor is not None,
        "last_error": server_state.airsim_last_error,
    }


async def airsim_takeoff(altitude: float = 10.0) -> dict:
    """Take off to specified altitude."""
    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(status_code=503, detail="AirSim not connected")

    try:
        logger.info("takeoff_api_called", altitude=altitude)
        success = await bridge.takeoff(altitude)
        logger.info("takeoff_api_result", success=success)
        return {
            "status": "completed" if success else "failed",
            "action": "takeoff",
            "altitude_agl": altitude,
            "bridge_connected": bridge.connected,
        }
    except Exception as exc:
        logger.exception("takeoff_api_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def airsim_land() -> dict:
    """Land at current position."""
    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(status_code=503, detail="AirSim not connected")

    try:
        success = await bridge.land()
        return {
            "status": "completed" if success else "failed",
            "action": "land",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def airsim_move(x: float, y: float, z: float, velocity: float = 5.0) -> dict:
    """Move to NED position."""
    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(status_code=503, detail="AirSim not connected")

    try:
        logger.info("airsim_move_called", x=x, y=y, z=z, velocity=velocity)

        # Enable API control directly here as a test
        import asyncio
        await asyncio.to_thread(
            bridge.client.enableApiControl, True, bridge.config.vehicle_name
        )
        logger.info("airsim_move_api_control_enabled")

        if hasattr(bridge, "move_to_position_with_obstacle_avoidance"):
            success = await bridge.move_to_position_with_obstacle_avoidance(
                x,
                y,
                z,
                velocity=velocity,
                obstacle_distance_m=15.0,
                avoidance_step_m=10.0,
            )
        else:
            success = await bridge.move_to_position(x, y, z, velocity)
        logger.info("airsim_move_result", success=success)
        return {
            "status": "completed" if success else "failed",
            "action": "move",
            "position": {"x": x, "y": y, "z": z},
            "velocity": velocity,
        }
    except Exception as exc:
        logger.exception("airsim_move_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def airsim_hover() -> dict:
    """Hold current position."""
    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(status_code=503, detail="AirSim not connected")

    try:
        success = await bridge.hover()
        return {
            "status": "completed" if success else "failed",
            "action": "hover",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def airsim_rtl() -> dict:
    """Return to launch position and land."""
    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(status_code=503, detail="AirSim not connected")

    try:
        success = await bridge.return_to_launch()
        return {
            "status": "completed" if success else "failed",
            "action": "rtl",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def airsim_orbit(
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = -30,
    radius: float = 20.0,
    duration: float = 30.0,
) -> dict:
    """Orbit around a point."""
    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(status_code=503, detail="AirSim not connected")

    try:
        success = await bridge.orbit(center_x, center_y, center_z, radius, duration=duration)
        return {
            "status": "completed" if success else "failed",
            "action": "orbit",
            "center": {"x": center_x, "y": center_y, "z": center_z},
            "radius": radius,
            "duration": duration,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def airsim_flight_status() -> dict:
    """Get flight executor status."""
    executor = server_state.airsim_action_executor
    bridge = server_state.airsim_bridge
    geo_ref = server_state.airsim_geo_ref

    return {
        "bridge_connected": bridge is not None and bridge.connected,
        "executor_available": executor is not None,
        "is_flying": executor.is_flying if executor else False,
        "current_action": executor.current_action if executor else None,
        "geo_reference": {
            "latitude": geo_ref.latitude,
            "longitude": geo_ref.longitude,
            "altitude": geo_ref.altitude,
        } if geo_ref else None,
    }


async def start_camera_stream(
    drone_id: str = "Drone1",
    fps: float = 10.0,
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Start streaming camera frames to Unreal clients."""
    fps = max(1.0, min(30.0, fps))

    if drone_id in server_state.camera_streaming:
        return {
            "status": "already_streaming",
            "drone_id": drone_id,
            "fps": fps,
        }

    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(
            status_code=503,
            detail="AirSim bridge not connected",
        )

    task = asyncio.create_task(_camera_stream_loop(drone_id, fps))
    server_state.camera_streaming[drone_id] = task
    server_state.camera_stream_sequence[drone_id] = 0

    logger.info("camera_stream_started", drone_id=drone_id, fps=fps)

    return {
        "status": "streaming",
        "drone_id": drone_id,
        "fps": fps,
        "unreal_clients": unreal_manager.active_connections,
    }


async def stop_camera_stream(
    drone_id: str = "Drone1",
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Stop camera frame streaming."""
    if drone_id not in server_state.camera_streaming:
        return {
            "status": "not_streaming",
            "drone_id": drone_id,
        }

    task = server_state.camera_streaming.pop(drone_id)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    frames_sent = server_state.camera_stream_sequence.pop(drone_id, 0)
    logger.info("camera_stream_stopped", drone_id=drone_id, frames_sent=frames_sent)

    return {
        "status": "stopped",
        "drone_id": drone_id,
        "frames_sent": frames_sent,
    }


async def get_camera_snapshot(
    drone_id: str,
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Capture a single camera snapshot."""
    bridge = server_state.airsim_bridge
    if not bridge or not bridge.connected:
        raise HTTPException(
            status_code=503,
            detail="AirSim bridge not connected",
        )

    try:
        frame_result = await bridge.capture_frame_synchronized(include_image=True)

        if not frame_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Capture failed: {frame_result.error}",
            )

        if not frame_result.image_path:
            raise HTTPException(
                status_code=500,
                detail="No image captured",
            )

        image_data = frame_result.image_path.read_bytes()
        image_b64 = base64.b64encode(image_data).decode("ascii")

        return {
            "drone_id": drone_id,
            "timestamp_ms": frame_result.server_timestamp_ms,
            "image_base64": image_b64,
            "capture_latency_ms": frame_result.capture_latency_ms,
            "telemetry": frame_result.telemetry.model_dump() if frame_result.telemetry else None,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def get_camera_status() -> dict:
    """Get camera streaming status."""
    bridge = server_state.airsim_bridge

    available_vehicles: list[str] = []
    if bridge and hasattr(bridge, "vehicle_names"):
        available_vehicles = list(bridge.vehicle_names)
    elif scenario_runner_state.runner and scenario_runner_state.runner.scenario:
        available_vehicles = [
            d.drone_id for d in scenario_runner_state.runner.scenario.drones
        ]

    return {
        "bridge_connected": bridge is not None and bridge.connected,
        "active_streams": list(server_state.camera_streaming.keys()),
        "stream_sequences": dict(server_state.camera_stream_sequence),
        "unreal_clients": unreal_manager.active_connections,
        "available_vehicles": available_vehicles,
    }


async def get_camera_summary() -> dict:
    """Return latest AirSim camera/depth capture metadata."""
    bridge = server_state.airsim_bridge
    return {
        "bridge_connected": bridge is not None and bridge.connected,
        "camera_name": bridge.config.camera_name if bridge else None,
        "last_depth_capture": server_state.last_depth_capture,
        "navigation_map": server_state.navigation_map,
    }


async def get_dock_status() -> dict:
    """Get dock station status."""
    dock = server_state.world_model.get_dock()
    if not dock:
        return {"status": "no_dock_configured"}

    return {
        "dock_id": "dock_main",
        "status": dock.status.value,
        "position": {
            "latitude": dock.position.latitude,
            "longitude": dock.position.longitude,
            "altitude_m": dock.position.altitude_msl,
        },
        "current_vehicle_id": dock.current_vehicle_id,
        "charge_rate_percent_per_minute": dock.charge_rate_percent_per_minute,
    }


async def update_dock_status(
    status: str,
    drone_id: str | None = None,
    charge_percent: float | None = None,
    landing_pad_active: bool = False,
    charging_animation: bool = False,
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Update dock station status and broadcast to Unreal."""
    dock = server_state.world_model.get_dock()
    if not dock:
        raise HTTPException(status_code=404, detail="No dock configured")

    try:
        new_status = DockStatus(status)
        server_state.world_model.set_dock(dock.position, new_status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid status. Must be one of: "
                f"{[s.value for s in DockStatus]}"
            ),
        ) from None

    await broadcast_dock_state(
        dock_id="dock_main",
        status=status,
        latitude=dock.position.latitude,
        longitude=dock.position.longitude,
        altitude_m=dock.position.altitude_msl,
        docked_drone_id=drone_id,
        charge_percent=charge_percent,
        landing_pad_active=landing_pad_active,
        charging_animation=charging_animation,
    )

    return {
        "dock_id": "dock_main",
        "status": status,
        "docked_drone_id": drone_id,
        "charge_percent": charge_percent,
        "broadcast_sent": unreal_manager.active_connections > 0,
    }


async def dock_approach(
    drone_id: str,
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Signal that a drone is approaching the dock."""
    dock = server_state.world_model.get_dock()

    if unreal_manager.active_connections > 0:
        await unreal_manager.broadcast({
            "type": UnrealMessageType.DOCK_APPROACH.value,
            "drone_id": drone_id,
            "dock_id": "dock_main",
            "timestamp_ms": time.time() * 1000,
        })

        if dock:
            await broadcast_dock_state(
                dock_id="dock_main",
                status="available",
                latitude=dock.position.latitude,
                longitude=dock.position.longitude,
                altitude_m=dock.position.altitude_msl,
                landing_pad_active=True,
            )

    return {
        "status": "approach_signaled",
        "drone_id": drone_id,
    }


async def update_battery(
    drone_id: str,
    percent: float,
    voltage: float = 22.2,
    current: float = -5.0,
    is_charging: bool = False,
    time_remaining_s: float | None = None,
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Update battery state and broadcast to Unreal."""
    await broadcast_battery_update(
        drone_id=drone_id,
        percent=percent,
        voltage=voltage,
        current=current,
        is_charging=is_charging,
        time_remaining_s=time_remaining_s,
    )

    return {
        "drone_id": drone_id,
        "percent": percent,
        "is_charging": is_charging,
        "broadcast_sent": unreal_manager.active_connections > 0,
    }


async def update_environment(
    hour: int = 12,
    rain: float = 0.0,
    fog: float = 0.0,
    dust: float = 0.0,
    snow: float = 0.0,
    wind_speed_ms: float = 3.0,
    wind_direction_deg: float = 180.0,
    visibility_m: float = 10000.0,
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Manually update environment and broadcast to AirSim and Unreal."""
    hour = max(0, min(23, hour))
    rain = max(0.0, min(1.0, rain))
    fog = max(0.0, min(1.0, fog))
    dust = max(0.0, min(1.0, dust))
    snow = max(0.0, min(1.0, snow))

    mapped = {
        "hour": hour,
        "is_daylight": 6 <= hour <= 20,
        "rain": rain,
        "fog": fog,
        "dust": dust,
        "snow": snow,
        "wind_speed_ms": wind_speed_ms,
        "wind_direction_deg": wind_direction_deg,
        "visibility_m": visibility_m,
    }

    server_state.airsim_env_last = mapped
    await _apply_airsim_environment(mapped)

    return {
        "status": "updated",
        "environment": mapped,
        "airsim_connected": _airsim_bridge_connected(),
        "unreal_clients": unreal_manager.active_connections,
    }


async def spawn_anomaly_marker(
    anomaly_id: str,
    asset_id: str,
    severity: float,
    latitude: float,
    longitude: float,
    altitude_m: float = 20.0,
    label: str = "",
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Spawn an anomaly marker in the Unreal scene."""
    await broadcast_spawn_anomaly_marker(
        anomaly_id=anomaly_id,
        asset_id=asset_id,
        severity=severity,
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude_m,
        label=label,
    )

    return {
        "status": "spawned",
        "anomaly_id": anomaly_id,
        "asset_id": asset_id,
        "severity": severity,
        "broadcast_sent": unreal_manager.active_connections > 0,
    }


async def clear_anomaly_markers(
    _auth: dict = Depends(auth_handler),
) -> dict:
    """Clear all anomaly markers from the Unreal scene."""
    if unreal_manager.active_connections > 0:
        await unreal_manager.broadcast({
            "type": UnrealMessageType.CLEAR_ANOMALY_MARKERS.value,
            "timestamp_ms": time.time() * 1000,
        })

    return {
        "status": "cleared",
        "broadcast_sent": unreal_manager.active_connections > 0,
    }


def register_airsim_routes(app: FastAPI) -> None:
    """Register AirSim, Unreal, and camera routes."""
    app.get("/api/airsim/status")(get_airsim_status)
    app.get("/api/airsim/environment")(get_airsim_environment)
    app.post("/api/airsim/scene/sync")(sync_airsim_scene)
    app.post("/api/unreal/sync-scene")(sync_unreal_scene)
    app.get("/api/unreal/scene-state")(get_unreal_scene_state)
    async def manual_fly_scenario() -> dict:
        """Manually trigger flying through all scenario assets."""
        if not scenario_run_state.running:
            raise HTTPException(status_code=400, detail="No scenario running")

        executor = server_state.airsim_action_executor
        if not executor:
            raise HTTPException(status_code=503, detail="AirSim executor not available")

        scenario = get_scenario(scenario_run_state.scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")

        async def fly_mission() -> None:
            try:
                logger.info("manual_fly_starting", asset_count=len(scenario.assets))

                # Sync assets to Unreal at start
                await broadcast_scenario_scene(scenario, include_defects=True)
                logger.info("manual_fly_scene_synced")

                # Battery simulation - start at 100%
                battery_percent = 100.0
                drain_per_minute = 2.0  # 2% per minute of flight

                async def update_battery(drain_amount: float) -> None:
                    nonlocal battery_percent
                    battery_percent = max(0, battery_percent - drain_amount)
                    await broadcast_battery_update(
                        drone_id="Drone1",
                        percent=battery_percent,
                        voltage=22.2 + (battery_percent / 100) * 3.0,
                        current=-5.0,
                        is_charging=False,
                    )

                # Takeoff
                if not executor.is_flying:
                    result = await executor.execute({
                        "action": "takeoff",
                        "parameters": {"altitude_agl": 30.0},
                        "reasoning": "Manual fly - takeoff",
                        "confidence": 1.0,
                    })
                    logger.info("manual_fly_takeoff_result", status=result.status.value)
                    if result.status.value != "completed":
                        logger.error("manual_fly_takeoff_failed", error=result.error)
                        return
                    await update_battery(1.0)  # Takeoff uses 1% battery
                    await asyncio.sleep(2.0)

                # Fly to each asset
                for i, asset in enumerate(scenario.assets):
                    logger.info(
                        f"manual_fly_asset_{i+1}",
                        asset_id=asset.asset_id,
                        lat=asset.latitude,
                        lon=asset.longitude,
                    )

                    # Update battery before each asset
                    await update_battery(drain_per_minute * 0.5)  # ~30s per asset

                    result = await executor.execute({
                        "action": "inspect_asset",
                        "target_asset": {
                            "asset_id": asset.asset_id,
                            "name": asset.name,
                            "latitude": asset.latitude,
                            "longitude": asset.longitude,
                            "inspection_altitude_agl": 30.0,
                            "orbit_radius_m": 20.0,
                            "dwell_time_s": 10.0,
                        },
                        "reasoning": f"Manual fly - asset {i+1}/{len(scenario.assets)}",
                        "confidence": 1.0,
                    })
                    logger.info(
                        f"manual_fly_asset_{i+1}_result",
                        status=result.status.value,
                        error=result.error,
                        details=result.details,
                    )

                # Return home and land
                logger.info("manual_fly_returning_home")
                await update_battery(drain_per_minute * 0.3)  # Return trip

                result = await executor.execute({
                    "action": "return_low_battery",
                    "reasoning": "Manual fly complete - returning to dock",
                    "confidence": 1.0,
                })
                logger.info("manual_fly_return_result", status=result.status.value)

                # Wait a moment then update battery to charging
                await asyncio.sleep(2.0)
                await broadcast_battery_update(
                    drone_id="Drone1",
                    percent=battery_percent,
                    voltage=22.2 + (battery_percent / 100) * 3.0,
                    current=2.0,  # Positive = charging
                    is_charging=True,
                )

                # Update dock status
                await broadcast_dock_state(
                    dock_id="dock_main",
                    status="charging",
                    latitude=scenario.drones[0].latitude if scenario.drones else 47.641468,
                    longitude=scenario.drones[0].longitude if scenario.drones else -122.140165,
                    altitude_m=0.0,
                    docked_drone_id="Drone1",
                    charge_percent=battery_percent,
                    charging_animation=True,
                )

                logger.info("manual_fly_complete", final_battery=battery_percent)

            except Exception as exc:
                logger.exception("manual_fly_error", error=str(exc))

        asyncio.create_task(fly_mission())
        return {
            "status": "started",
            "scenario_id": scenario_run_state.scenario_id,
            "asset_count": len(scenario.assets),
        }

    async def test_move_velocity() -> dict:
        """Test moving by velocity - simpler than position."""
        bridge = server_state.airsim_bridge
        if not bridge or not bridge.connected:
            raise HTTPException(status_code=503, detail="AirSim not connected")

        try:
            # Enable API control
            await asyncio.to_thread(
                bridge.client.enableApiControl, True, bridge.config.vehicle_name
            )

            # Move forward at 5 m/s for 3 seconds (non-blocking)
            logger.info("test_move_velocity: moving forward")
            bridge.client.moveByVelocityAsync(5, 0, 0, 3, vehicle_name=bridge.config.vehicle_name)

            # Wait for the movement to happen
            await asyncio.sleep(3.5)

            return {"status": "completed", "action": "test_move_velocity"}
        except Exception as exc:
            logger.exception("test_move_velocity_error", error=str(exc))
            return {"status": "failed", "error": str(exc)}

    async def debug_takeoff() -> dict:
        """Debug takeoff - test each step individually."""
        bridge = server_state.airsim_bridge
        if not bridge or not bridge.connected:
            raise HTTPException(status_code=503, detail="AirSim not connected")

        steps = []
        try:
            # Step 1: Enable API control
            await asyncio.to_thread(
                bridge.client.enableApiControl, True, bridge.config.vehicle_name
            )
            steps.append("enableApiControl: OK")

            # Step 2: Arm
            await asyncio.to_thread(
                bridge.client.armDisarm, True, bridge.config.vehicle_name
            )
            steps.append("armDisarm: OK")

            # Step 3: Takeoff (non-blocking)
            bridge.client.takeoffAsync(timeout_sec=30, vehicle_name=bridge.config.vehicle_name)
            steps.append("takeoffAsync: sent")

            # Step 4: Wait and check altitude
            await asyncio.sleep(3)
            state = bridge.client.getMultirotorState(vehicle_name=bridge.config.vehicle_name)
            alt = -state.kinematics_estimated.position.z_val
            steps.append(f"altitude after 3s: {alt:.2f}m")

            # Step 5: Move up with velocity if not rising
            if alt < 1.0:
                steps.append("trying moveByVelocityAsync to go up")
                bridge.client.moveByVelocityAsync(0, 0, -5, 5, vehicle_name=bridge.config.vehicle_name)
                await asyncio.sleep(5)
                state = bridge.client.getMultirotorState(vehicle_name=bridge.config.vehicle_name)
                alt = -state.kinematics_estimated.position.z_val
                steps.append(f"altitude after velocity: {alt:.2f}m")

            return {"status": "completed", "steps": steps, "final_altitude": alt}
        except Exception as exc:
            logger.exception("debug_takeoff_error", error=str(exc))
            steps.append(f"ERROR: {str(exc)}")
            return {"status": "failed", "steps": steps, "error": str(exc)}

    async def debug_fly_to(north: float = 50, east: float = 50, down: float = -30) -> dict:
        """Debug: fly to specific NED coordinates."""
        bridge = server_state.airsim_bridge
        if not bridge or not bridge.connected:
            raise HTTPException(status_code=503, detail="AirSim not connected")

        steps = []
        try:
            # Ensure connected
            await bridge.ensure_connected()
            steps.append("connected")

            # Get current position
            state = bridge.client.getMultirotorState(vehicle_name=bridge.config.vehicle_name)
            pos = state.kinematics_estimated.position
            steps.append(f"current pos: N={pos.x_val:.1f}, E={pos.y_val:.1f}, D={pos.z_val:.1f}")

            # Enable API control and arm
            bridge.client.enableApiControl(True, bridge.config.vehicle_name)
            bridge.client.armDisarm(True, bridge.config.vehicle_name)
            steps.append("armed")

            # Takeoff first if on ground
            if -pos.z_val < 1.0:
                bridge.client.takeoffAsync(timeout_sec=10, vehicle_name=bridge.config.vehicle_name)
                await asyncio.sleep(5)
                state = bridge.client.getMultirotorState(vehicle_name=bridge.config.vehicle_name)
                alt = -state.kinematics_estimated.position.z_val
                steps.append(f"takeoff complete, alt={alt:.1f}m")

            # Now move to target position
            steps.append(f"moving to N={north}, E={east}, D={down}")
            bridge.client.moveToPositionAsync(
                north, east, down, 5.0,
                timeout_sec=30,
                vehicle_name=bridge.config.vehicle_name
            )

            # Wait and check progress
            for i in range(6):
                await asyncio.sleep(2)
                state = bridge.client.getMultirotorState(vehicle_name=bridge.config.vehicle_name)
                pos = state.kinematics_estimated.position
                steps.append(f"t={i*2}s: N={pos.x_val:.1f}, E={pos.y_val:.1f}, D={pos.z_val:.1f}")

            return {"status": "completed", "steps": steps}
        except Exception as exc:
            logger.exception("debug_fly_to_error", error=str(exc))
            steps.append(f"ERROR: {str(exc)}")
            return {"status": "failed", "steps": steps, "error": str(exc)}

    async def reset_drone(altitude: float = 5.0) -> dict:
        """Reset drone position to fix stuck/underground states."""
        bridge = server_state.airsim_bridge
        if not bridge:
            raise HTTPException(status_code=503, detail="AirSim bridge not initialized")

        try:
            logger.info("reset_drone_requested", altitude=altitude)
            success = await bridge.reset_position(altitude_agl=altitude, reset_to_origin=True)

            if success:
                # Get new position to confirm
                await asyncio.sleep(0.5)
                state = bridge.client.getMultirotorState(vehicle_name=bridge.config.vehicle_name)
                pos = state.kinematics_estimated.position
                return {
                    "status": "success",
                    "message": f"Drone reset to altitude {altitude}m",
                    "position": {
                        "north": pos.x_val,
                        "east": pos.y_val,
                        "down": pos.z_val,
                        "altitude_agl": -pos.z_val,
                    }
                }
            else:
                return {"status": "failed", "message": "Reset failed - check logs"}
        except Exception as exc:
            logger.exception("reset_drone_error", error=str(exc))
            return {"status": "failed", "error": str(exc)}

    async def spawn_scene() -> dict:
        """Spawn dock and assets in AirSim for the current scenario."""
        bridge = server_state.airsim_bridge
        if not bridge:
            raise HTTPException(status_code=503, detail="AirSim bridge not initialized")

        geo_ref = server_state.airsim_geo_ref
        if not geo_ref:
            raise HTTPException(status_code=503, detail="Geo reference not initialized")

        scenario = None
        if scenario_run_state.running and scenario_run_state.scenario_id:
            scenario = get_scenario(scenario_run_state.scenario_id)

        if not scenario:
            raise HTTPException(status_code=400, detail="No scenario running")

        try:
            # Convert assets to dict format
            asset_dicts = [
                {
                    "latitude": a.latitude,
                    "longitude": a.longitude,
                    "name": a.name,
                    "asset_id": a.asset_id,
                    "asset_type": a.asset_type,
                }
                for a in scenario.assets
            ]

            results = await bridge.spawn_scene_objects(
                dock_ned=(0.0, 0.0, 0.0),
                assets=asset_dicts,
                geo_ref=geo_ref,
            )

            return {
                "status": "success",
                "dock_spawned": results.get("dock", False),
                "assets_spawned": len([a for a in results.get("assets", []) if a.get("success")]),
                "total_assets": len(scenario.assets),
                "details": results,
            }
        except Exception as exc:
            logger.exception("spawn_scene_error", error=str(exc))
            return {"status": "failed", "error": str(exc)}

    app.post("/api/airsim/spawn_scene")(spawn_scene)
    app.post("/api/airsim/reset")(reset_drone)
    app.post("/api/airsim/debug_fly_to")(debug_fly_to)
    app.post("/api/airsim/debug_takeoff")(debug_takeoff)
    app.post("/api/airsim/test_move")(test_move_velocity)
    app.post("/api/airsim/fly_scenario")(manual_fly_scenario)
    app.post("/api/airsim/start")(start_airsim)
    app.post("/api/airsim/reconnect")(reconnect_airsim)
    app.post("/api/airsim/flight/takeoff")(airsim_takeoff)
    app.post("/api/airsim/flight/land")(airsim_land)
    app.post("/api/airsim/flight/move")(airsim_move)
    app.post("/api/airsim/flight/hover")(airsim_hover)
    app.post("/api/airsim/flight/rtl")(airsim_rtl)
    app.post("/api/airsim/flight/orbit")(airsim_orbit)
    app.get("/api/airsim/flight/status")(airsim_flight_status)
    app.post("/api/camera/start_stream")(start_camera_stream)
    app.post("/api/camera/stop_stream")(stop_camera_stream)
    app.get("/api/camera/snapshot/{drone_id}")(get_camera_snapshot)
    app.get("/api/camera/status")(get_camera_status)
    app.get("/api/camera")(get_camera_summary)
    app.get("/api/dock/status")(get_dock_status)
    app.post("/api/dock/update")(update_dock_status)
    app.post("/api/dock/approach/{drone_id}")(dock_approach)
    app.post("/api/battery/update/{drone_id}")(update_battery)
    app.post("/api/environment/update")(update_environment)
    app.post("/api/anomaly/spawn")(spawn_anomaly_marker)
    app.post("/api/anomaly/clear")(clear_anomaly_markers)
