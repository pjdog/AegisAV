"""Scenario API routes and execution helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import structlog
from fastapi import FastAPI, HTTPException

from agent.edge_config import EdgeComputeProfile, default_edge_compute_config
from agent.server.airsim_settings import update_airsim_settings
from agent.server.airsim_support import (
    _airsim_bridge_connected,
    _apply_airsim_environment,
    _execute_airsim_action,
    _map_scenario_environment,
    _schedule_airsim_connect,
    _schedule_airsim_environment,
    _sync_airsim_scene,
    _update_airsim_georef_for_scenario,
    apply_navigation_map_to_executors,
    broadcast_battery_update,
    broadcast_dock_state,
    broadcast_scenario_scene,
    schedule_airsim_restart,
    seed_navigation_map_from_assets,
    sync_multi_drone_scenario,
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
)
from agent.server.unreal_stream import (
    UnrealMessageType,
    thinking_tracker,
    unreal_manager,
)
from agent.server.world_model import Anomaly, Asset, AssetType, DockStatus
from autonomy.vehicle_state import Position
from mapping.manifest import ManifestBuilder, SensorCalibration
from mapping.map_fusion import MapFusion, MapFusionConfig
from mapping.slam_runner import run as run_slam_runner
from mapping.splat_trainer import SplatTrainingConfig, train_splat

logger = structlog.get_logger(__name__)
log = logging.getLogger(__name__)


class ClearDirResult(TypedDict):
    """Result of clearing a mapping directory."""

    label: str
    path: str
    removed_entries: int
    skipped: bool
    reason: str | None
    errors: list[str]


class ResetMappingResult(TypedDict, total=False):
    """Result of resetting mapping state."""

    skipped: bool
    reason: str
    reset: bool
    dir_results: list[ClearDirResult]
    error: str


def _is_within_base_dir(path: Path, base_dir: Path) -> bool:
    """Check if a path is contained within a base directory.

    Args:
        path: The path to check.
        base_dir: The base directory that should contain the path.

    Returns:
        True if path is within base_dir, False otherwise.
    """
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def _clear_mapping_dir(path: Path, base_dir: Path, label: str) -> ClearDirResult:
    """Clear mapping artifacts in a directory, guarded by base_dir.

    Args:
        path: Directory path to clear.
        base_dir: Base directory for safety check (path must be within).
        label: Human-readable label for this directory.

    Returns:
        Result dict with removal statistics and any errors.
    """
    resolved = path
    if not resolved.is_absolute():
        resolved = (base_dir / resolved).resolve()
    else:
        resolved = resolved.resolve()

    errors: list[str] = []
    removed_entries = 0

    if not resolved.exists():
        return ClearDirResult(
            label=label,
            path=str(resolved),
            removed_entries=0,
            skipped=True,
            reason="missing",
            errors=[],
        )
    if not resolved.is_dir():
        return ClearDirResult(
            label=label,
            path=str(resolved),
            removed_entries=0,
            skipped=True,
            reason="not_dir",
            errors=[],
        )
    if not _is_within_base_dir(resolved, base_dir):
        return ClearDirResult(
            label=label,
            path=str(resolved),
            removed_entries=0,
            skipped=True,
            reason="outside_repo",
            errors=[],
        )

    for child in resolved.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed_entries += 1
        except Exception as exc:
            errors.append(f"{child.name}: {exc}")

    resolved.mkdir(parents=True, exist_ok=True)
    return ClearDirResult(
        label=label,
        path=str(resolved),
        removed_entries=removed_entries,
        skipped=False,
        reason=None,
        errors=errors,
    )


async def _reset_mapping_state(scenario_id: str) -> ResetMappingResult:
    """Clear in-memory and on-disk mapping state before a scenario run.

    Args:
        scenario_id: The scenario ID for logging context.

    Returns:
        Result dict indicating success/skip status and any directory operations.
    """
    config = get_config_manager().config
    mapping_cfg = config.mapping
    if not mapping_cfg.enabled:
        return ResetMappingResult(skipped=True, reason="mapping_disabled")
    if not getattr(mapping_cfg, "reset_on_scenario_start", True):
        return ResetMappingResult(skipped=True, reason="reset_disabled")

    map_service = server_state.map_update_service
    try:
        if map_service:
            await map_service.stop()
            map_service.reset_state()

        server_state.slam_status = None
        server_state.splat_artifacts = None
        server_state.slam_pose_graph_summary = None
        server_state.map_update_last_error = None
        server_state.map_update_last_error_at = None
        server_state.fused_map_artifact = None
        server_state.navigation_map = None
        server_state.last_valid_navigation_map = None
        server_state.last_depth_capture = None
        server_state.last_vision_observation = None
        server_state.map_gate_history = []
        server_state.preflight_status = None
        if server_state.preflight_task and not server_state.preflight_task.done():
            server_state.preflight_task.cancel()
        server_state.preflight_task = None

        if server_state.airsim_action_executor:
            server_state.airsim_action_executor.set_avoid_zones([])

        base_dir = Path(__file__).resolve().parents[2]
        dir_results = [
            _clear_mapping_dir(Path(mapping_cfg.slam_dir), base_dir, "slam_runs"),
            _clear_mapping_dir(Path(mapping_cfg.splat_dir), base_dir, "splat_scenes"),
            _clear_mapping_dir(Path(mapping_cfg.fused_map_dir), base_dir, "fused_maps"),
        ]

        if map_service and mapping_cfg.enabled:
            await map_service.start()

        if mapping_cfg.preflight_enabled:
            await _broadcast_preflight_status(
                scenario_id,
                "pending",
                "Preflight mapping pending.",
                details={
                    "mapping_status": "pending",
                    "slam_status": "pending",
                },
            )

        logger.info(
            "mapping_state_reset",
            scenario_id=scenario_id,
            dir_results=dir_results,
        )
        return ResetMappingResult(reset=True, dir_results=dir_results)
    except Exception as exc:
        if map_service and mapping_cfg.enabled:
            try:
                await map_service.start()
            except Exception as restart_exc:
                logger.warning(
                    "map_update_restart_failed",
                    scenario_id=scenario_id,
                    error=str(restart_exc),
                )
        logger.warning(
            "mapping_state_reset_failed",
            scenario_id=scenario_id,
            error=str(exc),
        )
        return ResetMappingResult(reset=False, error=str(exc))


async def _broadcast_preflight_status(
    scenario_id: str,
    status: str,
    message: str,
    severity: EventSeverity = EventSeverity.INFO,
    details: dict | None = None,
) -> None:
    payload = {
        "type": "preflight_status",
        "scenario_id": scenario_id,
        "status": status,
        "message": message,
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
    }

    # Update server_state for dashboard polling
    server_state.preflight_status = {
        "scenario_id": scenario_id,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        **(details or {}),
    }

    try:
        await unreal_manager.broadcast(payload)
    except Exception as exc:
        logger.warning("preflight_status_unreal_failed", error=str(exc))

    await connection_manager.broadcast(
        Event(
            event_type=EventType.PREFLIGHT_STATUS,
            timestamp=datetime.now(),
            data=payload,
            severity=severity,
        )
    )
    log.info("Preflight %s: %s", status, message)


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
                "agent_label": decision.get("agent_label") or "Drone AG",
                "decision_id": decision.get("decision_id"),
                "action": action,
                "confidence": confidence,
                "reasoning": reason,
                "risk_level": decision.get("risk_level"),
                "risk_score": decision.get("risk_score"),
                "battery_percent": decision.get("battery_percent"),
                "drone_id": drone_id,
                "drone_name": decision.get("drone_name"),
                "elapsed_s": decision.get("elapsed_s"),
                "reasoning_context": decision.get("reasoning_context"),
                "alternatives": decision.get("alternatives"),
                "critic_validation": decision.get("critic_validation"),
                "target_asset": decision.get("target_asset"),
            },
            severity=EventSeverity.INFO,
        )
    )

    # Also broadcast to Unreal clients (overlay connects to /ws/unreal, not /ws)
    await unreal_manager.broadcast({
        "type": "server_decision",
        "drone_id": drone_id,
        "action": action,
        "confidence": confidence,
        "reasoning": reason,
        "risk_level": decision.get("risk_level"),
        "risk_score": decision.get("risk_score"),
        "battery_percent": decision.get("battery_percent"),
    })

    executor = server_state.airsim_action_executor

    def _fleet_state_value() -> str | None:
        fleet_bridge = server_state.fleet_bridge
        state = getattr(fleet_bridge, "state", None) if fleet_bridge else None
        if hasattr(state, "value"):
            return str(state.value)
        if isinstance(state, str):
            return state
        return str(state) if state is not None else None

    def _fleet_ready() -> bool:
        if not server_state.fleet_bridge_enabled or not server_state.fleet_bridge:
            return False
        state_value = _fleet_state_value()
        return bool(state_value and state_value.lower() == "ready")

    def _scenario_is_multi() -> bool:
        runner = scenario_runner_state.runner
        scenario = getattr(getattr(runner, "run_state", None), "scenario", None)
        if scenario:
            return len(scenario.drones) > 1
        return False

    async def _wait_for_fleet_ready(
        timeout_s: float = 20.0,
        poll_s: float = 0.5,
    ) -> bool:
        deadline = time.monotonic() + max(0.5, timeout_s)
        while time.monotonic() < deadline:
            if _fleet_ready():
                return True
            await asyncio.sleep(poll_s)
        return False

    fleet_ready = _fleet_ready()
    multi_expected = _scenario_is_multi()
    if action.lower() not in ("none", "wait"):
        exec_decision = dict(decision)
        exec_decision["drone_id"] = drone_id

        if multi_expected and not fleet_ready:

            async def _deferred_exec() -> None:
                ready = await _wait_for_fleet_ready()
                if not ready:
                    logger.warning(
                        "airsim_action_skipped_fleet_not_ready",
                        action=action,
                        drone_id=drone_id,
                    )
                    return
                await _execute_airsim_action(exec_decision)

            asyncio.create_task(_deferred_exec())
            return

        if fleet_ready:
            try:
                asyncio.create_task(_execute_airsim_action(exec_decision))
            except Exception as exc:
                logger.warning(
                    "airsim_action_schedule_failed",
                    action=action,
                    error=str(exc),
                )
            return

        if not executor:
            config = get_config_manager().config
            mapping_cfg = config.mapping
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


def _assign_assets_round_robin(assets: list, drone_ids: list[str]) -> dict[str, list]:
    assignments = {drone_id: [] for drone_id in drone_ids}
    if not drone_ids:
        return assignments
    for idx, asset in enumerate(assets):
        assignments[drone_ids[idx % len(drone_ids)]].append(asset)
    return assignments


def _collect_preflight_path_points(
    assets: list, geo_ref, altitude_agl: float, start_n: float, start_e: float
) -> list[tuple[float, float]]:
    path_points = [(start_n, start_e)]
    for asset in assets:
        n, e, _ = geo_ref.gps_to_ned(
            asset.latitude,
            asset.longitude,
            geo_ref.altitude + altitude_agl,
        )
        path_points.append((n, e))
    path_points.append((0.0, 0.0))
    return path_points


def _merge_preflight_sessions(
    combined_dir: Path,
    session_dirs: list[Path],
    drone_ids: list[str],
) -> int:
    frames_dir = combined_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for drone_id, session_dir in zip(drone_ids, session_dirs, strict=False):
        source_frames = session_dir / "frames"
        if not source_frames.exists():
            continue
        for json_path in source_frames.glob("*.json"):
            stem = f"{drone_id}_{json_path.stem}"
            (frames_dir / f"{stem}.json").write_text(json_path.read_text())
            for ext in (".png", ".jpg", ".jpeg"):
                src = source_frames / f"{json_path.stem}{ext}"
                if src.exists():
                    shutil.copy2(src, frames_dir / f"{stem}{ext}")
                    break
            for suffix in ("_depth.npy", "_depth.png", "_depth.exr"):
                src = source_frames / f"{json_path.stem}{suffix}"
                if src.exists():
                    shutil.copy2(src, frames_dir / f"{stem}{suffix}")
            copied += 1

    return copied


async def _train_splat_from_pose_graph(
    pose_graph_path: Path,
    mapping_cfg,
    scenario_id: str,
    run_id: str,
) -> None:
    if not mapping_cfg.splat_auto_train:
        return
    if not pose_graph_path.exists():
        logger.warning("splat_auto_train_missing_pose_graph", path=str(pose_graph_path))
        return

    backend = getattr(mapping_cfg, "splat_backend", "stub") or "stub"
    iterations = int(getattr(mapping_cfg, "splat_iterations", 7000))
    config = SplatTrainingConfig(
        backend=backend,
        iterations=iterations,
        max_points=int(mapping_cfg.max_points),
        min_points=int(mapping_cfg.min_points),
    )
    try:
        result = await asyncio.to_thread(
            train_splat,
            pose_graph_path=pose_graph_path,
            run_id=run_id,
            scenario_id=scenario_id,
            config=config,
        )
    except Exception as exc:
        logger.warning("splat_auto_train_failed", error=str(exc))
        return

    if not result.success or not result.scene_path:
        logger.warning(
            "splat_auto_train_failed",
            error=result.error_message or "unknown_error",
        )
        return

    try:
        scene_data = json.loads(result.scene_path.read_text())
        scene_data["scene_path"] = str(result.scene_path)
        server_state.splat_artifacts = scene_data
        logger.info(
            "splat_auto_train_complete",
            backend=backend,
            run_id=run_id,
            scene_path=str(result.scene_path),
        )
    except Exception as exc:
        logger.warning("splat_auto_train_scene_parse_failed", error=str(exc))


async def _run_preflight_slam_mapping(scenario, executor) -> None:
    """Capture a short SLAM mapping pass between dock and first target."""
    config = get_config_manager().config
    mapping_cfg = config.mapping
    if not mapping_cfg.enabled or not mapping_cfg.preflight_enabled:
        return
    if not scenario.assets:
        return

    if server_state.fleet_bridge_enabled and server_state.fleet_bridge and len(scenario.drones) > 1:
        await _run_preflight_slam_mapping_multi(scenario)
        return

    bridge = server_state.airsim_bridge
    if not bridge or not getattr(bridge, "connected", False):
        logger.info("preflight_mapping_skipped", reason="airsim_not_connected")
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "skipped",
            "Preflight mapping skipped (AirSim not connected).",
            severity=EventSeverity.WARNING,
        )
        return

    geo_ref = getattr(executor, "geo_ref", None) or server_state.airsim_geo_ref
    if not geo_ref:
        logger.info("preflight_mapping_skipped", reason="missing_geo_ref")
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "skipped",
            "Preflight mapping skipped (missing geo reference).",
            severity=EventSeverity.WARNING,
        )
        return

    altitude_agl = float(mapping_cfg.preflight_altitude_agl)
    timeout_s = float(getattr(mapping_cfg, "preflight_timeout_s", 180.0))
    move_timeout_s = float(getattr(mapping_cfg, "preflight_move_timeout_s", 30.0))
    wait_timeout_s = move_timeout_s + 10.0
    retry_count = max(0, int(getattr(mapping_cfg, "preflight_retry_count", 0)))
    retry_delay_s = float(getattr(mapping_cfg, "preflight_retry_delay_s", 5.0))

    start_pos = await bridge.get_position()
    if start_pos:
        start_n = float(start_pos.x_val)
        start_e = float(start_pos.y_val)
    else:
        start_n, start_e = 0.0, 0.0

    path_points = [(start_n, start_e)]
    for asset in scenario.assets:
        n, e, _ = geo_ref.gps_to_ned(
            asset.latitude,
            asset.longitude,
            geo_ref.altitude + altitude_agl,
        )
        path_points.append((n, e))

    path_points.append((0.0, 0.0))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("data/maps") / f"sequence_{scenario.scenario_id}_preflight_{stamp}"
    output_dir = Path(mapping_cfg.slam_dir) / f"run_{scenario.scenario_id}_preflight_{stamp}"

    dataset_id = f"{scenario.scenario_id}_preflight"
    sequence_id = f"{scenario.scenario_id}_preflight_{stamp}"
    manifest_builder = ManifestBuilder(
        dataset_id=dataset_id,
        sequence_id=sequence_id,
        output_dir=session_dir,
        sensor_type="airsim",
    )
    manifest_builder.set_origin(geo_ref.latitude, geo_ref.longitude, geo_ref.altitude)
    manifest_builder.manifest.metadata.update({
        "scenario_id": scenario.scenario_id,
        "capture_type": "preflight",
    })
    calibration_set = False

    def _quat_to_euler_deg(
        qw: float, qx: float, qy: float, qz: float
    ) -> tuple[float, float, float]:
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    def _rel_path(path_value: str | None) -> str | None:
        if not path_value:
            return None
        try:
            return str(Path(path_value).resolve().relative_to(session_dir.resolve()))
        except Exception:
            return str(Path(path_value).name)

    def _timestamp_s(metadata: dict) -> float:
        timestamp_ns = metadata.get("timestamp_ns")
        if timestamp_ns:
            return float(timestamp_ns) / 1e9
        timestamp_str = metadata.get("timestamp")
        if timestamp_str:
            try:
                return datetime.fromisoformat(timestamp_str).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    # Estimate total captures based on path segments and step size
    step_m = max(1.0, float(mapping_cfg.preflight_step_m))
    estimated_captures = 0
    for seg_idx in range(len(path_points) - 1):
        seg_start = path_points[seg_idx]
        seg_end = path_points[seg_idx + 1]
        distance = math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        estimated_captures += max(1, int(math.ceil(distance / step_m))) + 1

    capture_count = [0]  # Mutable for inner function access

    logger.info(
        "preflight_mapping_start",
        scenario_id=scenario.scenario_id,
        target_id=scenario.assets[0].asset_id,
        steps=len(path_points),
        estimated_captures=estimated_captures,
    )
    await _broadcast_preflight_status(
        scenario.scenario_id,
        "started",
        f"Preflight mapping started ({len(path_points)} path points).",
        details={
            "steps": len(path_points),
            "total_captures": estimated_captures,
            "capture_count": 0,
            "mapping_status": "started",
            "slam_status": "pending",
        },
    )

    move_with_avoidance = getattr(bridge, "move_to_position_with_obstacle_avoidance", None)
    recovery_timeout_s = float(getattr(mapping_cfg, "preflight_recovery_timeout_s", 60.0))

    async def _recover_preflight(reason: str, error: str | None = None) -> None:
        """Attempt safe recovery after preflight mapping failure.

        Recovery steps:
        1. Broadcast recovery status
        2. Try to move back to home position (with timeout)
        3. Fall back to hover if move fails
        4. Report final recovery status
        """
        recovery_success = False
        try:
            error_snippet = ""
            if error:
                trimmed = str(error).strip().replace("\n", " ")
                error_snippet = f": {trimmed[:120]}"
            await _broadcast_preflight_status(
                scenario.scenario_id,
                "recovering",
                f"Preflight recovery initiated ({reason}{error_snippet}).",
                severity=EventSeverity.WARNING,
                details={
                    "reason": reason,
                    "error": str(error) if error else None,
                    "capture_count": capture_count[0],
                    "mapping_status": "recovering",
                    "slam_status": "failed",
                },
            )

            # Try to move back to home with timeout
            try:
                await asyncio.wait_for(
                    bridge.move_to_position(0.0, 0.0, -altitude_agl, velocity=2.0),
                    timeout=recovery_timeout_s,
                )
                logger.info(
                    "preflight_recovery_move_complete",
                    scenario_id=scenario.scenario_id,
                    reason=reason,
                )
                recovery_success = True
            except asyncio.TimeoutError:
                logger.warning(
                    "preflight_recovery_move_timeout",
                    scenario_id=scenario.scenario_id,
                    timeout_s=recovery_timeout_s,
                )
                # Fallback: just try to hover in place
                if executor and executor.is_flying:
                    try:
                        await bridge.hover()
                        logger.info(
                            "preflight_recovery_hover_fallback", scenario_id=scenario.scenario_id
                        )
                        recovery_success = True
                    except Exception as hover_exc:
                        logger.error("preflight_recovery_hover_failed", error=str(hover_exc))

            # Ensure we're hovering at the end
            if executor and executor.is_flying:
                await bridge.hover()

            # Broadcast recovery complete
            if recovery_success:
                await _broadcast_preflight_status(
                    scenario.scenario_id,
                    "idle",
                    "Recovery complete. Drone returned to safe position.",
                    severity=EventSeverity.INFO,
                    details={
                        "reason": reason,
                        "capture_count": capture_count[0],
                        "mapping_status": "idle",
                        "slam_status": "failed",
                    },
                )

        except Exception as exc:
            logger.warning(
                "preflight_recovery_failed",
                scenario_id=scenario.scenario_id,
                error=str(exc),
            )
            await _broadcast_preflight_status(
                scenario.scenario_id,
                "recovery_failed",
                f"Recovery failed: {str(exc)[:50]}",
                severity=EventSeverity.ERROR,
                details={
                    "error": str(exc),
                    "reason": reason,
                    "mapping_status": "recovery_failed",
                },
            )

    async def _perform_mapping() -> None:
        nonlocal calibration_set
        move_failures = 0
        capture_failures = 0
        max_move_failures = max(1, int(getattr(mapping_cfg, "preflight_max_move_failures", 3)))
        max_capture_failures = max(
            1, int(getattr(mapping_cfg, "preflight_max_capture_failures", 5))
        )
        if executor and not executor.is_flying:
            await executor.execute({
                "action": "takeoff",
                "parameters": {"altitude_agl": altitude_agl},
                "reasoning": f"Preflight mapping takeoff for {scenario.scenario_id}",
                "confidence": 1.0,
            })

        for seg_idx in range(len(path_points) - 1):
            seg_start = path_points[seg_idx]
            seg_end = path_points[seg_idx + 1]
            distance = math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
            step_m = max(1.0, float(mapping_cfg.preflight_step_m))
            steps = max(1, int(math.ceil(distance / step_m)))

            for idx in range(steps + 1):
                frac = idx / steps
                n = seg_start[0] + (seg_end[0] - seg_start[0]) * frac
                e = seg_start[1] + (seg_end[1] - seg_start[1]) * frac
                try:
                    if move_with_avoidance:
                        move_ok = await asyncio.wait_for(
                            move_with_avoidance(
                                n,
                                e,
                                -altitude_agl,
                                velocity=float(mapping_cfg.preflight_velocity_ms),
                            ),
                            timeout=wait_timeout_s,
                        )
                    else:
                        move_ok = await asyncio.wait_for(
                            bridge.move_to_position(
                                n,
                                e,
                                -altitude_agl,
                                velocity=float(mapping_cfg.preflight_velocity_ms),
                                timeout=move_timeout_s,
                            ),
                            timeout=wait_timeout_s,
                        )

                    if not move_ok:
                        raise RuntimeError(
                            "move_to_position returned False (check AirSim connection or limits)"
                        )
                except asyncio.TimeoutError:
                    move_failures += 1
                    error_msg = f"Move timed out after {wait_timeout_s:.0f}s"
                    logger.warning(
                        "preflight_move_failed",
                        scenario_id=scenario.scenario_id,
                        error=error_msg,
                        error_type="TimeoutError",
                        target_n=n,
                        target_e=e,
                        target_altitude_agl=altitude_agl,
                        timeout_s=move_timeout_s,
                        failures=move_failures,
                    )
                    await _broadcast_preflight_status(
                        scenario.scenario_id,
                        "capturing",
                        f"Move failed ({move_failures}/{max_move_failures}). Continuing.",
                        severity=EventSeverity.WARNING,
                        details={
                            "capture_count": capture_count[0],
                            "total_captures": estimated_captures,
                            "mapping_status": "capturing",
                            "slam_status": "pending",
                            "move_failures": move_failures,
                            "last_move_error": error_msg,
                            "last_move_target": {
                                "north_m": n,
                                "east_m": e,
                                "altitude_agl_m": altitude_agl,
                            },
                        },
                    )
                    if move_failures >= max_move_failures:
                        logger.warning(
                            "preflight_move_failures_exceeded",
                            scenario_id=scenario.scenario_id,
                            failures=move_failures,
                        )
                        break
                    continue
                except Exception as exc:
                    move_failures += 1
                    logger.warning(
                        "preflight_move_failed",
                        scenario_id=scenario.scenario_id,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        target_n=n,
                        target_e=e,
                        target_altitude_agl=altitude_agl,
                        timeout_s=move_timeout_s,
                        failures=move_failures,
                    )
                    await _broadcast_preflight_status(
                        scenario.scenario_id,
                        "capturing",
                        f"Move failed ({move_failures}/{max_move_failures}). Continuing.",
                        severity=EventSeverity.WARNING,
                        details={
                            "capture_count": capture_count[0],
                            "total_captures": estimated_captures,
                            "mapping_status": "capturing",
                            "slam_status": "pending",
                            "move_failures": move_failures,
                            "last_move_error": str(exc),
                            "last_move_target": {
                                "north_m": n,
                                "east_m": e,
                                "altitude_agl_m": altitude_agl,
                            },
                        },
                    )
                    if move_failures >= max_move_failures:
                        logger.warning(
                            "preflight_move_failures_exceeded",
                            scenario_id=scenario.scenario_id,
                            failures=move_failures,
                        )
                        break
                    continue

                try:
                    capture = await bridge.capture_mapping_bundle(
                        output_dir=session_dir,
                        include_depth=True,
                        include_imu=True,
                    )
                except Exception as exc:
                    capture = {"success": False, "error": str(exc)}
                    logger.warning(
                        "preflight_capture_failed",
                        scenario_id=scenario.scenario_id,
                        error=str(exc),
                    )
                if not capture.get("success"):
                    # Log why capture failed if not an exception
                    error_reason = capture.get("error", "unknown")
                    logger.debug(
                        "preflight_capture_not_successful",
                        scenario_id=scenario.scenario_id,
                        error=error_reason,
                    )
                if capture.get("success"):
                    telemetry = capture.get("telemetry") or {}
                    pose = telemetry.get("pose") if isinstance(telemetry, dict) else None
                    position = None
                    orientation = None
                    if pose:
                        pos_data = pose.get("position") or {}
                        ori = pose.get("orientation") or {}
                        position = [
                            float(pos_data.get("x", 0.0)),
                            float(pos_data.get("y", 0.0)),
                            float(pos_data.get("z", 0.0)),
                        ]
                        orientation = list(
                            _quat_to_euler_deg(
                                float(ori.get("w", 1.0)),
                                float(ori.get("x", 0.0)),
                                float(ori.get("y", 0.0)),
                                float(ori.get("z", 0.0)),
                            )
                        )

                    if not calibration_set:
                        cam = capture.get("camera") or {}
                        intrinsics = cam.get("intrinsics") or {}
                        resolution = cam.get("resolution") or [0, 0]
                        calibration = SensorCalibration(
                            fx=float(intrinsics.get("fx", 0.0)),
                            fy=float(intrinsics.get("fy", 0.0)),
                            cx=float(intrinsics.get("cx", 0.0)),
                            cy=float(intrinsics.get("cy", 0.0)),
                            width=int(
                                intrinsics.get("width")
                                or (resolution[0] if len(resolution) > 0 else 0)
                            ),
                            height=int(
                                intrinsics.get("height")
                                or (resolution[1] if len(resolution) > 1 else 0)
                            ),
                        )
                        manifest_builder.set_calibration(calibration)
                        calibration_set = True

                    manifest_builder.add_frame(
                        timestamp_s=_timestamp_s(capture),
                        image_path=_rel_path(capture.get("files", {}).get("rgb")) or "",
                        depth_path=_rel_path(capture.get("files", {}).get("depth")),
                        position=position,
                        orientation=orientation,
                    )
                    capture_count[0] += 1
                    # Broadcast capture progress every 5 captures to reduce overhead
                    if capture_count[0] % 5 == 0 or capture_count[0] == estimated_captures:
                        await _broadcast_preflight_status(
                            scenario.scenario_id,
                            "capturing",
                            f"Captured {capture_count[0]}/{estimated_captures} frames.",
                            details={
                                "capture_count": capture_count[0],
                                "total_captures": estimated_captures,
                                "mapping_status": "capturing",
                                "slam_status": "pending",
                            },
                        )
                else:
                    capture_failures += 1
                    await _broadcast_preflight_status(
                        scenario.scenario_id,
                        "capturing",
                        f"Capture failed ({capture_failures}/{max_capture_failures}). Continuing.",
                        severity=EventSeverity.WARNING,
                        details={
                            "capture_count": capture_count[0],
                            "total_captures": estimated_captures,
                            "mapping_status": "capturing",
                            "slam_status": "pending",
                            "capture_failures": capture_failures,
                        },
                    )
                    if capture_failures >= max_capture_failures:
                        logger.warning(
                            "preflight_capture_failures_exceeded",
                            scenario_id=scenario.scenario_id,
                            failures=capture_failures,
                        )
                        break
                await asyncio.sleep(float(mapping_cfg.preflight_capture_interval_s))

            if move_failures >= max_move_failures or capture_failures >= max_capture_failures:
                break

        # Broadcast SLAM starting
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "processing",
            f"Running SLAM on {capture_count[0]} frames...",
            details={
                "capture_count": capture_count[0],
                "total_captures": estimated_captures,
                "mapping_status": "processing",
                "slam_status": "running",
            },
        )

        backend = getattr(mapping_cfg, "slam_backend", "telemetry") or "telemetry"
        args = SimpleNamespace(
            input_dir=str(session_dir),
            output_dir=str(output_dir),
            backend=backend,
            allow_telemetry_fallback=bool(getattr(mapping_cfg, "slam_allow_fallback", True)),
            no_pointcloud=False,
            max_points=mapping_cfg.max_points,
            depth_subsample=6,
            min_time_interval_s=None,
            max_time_interval_s=None,
            min_translation_m=None,
            min_rotation_deg=None,
            velocity_threshold_ms=None,
            blur_threshold=None,
            min_feature_count=None,
            overlap_target=None,
        )
        if capture_count[0] == 0:
            # No captures collected - skip SLAM and continue gracefully
            logger.warning(
                "preflight_mapping_no_captures",
                scenario_id=scenario.scenario_id,
                message="No captures collected - check AirSim camera connection",
            )
            await _broadcast_preflight_status(
                scenario.scenario_id,
                "complete",
                "Preflight completed without map (no captures collected).",
                severity=EventSeverity.WARNING,
                details={
                    "capture_count": 0,
                    "total_captures": estimated_captures,
                    "mapping_status": "skipped",
                    "slam_status": "skipped",
                    "quality": 0,
                },
            )
            return  # Exit gracefully without raising error

        slam_result = await asyncio.to_thread(run_slam_runner, args)
        map_points = output_dir / "map_points.ply"
        if slam_result != 0 and not map_points.exists():
            raise RuntimeError(f"SLAM runner failed (backend={backend}, code={slam_result})")

        # Broadcast SLAM complete, starting fusion
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "processing",
            "SLAM complete. Fusing navigation map...",
            details={
                "capture_count": capture_count[0],
                "total_captures": estimated_captures,
                "mapping_status": "fusing",
                "slam_status": "complete",
            },
        )

        map_points = output_dir / "map_points.ply"
        if map_points.exists():
            fusion = MapFusion(
                MapFusionConfig(
                    resolution_m=mapping_cfg.map_resolution_m,
                    tile_size_cells=getattr(mapping_cfg, "tile_size_cells", 120),
                    voxel_size_m=mapping_cfg.voxel_size_m,
                    max_points=mapping_cfg.max_points,
                    min_points=mapping_cfg.min_points,
                )
            )
            fused = fusion.build_navigation_map(
                point_cloud_path=map_points,
                map_id=output_dir.name,
                scenario_id=scenario.scenario_id,
                source="slam_preflight",
            )
            nav_map = fused.navigation_map
            metadata = nav_map.get("metadata", {})
            metadata["point_cloud_path"] = str(map_points)
            nav_map["metadata"] = metadata
            server_state.navigation_map = nav_map
            apply_navigation_map_to_executors(nav_map)

        pose_graph_path = output_dir / "pose_graph.json"
        await _train_splat_from_pose_graph(
            pose_graph_path,
            mapping_cfg,
            scenario.scenario_id,
            output_dir.name,
        )

    try:
        attempts = retry_count + 1
        for attempt in range(1, attempts + 1):
            try:
                await asyncio.wait_for(_perform_mapping(), timeout=timeout_s)
                logger.info("preflight_mapping_complete", scenario_id=scenario.scenario_id)
                # Extract quality from navigation map if available
                quality = None
                obstacle_count = 0
                if server_state.navigation_map:
                    meta = server_state.navigation_map.get("metadata", {})
                    quality = meta.get("map_quality_score")
                    obstacle_count = len(server_state.navigation_map.get("obstacles", []))
                await _broadcast_preflight_status(
                    scenario.scenario_id,
                    "complete",
                    "Preflight mapping completed successfully.",
                    details={
                        "attempt": attempt,
                        "capture_count": capture_count[0],
                        "total_captures": estimated_captures,
                        "mapping_status": "complete",
                        "slam_status": "complete",
                        "quality": quality,
                        "obstacle_count": obstacle_count,
                    },
                )
                return
            except asyncio.TimeoutError:
                logger.warning("preflight_mapping_timeout", scenario_id=scenario.scenario_id)
                await _broadcast_preflight_status(
                    scenario.scenario_id,
                    "timeout",
                    "Preflight mapping timed out.",
                    severity=EventSeverity.WARNING,
                    details={
                        "attempt": attempt,
                        "timeout_s": timeout_s,
                        "capture_count": capture_count[0],
                        "total_captures": estimated_captures,
                        "mapping_status": "timeout",
                        "slam_status": "failed",
                    },
                )
                await _recover_preflight("timeout")
            except Exception as exc:
                logger.warning(
                    "preflight_mapping_failed",
                    scenario_id=scenario.scenario_id,
                    error=str(exc),
                )
                await _broadcast_preflight_status(
                    scenario.scenario_id,
                    "failed",
                    f"Preflight mapping failed: {str(exc)[:100]}",
                    severity=EventSeverity.ERROR,
                    details={
                        "attempt": attempt,
                        "error": str(exc),
                        "capture_count": capture_count[0],
                        "total_captures": estimated_captures,
                        "mapping_status": "failed",
                        "slam_status": "failed",
                    },
                )
                await _recover_preflight("failure", error=str(exc))

            if attempt < attempts:
                await _broadcast_preflight_status(
                    scenario.scenario_id,
                    "retrying",
                    f"Retrying preflight mapping (attempt {attempt + 1}/{attempts}).",
                    severity=EventSeverity.WARNING,
                    details={
                        "attempt": attempt + 1,
                        "capture_count": 0,
                        "total_captures": estimated_captures,
                        "mapping_status": "retrying",
                        "slam_status": "pending",
                    },
                )
                await asyncio.sleep(retry_delay_s)
    finally:
        if manifest_builder.manifest.frames:
            manifest_builder.save()


async def _run_preflight_slam_mapping_multi(scenario) -> None:
    """Run preflight mapping using all available drones and merge captures."""
    config = get_config_manager().config
    mapping_cfg = config.mapping
    if not mapping_cfg.enabled or not mapping_cfg.preflight_enabled:
        return
    if not scenario.assets:
        return

    fleet_bridge = server_state.fleet_bridge
    if not fleet_bridge:
        logger.info("preflight_mapping_skipped", reason="fleet_bridge_missing")
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "skipped",
            "Preflight mapping skipped (fleet bridge unavailable).",
            severity=EventSeverity.WARNING,
        )
        return

    geo_ref = server_state.airsim_geo_ref
    if not geo_ref:
        logger.info("preflight_mapping_skipped", reason="missing_geo_ref")
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "skipped",
            "Preflight mapping skipped (missing geo reference).",
            severity=EventSeverity.WARNING,
        )
        return

    drone_entries: list[tuple[str, object, object]] = []
    for drone in scenario.drones:
        bridge = fleet_bridge.get_bridge(drone.drone_id)
        executor = fleet_bridge.get_executor(drone.drone_id)
        if not bridge or not executor or not getattr(bridge, "connected", False):
            logger.warning(
                "preflight_mapping_drone_unavailable",
                scenario_id=scenario.scenario_id,
                drone_id=drone.drone_id,
            )
            continue
        drone_entries.append((drone.drone_id, bridge, executor))

    if not drone_entries:
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "skipped",
            "Preflight mapping skipped (no drones available).",
            severity=EventSeverity.WARNING,
        )
        return

    drone_ids = [drone_id for drone_id, _, _ in drone_entries]
    assignments = _assign_assets_round_robin(scenario.assets, drone_ids)

    altitude_agl = float(mapping_cfg.preflight_altitude_agl)
    timeout_s = float(getattr(mapping_cfg, "preflight_timeout_s", 180.0))
    move_timeout_s = float(getattr(mapping_cfg, "preflight_move_timeout_s", 30.0))
    wait_timeout_s = move_timeout_s + 10.0
    step_m = max(1.0, float(mapping_cfg.preflight_step_m))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    await _broadcast_preflight_status(
        scenario.scenario_id,
        "started",
        f"Preflight mapping started (multi-drone: {len(drone_entries)}).",
        details={
            "drone_count": len(drone_entries),
            "asset_count": len(scenario.assets),
            "mapping_status": "started",
            "slam_status": "pending",
        },
    )

    async def _capture_for_drone(
        drone_id: str,
        bridge,
        executor,
        assets: list,
    ) -> dict[str, object]:
        start_pos = await bridge.get_position()
        if start_pos:
            start_n = float(start_pos.x_val)
            start_e = float(start_pos.y_val)
        else:
            start_n, start_e = 0.0, 0.0

        path_points = _collect_preflight_path_points(
            assets, geo_ref, altitude_agl, start_n, start_e
        )
        estimated_captures = 0
        for seg_idx in range(len(path_points) - 1):
            seg_start = path_points[seg_idx]
            seg_end = path_points[seg_idx + 1]
            distance = math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
            estimated_captures += max(1, int(math.ceil(distance / step_m))) + 1

        session_dir = (
            Path("data/maps") / f"sequence_{scenario.scenario_id}_preflight_{drone_id}_{stamp}"
        )
        session_dir.mkdir(parents=True, exist_ok=True)

        move_failures = 0
        capture_failures = 0
        max_move_failures = max(1, int(getattr(mapping_cfg, "preflight_max_move_failures", 3)))
        max_capture_failures = max(
            1, int(getattr(mapping_cfg, "preflight_max_capture_failures", 5))
        )

        if executor and not executor.is_flying:
            await executor.execute({
                "action": "takeoff",
                "parameters": {"altitude_agl": altitude_agl},
                "reasoning": f"Preflight mapping takeoff for {scenario.scenario_id} ({drone_id})",
                "confidence": 1.0,
            })

        capture_count = 0
        move_with_avoidance = getattr(bridge, "move_to_position_with_obstacle_avoidance", None)

        for seg_idx in range(len(path_points) - 1):
            seg_start = path_points[seg_idx]
            seg_end = path_points[seg_idx + 1]
            distance = math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
            steps = max(1, int(math.ceil(distance / step_m)))

            for idx in range(steps + 1):
                frac = idx / steps
                n = seg_start[0] + (seg_end[0] - seg_start[0]) * frac
                e = seg_start[1] + (seg_end[1] - seg_start[1]) * frac
                try:
                    if move_with_avoidance:
                        move_ok = await asyncio.wait_for(
                            move_with_avoidance(
                                n,
                                e,
                                -altitude_agl,
                                velocity=float(mapping_cfg.preflight_velocity_ms),
                            ),
                            timeout=wait_timeout_s,
                        )
                    else:
                        move_ok = await asyncio.wait_for(
                            bridge.move_to_position(
                                n,
                                e,
                                -altitude_agl,
                                velocity=float(mapping_cfg.preflight_velocity_ms),
                                timeout=move_timeout_s,
                            ),
                            timeout=wait_timeout_s,
                        )
                    if not move_ok:
                        raise RuntimeError("move_to_position returned False")
                except Exception as exc:
                    move_failures += 1
                    logger.warning(
                        "preflight_move_failed",
                        scenario_id=scenario.scenario_id,
                        drone_id=drone_id,
                        error=str(exc),
                        failures=move_failures,
                    )
                    if move_failures >= max_move_failures:
                        break
                    continue

                try:
                    capture = await bridge.capture_mapping_bundle(
                        output_dir=session_dir,
                        include_depth=True,
                        include_imu=True,
                    )
                except Exception as exc:
                    capture = {"success": False, "error": str(exc)}
                if capture.get("success"):
                    capture_count += 1
                else:
                    capture_failures += 1
                    if capture_failures >= max_capture_failures:
                        break
                await asyncio.sleep(float(mapping_cfg.preflight_capture_interval_s))

            if move_failures >= max_move_failures or capture_failures >= max_capture_failures:
                break

        if executor and executor.is_flying:
            await bridge.hover()

        return {
            "drone_id": drone_id,
            "session_dir": session_dir,
            "capture_count": capture_count,
            "estimated_captures": estimated_captures,
            "move_failures": move_failures,
            "capture_failures": capture_failures,
        }

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[
                _capture_for_drone(drone_id, bridge, executor, assignments.get(drone_id, []))
                for drone_id, bridge, executor in drone_entries
            ]),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "preflight_mapping_timeout",
            scenario_id=scenario.scenario_id,
            timeout_s=timeout_s,
        )
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "timeout",
            f"Preflight mapping timed out after {timeout_s:.0f}s.",
            severity=EventSeverity.WARNING,
        )
        return

    total_captures = sum(int(res.get("capture_count", 0)) for res in results)
    if total_captures == 0:
        await _broadcast_preflight_status(
            scenario.scenario_id,
            "complete",
            "Preflight completed without map (no captures collected).",
            severity=EventSeverity.WARNING,
            details={
                "capture_count": 0,
                "total_captures": 0,
                "mapping_status": "skipped",
                "slam_status": "skipped",
                "quality": 0,
            },
        )
        return

    combined_session_dir = Path("data/maps") / f"sequence_{scenario.scenario_id}_preflight_{stamp}"
    combined_session_dir.mkdir(parents=True, exist_ok=True)
    session_dirs = [Path(res["session_dir"]) for res in results]
    merged = _merge_preflight_sessions(combined_session_dir, session_dirs, drone_ids)
    if merged == 0:
        logger.warning("preflight_merge_failed", scenario_id=scenario.scenario_id)
        return

    await _broadcast_preflight_status(
        scenario.scenario_id,
        "processing",
        f"Running SLAM on {total_captures} frames from {len(drone_entries)} drones...",
        details={
            "capture_count": total_captures,
            "total_captures": total_captures,
            "mapping_status": "processing",
            "slam_status": "running",
        },
    )

    output_dir = Path(mapping_cfg.slam_dir) / f"run_{scenario.scenario_id}_preflight_{stamp}"
    backend = getattr(mapping_cfg, "slam_backend", "telemetry") or "telemetry"
    args = SimpleNamespace(
        input_dir=str(combined_session_dir),
        output_dir=str(output_dir),
        backend=backend,
        allow_telemetry_fallback=bool(getattr(mapping_cfg, "slam_allow_fallback", True)),
        no_pointcloud=False,
        max_points=mapping_cfg.max_points,
        depth_subsample=6,
        min_time_interval_s=None,
        max_time_interval_s=None,
        min_translation_m=None,
        min_rotation_deg=None,
        velocity_threshold_ms=None,
        blur_threshold=None,
        min_feature_count=None,
        overlap_target=None,
    )

    slam_result = await asyncio.to_thread(run_slam_runner, args)
    map_points = output_dir / "map_points.ply"
    if slam_result != 0 and not map_points.exists():
        raise RuntimeError(f"SLAM runner failed (backend={backend}, code={slam_result})")

    await _broadcast_preflight_status(
        scenario.scenario_id,
        "processing",
        "SLAM complete. Fusing navigation map...",
        details={
            "capture_count": total_captures,
            "total_captures": total_captures,
            "mapping_status": "fusing",
            "slam_status": "complete",
        },
    )

    if map_points.exists():
        fusion = MapFusion(
            MapFusionConfig(
                resolution_m=mapping_cfg.map_resolution_m,
                tile_size_cells=getattr(mapping_cfg, "tile_size_cells", 120),
                voxel_size_m=mapping_cfg.voxel_size_m,
                max_points=mapping_cfg.max_points,
                min_points=mapping_cfg.min_points,
            )
        )
        fused = fusion.build_navigation_map(
            point_cloud_path=map_points,
            map_id=output_dir.name,
            scenario_id=scenario.scenario_id,
            source="slam_preflight_multi",
        )
        nav_map = fused.navigation_map
        metadata = nav_map.get("metadata", {})
        metadata["point_cloud_path"] = str(map_points)
        nav_map["metadata"] = metadata
        server_state.navigation_map = nav_map
        apply_navigation_map_to_executors(nav_map)

    pose_graph_path = output_dir / "pose_graph.json"
    await _train_splat_from_pose_graph(
        pose_graph_path,
        mapping_cfg,
        scenario.scenario_id,
        output_dir.name,
    )

    quality = None
    obstacle_count = 0
    if server_state.navigation_map:
        meta = server_state.navigation_map.get("metadata", {})
        quality = meta.get("map_quality_score")
        obstacle_count = len(server_state.navigation_map.get("obstacles", []))
    await _broadcast_preflight_status(
        scenario.scenario_id,
        "complete",
        "Preflight mapping completed successfully.",
        details={
            "capture_count": total_captures,
            "total_captures": total_captures,
            "mapping_status": "complete",
            "slam_status": "complete",
            "quality": quality,
            "obstacle_count": obstacle_count,
        },
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
                    "altitude_m": a.altitude_m,
                    "inspection_altitude_agl": a.inspection_altitude_agl,
                    "orbit_radius_m": a.orbit_radius_m,
                    "dwell_time_s": a.dwell_time_s,
                    "scale": a.scale,
                    "rotation_deg": a.rotation_deg,
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
                    detail=(f"Invalid edge profile: {edge_profile}. Valid: {valid_profiles}"),
                )

            time_scale = max(0.5, min(5.0, time_scale))

            scenario_log_dir = server_state.log_dir / "scenarios"
            runner = ScenarioRunner(
                log_dir=scenario_log_dir,
                enable_llm=bool(config.agent.use_llm),
                use_advanced_engine=bool(config.agent.use_llm),
            )
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

            await _reset_mapping_state(scenario_id)

            # Seed the shared world model so live decisions see scenario assets.
            server_state.world_model.reset_assets()
            for asset in scenario.assets:
                asset_type = (
                    AssetType(asset.asset_type)
                    if asset.asset_type in [e.value for e in AssetType]
                    else AssetType.OTHER
                )
                server_state.world_model.add_asset(
                    Asset(
                        asset_id=asset.asset_id,
                        name=asset.name,
                        asset_type=asset_type,
                        position=Position(
                            latitude=asset.latitude,
                            longitude=asset.longitude,
                            altitude_msl=float(asset.altitude_m or 0.0),
                        ),
                        priority=asset.priority,
                        inspection_altitude_agl=asset.inspection_altitude_agl,
                        orbit_radius_m=asset.orbit_radius_m,
                        dwell_time_s=asset.dwell_time_s,
                    )
                )
                if asset.has_anomaly:
                    server_state.world_model.add_anomaly(
                        Anomaly(
                            anomaly_id=f"anom_{asset.asset_id}",
                            asset_id=asset.asset_id,
                            detected_at=datetime.now(),
                            severity=asset.anomaly_severity,
                            description=f"Pre-existing anomaly on {asset.name}",
                        )
                    )
            server_state.world_model.start_mission(
                mission_id=f"scenario_{scenario.scenario_id}",
                mission_name=scenario.name,
            )

            mapping_cfg = config.mapping
            if mapping_cfg.enabled and mapping_cfg.preflight_enabled:
                if server_state.preflight_task and not server_state.preflight_task.done():
                    server_state.preflight_task.cancel()
                preflight_task = asyncio.create_task(
                    _run_preflight_slam_mapping(scenario, server_state.airsim_action_executor)
                )
                server_state.preflight_task = preflight_task

                def _clear_preflight_task(task: asyncio.Task) -> None:
                    if server_state.preflight_task is task:
                        server_state.preflight_task = None

                preflight_task.add_done_callback(_clear_preflight_task)
                try:
                    # Shield from HTTP request cancellation - preflight should complete
                    await asyncio.shield(preflight_task)
                except asyncio.CancelledError:
                    # HTTP request was cancelled but preflight continues in background
                    logger.warning(
                        "preflight_http_cancelled",
                        scenario_id=scenario_id,
                        message="HTTP request cancelled, preflight continues in background",
                    )
                except Exception as exc:
                    logger.warning(
                        "preflight_blocking_failed",
                        scenario_id=scenario_id,
                        error=str(exc),
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
                    apply_navigation_map_to_executors(navigation_map)
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

            airsim_settings_update: dict[str, object] | None = None
            airsim_restart: dict[str, object] | None = None
            if config.simulation.airsim_enabled:
                airsim_settings_update = update_airsim_settings(
                    config,
                    scenario_drone_ids=[drone.drone_id for drone in scenario.drones],
                    reason=f"scenario_start:{scenario_id}",
                )
                if (
                    airsim_settings_update
                    and airsim_settings_update.get("updated")
                    and config.simulation.airsim_auto_restart_on_scenario_change
                ):
                    airsim_restart = schedule_airsim_restart(
                        reason=f"scenario_start:{scenario_id}",
                    )

            airsim_sync: dict[str, object] | None = None
            if config.simulation.airsim_enabled:
                restart_pending = bool(
                    airsim_restart
                    and (airsim_restart.get("scheduled") or airsim_restart.get("restarted"))
                )
                if restart_pending:
                    if len(scenario.drones) > 1:

                        async def _sync_multi_after_restart() -> None:
                            await asyncio.sleep(6.0)
                            try:
                                await sync_multi_drone_scenario(scenario)
                            except Exception as exc:
                                logger.warning(
                                    "airsim_multi_sync_after_restart_failed",
                                    scenario_id=scenario_id,
                                    error=str(exc),
                                )

                        asyncio.create_task(_sync_multi_after_restart())
                    else:
                        asyncio.create_task(_sync_airsim_scene(scenario, wait_for_connect=True))

                    airsim_sync = {"scheduled": True, "reason": "airsim_restarting"}
                elif len(scenario.drones) > 1:
                    airsim_sync = await sync_multi_drone_scenario(scenario)
                else:
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
                    apply_navigation_map_to_executors(server_state.navigation_map)

            if (
                config.simulation.airsim_enabled
                and len(scenario.drones) > 1
                and config.simulation.require_fleet_for_multi_drone
            ):
                failed_drones = []
                if isinstance(airsim_sync, dict):
                    failed_drones = airsim_sync.get("failed_drones") or []
                multi_ready = bool(airsim_sync and airsim_sync.get("multi_drone"))
                # Don't fail if AirSim restart is pending - the re-sync will happen after restart
                restart_scheduled = bool(
                    airsim_sync
                    and airsim_sync.get("scheduled")
                    and airsim_sync.get("reason") == "airsim_restarting"
                )
                if (not multi_ready and not restart_scheduled) or failed_drones:
                    scenario_runner_state.is_running = False
                    if scenario_runner_state.run_task:
                        try:
                            if not scenario_runner_state.run_task.done():
                                scenario_runner_state.run_task.cancel()
                                await asyncio.wait_for(scenario_runner_state.run_task, timeout=2.0)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "scenario_runner_cancel_timeout", scenario_id=scenario_id
                            )
                        except asyncio.CancelledError:
                            # Expected when cancelling the task
                            pass
                        except Exception as exc:
                            logger.warning(
                                "scenario_runner_cancel_failed",
                                scenario_id=scenario_id,
                                error=str(exc),
                            )
                        scenario_runner_state.run_task = None
                    if scenario_runner_state.runner is runner:
                        scenario_runner_state.runner = None
                    scenario_run_state.running = False
                    scenario_run_state.scenario_id = None
                    scenario_run_state.start_time = None
                    detail = {
                        "reason": "multi_drone_fleet_not_ready",
                        "failed_drones": failed_drones,
                        "sync": airsim_sync,
                    }
                    logger.error("multi_drone_fleet_required", **detail)
                    raise HTTPException(status_code=409, detail=detail)

            logger.info(
                "scenario_start_success",
                scenario_id=scenario_id,
                run_id=runner.run_id,
            )

            # Auto-fly mission: automatically takeoff and fly through all assets
            if mode == "live" and len(scenario.drones) == 1:

                async def auto_fly_mission() -> None:
                    """Automatically fly through all scenario assets."""
                    try:
                        # Wait for AirSim executor to become available (up to 30 seconds)
                        wait_attempts = 0
                        max_wait_attempts = 30
                        while (
                            server_state.airsim_action_executor is None
                            and wait_attempts < max_wait_attempts
                        ):
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

                        if server_state.preflight_task:
                            try:
                                await server_state.preflight_task
                            except asyncio.CancelledError:
                                raise
                            except Exception as exc:
                                logger.warning(
                                    "preflight_task_failed",
                                    scenario_id=scenario_id,
                                    error=str(exc),
                                )
                        else:
                            await _run_preflight_slam_mapping(scenario, executor)

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
                "airsim_settings": airsim_settings_update,
                "airsim_restart": airsim_restart,
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
