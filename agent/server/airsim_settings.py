"""AirSim settings.json helpers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from agent.server.config_manager import AegisConfig

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS_VERSION = 1.2
DEFAULT_SIM_MODE = "Multirotor"
DEFAULT_CLOCK_TYPE = "SteppableClock"
DEFAULT_VEHICLE_PREFIX = "Drone"
DEFAULT_CAMERA_FOV = 90

BASE_UDP_PORT = 9003
BASE_SITL_PORT = 5760
BASE_CONTROL_PORT = 14550


def _is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def _resolve_windows_user() -> str | None:
    for key in ("WINDOWS_USER", "WINUSER"):
        value = os.environ.get(key)
        if value:
            return value
    if _is_wsl():
        try:
            result = subprocess.run(  # noqa: S603, S607
                ["cmd.exe", "/c", "echo", "%USERNAME%"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            candidate = result.stdout.strip().strip("\r")
            if candidate:
                return candidate
        except Exception:
            return None
    return None


def _resolve_settings_path(config: AegisConfig) -> Path | None:
    path_value = config.simulation.airsim_settings_path
    if not path_value:
        path_value = os.environ.get("AEGIS_AIRSIM_SETTINGS_PATH") or os.environ.get(
            "AIRSIM_SETTINGS_PATH"
        )
    if path_value:
        return Path(path_value).expanduser()

    if _is_wsl():
        windows_user = _resolve_windows_user()
        if windows_user:
            candidate = (
                Path("/mnt/c/Users") / windows_user / "Documents" / "AirSim" / "settings.json"
            )
            if candidate.parent.exists():
                return candidate

    return Path.home() / "Documents" / "AirSim" / "settings.json"


def _coerce_camera_resolution(config: AegisConfig) -> tuple[int, int]:
    resolution = config.vision.camera_resolution
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        try:
            width = int(resolution[0])
            height = int(resolution[1])
            if width > 0 and height > 0:
                return width, height
        except (TypeError, ValueError):
            pass
    return 1920, 1080


def _collect_vehicle_names(
    config: AegisConfig,
    scenario_drone_ids: Iterable[str] | None,
) -> list[str]:
    names: list[str] = []

    primary = (config.simulation.airsim_vehicle_name or "").strip()
    if primary:
        names.append(primary)

    for name in config.simulation.airsim_vehicle_mapping.values():
        if name and name not in names:
            names.append(name)

    scenario_ids = list(scenario_drone_ids) if scenario_drone_ids else []
    scenario_count = len(scenario_ids)

    vehicle_type = (config.simulation.airsim_vehicle_type or "").strip().lower()
    arducopter = vehicle_type == "arducopter"
    require_fleet = bool(getattr(config.simulation, "require_fleet_for_multi_drone", False))
    force_multi = bool(scenario_count > 1 and require_fleet)
    allow_multi = (not arducopter) or config.simulation.sitl_multi_vehicle or force_multi

    if force_multi and arducopter and not config.simulation.sitl_multi_vehicle:
        logger.warning(
            "airsim_settings_force_multi_vehicle",
            scenario_count=scenario_count,
            reason="require_fleet_for_multi_drone",
        )

    if allow_multi:
        desired_count = max(config.simulation.max_drones, scenario_count, len(names), 1)
    else:
        desired_count = max(len(names), 1)

    for index in range(1, desired_count + 1):
        candidate = f"{DEFAULT_VEHICLE_PREFIX}{index}"
        if candidate not in names:
            names.append(candidate)

    if not allow_multi and len(names) > 1:
        logger.warning(
            "airsim_settings_multi_vehicle_disabled",
            requested=len(names),
            reason="sitl_multi_vehicle_disabled",
        )
        names = names[:1]

    return names


def _build_camera_settings(width: int, height: int) -> dict[str, dict[str, object]]:
    capture = [
        {"ImageType": 0, "Width": width, "Height": height, "FOV_Degrees": DEFAULT_CAMERA_FOV}
    ]
    return {
        "front_center": {
            "CaptureSettings": capture,
            "X": 0.5,
            "Y": 0,
            "Z": -0.3,
            "Pitch": -15,
            "Roll": 0,
            "Yaw": 0,
        },
        "bottom": {
            "CaptureSettings": capture,
            "X": 0,
            "Y": 0,
            "Z": 0.5,
            "Pitch": -90,
            "Roll": 0,
            "Yaw": 0,
        },
    }


def _build_vehicle_payload(
    *,
    index: int,
    config: AegisConfig,
    width: int,
    height: int,
) -> dict[str, object]:
    vehicle_type = (config.simulation.airsim_vehicle_type or "").strip()
    if not vehicle_type:
        vehicle_type = "ArduCopter" if config.simulation.sitl_enabled else "SimpleFlight"
    vehicle_type = vehicle_type.strip()

    payload: dict[str, object] = {
        "VehicleType": vehicle_type,
        "AutoCreate": True,
        "Cameras": _build_camera_settings(width, height),
        "X": 0,
        "Y": float(index * 3),
        "Z": 0,
        "Pitch": 0,
        "Roll": 0,
        "Yaw": 0,
    }

    if vehicle_type.lower() == "arducopter":
        payload.update(
            {
                "UseSerial": False,
                "LocalHostIp": "127.0.0.1",
                "UdpIp": "127.0.0.1",
                "UdpPort": BASE_UDP_PORT + index,
                "SitlPort": BASE_SITL_PORT + index,
                "ControlPort": BASE_CONTROL_PORT + index,
            }
        )

    return payload


def _build_settings_payload(
    config: AegisConfig,
    *,
    vehicle_names: list[str],
) -> dict[str, object]:
    width, height = _coerce_camera_resolution(config)
    vehicles = {}
    for idx, name in enumerate(vehicle_names):
        vehicles[name] = _build_vehicle_payload(
            index=idx,
            config=config,
            width=width,
            height=height,
        )

    primary = vehicle_names[0] if vehicle_names else "Drone1"
    return {
        "SettingsVersion": DEFAULT_SETTINGS_VERSION,
        "SimMode": DEFAULT_SIM_MODE,
        "ClockType": DEFAULT_CLOCK_TYPE,
        "Vehicles": vehicles,
        "SubWindows": [
            {"WindowID": 0, "CameraName": "front_center", "ImageType": 0, "VehicleName": primary}
        ],
    }


def _summarize_settings(payload: dict[str, Any]) -> dict[str, Any]:
    vehicles = payload.get("Vehicles", {}) or {}
    vehicle_names = list(vehicles.keys())
    vehicle_types = {name: (data.get("VehicleType") if isinstance(data, dict) else None)
                     for name, data in vehicles.items()}
    camera_resolution = None
    if vehicle_names:
        first_vehicle = vehicles.get(vehicle_names[0], {})
        cameras = first_vehicle.get("Cameras", {}) if isinstance(first_vehicle, dict) else {}
        front = cameras.get("front_center", {}) if isinstance(cameras, dict) else {}
        captures = front.get("CaptureSettings", []) if isinstance(front, dict) else []
        if captures and isinstance(captures, list):
            first_capture = captures[0] if captures else {}
            if isinstance(first_capture, dict):
                width = first_capture.get("Width")
                height = first_capture.get("Height")
                if width and height:
                    camera_resolution = [width, height]

    return {
        "settings_version": payload.get("SettingsVersion"),
        "sim_mode": payload.get("SimMode"),
        "clock_type": payload.get("ClockType"),
        "vehicle_count": len(vehicle_names),
        "vehicle_names": vehicle_names,
        "vehicle_types": vehicle_types,
        "camera_resolution": camera_resolution,
    }


def _diff_settings_summary(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not previous:
        return {"initial": {"from": None, "to": current}}
    diff: dict[str, dict[str, Any]] = {}
    for key, value in current.items():
        prior = previous.get(key)
        if prior != value:
            diff[key] = {"from": prior, "to": value}
    return diff


def update_airsim_settings(
    config: AegisConfig,
    *,
    scenario_drone_ids: Iterable[str] | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    """Update AirSim settings.json based on configuration and scenario data."""
    if not config.simulation.airsim_enabled:
        return {"updated": False, "reason": "airsim_disabled"}
    if not config.simulation.airsim_auto_update_settings:
        return {"updated": False, "reason": "auto_update_disabled"}

    settings_path = _resolve_settings_path(config)
    if settings_path is None:
        return {"updated": False, "reason": "settings_path_unresolved"}

    vehicle_names = _collect_vehicle_names(config, scenario_drone_ids)
    payload = _build_settings_payload(config, vehicle_names=vehicle_names)
    normalized_payload = json.dumps(payload, sort_keys=True)

    existing_payload = None
    existing_dict: dict[str, Any] | None = None
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                existing_dict = json.load(f)
                existing_payload = json.dumps(existing_dict, sort_keys=True)
        except (OSError, json.JSONDecodeError):
            existing_payload = None
            existing_dict = None

    summary = _summarize_settings(payload)
    previous_summary = _summarize_settings(existing_dict) if existing_dict else None
    summary_diff = _diff_settings_summary(previous_summary, summary)

    if existing_payload == normalized_payload:
        return {
            "updated": False,
            "reason": reason or "no_change",
            "path": str(settings_path),
            "vehicles": vehicle_names,
            "summary": summary,
        }

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if settings_path.exists():
        backup_path = settings_path.with_suffix(".json.backup")
        try:
            shutil.copy2(settings_path, backup_path)
        except OSError as exc:
            logger.warning("airsim_settings_backup_failed", error=str(exc))
            backup_path = None

    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except OSError as exc:
        logger.error("airsim_settings_write_failed", error=str(exc))
        return {
            "updated": False,
            "reason": reason or "write_failed",
            "path": str(settings_path),
            "error": str(exc),
            "summary": summary,
        }

    logger.info(
        "airsim_settings_updated",
        reason=reason or "updated",
        path=str(settings_path),
        vehicles=len(vehicle_names),
        summary=summary,
        summary_diff=summary_diff,
    )
    return {
        "updated": True,
        "reason": reason or "updated",
        "path": str(settings_path),
        "vehicles": vehicle_names,
        "backup": str(backup_path) if backup_path else None,
        "note": "Restart AirSim to apply settings changes.",
        "summary": summary,
        "previous_summary": previous_summary,
        "summary_diff": summary_diff,
    }
