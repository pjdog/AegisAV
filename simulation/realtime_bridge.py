"""Real-Time AirSim Bridge for Unreal Visualization.

Provides async-safe, high-frequency communication between AirSim/Unreal and AegisAV.
Designed for sub-100ms latency with frame synchronization and telemetry streaming.

Key improvements over base bridge:
- All AirSim calls wrapped in _run_in_airsim_thread() to avoid blocking
- Parallel fetch of image + pose + IMU for synchronized data
- High-frequency telemetry streaming (30-50 Hz)
- Frame sequence numbers for ordering and deduplication
- Async producer queues for backpressure handling
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable

# Use a SINGLE dedicated thread for ALL AirSim operations
# The cosysairsim library uses msgpackrpc/tornado which has its own event loop
# that conflicts with asyncio. All AirSim calls must happen in the SAME thread.
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_airsim_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airsim")

try:
    # Use Cosys-AirSim (actively maintained fork with UE 5.5 support)
    # Install: pip install cosysairsim
    # https://github.com/Cosys-Lab/Cosys-AirSim
    import cosysairsim as airsim

    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    airsim = None  # type: ignore[assignment]

from pydantic import BaseModel

# Optional asset manager integration
try:
    from simulation.asset_manager import (
        AssetManager,
        AssetType,
        get_asset_manager,
    )

    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    ASSET_MANAGER_AVAILABLE = False
    AssetManager = None  # type: ignore
    AssetType = None  # type: ignore
    get_asset_manager = None  # type: ignore

logger = logging.getLogger(__name__)


# Global client reference that lives in the airsim thread
_airsim_client: Any = None
_airsim_client_lock = threading.Lock()

TRACEABLE_AIRSIM_METHODS = {
    "enableApiControl",
    "armDisarm",
    "takeoffAsync",
    "landAsync",
    "hoverAsync",
    "moveToPositionAsync",
    "moveByVelocityAsync",
    "simEnableWeather",
    "simSetWeatherParameter",
    "simSetTimeOfDay",
    "simSetWind",
    "simSetVehiclePose",
    "simSetVehiclePoseAsync",
    "simSpawnObject",
    "simDestroyObject",
    "simSetWorldLightIntensity",
    "simSetLightIntensity",
}


def _call_airsim_method(method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a method on the global AirSim client from within the dedicated thread.

    This function runs IN the airsim thread and accesses the client directly.
    """
    global _airsim_client
    if _airsim_client is None:
        raise RuntimeError("AirSim client not connected")
    method = getattr(_airsim_client, method_name)
    return method(*args, **kwargs)


async def _run_in_airsim_thread(
    func_or_method: Callable[..., Any] | str, *args: Any, **kwargs: Any
) -> Any:
    """Run a blocking AirSim function in the dedicated single thread.

    IMPORTANT: The cosysairsim library uses msgpackrpc/tornado which maintains
    its own event loop state per-thread. ALL AirSim client operations must
    happen in the SAME thread to avoid 'Cannot run event loop' errors.

    Args:
        func_or_method: Either a callable (for standalone functions) or a string
                       method name to call on the global client.
        *args, **kwargs: Arguments to pass to the function/method.
    """
    loop = asyncio.get_running_loop()

    if isinstance(func_or_method, str):
        # It's a method name - call it on the global client
        def _call():
            return _call_airsim_method(func_or_method, *args, **kwargs)

        return await loop.run_in_executor(_airsim_executor, _call)
    else:
        # It's a callable function - call it directly
        def _call():
            return func_or_method(*args, **kwargs)

        return await loop.run_in_executor(_airsim_executor, _call)


def _select_vehicle_name(requested: str, vehicles: list[str]) -> tuple[str, str | None]:
    if not vehicles:
        return requested, None
    if requested in vehicles:
        return requested, None
    return vehicles[0], f"Requested vehicle '{requested}' not found; using '{vehicles[0]}'"


# =============================================================================
# Data Models for Real-Time Streaming
# =============================================================================


class TelemetryType(str, Enum):
    """Types of telemetry messages."""

    POSE = "pose"
    IMU = "imu"
    FRAME = "frame"
    STATE = "state"
    FULL_SYNC = "full_sync"  # Combined pose + IMU + state


class Vector3(BaseModel):
    """3D vector for position, velocity, acceleration."""

    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        """Return vector as [x, y, z]."""
        return [self.x, self.y, self.z]


class Quaternion(BaseModel):
    """Quaternion for orientation."""

    w: float
    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        """Return quaternion as [w, x, y, z]."""
        return [self.w, self.x, self.y, self.z]


class IMUData(BaseModel):
    """IMU sensor data."""

    timestamp_ns: int
    linear_acceleration: Vector3
    angular_velocity: Vector3
    orientation: Quaternion


class PoseData(BaseModel):
    """Vehicle pose (position + orientation)."""

    timestamp_ns: int
    position: Vector3  # NED coordinates in meters
    orientation: Quaternion
    linear_velocity: Vector3
    angular_velocity: Vector3


class TelemetryFrame(BaseModel):
    """A synchronized telemetry frame with all sensor data."""

    sequence: int
    server_timestamp_ms: float
    airsim_timestamp_ns: int
    pose: PoseData
    imu: IMUData | None = None
    battery_percent: float = 100.0
    landed_state: str = "flying"
    latency_ms: float = 0.0

    def to_broadcast_dict(self) -> dict[str, Any]:
        """Convert to dict suitable for WebSocket broadcast."""
        return {
            "type": "airsim_telemetry",
            "sequence": self.sequence,
            "timestamp_ms": self.server_timestamp_ms,
            "latency_ms": self.latency_ms,
            "position": self.pose.position.to_list(),
            "orientation": self.pose.orientation.to_list(),
            "velocity": self.pose.linear_velocity.to_list(),
            "angular_velocity": self.pose.angular_velocity.to_list(),
            "imu": {
                "acceleration": self.imu.linear_acceleration.to_list(),
                "gyro": self.imu.angular_velocity.to_list(),
            }
            if self.imu
            else None,
            "battery_percent": self.battery_percent,
            "landed_state": self.landed_state,
        }


class FrameCaptureResult(BaseModel):
    """Result of a synchronized frame capture."""

    sequence: int
    success: bool
    timestamp: datetime
    server_timestamp_ms: float
    image_path: Path | None = None
    telemetry: TelemetryFrame | None = None
    capture_latency_ms: float = 0.0
    pose_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    error: str | None = None


@dataclass
class AirSimCallRecord:
    """Lightweight trace record for AirSim RPC calls."""

    timestamp_ms: float
    method: str
    args: list[str]
    kwargs: dict[str, str]
    duration_ms: float
    ok: bool
    error: str | None
    vehicle_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "method": self.method,
            "args": self.args,
            "kwargs": self.kwargs,
            "duration_ms": self.duration_ms,
            "ok": self.ok,
            "error": self.error,
            "vehicle_name": self.vehicle_name,
        }


@dataclass
class RealtimeBridgeConfig:
    """Configuration for real-time AirSim bridge."""

    # Connection
    host: str = "127.0.0.1"
    vehicle_name: str = "Drone1"
    camera_name: str = "front_center"

    # Capture settings
    target_fps: int = 30  # Target frame rate
    resolution: tuple[int, int] = (1280, 720)  # Lower res for speed
    compress: bool = True  # Use JPEG for faster transfer
    save_images: bool = False  # Don't save by default for speed
    output_dir: Path = field(default_factory=lambda: Path("data/vision/realtime"))
    mapping_output_dir: Path = field(default_factory=lambda: Path("data/maps"))

    # Telemetry settings
    telemetry_hz: int = 50  # Telemetry polling rate
    include_imu: bool = True

    # Battery simulation (used if AirSim does not provide battery telemetry)
    battery_sim_enabled: bool = True
    battery_initial_percent: float = 100.0
    battery_min_percent: float = 5.0
    battery_max_percent: float = 100.0
    battery_drain_hover_percent_per_min: float = 6.0
    battery_drain_move_percent_per_m: float = 0.015
    battery_charge_percent_per_min: float = 25.0
    battery_aggressive_multiplier: float = 1.3
    battery_low_speed_threshold_ms: float = 0.5

    # Queue settings
    max_queue_size: int = 100
    drop_old_frames: bool = True  # Drop old frames if queue full

    # Timing
    sync_timeout_ms: float = 100.0  # Max time to wait for synchronized data
    rpc_timeout_s: float = 10.0  # RPC call timeout to avoid hanging on connect

    # Landing behavior
    landing_soft_altitude_m: float = 1.5
    landing_soft_velocity_ms: float = 0.8
    landing_soft_timeout_s: float = 15.0

    # AirSim call tracing (low-volume, command-focused)
    trace_airsim_calls: bool = True
    trace_call_history_size: int = 200
    trace_call_max_value_len: int = 140


# =============================================================================
# Real-Time Bridge Implementation
# =============================================================================


class RealtimeAirSimBridge:
    """Async-safe, high-frequency AirSim bridge for real-time visualization.

    Features:
    - Non-blocking I/O via _run_in_airsim_thread()
    - Parallel data fetching for synchronization
    - High-frequency telemetry streaming
    - Frame sequence numbers
    - Async producer/consumer queues

    Example:
        bridge = RealtimeAirSimBridge(config)
        await bridge.connect()

        # Start telemetry stream
        async for frame in bridge.telemetry_stream():
            await websocket.send_json(frame.to_broadcast_dict())
    """

    def __init__(self, config: RealtimeBridgeConfig | None = None) -> None:
        if not AIRSIM_AVAILABLE:
            raise ImportError("airsim package not installed")

        self.config = config or RealtimeBridgeConfig()
        self.client: airsim.MultirotorClient | None = None
        self.connected = False
        self.vehicle_names: list[str] = []
        self._last_yaw_deg: float | None = None
        self._last_collision_stamp: int | None = None
        self._mapping_session_dir: Path | None = None

        # Sequence counter (monotonic)
        self._sequence = 0
        self._sequence_lock = asyncio.Lock()

        # Telemetry queue
        self._telemetry_queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )

        # Frame queue (for image captures)
        self._frame_queue: asyncio.Queue[FrameCaptureResult] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )

        # Streaming control
        self._streaming = False
        self._stream_task: asyncio.Task | None = None

        # Callbacks for event dispatch
        self._on_telemetry: list[Callable[[TelemetryFrame], Awaitable[None]]] = []
        self._on_frame: list[Callable[[FrameCaptureResult], Awaitable[None]]] = []

        # Metrics
        self._total_frames = 0
        self._dropped_frames = 0
        self._avg_latency_ms = 0.0
        self._latency_samples: deque[float] = deque(maxlen=100)

        # AirSim call tracing
        self._call_history: deque[AirSimCallRecord] = deque(
            maxlen=self.config.trace_call_history_size
        )

        # Battery simulation state
        self._battery_percent = float(self.config.battery_initial_percent)
        self._battery_last_update = time.time()

        # Ensure output dir exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.mapping_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RealtimeAirSimBridge initialized (target: {self.config.target_fps} FPS)")

    def _read_airsim_battery_percent(self, state_raw: Any) -> float | None:
        candidates = [
            getattr(state_raw, "battery_percent", None),
            getattr(state_raw, "battery_percentage", None),
            getattr(state_raw, "batteryRemaining", None),
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                percent = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= percent <= 1.0:
                return percent * 100.0
            if 0.0 <= percent <= 100.0:
                return percent

        battery_state = getattr(state_raw, "battery", None)
        if battery_state is None:
            return None
        for attr in ("remaining", "remaining_percent", "percent", "percentage"):
            value = getattr(battery_state, attr, None)
            if value is None:
                continue
            try:
                percent = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= percent <= 1.0:
                return percent * 100.0
            if 0.0 <= percent <= 100.0:
                return percent
        return None

    def _simulate_battery(self, speed_m_s: float, landed_state: str) -> float:
        now = time.time()
        dt = max(0.0, now - self._battery_last_update)
        self._battery_last_update = now

        if not self.config.battery_sim_enabled:
            return self._battery_percent

        if landed_state == "landed":
            charge_per_s = self.config.battery_charge_percent_per_min / 60.0
            self._battery_percent = min(
                self.config.battery_max_percent,
                self._battery_percent + charge_per_s * dt,
            )
            return self._battery_percent

        hover_drain_per_s = self.config.battery_drain_hover_percent_per_min / 60.0
        if speed_m_s < self.config.battery_low_speed_threshold_ms:
            move_drain_per_s = 0.0
        else:
            move_drain_per_s = speed_m_s * self.config.battery_drain_move_percent_per_m
        drain_per_s = hover_drain_per_s + move_drain_per_s
        drain_per_s *= max(0.1, self.config.battery_aggressive_multiplier)

        self._battery_percent = max(
            self.config.battery_min_percent,
            self._battery_percent - drain_per_s * dt,
        )
        return self._battery_percent

    @staticmethod
    def _angle_delta_deg(a: float, b: float) -> float:
        """Smallest signed angle difference (degrees)."""
        return (a - b + 180.0) % 360.0 - 180.0

    @staticmethod
    def _normalize_yaw_deg(yaw_deg: float) -> float:
        """Normalize yaw to [-180, 180) degrees."""
        return (yaw_deg + 180.0) % 360.0 - 180.0

    def _camera_candidates(self) -> list[str]:
        """Return ordered camera name fallbacks for depth capture."""
        candidates = [
            self.config.camera_name,
            "front_center",
            "0",
            "front",
            "front_left",
            "front_right",
        ]
        return [name for name in dict.fromkeys(candidates) if name]

    async def _align_yaw_to_vector(
        self,
        dx: float,
        dy: float,
        min_change_deg: float = 15.0,
    ) -> None:
        """Rotate to face the movement vector so the depth camera looks ahead."""
        if not self.connected or not self.client or not hasattr(self.client, "rotateToYawAsync"):
            return

        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return

        desired_yaw = self._normalize_yaw_deg(math.degrees(math.atan2(dy, dx)))
        if self._last_yaw_deg is not None:
            delta = abs(self._angle_delta_deg(desired_yaw, self._last_yaw_deg))
            if delta < min_change_deg:
                return

        vehicle_name = self.config.vehicle_name

        def _rotate() -> None:
            global _airsim_client
            try:
                result = _airsim_client.rotateToYawAsync(
                    desired_yaw,
                    timeout_sec=2.0,
                    vehicle_name=vehicle_name,
                )
            except TypeError:
                result = _airsim_client.rotateToYawAsync(desired_yaw, timeout_sec=2.0)
            if hasattr(result, "join"):
                try:
                    result.join(timeout_sec=2.0)
                except Exception:
                    result.join()

        await _run_in_airsim_thread(_rotate)
        self._last_yaw_deg = desired_yaw

    async def _get_collision_info(self) -> dict[str, Any] | None:
        """Fetch collision info from AirSim if available."""
        if not self.connected or not self.client:
            return None
        if not hasattr(self.client, "simGetCollisionInfo"):
            return None

        try:
            info = await _run_in_airsim_thread(
                self.client.simGetCollisionInfo, self.config.vehicle_name
            )
        except TypeError:
            info = await _run_in_airsim_thread(self.client.simGetCollisionInfo)
        except Exception as exc:
            if self._is_connection_error(exc):
                self.connected = False
                self.client = None
                return None
            logger.warning("collision_info_failed: %s", exc)
            return None

        has_collided = bool(getattr(info, "has_collided", False))
        timestamp = getattr(info, "time_stamp", None)
        object_name = getattr(info, "object_name", None)
        return {
            "has_collided": has_collided,
            "time_stamp": timestamp,
            "object_name": object_name,
        }

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "not connected" in message
            or "client is closed" in message
            or ("connection" in message and "closed" in message)
        )

    def _summarize_value(self, value: Any) -> str:
        if value is None or isinstance(value, (bool, int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            items = [self._summarize_value(v) for v in value[:6]]
            if len(value) > 6:
                items.append("...")
            return f"[{', '.join(items)}]"
        if hasattr(value, "x_val") and hasattr(value, "y_val") and hasattr(value, "z_val"):
            try:
                return f"Vector3({value.x_val:.2f}, {value.y_val:.2f}, {value.z_val:.2f})"
            except Exception:
                pass
        if all(hasattr(value, attr) for attr in ("w_val", "x_val", "y_val", "z_val")):
            try:
                return (
                    f"Quat({value.w_val:.2f}, {value.x_val:.2f}, "
                    f"{value.y_val:.2f}, {value.z_val:.2f})"
                )
            except Exception:
                pass
        if hasattr(value, "position") and hasattr(value, "orientation"):
            pos = getattr(value, "position", None)
            ori = getattr(value, "orientation", None)
            return f"Pose(pos={self._summarize_value(pos)}, ori={self._summarize_value(ori)})"
        if hasattr(value, "is_rate") and hasattr(value, "yaw_or_rate"):
            try:
                return f"YawMode(is_rate={bool(value.is_rate)}, yaw_or_rate={float(value.yaw_or_rate):.2f})"
            except Exception:
                pass
        text = repr(value)
        return text

    def _summarize_args(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[list[str], dict[str, str]]:
        max_len = int(self.config.trace_call_max_value_len)
        arg_list = [self._summarize_value(arg) for arg in args]
        kw_map = {key: self._summarize_value(val) for key, val in kwargs.items()}
        arg_list = [val[:max_len] if len(val) > max_len else val for val in arg_list]
        kw_map = {
            key: (val[:max_len] if len(val) > max_len else val) for key, val in kw_map.items()
        }
        return arg_list, kw_map

    def _record_airsim_call(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        duration_ms: float,
        ok: bool,
        error: str | None,
    ) -> None:
        if not self.config.trace_airsim_calls:
            return
        if ok and method not in TRACEABLE_AIRSIM_METHODS:
            return

        args_summary, kwargs_summary = self._summarize_args(args, kwargs)
        record = AirSimCallRecord(
            timestamp_ms=time.time() * 1000,
            method=method,
            args=args_summary,
            kwargs=kwargs_summary,
            duration_ms=duration_ms,
            ok=ok,
            error=error,
            vehicle_name=self.config.vehicle_name,
        )
        self._call_history.append(record)

        if not ok:
            logger.warning(
                "airsim_call_failed method=%s duration_ms=%.1f error=%s args=%s kwargs=%s",
                method,
                duration_ms,
                error,
                args_summary,
                kwargs_summary,
            )

    async def _call_airsim(
        self,
        method: str,
        *args: Any,
        timeout: float | None = None,
        record: bool = True,
        **kwargs: Any,
    ) -> Any:
        start = time.perf_counter()
        ok = True
        error = None
        try:
            call = _run_in_airsim_thread(method, *args, **kwargs)
            if timeout is not None:
                return await asyncio.wait_for(call, timeout=timeout)
            return await call
        except asyncio.TimeoutError:
            ok = False
            error = "timeout"
            raise
        except Exception as exc:
            ok = False
            error = str(exc)
            raise
        finally:
            if record:
                duration_ms = (time.perf_counter() - start) * 1000.0
                self._record_airsim_call(method, args, kwargs, duration_ms, ok, error)

    def get_call_history(self, limit: int = 200) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        history = list(self._call_history)
        return [record.to_dict() for record in history[-limit:]]

    async def _next_sequence(self) -> int:
        """Get next sequence number (thread-safe)."""
        async with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to AirSim simulator (non-blocking)."""
        global _airsim_client
        try:
            logger.info("Connecting to AirSim...")

            # Capture config values for use in thread
            host = self.config.host
            vehicle_name = self.config.vehicle_name
            camera_name = self.config.camera_name

            # Run ALL connection steps in a single thread call
            # The cosysairsim library uses msgpackrpc/tornado which has its own event loop
            def _connect_and_setup() -> tuple[
                airsim.MultirotorClient, str, list[str], str | None, float
            ]:
                global _airsim_client
                timeout_s = max(1.0, float(self.config.rpc_timeout_s))
                client = airsim.MultirotorClient(ip=host, timeout_value=timeout_s)
                client.confirmConnection()
                vehicles: list[str] = []
                selection_note: str | None = None
                if hasattr(client, "listVehicles"):
                    try:
                        vehicles = list(client.listVehicles())
                    except Exception as exc:
                        selection_note = f"listVehicles failed: {exc}"

                selected, selection_reason = _select_vehicle_name(vehicle_name, vehicles)
                if selection_reason:
                    selection_note = (
                        selection_reason
                        if not selection_note
                        else f"{selection_note}; {selection_reason}"
                    )
                client.enableApiControl(True, selected)

                # Get camera info in the same thread
                try:
                    camera_info = client.simGetCameraInfo(camera_name, selected)
                    fov = camera_info.fov
                except Exception:
                    fov = 90.0  # Default FOV

                # Set global client for use by other thread calls
                _airsim_client = client
                return client, selected, vehicles, selection_note, fov

            # Execute in dedicated thread
            loop = asyncio.get_running_loop()
            self.client, selected_name, vehicles, selection_note, fov = await loop.run_in_executor(
                _airsim_executor, _connect_and_setup
            )
            self.config.vehicle_name = selected_name
            self.vehicle_names = vehicles
            self.connected = True

            if vehicles:
                logger.info("AirSim vehicles detected: %s", vehicles)
            if selection_note:
                logger.warning("AirSim vehicle selection note: %s", selection_note)
            logger.info(
                "Connected to AirSim (camera FOV: %s, vehicle: %s)",
                fov,
                self.config.vehicle_name,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to connect to AirSim: {e}")
            await self._cleanup_client()
            self.client = None
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        await self.stop_streaming()

        await self._cleanup_client()
        self.client = None

        self.connected = False
        logger.info("Disconnected from AirSim")

    async def _cleanup_client(self) -> None:
        """Release API control and close any underlying RPC connections."""
        client = self.client
        if not client:
            return

        vehicle_name = self.config.vehicle_name

        def _cleanup() -> None:
            global _airsim_client
            try:
                try:
                    client.enableApiControl(False, vehicle_name)
                except TypeError:
                    client.enableApiControl(False)
            except Exception as exc:
                logger.warning("Failed to disable API control: %s", exc)

            for handle in (
                client,
                getattr(client, "client", None),
                getattr(client, "_client", None),
            ):
                if handle and hasattr(handle, "close"):
                    try:
                        handle.close()
                    except Exception as exc:
                        logger.debug("Failed to close AirSim handle: %s", exc)

            if _airsim_client is client:
                _airsim_client = None

        try:
            await _run_in_airsim_thread(_cleanup)
        except Exception as exc:
            logger.warning("Failed to cleanup AirSim client: %s", exc)

    async def ensure_connected(self) -> bool:
        """Check connection and reconnect if needed.

        Returns:
            True if connected (or successfully reconnected)
        """
        if not self.client:
            logger.info("No client, attempting to connect...")
            return await self.connect()

        # Try a simple ping to check if connection is alive (with timeout)
        try:
            await asyncio.wait_for(
                _run_in_airsim_thread("getMultirotorState", vehicle_name=self.config.vehicle_name),
                timeout=5.0,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Connection check timed out, reconnecting...")
            was_streaming = self._streaming
            await self.disconnect()
            connected = await self.connect()
            if connected and was_streaming:
                await self.start_streaming()
            return connected
        except Exception as e:
            logger.warning(f"Connection check failed: {e}, reconnecting...")
            was_streaming = self._streaming
            await self.disconnect()
            connected = await self.connect()
            if connected and was_streaming:
                await self.start_streaming()
            return connected

    async def get_position(self) -> Any | None:
        """Get current drone position (async-safe).

        Returns:
            Position object with x_val, y_val, z_val attributes, or None if unavailable.
        """
        if not self.client:
            return None
        try:
            state = await asyncio.wait_for(
                _run_in_airsim_thread("getMultirotorState", vehicle_name=self.config.vehicle_name),
                timeout=3.0,
            )
            return state.kinematics_estimated.position
        except Exception as e:
            if self._is_connection_error(e):
                self.connected = False
                self.client = None
            logger.warning(f"Failed to get position: {e}")
            return None

    async def set_weather(
        self, rain: float = 0.0, snow: float = 0.0, fog: float = 0.0, dust: float = 0.0
    ) -> bool:
        """Set weather conditions in the simulation."""
        if not self.connected or not self.client:
            return False

        try:
            await self._call_airsim("simEnableWeather", True)
            await self._call_airsim("simSetWeatherParameter", airsim.WeatherParameter.Rain, rain)
            await self._call_airsim("simSetWeatherParameter", airsim.WeatherParameter.Snow, snow)
            await self._call_airsim("simSetWeatherParameter", airsim.WeatherParameter.Fog, fog)
            await self._call_airsim("simSetWeatherParameter", airsim.WeatherParameter.Dust, dust)
            logger.info(
                "Weather set (realtime bridge)",
                extra={"rain": rain, "snow": snow, "fog": fog, "dust": dust},
            )
            return True
        except Exception as exc:
            logger.error("Failed to set weather (realtime bridge): %s", exc)
            return False

    async def set_time_of_day(
        self, hour: int = 12, is_enabled: bool = True, celestial_clock_speed: float = 1.0
    ) -> bool:
        """Set time of day in the simulation."""
        if not self.connected or not self.client:
            return False

        try:
            start_time = f"2024-06-15 {hour:02d}:00:00"
            await self._call_airsim(
                "simSetTimeOfDay",
                is_enabled=is_enabled,
                start_datetime=start_time,
                celestial_clock_speed=celestial_clock_speed,
            )
            logger.info("Time of day set (realtime bridge): %s:00", hour)
            return True
        except Exception as exc:
            logger.error("Failed to set time of day (realtime bridge): %s", exc)
            return False

    async def set_wind(self, speed_ms: float, direction_deg: float) -> bool:
        """Set wind vector in the simulation (if supported)."""
        if not self.connected or not self.client:
            return False
        if not hasattr(self.client, "simSetWind"):
            return False

        try:
            radians = math.radians(direction_deg)
            north = speed_ms * math.cos(radians)
            east = speed_ms * math.sin(radians)
            wind = airsim.Vector3r(north, east, 0.0)
            await self._call_airsim("simSetWind", wind)
            logger.info(
                "Wind set (realtime bridge)",
                extra={"speed_ms": speed_ms, "direction_deg": direction_deg},
            )
            return True
        except Exception as exc:
            logger.error("Failed to set wind (realtime bridge): %s", exc)
            return False

    async def set_vehicle_pose(
        self,
        north: float,
        east: float,
        down: float,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        ignore_collision: bool = True,
    ) -> bool:
        """Set vehicle pose in NED coordinates."""
        if not self.connected or not self.client:
            return False
        if not (
            hasattr(self.client, "simSetVehiclePose")
            or hasattr(self.client, "simSetVehiclePoseAsync")
        ):
            logger.warning("AirSim client does not support simSetVehiclePose")
            return False

        quat = airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)
        if hasattr(airsim, "to_quaternion"):
            try:
                quat = airsim.to_quaternion(
                    math.radians(pitch_deg),
                    math.radians(roll_deg),
                    math.radians(yaw_deg),
                )
            except Exception:
                pass

        pose = airsim.Pose(airsim.Vector3r(north, east, down), quat)
        vehicle_name = self.config.vehicle_name

        try:
            start = time.perf_counter()
            # Use sync version if available, otherwise async without .join()
            if hasattr(self.client, "simSetVehiclePose"):

                def _set_pose() -> None:
                    global _airsim_client
                    try:
                        _airsim_client.simSetVehiclePose(pose, ignore_collision, vehicle_name)
                    except TypeError:
                        _airsim_client.simSetVehiclePose(pose, ignore_collision)

                await _run_in_airsim_thread(_set_pose)
            elif hasattr(self.client, "simSetVehiclePoseAsync"):
                # Just send the command without waiting - use thread
                def _set_pose_async() -> None:
                    global _airsim_client
                    _airsim_client.simSetVehiclePoseAsync(pose, ignore_collision, vehicle_name)

                await _run_in_airsim_thread(_set_pose_async)
                await asyncio.sleep(0.1)  # Brief pause for pose to be applied
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._record_airsim_call(
                "simSetVehiclePose",
                (north, east, down, yaw_deg, pitch_deg, roll_deg, ignore_collision),
                {"vehicle_name": vehicle_name},
                duration_ms,
                True,
                None,
            )
            logger.info(
                "Vehicle pose set (realtime bridge)",
                extra={"north": north, "east": east, "down": down},
            )
            return True
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0 if "start" in locals() else 0.0
            self._record_airsim_call(
                "simSetVehiclePose",
                (north, east, down, yaw_deg, pitch_deg, roll_deg, ignore_collision),
                {"vehicle_name": vehicle_name},
                duration_ms,
                False,
                str(exc),
            )
            logger.error("Failed to set vehicle pose (realtime bridge): %s", exc)
            return False

    async def reset_position(
        self,
        altitude_agl: float = 5.0,
        reset_to_origin: bool = True,
    ) -> bool:
        """Reset drone position to a safe location above ground.

        Use this to recover from stuck/underground states.

        Args:
            altitude_agl: Altitude above ground level in meters
            reset_to_origin: If True, reset to origin (0,0). If False, reset at current XY.

        Returns:
            True if reset successful
        """
        if not self.client:
            connected = await self.connect()
            if not connected:
                logger.error("Cannot reset: failed to connect")
                return False

        try:
            logger.info(f"Resetting drone position (altitude: {altitude_agl}m AGL)")

            # First, try to get current position if we want to keep XY
            north, east = 0.0, 0.0
            if not reset_to_origin:
                try:
                    state = await _run_in_airsim_thread(
                        "getMultirotorState", vehicle_name=self.config.vehicle_name
                    )
                    north = state.kinematics_estimated.position.x_val
                    east = state.kinematics_estimated.position.y_val
                except Exception:
                    pass  # Use origin if we can't get position

            # Reset using simReset first to clear physics state
            try:
                await _run_in_airsim_thread("reset")
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"simReset failed (may be expected): {e}")

            # Reconnect after reset - wrap in thread
            def _reconnect():
                global _airsim_client
                client = airsim.MultirotorClient(ip=self.config.host)
                client.confirmConnection()
                client.enableApiControl(True, self.config.vehicle_name)
                _airsim_client = client
                return client

            await self._cleanup_client()
            self.client = await _run_in_airsim_thread(_reconnect)

            # Set position above ground (negative Z in NED = altitude)
            down = -abs(altitude_agl)  # Ensure negative (above ground)
            success = await self.set_vehicle_pose(
                north=north,
                east=east,
                down=down,
                yaw_deg=0.0,
                ignore_collision=True,
            )

            if success:
                logger.info(f"Drone reset to ({north:.1f}, {east:.1f}, {down:.1f})")
                # Small delay for physics to settle
                await asyncio.sleep(0.3)
                return True
            else:
                logger.error("Failed to set vehicle pose during reset")
                return False

        except Exception as exc:
            logger.exception("Reset position failed: %s", exc)
            return False

    async def spawn_object(
        self,
        object_name: str,
        asset_name: str,
        north: float,
        east: float,
        down: float,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        physics_enabled: bool = False,
    ) -> bool:
        """Spawn a 3D object in AirSim at NED coordinates.

        Args:
            object_name: Unique name for this object instance
            asset_name: Unreal asset name (e.g., 'Sphere', 'Cube', 'Cylinder')
            north: North position in meters
            east: East position in meters
            down: Down position in meters (negative = above ground)
            scale: Scale factors (x, y, z)
            physics_enabled: Whether to enable physics simulation

        Returns:
            True if object spawned successfully
        """
        if not self.client:
            logger.error("Cannot spawn object: not connected")
            return False

        try:
            pose = airsim.Pose(airsim.Vector3r(north, east, down), airsim.Quaternionr(0, 0, 0, 1))
            scale_vec = airsim.Vector3r(scale[0], scale[1], scale[2])

            # Use simSpawnObject if available
            if hasattr(self.client, "simSpawnObject"):
                result = await self._call_airsim(
                    "simSpawnObject",
                    object_name,
                    asset_name,
                    pose,
                    scale_vec,
                    physics_enabled,
                    timeout=5.0,
                )
                if result:
                    logger.info(
                        f"Spawned object '{object_name}' at NED({north:.1f}, {east:.1f}, {down:.1f})"
                    )
                    return True
                else:
                    logger.warning(f"simSpawnObject returned False for '{object_name}'")
                    return False
            else:
                logger.warning("simSpawnObject not available in this AirSim version")
                return False

        except asyncio.TimeoutError:
            logger.error(f"Spawn object '{object_name}' timed out")
            return False
        except Exception as exc:
            logger.error(f"Failed to spawn object '{object_name}': {exc}")
            return False

    async def destroy_object(self, object_name: str) -> bool:
        """Destroy/remove a spawned object from AirSim.

        Args:
            object_name: Name of the object to destroy

        Returns:
            True if object destroyed successfully
        """
        if not self.client:
            return False

        try:
            if hasattr(self.client, "simDestroyObject"):
                result = await self._call_airsim(
                    "simDestroyObject",
                    object_name,
                    timeout=5.0,
                )
                if result:
                    logger.info(f"Destroyed object '{object_name}'")
                return result
            return False
        except Exception as exc:
            logger.warning(f"Failed to destroy object '{object_name}': {exc}")
            return False

    async def list_available_assets(self) -> list[str]:
        """List all available spawnable assets in the current environment.

        Returns:
            List of asset names that can be used with spawn_object
        """
        if not self.client:
            return []

        try:
            if hasattr(self.client, "simListAssets"):
                assets = await asyncio.wait_for(_run_in_airsim_thread("simListAssets"), timeout=5.0)
                logger.info(f"Found {len(assets)} available assets")
                return list(assets)
            else:
                # Return known primitives as fallback
                return ["Cube", "Sphere", "Cylinder", "Cone"]
        except Exception as exc:
            logger.warning(f"Failed to list assets: {exc}")
            return ["Cube", "Sphere", "Cylinder", "Cone"]

    async def spawn_landing_dock(
        self,
        north: float = 0.0,
        east: float = 0.0,
        down: float = 0.0,
        with_lights: bool = True,
        use_custom_assets: bool = True,
    ) -> dict:
        """Spawn an enhanced landing dock with visual markers.

        If custom assets are available (downloaded via asset_manager), uses
        realistic 3D models. Otherwise falls back to primitives.

        Creates a multi-part dock consisting of:
        - Main landing pad (custom helipad model or flat cylinder)
        - Center marker (sphere)
        - Corner markers (4 small spheres)
        - Optional landing lights (if supported)

        Args:
            north: North position in NED
            east: East position in NED
            down: Down position (negative = above ground)
            with_lights: Whether to add lights (if available)
            use_custom_assets: Try to use custom 3D models if available

        Returns:
            Dict with spawn results for each component
        """
        results = {
            "main_pad": False,
            "center_marker": False,
            "corner_markers": [],
            "lights": [],
            "total_success": 0,
            "using_custom_asset": False,
        }

        # Try to use custom helipad asset if available
        helipad_asset_name = "Cylinder"  # Default fallback
        helipad_scale = (4.0, 4.0, 0.3)

        if use_custom_assets and ASSET_MANAGER_AVAILABLE:
            try:
                manager = get_asset_manager()
                asset_name, metadata = manager.get_asset_for_spawn(AssetType.HELIPAD)
                if asset_name and asset_name != "Cylinder":
                    helipad_asset_name = asset_name
                    # Custom assets typically need different scale
                    if metadata and metadata.scale:
                        helipad_scale = metadata.scale
                    else:
                        helipad_scale = (1.0, 1.0, 1.0)
                    results["using_custom_asset"] = True
                    logger.info(f"Using custom helipad asset: {asset_name}")
            except Exception as e:
                logger.debug(f"Could not get custom helipad asset: {e}")

        # Main landing pad
        results["main_pad"] = await self.spawn_object(
            "Dock_MainPad",
            helipad_asset_name,
            north,
            east,
            down,
            scale=helipad_scale,
            physics_enabled=False,
        )
        if results["main_pad"]:
            results["total_success"] += 1

        # Center marker - small elevated sphere (landing target)
        results["center_marker"] = await self.spawn_object(
            "Dock_CenterMarker",
            "Sphere",
            north,
            east,
            down - 0.5,  # Slightly above pad
            scale=(0.5, 0.5, 0.5),
            physics_enabled=False,
        )
        if results["center_marker"]:
            results["total_success"] += 1

        # Corner markers - 4 small spheres at corners for visual reference
        corner_offset = 3.0  # Distance from center
        corners = [
            (north + corner_offset, east + corner_offset, "NE"),
            (north + corner_offset, east - corner_offset, "NW"),
            (north - corner_offset, east + corner_offset, "SE"),
            (north - corner_offset, east - corner_offset, "SW"),
        ]

        for cn, ce, label in corners:
            success = await self.spawn_object(
                f"Dock_Corner_{label}",
                "Sphere",
                cn,
                ce,
                down - 0.3,
                scale=(0.3, 0.3, 0.3),
                physics_enabled=False,
            )
            results["corner_markers"].append({"label": label, "success": success})
            if success:
                results["total_success"] += 1

        # Add landing lights if requested and available
        if with_lights:
            # Try to spawn point lights at corners
            for cn, ce, label in corners:
                try:
                    light_success = await self.spawn_object(
                        f"Dock_Light_{label}",
                        "PointLightBP",
                        cn,
                        ce,
                        down - 1.0,  # Above corner markers
                        scale=(1.0, 1.0, 1.0),
                        physics_enabled=False,
                    )
                    if light_success:
                        results["lights"].append({"label": label, "success": True})
                        results["total_success"] += 1
                        # Try to set light intensity
                        try:
                            await self._call_airsim(
                                "simSetWorldLightIntensity",
                                f"Dock_Light_{label}",
                                100.0,
                                timeout=5.0,
                            )
                        except Exception:
                            try:
                                await self._call_airsim(
                                    "simSetLightIntensity",
                                    f"Dock_Light_{label}",
                                    100.0,
                                    timeout=5.0,
                                )
                            except Exception:
                                pass  # Light intensity API may not be available
                except Exception as e:
                    logger.debug(f"Could not spawn light {label}: {e}")

        logger.info(
            f"Landing dock spawned: {results['total_success']} components at "
            f"NED({north:.1f}, {east:.1f}, {down:.1f})"
        )
        return results

    async def spawn_wind_turbine_marker(
        self,
        name: str,
        north: float,
        east: float,
        down: float = -5.0,
        height: float = 30.0,
    ) -> dict:
        """Spawn a wind turbine representation using primitives.

        Creates a tall cylinder (tower) with a sphere on top (hub).

        Args:
            name: Unique name for this turbine
            north: North position in NED
            east: East position in NED
            down: Base position (negative = above ground)
            height: Height of the tower

        Returns:
            Dict with spawn results
        """
        results = {"tower": False, "hub": False, "blades": []}
        safe_name = name.replace(" ", "_").replace("-", "_")

        # Tower - tall thin cylinder
        results["tower"] = await self.spawn_object(
            f"Turbine_{safe_name}_Tower",
            "Cylinder",
            north,
            east,
            down - height / 2,
            scale=(1.0, 1.0, height / 2),  # Tall and thin
            physics_enabled=False,
        )

        # Hub - sphere at top
        results["hub"] = await self.spawn_object(
            f"Turbine_{safe_name}_Hub",
            "Sphere",
            north,
            east,
            down - height,
            scale=(2.0, 2.0, 2.0),
            physics_enabled=False,
        )

        # Blades - 3 thin cubes rotated (simplified representation)
        blade_length = 15.0
        for i, angle in enumerate([0, 120, 240]):
            rad = math.radians(angle)
            blade_n = north + math.cos(rad) * blade_length / 2
            blade_e = east + math.sin(rad) * blade_length / 2
            success = await self.spawn_object(
                f"Turbine_{safe_name}_Blade{i}",
                "Cube",
                blade_n,
                blade_e,
                down - height,
                scale=(blade_length, 0.5, 0.2),
                physics_enabled=False,
            )
            results["blades"].append(success)

        return results

    async def spawn_solar_panel_marker(
        self,
        name: str,
        north: float,
        east: float,
        down: float = -2.0,
        rows: int = 2,
        cols: int = 3,
    ) -> dict:
        """Spawn a solar panel array representation using primitives.

        Creates a grid of flat cubes representing solar panels.

        Args:
            name: Unique name for this array
            north: North position in NED
            east: East position in NED
            down: Base position (negative = above ground)
            rows: Number of panel rows
            cols: Number of panel columns

        Returns:
            Dict with spawn results
        """
        results = {"panels": [], "success_count": 0}
        safe_name = name.replace(" ", "_").replace("-", "_")

        panel_width = 2.0
        panel_depth = 1.5
        spacing = 0.5

        for row in range(rows):
            for col in range(cols):
                panel_n = north + row * (panel_depth + spacing)
                panel_e = east + col * (panel_width + spacing)

                success = await self.spawn_object(
                    f"Solar_{safe_name}_Panel_{row}_{col}",
                    "Cube",
                    panel_n,
                    panel_e,
                    down,
                    scale=(panel_depth, panel_width, 0.1),  # Flat
                    physics_enabled=False,
                )
                results["panels"].append({
                    "row": row,
                    "col": col,
                    "success": success,
                })
                if success:
                    results["success_count"] += 1

        return results

    async def spawn_scene_objects(
        self,
        dock_ned: tuple[float, float, float] | None = None,
        assets: list[dict] | None = None,
        geo_ref=None,
    ) -> dict:
        """Spawn dock and asset markers in AirSim.

        Now uses enhanced visual representations for dock and assets.

        Args:
            dock_ned: Dock position in NED (or None for origin)
            assets: List of asset dicts with lat/lon/name
            geo_ref: GeoReference for GPS to NED conversion

        Returns:
            Dict with spawn results
        """
        results = {"dock": {}, "assets": [], "errors": []}

        # Spawn enhanced dock at origin (or specified position)
        dock_n, dock_e, dock_d = dock_ned or (0.0, 0.0, 0.0)
        results["dock"] = await self.spawn_landing_dock(dock_n, dock_e, dock_d, with_lights=True)

        # Spawn asset markers with enhanced visuals
        if assets and geo_ref:
            for asset in assets:
                try:
                    lat = asset.get("latitude")
                    lon = asset.get("longitude")
                    name = asset.get("name", asset.get("asset_id", "unknown"))
                    asset_type = asset.get("asset_type", "unknown").lower()

                    if lat is None or lon is None:
                        continue

                    # Convert GPS to NED
                    north, east, down = geo_ref.gps_to_ned(lat, lon, geo_ref.altitude)

                    # Choose representation based on asset type
                    if "turbine" in asset_type or "wind" in asset_type:
                        spawn_result = await self.spawn_wind_turbine_marker(
                            name, north, east, down=-5.0, height=30.0
                        )
                        results["assets"].append({
                            "name": name,
                            "type": "wind_turbine",
                            "result": spawn_result,
                            "ned": (north, east, -5.0),
                        })
                    elif "solar" in asset_type or "panel" in asset_type:
                        spawn_result = await self.spawn_solar_panel_marker(
                            name, north, east, down=-2.0
                        )
                        results["assets"].append({
                            "name": name,
                            "type": "solar_array",
                            "result": spawn_result,
                            "ned": (north, east, -2.0),
                        })
                    else:
                        # Generic marker - sphere for unknown types
                        success = await self.spawn_object(
                            f"Asset_{name.replace(' ', '_')}",
                            "Sphere",
                            north,
                            east,
                            -5.0,
                            scale=(3.0, 3.0, 3.0),
                            physics_enabled=False,
                        )
                        results["assets"].append({
                            "name": name,
                            "type": "generic",
                            "success": success,
                            "ned": (north, east, -5.0),
                        })
                except Exception as e:
                    results["errors"].append(f"{name}: {e}")

        dock_success = results["dock"].get("total_success", 0)
        asset_count = len(results["assets"])
        logger.info(f"Scene spawned: dock={dock_success} components, " f"assets={asset_count}")
        return results

    # -------------------------------------------------------------------------
    # Flight Control Methods
    # -------------------------------------------------------------------------

    async def arm(self) -> bool:
        """Arm the drone motors.

        Returns:
            True if successfully armed
        """
        # Ensure we have a valid connection
        if not await self.ensure_connected():
            logger.error("Arm failed: could not connect")
            return False

        try:
            # Enable API control first (with timeout)
            await self._call_airsim(
                "enableApiControl",
                True,
                self.config.vehicle_name,
                timeout=5.0,
            )
            logger.info("API control enabled")

            # Then arm (with timeout)
            await self._call_airsim(
                "armDisarm",
                True,
                self.config.vehicle_name,
                timeout=5.0,
            )
            logger.info("Drone armed")
            return True
        except asyncio.TimeoutError:
            logger.error("Arm timed out")
            return False
        except Exception as exc:
            logger.exception("Failed to arm drone: %s", exc)
            return False

    async def disarm(self) -> bool:
        """Disarm the drone motors.

        Returns:
            True if successfully disarmed
        """
        if not self.client:
            return False

        try:
            await self._call_airsim(
                "armDisarm",
                False,
                self.config.vehicle_name,
                timeout=5.0,
            )
            logger.info("Drone disarmed")
            return True
        except asyncio.TimeoutError:
            logger.error("Disarm timed out")
            return False
        except Exception as exc:
            logger.error("Failed to disarm drone: %s", exc)
            return False

    async def takeoff(self, altitude: float = 10.0, timeout: float = 30.0) -> bool:
        """Take off to specified altitude.

        Args:
            altitude: Target altitude in meters AGL (above ground level)
            timeout: Maximum time to wait for takeoff completion

        Returns:
            True if takeoff completed successfully
        """
        # Ensure we have a valid connection
        if not await self.ensure_connected():
            logger.error("Takeoff failed: could not connect")
            return False

        try:
            # Arm first
            armed = await self.arm()
            if not armed:
                logger.error("Failed to arm drone before takeoff")
                return False

            logger.info(f"Taking off to {altitude}m AGL")

            # Start takeoff (non-blocking) - wrap in thread to avoid blocking
            await self._call_airsim(
                "takeoffAsync",
                timeout_sec=timeout,
                vehicle_name=self.config.vehicle_name,
                timeout=5.0,
            )

            # Wait for takeoff to complete by checking altitude directly (with timeouts)
            start_time = time.time()
            timeout_count = 0
            while time.time() - start_time < timeout:
                try:
                    # Guard against None client
                    if self.client is None:
                        logger.warning("Client became None during takeoff, reconnecting...")
                        if not await self.connect():
                            logger.error("Failed to reconnect during takeoff")
                            return False
                        continue

                    # Use direct API call with timeout
                    state = await asyncio.wait_for(
                        _run_in_airsim_thread(
                            "getMultirotorState", vehicle_name=self.config.vehicle_name
                        ),
                        timeout=5.0,  # Increased timeout
                    )
                    current_alt = -state.kinematics_estimated.position.z_val
                    if current_alt > 1.0:  # At least 1m off ground
                        logger.info(f"Takeoff complete, altitude: {current_alt:.1f}m")
                        return True
                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count <= 2:  # Only log first couple warnings
                        logger.warning("Position check timed out during takeoff")
                except Exception as e:
                    logger.warning(f"Error checking altitude: {e}")
                await asyncio.sleep(0.5)  # Slower polling

            # Check final altitude (with timeout)
            try:
                state = await asyncio.wait_for(
                    _run_in_airsim_thread(
                        "getMultirotorState", vehicle_name=self.config.vehicle_name
                    ),
                    timeout=3.0,
                )
                current_alt = -state.kinematics_estimated.position.z_val
                logger.info(f"Takeoff timeout, final altitude: {current_alt:.1f}m")
                return current_alt > 0.5
            except (asyncio.TimeoutError, Exception):
                logger.warning("Takeoff timed out, could not verify altitude")
                return True  # Probably still taking off

        except Exception as exc:
            logger.exception("Takeoff failed: %s", exc)
            return False

    async def land(self, timeout: float = 30.0) -> bool:
        """Land at current position.

        Args:
            timeout: Maximum time to wait for landing

        Returns:
            True if landing completed successfully
        """
        if not await self.ensure_connected():
            logger.error("Land failed: could not connect")
            return False

        try:
            logger.info("Landing...")

            # Soft descent to avoid hard drops near the ground.
            try:
                state = await asyncio.wait_for(
                    _run_in_airsim_thread(
                        "getMultirotorState", vehicle_name=self.config.vehicle_name
                    ),
                    timeout=3.0,
                )
                pos = state.kinematics_estimated.position
                current_alt = -pos.z_val
                soft_alt = max(0.5, float(self.config.landing_soft_altitude_m))
                soft_velocity = max(0.2, float(self.config.landing_soft_velocity_ms))
                soft_timeout = min(float(self.config.landing_soft_timeout_s), timeout)
                if current_alt > soft_alt + 0.2:
                    await self.move_to_position(
                        pos.x_val,
                        pos.y_val,
                        -soft_alt,
                        velocity=soft_velocity,
                        timeout=soft_timeout,
                    )
                    await asyncio.sleep(0.5)
            except Exception as exc:
                logger.warning("Soft landing pre-descent skipped: %s", exc)

            # Start land command (non-blocking) - wrap in thread
            await self._call_airsim(
                "landAsync",
                timeout_sec=timeout,
                vehicle_name=self.config.vehicle_name,
                timeout=5.0,
            )

            # Wait for landing by polling altitude with timeouts
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    state = await asyncio.wait_for(
                        _run_in_airsim_thread(
                            "getMultirotorState", vehicle_name=self.config.vehicle_name
                        ),
                        timeout=3.0,
                    )
                    # Check if close to ground (z close to 0 in NED)
                    current_alt = -state.kinematics_estimated.position.z_val
                    if current_alt < 0.5:  # Less than 0.5m from ground
                        logger.info(f"Landing complete, altitude: {current_alt:.2f}m")
                        await self.disarm()
                        return True
                except asyncio.TimeoutError:
                    logger.warning("Position check timed out during landing")
                except Exception as e:
                    logger.warning(f"Error checking altitude during landing: {e}")
                await asyncio.sleep(0.3)

            try:
                state = await asyncio.wait_for(
                    _run_in_airsim_thread(
                        "getMultirotorState", vehicle_name=self.config.vehicle_name
                    ),
                    timeout=3.0,
                )
                current_alt = -state.kinematics_estimated.position.z_val
                if state.landed_state == airsim.LandedState.Landed or current_alt < 0.5:
                    logger.info("Landing complete, altitude: %.2fm", current_alt)
                else:
                    logger.warning("Landing timed out")
            except Exception as exc:
                logger.warning("Landing timed out: %s", exc)
            await self.disarm()
            return True  # Probably landed anyway

        except Exception as exc:
            logger.error("Landing failed: %s", exc)
            return False

    async def hover(self) -> bool:
        """Hold current position (hover in place).

        Returns:
            True if hover command sent successfully
        """
        if not self.connected or not self.client:
            return False

        try:
            # Send hover command (non-blocking) - wrap in thread
            await self._call_airsim(
                "hoverAsync",
                vehicle_name=self.config.vehicle_name,
                timeout=5.0,
            )
            logger.debug("Hover command sent")
            return True

        except Exception as exc:
            logger.error("Hover failed: %s", exc)
            return False

    async def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 5.0,
        timeout: float = 60.0,
        drivetrain: int = 0,
        yaw_mode: dict | None = None,
        max_distance: float = 2000.0,  # Maximum distance from origin in meters
    ) -> bool:
        """Move to a position in NED coordinates.

        Args:
            x: North position in meters (positive = north)
            y: East position in meters (positive = east)
            z: Down position in meters (negative = up, so -30 = 30m altitude)
            velocity: Flight speed in m/s
            timeout: Maximum time to wait for arrival
            drivetrain: 0=MaxDegreeOfFreedom, 1=ForwardOnly
            yaw_mode: Optional yaw mode dict, defaults to angle=0, is_rate=False
            max_distance: Maximum distance from origin (boundary check)

        Returns:
            True if destination reached
        """
        # Boundary check - prevent flying off the map
        distance_from_origin = math.sqrt(x * x + y * y)
        if distance_from_origin > max_distance:
            logger.error(
                f"Target position ({x:.1f}, {y:.1f}) is {distance_from_origin:.1f}m from origin, "
                f"exceeds max distance of {max_distance}m. Aborting move."
            )
            return False

        # Altitude check - prevent flying underground or too high
        if z > 0:
            logger.warning(
                f"Target altitude {z} is underground (positive Z in NED), clamping to ground level"
            )
            z = -1.0  # Just above ground
        if z < -500:
            logger.warning(f"Target altitude {-z}m is too high, clamping to 500m")
            z = -500.0

        # Ensure we have a valid connection
        if not await self.ensure_connected():
            logger.error("Move failed: could not connect")
            return False

        try:
            # Ensure API control is enabled (with timeout)
            await self._call_airsim(
                "enableApiControl",
                True,
                self.config.vehicle_name,
                timeout=5.0,
            )

            # Get current position to calculate yaw toward target (with timeout)
            state = await asyncio.wait_for(
                _run_in_airsim_thread("getMultirotorState", vehicle_name=self.config.vehicle_name),
                timeout=5.0,
            )
            current_pos = state.kinematics_estimated.position

            # Calculate yaw angle to face the target (in degrees)
            dx = x - current_pos.x_val
            dy = y - current_pos.y_val
            yaw_rad = math.atan2(dy, dx)
            yaw_deg = math.degrees(yaw_rad)

            if isinstance(drivetrain, airsim.DrivetrainType):
                drivetrain_mode = drivetrain
            else:
                drivetrain_mode = (
                    airsim.DrivetrainType.ForwardOnly
                    if int(drivetrain) == 1
                    else airsim.DrivetrainType.MaxDegreeOfFreedom
                )

            yaw_mode_obj = None
            if yaw_mode is None:
                yaw_mode_obj = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)
            elif isinstance(yaw_mode, airsim.YawMode):
                yaw_mode_obj = yaw_mode
            elif isinstance(yaw_mode, dict):
                yaw_mode_obj = airsim.YawMode(
                    is_rate=bool(yaw_mode.get("is_rate", False)),
                    yaw_or_rate=float(yaw_mode.get("yaw_or_rate", yaw_deg)),
                )
            else:
                yaw_mode_obj = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

            logger.info(
                "Moving to NED (%.1f, %.1f, %.1f) at %.1f m/s, yaw=%.1f°, drivetrain=%s",
                x,
                y,
                z,
                velocity,
                yaw_deg,
                drivetrain_mode,
            )

            # Start the move command (non-blocking) - wrap in thread
            await self._call_airsim(
                "moveToPositionAsync",
                x,
                y,
                z,
                velocity,
                timeout_sec=timeout,
                drivetrain=drivetrain_mode,
                yaw_mode=yaw_mode_obj,
                vehicle_name=self.config.vehicle_name,
                timeout=5.0,
            )

            # Wait for arrival by polling position with timeouts
            start_time = time.time()
            timeout_warnings = 0
            while time.time() - start_time < timeout:
                try:
                    state = await asyncio.wait_for(
                        _run_in_airsim_thread(
                            "getMultirotorState", vehicle_name=self.config.vehicle_name
                        ),
                        timeout=5.0,  # Increased timeout for slower connections
                    )
                    pos = state.kinematics_estimated.position
                    dx = abs(pos.x_val - x)
                    dy = abs(pos.y_val - y)
                    dz = abs(pos.z_val - z)
                    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                    if dist < 3.0:  # Within 3 meters
                        logger.info(f"Arrived at ({x:.1f}, {y:.1f}, {z:.1f})")
                        return True
                    # Log progress occasionally
                    elapsed = time.time() - start_time
                    if int(elapsed) % 5 == 0 and elapsed > 1:
                        logger.debug(
                            f"Moving... dist={dist:.1f}m, pos=({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})"
                        )
                except asyncio.TimeoutError:
                    timeout_warnings += 1
                    if timeout_warnings <= 3:  # Only log first few warnings
                        logger.warning("Position check timed out, continuing...")
                except Exception as e:
                    if self._is_connection_error(e):
                        logger.warning("Position check failed: AirSim client disconnected")
                        self.connected = False
                        self.client = None
                        return False
                    logger.warning(f"Error checking position: {e}")
                await asyncio.sleep(1.0)  # Slower polling to reduce load

            logger.info(
                f"Move timed out after {timeout}s (position check timeouts: {timeout_warnings})"
            )
            return True  # Return true anyway - drone is moving

        except Exception as exc:
            logger.exception("Move to position failed: %s", exc)
            return False

    async def move_by_velocity(
        self, vx: float, vy: float, vz: float, duration: float = 1.0
    ) -> bool:
        """Move by velocity vector for a duration.

        Args:
            vx: North velocity in m/s
            vy: East velocity in m/s
            vz: Down velocity in m/s (negative = up)
            duration: How long to maintain velocity in seconds

        Returns:
            True if command executed
        """
        if not self.connected or not self.client:
            return False

        try:
            # Send velocity command (non-blocking) - wrap in thread
            await self._call_airsim(
                "moveByVelocityAsync",
                vx,
                vy,
                vz,
                duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                vehicle_name=self.config.vehicle_name,
                timeout=5.0,
            )

            # Wait for the duration
            await asyncio.sleep(duration + 0.1)
            return True

        except Exception as exc:
            logger.error("Move by velocity failed: %s", exc)
            return False

    async def return_to_launch(self, altitude: float = 30.0) -> bool:
        """Return to launch position (origin) and land.

        Args:
            altitude: Altitude to fly at during return (meters AGL)

        Returns:
            True if RTL completed
        """
        if not self.connected or not self.client:
            return False

        try:
            logger.info("Returning to launch...")

            # Go to origin at safe altitude
            await self.move_to_position(0, 0, -altitude, velocity=5.0)

            # Descend and land
            await self.land()

            logger.info("Return to launch complete")
            return True

        except Exception as exc:
            logger.error("Return to launch failed: %s", exc)
            return False

    async def orbit(
        self,
        center_x: float,
        center_y: float,
        center_z: float,
        radius: float = 20.0,
        velocity: float = 3.0,
        duration: float = 30.0,
        clockwise: bool = True,
    ) -> bool:
        """Orbit around a point (for inspection).

        The drone faces INWARD toward the center during the orbit,
        allowing the forward camera to inspect the target.

        Args:
            center_x: Center north position (NED)
            center_y: Center east position (NED)
            center_z: Center down position (NED, negative = altitude)
            radius: Orbit radius in meters
            velocity: Tangential velocity in m/s
            duration: How long to orbit in seconds
            clockwise: True for clockwise, False for counter-clockwise

        Returns:
            True if orbit completed
        """
        # Ensure we have a valid connection
        if not await self.ensure_connected():
            logger.error("Orbit failed: could not connect")
            return False

        try:
            logger.info(
                f"Starting orbit: center=({center_x:.1f}, {center_y:.1f}, {center_z:.1f}), "
                f"radius={radius}m, duration={duration}s"
            )

            start_time = time.time()
            angular_velocity = velocity / radius
            if not clockwise:
                angular_velocity = -angular_velocity

            # Get starting angle based on current position (with timeout)
            try:
                state = await asyncio.wait_for(
                    _run_in_airsim_thread(
                        "getMultirotorState", vehicle_name=self.config.vehicle_name
                    ),
                    timeout=3.0,
                )
                pos = state.kinematics_estimated.position
                dx = pos.x_val - center_x
                dy = pos.y_val - center_y
                start_angle = math.atan2(dy, dx)
            except (asyncio.TimeoutError, Exception):
                start_angle = 0.0

            while (time.time() - start_time) < duration:
                elapsed = time.time() - start_time
                angle = start_angle + angular_velocity * elapsed

                # Calculate orbit position
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)

                # Calculate yaw to face CENTER (inward) for inspection
                # Angle from orbit point to center is opposite of orbit angle
                yaw_to_center = math.degrees(angle + math.pi)  # Face inward

                # Move to orbit point with yaw facing center - wrap in thread
                await _run_in_airsim_thread(
                    "moveToPositionAsync",
                    x,
                    y,
                    center_z,
                    velocity,
                    timeout_sec=2.0,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_to_center),
                    vehicle_name=self.config.vehicle_name,
                )

                await asyncio.sleep(0.5)

            # Hover at end
            await self.hover()
            logger.info("Orbit complete")
            return True

        except Exception as exc:
            logger.error("Orbit failed: %s", exc)
            return False

    async def go_home(self) -> bool:
        """Convenience method to return home (alias for return_to_launch)."""
        return await self.return_to_launch()

    # -------------------------------------------------------------------------
    # Synchronized Data Fetching
    # -------------------------------------------------------------------------

    async def get_synchronized_state(self) -> TelemetryFrame | None:
        """Fetch pose + IMU + state in parallel for synchronized telemetry.

        Returns:
            TelemetryFrame with all sensor data, or None on error
        """
        if not self.connected or not self.client:
            return None

        start_time = time.perf_counter()
        seq = await self._next_sequence()

        try:
            # Parallel fetch of all telemetry sources
            tasks = [
                _run_in_airsim_thread("simGetVehiclePose", self.config.vehicle_name),
                _run_in_airsim_thread("getMultirotorState", self.config.vehicle_name),
            ]

            if self.config.include_imu:
                tasks.append(
                    _run_in_airsim_thread(
                        "getImuData", imu_name="", vehicle_name=self.config.vehicle_name
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Telemetry fetch error: {r}")
                    return None

            pose_raw, state_raw = results[0], results[1]
            imu_raw = results[2] if len(results) > 2 else None

            # Convert pose
            pose = PoseData(
                timestamp_ns=int(time.time_ns()),
                position=Vector3(
                    x=pose_raw.position.x_val, y=pose_raw.position.y_val, z=pose_raw.position.z_val
                ),
                orientation=Quaternion(
                    w=pose_raw.orientation.w_val,
                    x=pose_raw.orientation.x_val,
                    y=pose_raw.orientation.y_val,
                    z=pose_raw.orientation.z_val,
                ),
                linear_velocity=Vector3(
                    x=state_raw.kinematics_estimated.linear_velocity.x_val,
                    y=state_raw.kinematics_estimated.linear_velocity.y_val,
                    z=state_raw.kinematics_estimated.linear_velocity.z_val,
                ),
                angular_velocity=Vector3(
                    x=state_raw.kinematics_estimated.angular_velocity.x_val,
                    y=state_raw.kinematics_estimated.angular_velocity.y_val,
                    z=state_raw.kinematics_estimated.angular_velocity.z_val,
                ),
            )

            # Convert IMU if available
            imu = None
            if imu_raw:
                imu = IMUData(
                    timestamp_ns=imu_raw.time_stamp,
                    linear_acceleration=Vector3(
                        x=imu_raw.linear_acceleration.x_val,
                        y=imu_raw.linear_acceleration.y_val,
                        z=imu_raw.linear_acceleration.z_val,
                    ),
                    angular_velocity=Vector3(
                        x=imu_raw.angular_velocity.x_val,
                        y=imu_raw.angular_velocity.y_val,
                        z=imu_raw.angular_velocity.z_val,
                    ),
                    orientation=Quaternion(
                        w=imu_raw.orientation.w_val,
                        x=imu_raw.orientation.x_val,
                        y=imu_raw.orientation.y_val,
                        z=imu_raw.orientation.z_val,
                    ),
                )

            # Determine landed state
            landed_state = "landed"
            if hasattr(state_raw, "landed_state"):
                if state_raw.landed_state == airsim.LandedState.Flying:
                    landed_state = "flying"
                elif state_raw.landed_state == airsim.LandedState.Landed:
                    landed_state = "landed"

            speed_m_s = math.sqrt(
                pose.linear_velocity.x**2 + pose.linear_velocity.y**2 + pose.linear_velocity.z**2
            )
            battery_percent = self._read_airsim_battery_percent(state_raw)
            if battery_percent is None:
                battery_percent = self._simulate_battery(speed_m_s, landed_state)
            else:
                self._battery_percent = battery_percent
                self._battery_last_update = time.time()

            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_latency_stats(latency_ms)

            return TelemetryFrame(
                sequence=seq,
                server_timestamp_ms=time.time() * 1000,
                airsim_timestamp_ns=int(time.time_ns()),
                pose=pose,
                imu=imu,
                battery_percent=battery_percent,
                landed_state=landed_state,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Failed to get synchronized state: {e}")
            return None

    async def capture_frame_synchronized(self, include_image: bool = True) -> FrameCaptureResult:
        """Capture image + pose + telemetry in synchronized manner.

        Args:
            include_image: Whether to capture image (set False for telemetry-only)

        Returns:
            FrameCaptureResult with synchronized data
        """
        if not self.connected or not self.client:
            return FrameCaptureResult(
                sequence=await self._next_sequence(),
                success=False,
                timestamp=datetime.now(),
                server_timestamp_ms=time.time() * 1000,
                error="Not connected",
            )

        start_time = time.perf_counter()
        seq = await self._next_sequence()

        try:
            # Build parallel tasks
            tasks = []

            # Image capture
            if include_image:
                tasks.append(
                    _run_in_airsim_thread(
                        "simGetImages",
                        [
                            airsim.ImageRequest(
                                self.config.camera_name,
                                airsim.ImageType.Scene,
                                pixels_as_float=False,
                                compress=self.config.compress,
                            )
                        ],
                        self.config.vehicle_name,
                    )
                )

            # Telemetry (always fetch)
            tasks.append(self.get_synchronized_state())

            # Execute in parallel
            t0 = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Parse results
            image_response = None
            telemetry = None
            image_path = None

            if include_image:
                if isinstance(results[0], Exception):
                    raise results[0]
                image_response = results[0]
                telemetry = results[1] if len(results) > 1 else None
            else:
                telemetry = results[0]

            capture_latency = (time.perf_counter() - t0) * 1000

            # Save image if requested
            if image_response and len(image_response) > 0 and self.config.save_images:
                response = image_response[0]
                if self.config.compress:
                    img_array = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img = Image.open(io.BytesIO(img_array))
                else:
                    img_array = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(
                        response.height, response.width, 3
                    )
                    img = Image.fromarray(img_array)

                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                filename = f"frame_{seq:08d}.jpg"
                image_path = self.config.output_dir / filename
                img.save(image_path, quality=85)

            total_latency = (time.perf_counter() - start_time) * 1000
            self._total_frames += 1

            return FrameCaptureResult(
                sequence=seq,
                success=True,
                timestamp=datetime.now(),
                server_timestamp_ms=time.time() * 1000,
                image_path=image_path,
                telemetry=telemetry,
                capture_latency_ms=capture_latency,
                total_latency_ms=total_latency,
            )

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return FrameCaptureResult(
                sequence=seq,
                success=False,
                timestamp=datetime.now(),
                server_timestamp_ms=time.time() * 1000,
                error=str(e),
            )

    def _get_mapping_session_dir(self, output_dir: Path | None) -> Path:
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "frames").mkdir(parents=True, exist_ok=True)
            return output_dir

        if self._mapping_session_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = self.config.mapping_output_dir / f"sequence_{stamp}"
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "frames").mkdir(parents=True, exist_ok=True)
            self._mapping_session_dir = session_dir

        return self._mapping_session_dir

    @staticmethod
    def _compute_intrinsics(width: int, height: int, fov_deg: float) -> dict[str, float]:
        fov_rad = math.radians(fov_deg)
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx
        return {
            "fx": fx,
            "fy": fy,
            "cx": width / 2.0,
            "cy": height / 2.0,
        }

    async def capture_mapping_bundle(
        self,
        output_dir: Path | None = None,
        include_depth: bool = True,
        include_imu: bool = True,
    ) -> dict[str, Any]:
        """Capture a synchronized mapping bundle (RGB + depth + pose/IMU metadata)."""
        if not self.connected or not self.client:
            return {"success": False, "error": "not_connected"}

        session_dir = self._get_mapping_session_dir(output_dir)
        frames_dir = session_dir / "frames"
        timestamp = datetime.now()
        timestamp_ns = time.time_ns()
        seq = await self._next_sequence()

        requests: list[Any] = [
            airsim.ImageRequest(
                self.config.camera_name,
                airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=self.config.compress,
            )
        ]
        if include_depth:
            requests.append(
                airsim.ImageRequest(
                    self.config.camera_name,
                    airsim.ImageType.DepthPerspective,
                    pixels_as_float=True,
                    compress=False,
                )
            )

        image_task = _run_in_airsim_thread(
            "simGetImages",
            requests,
            self.config.vehicle_name,
        )
        telemetry_task = self.get_synchronized_state()
        camera_task = _run_in_airsim_thread(
            "simGetCameraInfo", self.config.camera_name, self.config.vehicle_name
        )

        image_response, telemetry, camera_info = await asyncio.gather(
            image_task, telemetry_task, camera_task, return_exceptions=True
        )

        if isinstance(image_response, Exception):
            logger.error("Mapping capture image failed: %s", image_response)
            return {"success": False, "error": str(image_response)}
        if isinstance(telemetry, Exception):
            logger.warning("Mapping capture telemetry failed: %s", telemetry)
            telemetry = None
        if isinstance(camera_info, Exception):
            logger.warning("Mapping capture camera info failed: %s", camera_info)
            camera_info = None

        if not image_response:
            return {"success": False, "error": "no_image_response"}

        rgb_response = image_response[0]
        depth_response = image_response[1] if include_depth and len(image_response) > 1 else None

        frame_id = str(timestamp_ns)
        rgb_path = frames_dir / f"{frame_id}.png"
        if self.config.compress:
            img_array = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            img = Image.open(io.BytesIO(img_array))
        else:
            img_array = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8).reshape(
                rgb_response.height, rgb_response.width, 3
            )
            img = Image.fromarray(img_array)
        img.save(rgb_path)

        depth_path = None
        if depth_response:
            depth = np.array(depth_response.image_data_float, dtype=np.float32).reshape(
                depth_response.height, depth_response.width
            )
            depth_path = frames_dir / f"{frame_id}_depth.npy"
            np.save(depth_path, depth)

        intrinsics = self._compute_intrinsics(
            rgb_response.width, rgb_response.height, float(getattr(camera_info, "fov", 90.0))
        )
        camera_pose = getattr(camera_info, "pose", None)

        metadata = {
            "frame_id": frame_id,
            "sequence": seq,
            "timestamp": timestamp.isoformat(),
            "timestamp_ns": timestamp_ns,
            "camera": {
                "name": self.config.camera_name,
                "resolution": [rgb_response.width, rgb_response.height],
                "fov_deg": float(getattr(camera_info, "fov", 90.0)),
                "intrinsics": intrinsics,
                "pose": {
                    "position": {
                        "x": camera_pose.position.x_val,
                        "y": camera_pose.position.y_val,
                        "z": camera_pose.position.z_val,
                    },
                    "orientation": {
                        "w": camera_pose.orientation.w_val,
                        "x": camera_pose.orientation.x_val,
                        "y": camera_pose.orientation.y_val,
                        "z": camera_pose.orientation.z_val,
                    },
                }
                if camera_pose
                else None,
            },
            "telemetry": telemetry.model_dump(mode="json") if telemetry else None,
            "files": {
                "rgb": str(rgb_path),
                "depth": str(depth_path) if depth_path else None,
            },
        }

        if not include_imu and metadata["telemetry"]:
            metadata["telemetry"]["imu"] = None

        meta_path = frames_dir / f"{frame_id}.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        return {"success": True, "metadata_path": str(meta_path), **metadata}

    # -------------------------------------------------------------------------
    # Depth Camera for Obstacle Detection
    # -------------------------------------------------------------------------

    async def capture_depth(
        self,
        window_ratio: float = 0.3,
        include_frame: bool = False,
    ) -> dict[str, Any]:
        """Capture depth image and return obstacle information.

        Uses the front camera to detect obstacles ahead of the drone.
        Returns the minimum depth in a center window and obstacle directions.

        Args:
            window_ratio: Size of center window to analyze (0.3 = center 30%)

        Returns:
            Dict with:
                - success: bool
                - min_depth_m: minimum depth in meters (distance to closest obstacle)
                - obstacle_direction: 'left', 'right', 'center', 'above', 'below', or None
                - depths: dict with depth readings in different zones
        """
        if not self.connected or not self.client:
            return {"success": False, "error": "not_connected", "min_depth_m": None}

        try:
            # Request depth image (try fallback camera names if needed)
            responses = None
            response = None
            selected_camera = None
            last_error: str | None = None
            selected_type = None
            depth_types = [airsim.ImageType.DepthPerspective, airsim.ImageType.DepthPlanner]
            for camera_name in self._camera_candidates():
                for depth_type in depth_types:
                    try:
                        responses = await _run_in_airsim_thread(
                            "simGetImages",
                            [
                                airsim.ImageRequest(
                                    camera_name,
                                    depth_type,
                                    pixels_as_float=True,
                                    compress=False,
                                )
                            ],
                            self.config.vehicle_name,
                        )
                    except Exception as exc:
                        last_error = str(exc)
                        continue
                    if not responses:
                        continue
                    response = responses[0]
                    if response.width == 0 or response.height == 0:
                        continue
                    selected_camera = camera_name
                    selected_type = depth_type
                    break
                if response:
                    break

            if not response:
                return {
                    "success": False,
                    "error": last_error or "no_response",
                    "min_depth_m": None,
                }

            if selected_camera and selected_camera != self.config.camera_name:
                logger.warning(
                    "Depth camera fallback selected: %s (was %s)",
                    selected_camera,
                    self.config.camera_name,
                )
                self.config.camera_name = selected_camera

            # Parse depth data
            depth = np.array(response.image_data_float, dtype=np.float32)
            depth = depth.reshape(response.height, response.width)

            h, w = depth.shape

            # Analyze different zones
            center_h = int(h * window_ratio)
            center_w = int(w * window_ratio)
            h0 = (h - center_h) // 2
            w0 = (w - center_w) // 2

            # Center window
            center = depth[h0 : h0 + center_h, w0 : w0 + center_w]
            center_valid = center[np.isfinite(center) & (center > 0.5)]

            # Left/Right zones for steering
            left_zone = depth[h0 : h0 + center_h, :w0]
            right_zone = depth[h0 : h0 + center_h, w0 + center_w :]
            top_zone = depth[:h0, w0 : w0 + center_w]
            bottom_zone = depth[h0 + center_h :, w0 : w0 + center_w]

            def get_min_depth(zone: np.ndarray) -> float:
                valid = zone[np.isfinite(zone) & (zone > 0.5)]
                return float(np.min(valid)) if valid.size > 0 else 999.0

            depths = {
                "center": get_min_depth(center),
                "left": get_min_depth(left_zone),
                "right": get_min_depth(right_zone),
                "top": get_min_depth(top_zone),
                "bottom": get_min_depth(bottom_zone),
            }

            min_depth = depths["center"]

            # Determine obstacle direction for avoidance
            obstacle_direction = None
            if min_depth < 100.0:  # Obstacle detected ahead
                # Find clearest direction to steer
                if depths["left"] > depths["right"] + 5:
                    obstacle_direction = "right"  # Obstacle on right, steer left
                elif depths["right"] > depths["left"] + 5:
                    obstacle_direction = "left"  # Obstacle on left, steer right
                elif depths["top"] > depths["center"] + 10:
                    obstacle_direction = "below"  # Go up
                else:
                    obstacle_direction = "center"  # Obstacle directly ahead

            result = {
                "success": True,
                "min_depth_m": min_depth,
                "obstacle_direction": obstacle_direction,
                "depths": depths,
                "resolution": (response.width, response.height),
                "camera_name": selected_camera,
                "image_type": str(selected_type) if selected_type is not None else None,
            }
            if include_frame:
                try:
                    camera_info = await _run_in_airsim_thread(
                        "simGetCameraInfo", selected_camera, self.config.vehicle_name
                    )
                except Exception as exc:
                    if self._is_connection_error(exc):
                        self.connected = False
                        self.client = None
                        camera_info = None
                    else:
                        logger.warning("Depth camera info failed: %s", exc)
                        camera_info = None

                if camera_info:
                    fov_deg = float(getattr(camera_info, "fov", 90.0))
                    result["intrinsics"] = self._compute_intrinsics(
                        response.width, response.height, fov_deg
                    )
                    pose = getattr(camera_info, "pose", None)
                    if pose:
                        result["camera_pose"] = {
                            "position": {
                                "x": pose.position.x_val,
                                "y": pose.position.y_val,
                                "z": pose.position.z_val,
                            },
                            "orientation": {
                                "w": pose.orientation.w_val,
                                "x": pose.orientation.x_val,
                                "y": pose.orientation.y_val,
                                "z": pose.orientation.z_val,
                            },
                        }
                result["depth"] = depth

            return result

        except Exception as e:
            if self._is_connection_error(e):
                self.connected = False
                self.client = None
            logger.error(f"Depth capture failed: {e}")
            return {"success": False, "error": str(e), "min_depth_m": None}

    async def move_to_position_with_obstacle_avoidance(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        velocity: float = 5.0,
        obstacle_distance_m: float = 15.0,
        avoidance_step_m: float = 10.0,
        max_distance: float = 2000.0,  # Maximum distance from origin
    ) -> bool:
        """Move to position with real-time obstacle avoidance using depth camera.

        Instead of blindly flying to the target, this method:
        1. Moves in increments toward the target
        2. Continuously checks depth camera for obstacles
        3. If obstacle detected, stops and adjusts path (go around or above)

        Args:
            target_x: Target north position (NED)
            target_y: Target east position (NED)
            target_z: Target down position (NED, negative = altitude)
            velocity: Flight speed in m/s
            obstacle_distance_m: Distance at which to start avoiding obstacles
            avoidance_step_m: How far to move sideways/up when avoiding
            max_distance: Maximum distance from origin (boundary check)

        Returns:
            True if destination reached, False if aborted
        """
        # Boundary check - prevent flying off the map
        distance_from_origin = math.sqrt(target_x * target_x + target_y * target_y)
        if distance_from_origin > max_distance:
            logger.error(
                f"Target position ({target_x:.1f}, {target_y:.1f}) is {distance_from_origin:.1f}m from origin, "
                f"exceeds max distance of {max_distance}m. Aborting move."
            )
            return False

        # Altitude check
        if target_z > 0:
            logger.warning(f"Target altitude {target_z} is underground, clamping to ground level")
            target_z = -1.0
        if target_z < -500:
            logger.warning(f"Target altitude {-target_z}m is too high, clamping to 500m")
            target_z = -500.0

        # Ensure we have a valid connection
        if not await self.ensure_connected():
            logger.error("move_with_avoidance: could not connect")
            return False

        logger.info(
            f"move_with_avoidance: starting to ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
        )

        max_attempts = 50  # Maximum avoidance maneuvers
        attempts = 0
        collision_events = 0
        max_collision_events = 5
        depth_failures = 0
        max_depth_failures = 3

        while attempts < max_attempts:
            attempts += 1

            if attempts % 5 == 1:
                logger.debug(f"move_with_avoidance: attempt {attempts}/{max_attempts}")

            # Get current position using thread-safe API call
            try:
                state = await _run_in_airsim_thread(
                    "getMultirotorState", vehicle_name=self.config.vehicle_name
                )
                pos = state.kinematics_estimated.position
                current_x = pos.x_val
                current_y = pos.y_val
                current_z = pos.z_val
            except Exception as e:
                if self._is_connection_error(e):
                    self.connected = False
                    self.client = None
                    logger.error("Lost position during flight: AirSim client disconnected")
                    return False
                logger.error(f"Lost position during flight: {e}")
                return False

            # Calculate distance to target
            dx = target_x - current_x
            dy = target_y - current_y
            dz = target_z - current_z
            distance_to_target = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Check if we've arrived
            if distance_to_target < 5.0:
                logger.info(f"Arrived at target (distance: {distance_to_target:.1f}m)")
                await self.hover()
                return True

            collision = await self._get_collision_info()
            if collision and collision.get("has_collided"):
                stamp = collision.get("time_stamp")
                if stamp != self._last_collision_stamp:
                    self._last_collision_stamp = stamp
                    collision_events += 1
                logger.warning(
                    "Collision detected (count=%s, object=%s)",
                    collision_events,
                    collision.get("object_name"),
                )
                await self.hover()
                await asyncio.sleep(0.2)

                # Back off and climb before retrying
                norm = math.hypot(dx, dy)
                if norm < 0.1:
                    back_x = current_x
                    back_y = current_y - avoidance_step_m
                else:
                    back_x = current_x - (dx / norm) * avoidance_step_m
                    back_y = current_y - (dy / norm) * avoidance_step_m
                back_z = current_z - avoidance_step_m

                await self.move_to_position(
                    back_x,
                    back_y,
                    back_z,
                    velocity=velocity * 0.6,
                    timeout=15.0,
                )

                if collision_events >= max_collision_events:
                    logger.error("Collision recovery failed after %s attempts", collision_events)
                    return False
                continue

            # Align yaw to movement direction so the depth camera looks ahead
            await self._align_yaw_to_vector(dx, dy)

            # Check for obstacles ahead
            depth_result = await self.capture_depth()

            if not depth_result.get("success"):
                depth_failures += 1
                logger.warning(
                    "Depth capture unavailable (%s/%s): %s",
                    depth_failures,
                    max_depth_failures,
                    depth_result.get("error"),
                )
                if depth_failures >= max_depth_failures:
                    await self.hover()
                    logger.error("Depth capture failed repeatedly; aborting obstacle avoidance.")
                    return False
            else:
                depth_failures = 0
                min_depth = depth_result.get("min_depth_m", 999.0)
                obstacle_dir = depth_result.get("obstacle_direction")

                if min_depth is not None and min_depth < obstacle_distance_m:
                    logger.warning(
                        f"Obstacle detected at {min_depth:.1f}m! Direction: {obstacle_dir}"
                    )

                    # Stop and hover
                    await self.hover()
                    await asyncio.sleep(0.5)

                    # Determine avoidance maneuver
                    if obstacle_dir == "left":
                        # Obstacle on left, move right
                        avoid_x = current_x
                        avoid_y = current_y + avoidance_step_m
                        avoid_z = current_z
                        logger.info("Avoiding obstacle: moving RIGHT")
                    elif obstacle_dir == "right":
                        # Obstacle on right, move left
                        avoid_x = current_x
                        avoid_y = current_y - avoidance_step_m
                        avoid_z = current_z
                        logger.info("Avoiding obstacle: moving LEFT")
                    elif obstacle_dir in ("below", "center"):
                        # Obstacle ahead/below, go up
                        avoid_x = current_x
                        avoid_y = current_y
                        avoid_z = current_z - avoidance_step_m  # Negative = up
                        logger.info("Avoiding obstacle: moving UP")
                    else:
                        # Default: go up and slightly to the side
                        avoid_x = current_x
                        avoid_y = current_y + avoidance_step_m * 0.5
                        avoid_z = current_z - avoidance_step_m * 0.5
                        logger.info("Avoiding obstacle: moving UP and RIGHT")

                    # Execute avoidance maneuver
                    await self.move_to_position(
                        avoid_x,
                        avoid_y,
                        avoid_z,
                        velocity=velocity * 0.7,  # Slower during avoidance
                        timeout=15.0,
                    )
                    continue  # Re-check obstacles after avoidance

            # No obstacle or depth failed - move toward target
            # Move a segment toward target (not all the way at once)
            segment_distance = min(30.0, distance_to_target)
            if distance_to_target > 0.1:
                ratio = segment_distance / distance_to_target
                next_x = current_x + dx * ratio
                next_y = current_y + dy * ratio
                next_z = current_z + dz * ratio
            else:
                next_x, next_y, next_z = target_x, target_y, target_z

            logger.debug(
                f"Moving segment: ({next_x:.1f}, {next_y:.1f}, {next_z:.1f}), "
                f"remaining: {distance_to_target:.1f}m"
            )

            # Move this segment
            await self.move_to_position(next_x, next_y, next_z, velocity=velocity, timeout=30.0)

            # Brief pause to check obstacles again
            await asyncio.sleep(0.2)

        logger.warning(f"Obstacle avoidance failed after {max_attempts} attempts")
        return False

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------

    async def start_streaming(self) -> None:
        """Start background telemetry streaming."""
        if self._streaming:
            return

        self._streaming = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        logger.info(f"Started telemetry streaming at {self.config.telemetry_hz} Hz")

    async def stop_streaming(self) -> None:
        """Stop telemetry streaming."""
        self._streaming = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        logger.info("Stopped telemetry streaming")

    async def _stream_loop(self) -> None:
        """Background loop for high-frequency telemetry."""
        interval = 1.0 / self.config.telemetry_hz
        consecutive_errors = 0
        max_consecutive_errors = 10  # Stop after too many errors

        while self._streaming and self.connected:
            loop_start = time.perf_counter()

            try:
                frame = await self.get_synchronized_state()

                if frame:
                    consecutive_errors = 0  # Reset on success
                    # Try to put in queue, drop if full
                    try:
                        self._telemetry_queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        if self.config.drop_old_frames:
                            # Drop oldest frame
                            try:
                                self._telemetry_queue.get_nowait()
                                self._telemetry_queue.put_nowait(frame)
                                self._dropped_frames += 1
                            except asyncio.QueueEmpty:
                                pass

                    # Notify callbacks
                    for callback in self._on_telemetry:
                        try:
                            await callback(frame)
                        except Exception as e:
                            logger.warning(f"Telemetry callback error: {e}")
                else:
                    # Frame fetch returned None (likely connection issue)
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(
                            "Telemetry stream stopping: too many consecutive fetch errors"
                        )
                        break

            except Exception as e:
                logger.error(f"Stream loop error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning("Telemetry stream stopping: too many consecutive errors")
                    break

            # Maintain target frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def telemetry_stream(self) -> AsyncIterator[TelemetryFrame]:
        """Async generator for consuming telemetry frames.

        Yields:
            TelemetryFrame objects as they become available

        Example:
            async for frame in bridge.telemetry_stream():
                await websocket.send_json(frame.to_broadcast_dict())
        """
        while self._streaming or not self._telemetry_queue.empty():
            try:
                frame = await asyncio.wait_for(self._telemetry_queue.get(), timeout=1.0)
                yield frame
            except asyncio.TimeoutError:
                if not self._streaming:
                    break

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_telemetry(self, callback: Callable[[TelemetryFrame], Awaitable[None]]) -> None:
        """Register callback for telemetry updates."""
        self._on_telemetry.append(callback)

    def on_frame(self, callback: Callable[[FrameCaptureResult], Awaitable[None]]) -> None:
        """Register callback for frame captures."""
        self._on_frame.append(callback)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update running latency statistics."""
        self._latency_samples.append(latency_ms)
        self._avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def get_stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        return {
            "connected": self.connected,
            "streaming": self._streaming,
            "total_frames": self._total_frames,
            "dropped_frames": self._dropped_frames,
            "queue_size": self._telemetry_queue.qsize(),
            "avg_latency_ms": round(self._avg_latency_ms, 2),
            "target_hz": self.config.telemetry_hz,
            "sequence": self._sequence,
        }

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._streaming


# =============================================================================
# WebSocket Integration Helper
# =============================================================================


class TelemetryBroadcaster:
    """Bridges RealtimeAirSimBridge to WebSocket broadcast.

    Connects the telemetry stream to the server's ConnectionManager
    for real-time delivery to Unreal/dashboard clients.
    """

    def __init__(
        self, bridge: RealtimeAirSimBridge, broadcast_fn: Callable[[dict], Awaitable[None]]
    ) -> None:
        self.bridge = bridge
        self.broadcast = broadcast_fn
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start broadcasting telemetry."""
        if self._running:
            return

        self._running = True

        # Register callback with bridge
        async def on_frame(frame: TelemetryFrame) -> None:
            await self.broadcast(frame.to_broadcast_dict())

        self.bridge.on_telemetry(on_frame)

        # Start bridge streaming if not already
        await self.bridge.start_streaming()

        logger.info("TelemetryBroadcaster started")

    async def stop(self) -> None:
        """Stop broadcasting."""
        self._running = False
        await self.bridge.stop_streaming()
        logger.info("TelemetryBroadcaster stopped")


# =============================================================================
# Multi-Vehicle Bridge Factory
# =============================================================================


def create_multi_vehicle_bridges(
    vehicle_names: list[str],
    host: str = "127.0.0.1",
    base_config: RealtimeBridgeConfig | None = None,
) -> dict[str, RealtimeAirSimBridge]:
    """Create multiple bridge instances for multi-drone scenarios.

    Each bridge controls a single vehicle, allowing independent operation
    of multiple drones in the same AirSim environment.

    Args:
        vehicle_names: List of AirSim vehicle names (e.g., ["Drone1", "Drone2"])
        host: AirSim host address
        base_config: Base configuration to clone for each bridge

    Returns:
        Dict mapping vehicle_name -> bridge instance

    Example:
        bridges = create_multi_vehicle_bridges(["Drone1", "Drone2", "Drone3"])
        for name, bridge in bridges.items():
            await bridge.connect()
            print(f"Connected to {name}")
    """
    bridges = {}

    for vehicle_name in vehicle_names:
        # Clone base config or create new
        if base_config:
            config = RealtimeBridgeConfig(
                host=host,
                vehicle_name=vehicle_name,
                camera_name=base_config.camera_name,
                target_fps=base_config.target_fps,
                resolution=base_config.resolution,
                compress=base_config.compress,
                save_images=base_config.save_images,
                output_dir=base_config.output_dir / vehicle_name,
                mapping_output_dir=base_config.mapping_output_dir / vehicle_name,
                telemetry_hz=base_config.telemetry_hz,
                include_imu=base_config.include_imu,
                max_queue_size=base_config.max_queue_size,
                drop_old_frames=base_config.drop_old_frames,
                sync_timeout_ms=base_config.sync_timeout_ms,
            )
        else:
            config = RealtimeBridgeConfig(
                host=host,
                vehicle_name=vehicle_name,
            )

        bridges[vehicle_name] = RealtimeAirSimBridge(config)
        logger.debug(f"Created bridge for vehicle: {vehicle_name}")

    logger.info(f"Created {len(bridges)} vehicle bridges: {list(bridges.keys())}")
    return bridges


async def connect_all_bridges(bridges: dict[str, RealtimeAirSimBridge]) -> dict[str, bool]:
    """Connect all bridges in parallel.

    Args:
        bridges: Dict of vehicle_name -> bridge

    Returns:
        Dict of vehicle_name -> connection success
    """
    results = {}

    async def connect_one(name: str, bridge: RealtimeAirSimBridge) -> tuple[str, bool]:
        try:
            success = await bridge.connect()
            return name, success
        except Exception as e:
            logger.error(f"Failed to connect bridge for {name}: {e}")
            return name, False

    # Connect all in parallel
    tasks = [connect_one(name, bridge) for name, bridge in bridges.items()]
    for coro in asyncio.as_completed(tasks):
        name, success = await coro
        results[name] = success
        if success:
            logger.info(f"Bridge connected: {name}")
        else:
            logger.warning(f"Bridge failed to connect: {name}")

    return results


async def disconnect_all_bridges(bridges: dict[str, RealtimeAirSimBridge]) -> None:
    """Disconnect all bridges."""
    for name, bridge in bridges.items():
        try:
            await bridge.disconnect()
            logger.debug(f"Disconnected bridge: {name}")
        except Exception as e:
            logger.error(f"Error disconnecting bridge {name}: {e}")
