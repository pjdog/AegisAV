"""Real-Time AirSim Bridge for Unreal Visualization.

Provides async-safe, high-frequency communication between AirSim/Unreal and AegisAV.
Designed for sub-100ms latency with frame synchronization and telemetry streaming.

Key improvements over base bridge:
- All AirSim calls wrapped in asyncio.to_thread() to avoid blocking
- Parallel fetch of image + pose + IMU for synchronized data
- High-frequency telemetry streaming (30-50 Hz)
- Frame sequence numbers for ordering and deduplication
- Async producer queues for backpressure handling
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import airsim

    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    airsim = None

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
class RealtimeBridgeConfig:
    """Configuration for real-time AirSim bridge."""

    # Connection
    vehicle_name: str = "Drone1"
    camera_name: str = "front_center"

    # Capture settings
    target_fps: int = 30  # Target frame rate
    resolution: tuple[int, int] = (1280, 720)  # Lower res for speed
    compress: bool = True  # Use JPEG for faster transfer
    save_images: bool = False  # Don't save by default for speed
    output_dir: Path = field(default_factory=lambda: Path("data/vision/realtime"))

    # Telemetry settings
    telemetry_hz: int = 50  # Telemetry polling rate
    include_imu: bool = True

    # Queue settings
    max_queue_size: int = 100
    drop_old_frames: bool = True  # Drop old frames if queue full

    # Timing
    sync_timeout_ms: float = 100.0  # Max time to wait for synchronized data


# =============================================================================
# Real-Time Bridge Implementation
# =============================================================================


class RealtimeAirSimBridge:
    """Async-safe, high-frequency AirSim bridge for real-time visualization.

    Features:
    - Non-blocking I/O via asyncio.to_thread()
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

        # Ensure output dir exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RealtimeAirSimBridge initialized (target: {self.config.target_fps} FPS)")

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
        try:
            logger.info("Connecting to AirSim...")

            # Run blocking connection in thread pool
            def _connect() -> airsim.MultirotorClient:
                client = airsim.MultirotorClient()
                client.confirmConnection()
                client.enableApiControl(True, self.config.vehicle_name)
                return client

            self.client = await asyncio.to_thread(_connect)
            self.connected = True

            # Log camera info
            camera_info = await asyncio.to_thread(
                self.client.simGetCameraInfo, self.config.camera_name, self.config.vehicle_name
            )
            logger.info(f"Connected to AirSim (camera FOV: {camera_info.fov})")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to AirSim: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        await self.stop_streaming()

        if self.client:
            try:
                await asyncio.to_thread(
                    self.client.enableApiControl, False, self.config.vehicle_name
                )
            except Exception as exc:
                logger.warning("Failed to disable API control: %s", exc)
            self.client = None

        self.connected = False
        logger.info("Disconnected from AirSim")

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
                asyncio.to_thread(self.client.simGetVehiclePose, self.config.vehicle_name),
                asyncio.to_thread(self.client.getMultirotorState, self.config.vehicle_name),
            ]

            if self.config.include_imu:
                tasks.append(
                    asyncio.to_thread(
                        self.client.getImuData, imu_name="", vehicle_name=self.config.vehicle_name
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

            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_latency_stats(latency_ms)

            return TelemetryFrame(
                sequence=seq,
                server_timestamp_ms=time.time() * 1000,
                airsim_timestamp_ns=int(time.time_ns()),
                pose=pose,
                imu=imu,
                battery_percent=100.0,  # TODO: Simulate battery
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
                    asyncio.to_thread(
                        self.client.simGetImages,
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

        while self._streaming:
            loop_start = time.perf_counter()

            try:
                frame = await self.get_synchronized_state()

                if frame:
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

            except Exception as e:
                logger.error(f"Stream loop error: {e}")

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
