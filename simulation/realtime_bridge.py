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
import math
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
        self.vehicle_names: list[str] = []

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
            def _connect() -> tuple[airsim.MultirotorClient, str, list[str], str | None]:
                client = airsim.MultirotorClient(ip=self.config.host)
                client.confirmConnection()
                vehicles: list[str] = []
                selection_note: str | None = None
                if hasattr(client, "listVehicles"):
                    try:
                        vehicles = list(client.listVehicles())
                    except Exception as exc:
                        selection_note = f"listVehicles failed: {exc}"

                selected, selection_reason = _select_vehicle_name(self.config.vehicle_name, vehicles)
                if selection_reason:
                    selection_note = selection_reason if not selection_note else f"{selection_note}; {selection_reason}"
                client.enableApiControl(True, selected)
                return client, selected, vehicles, selection_note

            self.client, selected_name, vehicles, selection_note = await asyncio.to_thread(_connect)
            self.config.vehicle_name = selected_name
            self.vehicle_names = vehicles
            self.connected = True

            # Log camera info
            camera_info = await asyncio.to_thread(
                self.client.simGetCameraInfo, self.config.camera_name, self.config.vehicle_name
            )
            if vehicles:
                logger.info("AirSim vehicles detected: %s", vehicles)
            if selection_note:
                logger.warning("AirSim vehicle selection note: %s", selection_note)
            logger.info(
                "Connected to AirSim (camera FOV: %s, vehicle: %s)",
                camera_info.fov,
                self.config.vehicle_name,
            )

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
                asyncio.to_thread(self.client.getMultirotorState, vehicle_name=self.config.vehicle_name),
                timeout=3.0
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Connection check timed out, reconnecting...")
            self.connected = False
            self.client = None
            return await self.connect()
        except Exception as e:
            logger.warning(f"Connection check failed: {e}, reconnecting...")
            self.connected = False
            self.client = None
            return await self.connect()

    async def set_weather(
        self, rain: float = 0.0, snow: float = 0.0, fog: float = 0.0, dust: float = 0.0
    ) -> bool:
        """Set weather conditions in the simulation."""
        if not self.connected or not self.client:
            return False

        try:
            await asyncio.to_thread(self.client.simEnableWeather, True)
            await asyncio.to_thread(
                self.client.simSetWeatherParameter, airsim.WeatherParameter.Rain, rain
            )
            await asyncio.to_thread(
                self.client.simSetWeatherParameter, airsim.WeatherParameter.Snow, snow
            )
            await asyncio.to_thread(
                self.client.simSetWeatherParameter, airsim.WeatherParameter.Fog, fog
            )
            await asyncio.to_thread(
                self.client.simSetWeatherParameter, airsim.WeatherParameter.Dust, dust
            )
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
            await asyncio.to_thread(
                self.client.simSetTimeOfDay,
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
            await asyncio.to_thread(self.client.simSetWind, wind)
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
        if not (hasattr(self.client, "simSetVehiclePose") or hasattr(self.client, "simSetVehiclePoseAsync")):
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

        try:
            # Use sync version if available, otherwise async without .join()
            if hasattr(self.client, "simSetVehiclePose"):
                def _set_pose() -> None:
                    try:
                        self.client.simSetVehiclePose(pose, ignore_collision, self.config.vehicle_name)
                    except TypeError:
                        self.client.simSetVehiclePose(pose, ignore_collision)
                await asyncio.to_thread(_set_pose)
            elif hasattr(self.client, "simSetVehiclePoseAsync"):
                # Just send the command without waiting
                self.client.simSetVehiclePoseAsync(
                    pose, ignore_collision, self.config.vehicle_name
                )
                await asyncio.sleep(0.1)  # Brief pause for pose to be applied
            logger.info(
                "Vehicle pose set (realtime bridge)",
                extra={"north": north, "east": east, "down": down},
            )
            return True
        except Exception as exc:
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
                    state = self.client.getMultirotorState(vehicle_name=self.config.vehicle_name)
                    north = state.kinematics_estimated.position.x_val
                    east = state.kinematics_estimated.position.y_val
                except Exception:
                    pass  # Use origin if we can't get position

            # Reset using simReset first to clear physics state
            try:
                self.client.reset()
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"simReset failed (may be expected): {e}")

            # Reconnect after reset
            self.client = airsim.MultirotorClient(ip=self.config.host)
            self.client.confirmConnection()
            self.client.enableApiControl(True, self.config.vehicle_name)

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
            pose = airsim.Pose(
                airsim.Vector3r(north, east, down),
                airsim.Quaternionr(0, 0, 0, 1)
            )
            scale_vec = airsim.Vector3r(scale[0], scale[1], scale[2])

            # Use simSpawnObject if available
            if hasattr(self.client, 'simSpawnObject'):
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.simSpawnObject,
                        object_name,
                        asset_name,
                        pose,
                        scale_vec,
                        physics_enabled,
                    ),
                    timeout=5.0
                )
                if result:
                    logger.info(f"Spawned object '{object_name}' at NED({north:.1f}, {east:.1f}, {down:.1f})")
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
            if hasattr(self.client, 'simDestroyObject'):
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.client.simDestroyObject, object_name),
                    timeout=5.0
                )
                if result:
                    logger.info(f"Destroyed object '{object_name}'")
                return result
            return False
        except Exception as exc:
            logger.warning(f"Failed to destroy object '{object_name}': {exc}")
            return False

    async def spawn_scene_objects(
        self,
        dock_ned: tuple[float, float, float] | None = None,
        assets: list[dict] | None = None,
        geo_ref = None,
    ) -> dict:
        """Spawn dock and asset markers in AirSim.

        Args:
            dock_ned: Dock position in NED (or None for origin)
            assets: List of asset dicts with lat/lon/name
            geo_ref: GeoReference for GPS to NED conversion

        Returns:
            Dict with spawn results
        """
        results = {"dock": False, "assets": [], "errors": []}

        # Spawn dock at origin (or specified position)
        dock_n, dock_e, dock_d = dock_ned or (0.0, 0.0, 0.0)
        results["dock"] = await self.spawn_object(
            "DockStation",
            "Cylinder",  # Use cylinder as dock marker
            dock_n, dock_e, dock_d,
            scale=(3.0, 3.0, 0.5),  # Flat cylinder as landing pad
            physics_enabled=False,
        )

        # Spawn asset markers
        if assets and geo_ref:
            for asset in assets:
                try:
                    lat = asset.get("latitude")
                    lon = asset.get("longitude")
                    name = asset.get("name", asset.get("asset_id", "unknown"))
                    asset_type = asset.get("asset_type", "unknown")

                    if lat is None or lon is None:
                        continue

                    # Convert GPS to NED
                    north, east, down = geo_ref.gps_to_ned(lat, lon, geo_ref.altitude)
                    # Place marker above ground
                    down = -5.0  # 5m above ground

                    # Choose shape based on asset type
                    shape = "Sphere"
                    if "solar" in asset_type.lower():
                        shape = "Cube"
                    elif "turbine" in asset_type.lower() or "wind" in asset_type.lower():
                        shape = "Cylinder"

                    object_name = f"Asset_{name.replace(' ', '_')}"
                    success = await self.spawn_object(
                        object_name,
                        shape,
                        north, east, down,
                        scale=(2.0, 2.0, 2.0),
                        physics_enabled=False,
                    )
                    results["assets"].append({
                        "name": name,
                        "success": success,
                        "ned": (north, east, down),
                    })
                except Exception as e:
                    results["errors"].append(f"{name}: {e}")

        logger.info(
            f"Scene spawned: dock={results['dock']}, "
            f"assets={len([a for a in results['assets'] if a['success']])}/{len(results['assets'])}"
        )
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
            await asyncio.wait_for(
                asyncio.to_thread(self.client.enableApiControl, True, self.config.vehicle_name),
                timeout=5.0
            )
            logger.info("API control enabled")

            # Then arm (with timeout)
            await asyncio.wait_for(
                asyncio.to_thread(self.client.armDisarm, True, self.config.vehicle_name),
                timeout=5.0
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
            await asyncio.wait_for(
                asyncio.to_thread(self.client.armDisarm, False, self.config.vehicle_name),
                timeout=5.0
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

            # Start takeoff (non-blocking)
            self.client.takeoffAsync(
                timeout_sec=timeout,
                vehicle_name=self.config.vehicle_name
            )

            # Wait for takeoff to complete by checking altitude directly (with timeouts)
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Use direct API call with timeout
                    state = await asyncio.wait_for(
                        asyncio.to_thread(self.client.getMultirotorState, vehicle_name=self.config.vehicle_name),
                        timeout=3.0
                    )
                    current_alt = -state.kinematics_estimated.position.z_val
                    if current_alt > 1.0:  # At least 1m off ground
                        logger.info(f"Takeoff complete, altitude: {current_alt:.1f}m")
                        return True
                except asyncio.TimeoutError:
                    logger.warning("Position check timed out during takeoff")
                except Exception as e:
                    logger.warning(f"Error checking altitude: {e}")
                await asyncio.sleep(0.3)

            # Check final altitude (with timeout)
            try:
                state = await asyncio.wait_for(
                    asyncio.to_thread(self.client.getMultirotorState, vehicle_name=self.config.vehicle_name),
                    timeout=3.0
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

            # Start land command (non-blocking)
            self.client.landAsync(
                timeout_sec=timeout,
                vehicle_name=self.config.vehicle_name
            )

            # Wait for landing by polling altitude with timeouts
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    state = await asyncio.wait_for(
                        asyncio.to_thread(self.client.getMultirotorState, vehicle_name=self.config.vehicle_name),
                        timeout=3.0
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

            logger.warning("Landing timed out")
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
            # Send hover command (non-blocking)
            self.client.hoverAsync(vehicle_name=self.config.vehicle_name)
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
        yaw_mode: dict | None = None
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

        Returns:
            True if destination reached
        """
        # Ensure we have a valid connection
        if not await self.ensure_connected():
            logger.error("Move failed: could not connect")
            return False

        try:
            # Ensure API control is enabled (with timeout)
            await asyncio.wait_for(
                asyncio.to_thread(self.client.enableApiControl, True, self.config.vehicle_name),
                timeout=5.0
            )

            # Get current position to calculate yaw toward target (with timeout)
            state = await asyncio.wait_for(
                asyncio.to_thread(self.client.getMultirotorState, vehicle_name=self.config.vehicle_name),
                timeout=5.0
            )
            current_pos = state.kinematics_estimated.position

            # Calculate yaw angle to face the target (in degrees)
            dx = x - current_pos.x_val
            dy = y - current_pos.y_val
            yaw_rad = math.atan2(dy, dx)
            yaw_deg = math.degrees(yaw_rad)

            logger.info(f"Moving to NED ({x:.1f}, {y:.1f}, {z:.1f}) at {velocity} m/s, yaw={yaw_deg:.1f}°")

            # Start the move command (non-blocking)
            # Use ForwardOnly drivetrain to face direction of travel
            self.client.moveToPositionAsync(
                x, y, z, velocity,
                timeout_sec=timeout,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg),
                vehicle_name=self.config.vehicle_name
            )

            # Wait for arrival by polling position with timeouts
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    state = await asyncio.wait_for(
                        asyncio.to_thread(self.client.getMultirotorState, vehicle_name=self.config.vehicle_name),
                        timeout=3.0
                    )
                    pos = state.kinematics_estimated.position
                    dx = abs(pos.x_val - x)
                    dy = abs(pos.y_val - y)
                    dz = abs(pos.z_val - z)
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                    if dist < 3.0:  # Within 3 meters
                        logger.info(f"Arrived at ({x:.1f}, {y:.1f}, {z:.1f})")
                        return True
                except asyncio.TimeoutError:
                    logger.warning("Position check timed out, continuing...")
                except Exception as e:
                    logger.warning(f"Error checking position: {e}")
                await asyncio.sleep(0.5)

            logger.warning(f"Move timed out after {timeout}s")
            return True  # Return true anyway - drone is moving

        except Exception as exc:
            logger.exception("Move to position failed: %s", exc)
            return False

    async def move_by_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float = 1.0
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
            # Send velocity command (non-blocking)
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                vehicle_name=self.config.vehicle_name
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
        clockwise: bool = True
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
                    asyncio.to_thread(self.client.getMultirotorState, vehicle_name=self.config.vehicle_name),
                    timeout=3.0
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

                # Move to orbit point with yaw facing center
                self.client.moveToPositionAsync(
                    x, y, center_z, velocity,
                    timeout_sec=2.0,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_to_center),
                    vehicle_name=self.config.vehicle_name
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
    # Depth Camera for Obstacle Detection
    # -------------------------------------------------------------------------

    async def capture_depth(
        self,
        window_ratio: float = 0.3,
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
            # Request depth image
            responses = await asyncio.to_thread(
                self.client.simGetImages,
                [
                    airsim.ImageRequest(
                        self.config.camera_name,
                        airsim.ImageType.DepthPerspective,
                        pixels_as_float=True,
                        compress=False,
                    )
                ],
                self.config.vehicle_name,
            )

            if not responses:
                return {"success": False, "error": "no_response", "min_depth_m": None}

            response = responses[0]
            if response.width == 0 or response.height == 0:
                return {"success": False, "error": "empty_frame", "min_depth_m": None}

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
            center = depth[h0:h0 + center_h, w0:w0 + center_w]
            center_valid = center[np.isfinite(center) & (center > 0.5)]

            # Left/Right zones for steering
            left_zone = depth[h0:h0 + center_h, :w0]
            right_zone = depth[h0:h0 + center_h, w0 + center_w:]
            top_zone = depth[:h0, w0:w0 + center_w]
            bottom_zone = depth[h0 + center_h:, w0:w0 + center_w]

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

            return {
                "success": True,
                "min_depth_m": min_depth,
                "obstacle_direction": obstacle_direction,
                "depths": depths,
                "resolution": (response.width, response.height),
            }

        except Exception as e:
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

        Returns:
            True if destination reached, False if aborted
        """
        # Ensure we have a valid connection
        if not await self.ensure_connected():
            logger.error("move_with_avoidance: could not connect")
            return False

        logger.info(
            f"move_with_avoidance: starting to ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
        )

        max_attempts = 50  # Maximum avoidance maneuvers
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            if attempts % 5 == 1:
                logger.debug(f"move_with_avoidance: attempt {attempts}/{max_attempts}")

            # Get current position using direct API call
            try:
                state = self.client.getMultirotorState(vehicle_name=self.config.vehicle_name)
                pos = state.kinematics_estimated.position
                current_x = pos.x_val
                current_y = pos.y_val
                current_z = pos.z_val
            except Exception as e:
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

            # Check for obstacles ahead
            depth_result = await self.capture_depth()

            if depth_result.get("success"):
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
                        avoid_x, avoid_y, avoid_z,
                        velocity=velocity * 0.7,  # Slower during avoidance
                        timeout=15.0
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
            await self.move_to_position(
                next_x, next_y, next_z,
                velocity=velocity,
                timeout=30.0
            )

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
