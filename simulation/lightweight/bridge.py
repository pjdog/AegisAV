"""Lightweight Bridge - Drop-in replacement for AirSim.

Provides the same API as AirSimBridge but uses the lightweight physics
simulation instead. This allows testing without Unreal Engine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from autonomy.vehicle_state import (
    Position,
    VehicleState,
)
from simulation.lightweight.physics import DroneConfig, EnvironmentConfig
from simulation.lightweight.simulator import LightweightSim
from vision.data_models import CameraState, CameraStatus, CaptureResult

logger = logging.getLogger(__name__)


@dataclass
class LightweightCameraConfig:
    """Configuration for lightweight camera capture (matches AirSimCameraConfig API)."""

    camera_name: str = "front_center"
    resolution: tuple[int, int] = (1280, 720)
    fov_degrees: float = 90.0
    capture_interval_ms: int = 100
    output_dir: Path = field(default_factory=lambda: Path("data/vision/lightweight"))
    save_images: bool = True
    vehicle_name: str = "drone_001"


class LightweightBridge:
    """Bridge that provides AirSim-compatible API using lightweight physics.

    Drop-in replacement for AirSimBridge for testing without Unreal.

    Example:
        # Instead of:
        # bridge = AirSimBridge(config)

        # Use:
        bridge = LightweightBridge(config)
        await bridge.connect()

        # Same API
        state = await bridge.get_vehicle_state()
        result = await bridge.capture_frame()
    """

    def __init__(
        self,
        config: LightweightCameraConfig | None = None,
        simulator: LightweightSim | None = None,
    ) -> None:
        """Initialize the lightweight bridge.

        Args:
            config: Camera/capture configuration
            simulator: Optional existing simulator. Creates one if None.
        """
        self.config = config or LightweightCameraConfig()
        self._sim = simulator
        self._own_sim = simulator is None  # Track if we created the sim

        self.connected = False
        self.capture_count = 0
        self.last_capture_time: datetime | None = None

        # Ensure output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LightweightBridge initialized (vehicle: {self.config.vehicle_name})")

    async def connect(self) -> bool:
        """Connect to the simulator.

        Returns:
            True if connected successfully
        """
        try:
            if self._sim is None:
                # Create default simulator
                self._sim = self._create_default_sim()

            if not self._sim.is_running:
                await self._sim.start()

            # Ensure our vehicle exists
            if self.config.vehicle_name not in self._sim.drones:
                self._sim.add_drone(self.config.vehicle_name)

            self.connected = True
            logger.info("Connected to lightweight simulator")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from simulator."""
        if self._own_sim and self._sim:
            await self._sim.stop()
        self.connected = False
        logger.info("Disconnected from lightweight simulator")

    def _create_default_sim(self) -> LightweightSim:
        """Create a default simulator."""
        env = EnvironmentConfig(
            wind_speed_ms=5.0,
            wind_gust_intensity=0.2,
            wind_turbulence=0.1,
        )
        sim = LightweightSim(env_config=env)

        # Add vehicle
        sim.add_drone(self.config.vehicle_name, DroneConfig())

        # Add some assets
        for i in range(6):
            x = 30 + (i % 3) * 20
            y = 20 + (i // 3) * 20
            sim.add_asset(
                f"asset_{i:03d}",
                np.array([float(x), float(y), 0.0]),
                "solar_panel",
                has_anomaly=(i == 3),  # One has anomaly
                anomaly_severity=0.7 if i == 3 else 0.0,
            )

        return sim

    async def capture_frame(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> CaptureResult:
        """Capture a frame (generates synthetic image).

        Args:
            metadata: Optional metadata to include

        Returns:
            CaptureResult with image path and capture info
        """
        if not self.connected or not self._sim:
            return CaptureResult(
                success=False,
                timestamp=datetime.now(),
                image_path=None,
                camera_state=self._get_camera_state(error="Not connected"),
                metadata={"error": "Not connected"},
            )

        try:
            # Get drone state for image generation
            state = self._sim.get_vehicle_state(self.config.vehicle_name)
            if not state:
                return CaptureResult(
                    success=False,
                    timestamp=datetime.now(),
                    image_path=None,
                    camera_state=self._get_camera_state(error="Vehicle not found"),
                    metadata={"error": "Vehicle not found"},
                )

            # Generate synthetic image
            img = self._generate_synthetic_image(state)

            # Save image
            self.capture_count += 1
            timestamp = datetime.now()
            filename = (
                f"lightweight_{timestamp.strftime('%Y%m%d_%H%M%S')}_{self.capture_count:04d}.png"
            )
            image_path = self.config.output_dir / filename

            if self.config.save_images:
                img.save(image_path)

            self.last_capture_time = timestamp

            capture_metadata = {
                "source": "lightweight_sim",
                "camera": self.config.camera_name,
                "vehicle": self.config.vehicle_name,
                "resolution": self.config.resolution,
                "sim_time": self._sim.get_sim_time(),
                "position": {
                    "lat": state.position.latitude,
                    "lon": state.position.longitude,
                    "alt": state.position.altitude_msl,
                },
                **(metadata or {}),
            }

            return CaptureResult(
                success=True,
                timestamp=timestamp,
                image_path=image_path,
                camera_state=self._get_camera_state(),
                metadata=capture_metadata,
            )

        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return CaptureResult(
                success=False,
                timestamp=datetime.now(),
                image_path=None,
                camera_state=self._get_camera_state(error=str(e)),
                metadata={"error": str(e)},
            )

    def _generate_synthetic_image(self, state: VehicleState) -> Image.Image:
        """Generate a synthetic image based on drone position.

        Creates a simple overhead view with ground, assets, and drone shadow.
        For testing, can be replaced with more sophisticated rendering.
        """
        width, height = self.config.resolution
        img = Image.new("RGB", (width, height), color=(34, 80, 22))  # Green ground

        # Would add more sophisticated rendering here
        # For now, just a placeholder with position info

        draw = ImageDraw.Draw(img)

        # Draw grid
        for i in range(0, width, 50):
            draw.line([(i, 0), (i, height)], fill=(40, 90, 28), width=1)
        for i in range(0, height, 50):
            draw.line([(0, i), (width, i)], fill=(40, 90, 28), width=1)

        # Draw assets as rectangles
        center_x, center_y = width // 2, height // 2
        if self._sim:
            scale = 5  # pixels per meter

            for asset in self._sim.world.assets.values():
                # Position relative to drone
                dx = asset.position[1] - state.position.longitude
                dy = asset.position[0] - state.position.latitude

                ax = center_x + int(dx * scale)
                ay = center_y - int(dy * scale)

                if 0 <= ax < width and 0 <= ay < height:
                    color = (200, 50, 50) if asset.has_anomaly else (26, 35, 126)
                    draw.rectangle(
                        [ax - 15, ay - 10, ax + 15, ay + 10],
                        fill=color,
                        outline=(255, 255, 255),
                    )

        # Draw crosshair at center
        draw.line(
            [(center_x - 20, center_y), (center_x + 20, center_y)], fill=(255, 255, 0), width=2
        )
        draw.line(
            [(center_x, center_y - 20), (center_x, center_y + 20)], fill=(255, 255, 0), width=2
        )

        # Add info text
        info_text = (
            f"Alt: {state.position.altitude_msl:.1f}m | Bat: {state.battery.remaining_percent:.0f}%"
        )
        draw.text((10, 10), info_text, fill=(255, 255, 255))

        return img

    async def get_vehicle_state(self) -> VehicleState | None:
        """Get current vehicle state.

        Returns:
            VehicleState or None if not connected
        """
        if not self.connected or not self._sim:
            return None

        return self._sim.get_vehicle_state(self.config.vehicle_name)

    async def set_weather(
        self,
        rain: float = 0.0,
        snow: float = 0.0,
        fog: float = 0.0,
        dust: float = 0.0,
    ) -> bool:
        """Set weather conditions.

        Maps to wind/turbulence in lightweight sim.
        """
        if not self.connected or not self._sim:
            return False

        # Map weather to wind turbulence
        turbulence = max(rain, snow, fog, dust) * 0.5
        self._sim.set_wind(
            self._sim.env_config.wind_speed_ms,
            math.degrees(self._sim.env_config.wind_direction_rad),
            gust_intensity=turbulence,
            turbulence=turbulence,
        )

        logger.info(f"Weather set (mapped to turbulence: {turbulence})")
        return True

    async def set_time_of_day(
        self,
        hour: int = 12,
        _is_enabled: bool = True,
        _celestial_clock_speed: float = 1.0,
    ) -> bool:
        """Set time of day (no-op in lightweight sim)."""
        logger.info(f"Time of day set to {hour}:00 (visual only in lightweight)")
        return True

    # =========================================================================
    # Command API (same as simulator)
    # =========================================================================

    async def arm(self) -> bool:
        """Arm the vehicle."""
        if not self._sim:
            return False
        return self._sim.arm(self.config.vehicle_name)

    async def disarm(self) -> bool:
        """Disarm the vehicle."""
        if not self._sim:
            return False
        return self._sim.disarm(self.config.vehicle_name)

    async def takeoff(self, altitude: float = 10.0) -> bool:
        """Take off to specified altitude."""
        if not self._sim:
            return False
        return self._sim.takeoff(self.config.vehicle_name, altitude)

    async def land(self) -> bool:
        """Land the vehicle."""
        if not self._sim:
            return False
        return self._sim.land(self.config.vehicle_name)

    async def goto(self, position: Position, yaw: float = 0.0) -> bool:
        """Go to position."""
        if not self._sim:
            return False
        return self._sim.goto(self.config.vehicle_name, position, yaw)

    async def rtl(self) -> bool:
        """Return to launch."""
        if not self._sim:
            return False
        return self._sim.rtl(self.config.vehicle_name)

    def _get_camera_state(self, error: str | None = None) -> CameraState:
        """Build camera state object."""
        return CameraState(
            timestamp=datetime.now(),
            status=CameraStatus.ERROR if error else CameraStatus.READY,
            resolution=self.config.resolution,
            capture_format="RGB",
            total_captures=self.capture_count,
            last_capture_time=self.last_capture_time,
            error_message=error,
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected

    @property
    def simulator(self) -> LightweightSim | None:
        """Get the underlying simulator."""
        return self._sim


# Alias for drop-in replacement
LightweightCameraConfig.__doc__ = """Configuration matching AirSimCameraConfig."""
