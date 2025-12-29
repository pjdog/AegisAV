"""AirSim Bridge for AegisAV

Provides integration between AirSim (Unreal Engine) and the AegisAV vision pipeline.
Captures high-fidelity camera frames from the simulated environment.
"""

import io
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import airsim

    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    airsim = None

from PIL import Image

from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    Position,
    VehicleState,
    Velocity,
)
from vision.data_models import CameraState, CameraStatus, CaptureResult

logger = logging.getLogger(__name__)


class AirSimImageType(Enum):
    """AirSim camera image types."""

    SCENE = 0  # RGB scene
    DEPTH_PLANNER = 1  # Depth from planner perspective
    DEPTH_PERSPECTIVE = 2  # Depth from camera perspective
    DEPTH_VIS = 3  # Depth visualization
    SEGMENTATION = 5  # Semantic segmentation
    INFRARED = 7  # Infrared


@dataclass
class AirSimCameraConfig:
    """Configuration for AirSim camera capture."""

    camera_name: str = "front_center"
    image_type: AirSimImageType = AirSimImageType.SCENE
    resolution: tuple[int, int] = (1920, 1080)
    fov_degrees: float = 90.0
    capture_interval_ms: int = 100  # 10 FPS default
    output_dir: Path = field(default_factory=lambda: Path("data/vision/airsim"))
    save_images: bool = True
    compress: bool = False  # PNG vs JPEG
    vehicle_name: str = "Drone1"


class AirSimBridge:
    """Bridge between AirSim and AegisAV vision pipeline.

    Captures camera frames from the Unreal Engine rendered environment
    and converts them to the format expected by the vision pipeline.

    Example:
        bridge = AirSimBridge(config)
        await bridge.connect()

        # Capture frame
        result = await bridge.capture_frame()
        if result.success:
            # Pass to vision pipeline
            detection = await detector.analyze_image(result.image_path)
    """

    def __init__(self, config: AirSimCameraConfig | None = None) -> None:
        """Initialize AirSim bridge."""
        if not AIRSIM_AVAILABLE:
            raise ImportError("airsim package not installed. Install with: pip install airsim")

        self.config = config or AirSimCameraConfig()
        self.client: airsim.MultirotorClient | None = None
        self.connected = False
        self.capture_count = 0
        self.last_capture_time: datetime | None = None

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AirSimBridge initialized (camera: {self.config.camera_name})")

    async def connect(self) -> bool:
        """Connect to AirSim simulator.

        Returns:
            True if connected successfully
        """
        try:
            logger.info("Connecting to AirSim...")

            # Create client and confirm connection
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()

            # Enable API control
            self.client.enableApiControl(True, self.config.vehicle_name)

            self.connected = True
            logger.info("Connected to AirSim successfully")

            # Log camera info
            camera_info = self.client.simGetCameraInfo(
                self.config.camera_name, self.config.vehicle_name
            )
            logger.info(f"Camera info: FOV={camera_info.fov}, Pose={camera_info.pose}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to AirSim: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from AirSim."""
        if self.client:
            try:
                self.client.enableApiControl(False, self.config.vehicle_name)
            except Exception as exc:
                logger.warning("Failed to release API control: %s", exc)
            self.client = None
        self.connected = False
        logger.info("Disconnected from AirSim")

    async def capture_frame(self, metadata: dict[str, Any] | None = None) -> CaptureResult:
        """Capture a frame from the AirSim camera.

        Args:
            metadata: Optional metadata to include with capture

        Returns:
            CaptureResult with image path and capture info
        """
        if not self.connected or not self.client:
            return CaptureResult(
                success=False,
                timestamp=datetime.now(),
                image_path=None,
                camera_state=self._get_camera_state(error="Not connected"),
                metadata={"error": "Not connected to AirSim"},
            )

        try:
            start_time = time.time()

            # Request image from AirSim
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        self.config.camera_name,
                        self.config.image_type.value,
                        pixels_as_float=False,
                        compress=self.config.compress,
                    )
                ],
                self.config.vehicle_name,
            )

            if not responses or len(responses) == 0:
                return CaptureResult(
                    success=False,
                    timestamp=datetime.now(),
                    image_path=None,
                    camera_state=self._get_camera_state(error="No response"),
                    metadata={"error": "No image response from AirSim"},
                )

            response = responses[0]

            # Convert to numpy array
            if self.config.compress:
                # Compressed PNG/JPEG
                img_array = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img = Image.open(io.BytesIO(img_array))
            else:
                # Uncompressed RGB
                img_array = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(
                    response.height, response.width, 3
                )
                img = Image.fromarray(img_array)

            # Generate filename
            self.capture_count += 1
            timestamp = datetime.now()
            filename = f"airsim_{timestamp.strftime('%Y%m%d_%H%M%S')}_{self.capture_count:04d}.png"
            image_path = self.config.output_dir / filename

            # Save image
            if self.config.save_images:
                img.save(image_path)

            # Get vehicle pose for metadata
            pose = self.client.simGetVehiclePose(self.config.vehicle_name)
            vehicle_position = Position(
                latitude=pose.position.x_val,  # Note: AirSim uses NED, convert as needed
                longitude=pose.position.y_val,
                altitude_msl=abs(pose.position.z_val),
                altitude_agl=abs(pose.position.z_val),
            )

            capture_time = (time.time() - start_time) * 1000
            self.last_capture_time = timestamp

            # Build metadata
            capture_metadata = {
                "source": "airsim",
                "camera": self.config.camera_name,
                "vehicle": self.config.vehicle_name,
                "resolution": (response.width, response.height),
                "capture_time_ms": capture_time,
                "pose": {
                    "position": [pose.position.x_val, pose.position.y_val, pose.position.z_val],
                    "orientation": [
                        pose.orientation.w_val,
                        pose.orientation.x_val,
                        pose.orientation.y_val,
                        pose.orientation.z_val,
                    ],
                },
                "vehicle_position": {
                    "latitude": vehicle_position.latitude,
                    "longitude": vehicle_position.longitude,
                    "altitude_msl": vehicle_position.altitude_msl,
                    "altitude_agl": vehicle_position.altitude_agl,
                },
                **(metadata or {}),
            }

            logger.debug(f"Captured frame {self.capture_count} in {capture_time:.1f}ms")

            return CaptureResult(
                success=True,
                timestamp=timestamp,
                image_path=image_path,
                camera_state=self._get_camera_state(),
                metadata=capture_metadata,
            )

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return CaptureResult(
                success=False,
                timestamp=datetime.now(),
                image_path=None,
                camera_state=self._get_camera_state(error=str(e)),
                metadata={"error": str(e)},
            )

    async def get_vehicle_state(self) -> VehicleState | None:
        """Get current vehicle state from AirSim.

        Returns:
            VehicleState or None if not connected
        """
        if not self.connected or not self.client:
            return None

        try:
            # Get state from AirSim
            state = self.client.getMultirotorState(self.config.vehicle_name)
            pose = state.kinematics_estimated

            # Convert to our VehicleState format
            # Note: AirSim uses NED coordinates, may need conversion for real GPS
            return VehicleState(
                timestamp=datetime.now(),
                position=Position(
                    latitude=pose.position.x_val,
                    longitude=pose.position.y_val,
                    altitude_msl=abs(pose.position.z_val),
                    altitude_agl=abs(pose.position.z_val),
                ),
                velocity=Velocity(
                    north=pose.linear_velocity.x_val,
                    east=pose.linear_velocity.y_val,
                    down=pose.linear_velocity.z_val,
                ),
                attitude=Attitude(
                    roll=airsim.to_eularian_angles(pose.orientation)[0],
                    pitch=airsim.to_eularian_angles(pose.orientation)[1],
                    yaw=airsim.to_eularian_angles(pose.orientation)[2],
                ),
                battery=BatteryState(
                    voltage=22.2,  # Simulated
                    current=5.0,
                    remaining_percent=100.0,  # TODO: Simulate drain
                ),
                mode=FlightMode.GUIDED,
                armed=state.landed_state != airsim.LandedState.Landed,
                in_air=state.landed_state == airsim.LandedState.Flying,
            )

        except Exception as e:
            logger.error(f"Failed to get vehicle state: {e}")
            return None

    async def set_weather(
        self, rain: float = 0.0, snow: float = 0.0, fog: float = 0.0, dust: float = 0.0
    ) -> bool:
        """Set weather conditions in the simulation.

        Args:
            rain: Rain intensity 0-1
            snow: Snow intensity 0-1
            fog: Fog density 0-1
            dust: Dust density 0-1

        Returns:
            True if successful
        """
        if not self.connected or not self.client:
            return False

        try:
            self.client.simEnableWeather(True)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, snow)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog)
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, dust)

            logger.info(f"Weather set: rain={rain}, snow={snow}, fog={fog}, dust={dust}")
            return True

        except Exception as e:
            logger.error(f"Failed to set weather: {e}")
            return False

    async def set_time_of_day(
        self, hour: int = 12, is_enabled: bool = True, celestial_clock_speed: float = 1.0
    ) -> bool:
        """Set time of day in the simulation.

        Args:
            hour: Hour of day (0-23)
            is_enabled: Enable time of day simulation
            celestial_clock_speed: Speed multiplier for day/night cycle

        Returns:
            True if successful
        """
        if not self.connected or not self.client:
            return False

        try:
            start_time = f"2024-06-15 {hour:02d}:00:00"
            self.client.simSetTimeOfDay(
                is_enabled=is_enabled,
                start_datetime=start_time,
                celestial_clock_speed=celestial_clock_speed,
            )

            logger.info(f"Time of day set to {hour}:00")
            return True

        except Exception as e:
            logger.error(f"Failed to set time of day: {e}")
            return False

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
        """Check if connected to AirSim."""
        return self.connected and self.client is not None
