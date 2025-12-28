"""
Simulated Camera

Generates synthetic images for testing with probabilistic defect injection.
Useful for testing the vision pipeline without real camera hardware.
"""

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from vision.camera.base import BaseCameraConfig
from vision.data_models import CameraState, CameraStatus, CaptureResult, DetectionClass

logger = logging.getLogger(__name__)


@dataclass
class DefectConfig:
    """
    Configuration for probabilistic defect injection in simulated images.

    Each field represents the probability (0.0-1.0) of injecting that defect type.
    """

    crack_probability: float = 0.10  # 10% chance
    corrosion_probability: float = 0.08  # 8% chance
    structural_damage_probability: float = 0.03  # 3% chance
    discoloration_probability: float = 0.05  # 5% chance
    vegetation_probability: float = 0.05  # 5% chance
    damage_probability: float = 0.03  # 3% chance

    # Severity range for injected defects
    severity_min: float = 0.3
    severity_max: float = 0.9

    def get_total_probability(self) -> float:
        """Get total probability of any defect occurring."""
        return (
            self.crack_probability
            + self.corrosion_probability
            + self.structural_damage_probability
            + self.discoloration_probability
            + self.vegetation_probability
            + self.damage_probability
        )

    def sample_defect(self) -> tuple[DetectionClass, float] | None:
        """
        Sample a random defect based on configured probabilities.

        Returns:
            (defect_type, severity) tuple if defect occurs, None otherwise
        """
        rand = random.random()
        cumulative = 0.0

        # Check each defect type in order
        for defect_type, probability in [
            (DetectionClass.CRACK, self.crack_probability),
            (DetectionClass.CORROSION, self.corrosion_probability),
            (DetectionClass.STRUCTURAL_DAMAGE, self.structural_damage_probability),
            (DetectionClass.DISCOLORATION, self.discoloration_probability),
            (DetectionClass.VEGETATION_OVERGROWTH, self.vegetation_probability),
            (DetectionClass.STRUCTURAL_DAMAGE, self.damage_probability),
        ]:
            cumulative += probability
            if rand < cumulative:
                severity = random.uniform(self.severity_min, self.severity_max)
                return (defect_type, severity)

        return None


class SimulatedCameraConfig(BaseCameraConfig):
    """Configuration for simulated camera."""

    def __init__(
        self,
        resolution: tuple[int, int] = (1920, 1080),
        capture_format: str = "RGB",
        enabled: bool = True,
        output_dir: Path | str = "data/vision/simulated",
        defect_config: DefectConfig | None = None,
        save_images: bool = True,
    ):
        """
        Initialize simulated camera configuration.

        Args:
            resolution: Image resolution
            capture_format: Image format
            enabled: Whether camera is enabled
            output_dir: Directory to save simulated images
            defect_config: Defect injection configuration
            save_images: Whether to actually save image files (False for faster testing)
        """
        super().__init__(resolution, capture_format, enabled)
        self.output_dir = Path(output_dir)
        self.defect_config = defect_config or DefectConfig()
        self.save_images = save_images


class SimulatedCamera:
    """
    Simulated camera for testing.

    Generates synthetic images with configurable defect injection.
    Useful for testing vision pipeline without real hardware.
    """

    def __init__(self, config: SimulatedCameraConfig | None = None):
        """
        Initialize simulated camera.

        Args:
            config: Camera configuration
        """
        self.config = config or SimulatedCameraConfig()
        self.logger = logger
        self._capture_count = 0
        self._status = CameraStatus.OFFLINE
        self._last_capture_time: datetime | None = None
        self._error_message: str | None = None

        # Create output directory if saving images
        if self.config.save_images:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> bool:
        """
        Initialize the simulated camera.

        Always succeeds for simulated camera.

        Returns:
            True (always successful)
        """
        self.logger.info("Initializing simulated camera")
        self._status = CameraStatus.READY
        self._error_message = None
        return True

    async def capture(self, vehicle_state: dict | None = None) -> CaptureResult:
        """
        Capture a simulated image.

        Generates a synthetic image and optionally injects defects.

        Args:
            vehicle_state (dict | None): Optional vehicle state for metadata.

        Returns:
            CaptureResult: Capture result with image path and injected defect info
            (:class:`vision.data_models.CaptureResult`).
        """
        if self._status != CameraStatus.READY:
            return CaptureResult(
                success=False,
                camera_state=self.get_state(),
                error_message=f"Camera not ready (status: {self._status.value})",
            )

        self._status = CameraStatus.CAPTURING

        try:
            # Generate image path
            timestamp = datetime.now()
            image_filename = f"sim_capture_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.png"
            image_path = self.config.output_dir / image_filename

            # Sample for defect injection
            defect_info = self.config.defect_config.sample_defect()

            # Build metadata
            metadata: dict[str, Any] = {
                "capture_number": self._capture_count + 1,
                "simulated": True,
                "timestamp": timestamp.isoformat(),
            }

            if defect_info:
                defect_type, severity = defect_info
                metadata["injected_defect"] = {
                    "type": defect_type.value,
                    "severity": severity,
                }

            if vehicle_state:
                metadata["vehicle_state"] = vehicle_state

            # Generate synthetic image and save metadata
            if self.config.save_images:
                self._generate_synthetic_image(image_path, defect_info)
                # Save metadata sidecar file
                self._save_metadata(image_path, metadata)

            # Update state
            self._capture_count += 1
            self._last_capture_time = timestamp
            self._status = CameraStatus.READY

            return CaptureResult(
                success=True,
                timestamp=timestamp,
                image_path=image_path if self.config.save_images else None,
                camera_state=self.get_state(),
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Simulated capture failed: {e}")
            self._status = CameraStatus.ERROR
            self._error_message = str(e)

            return CaptureResult(
                success=False,
                camera_state=self.get_state(),
                error_message=str(e),
            )

    def get_state(self) -> CameraState:
        """
        Get current camera state.

        Returns:
            CameraState: Camera state snapshot (:class:`vision.data_models.CameraState`).
        """
        return CameraState(
            timestamp=datetime.now(),
            status=self._status,
            resolution=self.config.resolution,
            capture_format=self.config.capture_format,
            total_captures=self._capture_count,
            last_capture_time=self._last_capture_time,
            error_message=self._error_message,
        )

    async def shutdown(self) -> None:
        """
        Shutdown simulated camera.

        No-op for simulated camera, but logs the shutdown.
        """
        self.logger.info(f"Shutting down simulated camera (captured {self._capture_count} images)")
        self._status = CameraStatus.OFFLINE

    def _generate_synthetic_image(
        self, output_path: Path, defect_info: tuple[DetectionClass, float] | None
    ) -> None:
        """
        Generate a synthetic image file.

        Creates a simple colored image with text overlay indicating defect (if any).

        Args:
            output_path: Where to save the image
            defect_info: (defect_type, severity) if defect injected, None otherwise
        """
        if Image is None:
            # PIL not available - just create empty file
            output_path.touch()
            return

        width, height = self.config.resolution

        # Create base image with random color (simulating different lighting conditions)
        base_color = (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200),
        )
        image = Image.new("RGB", (width, height), base_color)

        # Add some noise to make it look more realistic
        noise = np.random.normal(0, 10, (height, width, 3))
        image_array = np.array(image) + noise
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_array)

        # If defect injected, modify image (for now, just save as-is)
        # In the future, could draw defect shapes, textures, etc.
        if defect_info:
            _defect_type, _severity = defect_info
            # TODO: Could draw simulated defects here
            # For now, defect info is stored in metadata

        # Save image
        image.save(output_path)
        self.logger.debug(f"Generated synthetic image: {output_path}")

    def _save_metadata(self, image_path: Path, metadata: dict[str, Any]) -> None:
        """
        Save metadata sidecar file for image.

        Args:
            image_path: Path to image file
            metadata: Metadata dictionary to save
        """
        metadata_path = image_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.logger.debug(f"Saved metadata sidecar: {metadata_path}")
