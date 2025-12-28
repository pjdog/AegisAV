"""
Camera Interface Protocol

Defines the common interface for all camera implementations.
Supports both simulated cameras (testing) and real cameras (deployment).
"""

from typing import Protocol, runtime_checkable

from vision.data_models import CameraState, CaptureResult


@runtime_checkable
class CameraInterface(Protocol):
    """
    Protocol defining the camera interface.

    All camera implementations (simulated, real hardware) must implement this interface.
    This allows for seamless switching between simulated and real cameras.
    """

    async def initialize(self) -> bool:
        """
        Initialize the camera hardware/simulation.

        Performs any necessary setup:
        - Hardware initialization (for real cameras)
        - Connection establishment
        - Configuration loading
        - Self-test

        Returns:
            True if initialization successful, False otherwise
        """
        ...

    async def capture(self, vehicle_state: dict | None = None) -> CaptureResult:
        """
        Capture an image from the camera.

        Args:
            vehicle_state (dict | None): Optional vehicle state dict for metadata
                (position, altitude, heading, etc.).

        Returns:
            CaptureResult: Capture result with image path and metadata
            (:class:`vision.data_models.CaptureResult`).

        Note:
            For simulated cameras, this may inject synthetic defects.
            For real cameras, this triggers actual image capture.
        """
        ...

    def get_state(self) -> CameraState:
        """
        Get current camera state.

        Returns:
            CameraState snapshot with current status, resolution, capture count, etc.
        """
        ...

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the camera.

        Performs cleanup:
        - Close hardware connections
        - Save any buffered data
        - Release resources
        """
        ...


class BaseCameraConfig:
    """
    Base configuration for camera implementations.

    Subclass this for specific camera type configurations.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (1920, 1080),
        capture_format: str = "RGB",
        enabled: bool = True,
    ):
        """
        Initialize camera configuration.

        Args:
            resolution: Image resolution (width, height)
            capture_format: Image format (RGB, RGBA, grayscale)
            enabled: Whether camera is enabled
        """
        self.resolution = resolution
        self.capture_format = capture_format
        self.enabled = enabled
