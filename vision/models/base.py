"""Model Interface Protocol.

Defines the common interface for all detection models.
Supports YOLO, custom models, and mock detectors.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from vision.data_models import DetectionResult


@runtime_checkable
class ModelInterface(Protocol):
    """Protocol defining the detection model interface.

    All model implementations (YOLO, custom, mock) must implement this interface.
    """

    async def initialize(self) -> bool:
        """Initialize the model.

        Performs:
        - Model loading
        - Weight loading
        - Device setup (CPU/GPU)
        - Warm-up inference

        Returns:
            True if initialization successful, False otherwise
        """
        ...

    async def analyze_image(self, image_path: Path) -> DetectionResult:
        """Analyze an image for defects/objects.

        Args:
            image_path: Path to image file to analyze

        Returns:
            DetectionResult: Detection result with all detections found
            (:class:`vision.data_models.DetectionResult`).

        Note:
            Should be fast for client-side models (<100ms target)
            Can be slower for server-side models (<500ms acceptable)
        """
        ...

    def get_model_info(self) -> dict:
        """Get model information.

        Returns:
            Dictionary with model name, version, device, etc.
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully shutdown the model.

        Cleanup:
        - Release GPU memory
        - Close model handles
        - Save any cached data
        """
        ...
