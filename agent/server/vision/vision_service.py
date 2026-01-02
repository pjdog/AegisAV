"""Vision Service.

Server-side vision orchestration.
Handles detailed analysis, anomaly creation, and observation tracking.
"""

import asyncio
import logging
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agent.server.models.vision_models import (
    CameraMetadata,
    VisionObservation,
)
from agent.server.vision.detector import SimulatedDetector
from agent.server.world_model import Anomaly, WorldModel
from autonomy.vehicle_state import Position
from mapping.splat_change_detection import SplatChangeConfig, detect_splat_change
from vision.data_models import BoundingBox, Detection, DetectionClass, DetectionResult
from vision.image_manager import ImageManager

if TYPE_CHECKING:
    from agent.server.unreal_stream import UnrealConnectionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionServiceConfig:
    """Configuration for server-side vision processing.

    Attributes:
        confidence_threshold (float): Minimum confidence to create an anomaly.
        severity_threshold (float): Minimum severity to create an anomaly.
    """

    confidence_threshold: float = 0.7
    severity_threshold: float = 0.4
    enable_splat_change_detection: bool = False
    splat_change_threshold: float = 0.85
    splat_change_min_age_s: float = 600.0
    splat_change_max_age_s: float = 3600.0


class VisionService:
    """Server-side vision service.

    Responsibilities:
    - Detailed image analysis
    - Anomaly creation and deduplication
    - Vision observation tracking
    - Integration with world model
    """

    def __init__(
        self,
        world_model: WorldModel,
        detector: SimulatedDetector | None = None,
        image_manager: ImageManager | None = None,
        config: VisionServiceConfig | None = None,
        unreal_manager: "UnrealConnectionManager | None" = None,
        splat_scene_provider: Callable[[], Path | None] | None = None,
    ) -> None:
        """Initialize vision service.

        Args:
            world_model (WorldModel): World model for anomaly creation.
            detector (SimulatedDetector | None): Detector instance (creates
                :class:`agent.server.vision.detector.SimulatedDetector` if None).
            image_manager (ImageManager | None): Image manager (creates default if None).
            config (VisionServiceConfig | None): Vision service configuration.
            unreal_manager (UnrealConnectionManager | None): Unreal connection manager
                for broadcasting anomaly detections.
        """
        self.world_model = world_model
        self.config = config or VisionServiceConfig()
        self.logger = logger
        self.unreal_manager = unreal_manager
        self.splat_scene_provider = splat_scene_provider

        # Create default instances if not provided
        if detector is None:
            detector = SimulatedDetector(
                defect_probability=0.15,  # 15% chance of detecting defects
                severity_range=(0.3, 0.9),
                confidence_boost=0.1,
            )

        if image_manager is None:
            image_manager = ImageManager(base_dir="data/vision/server_images")

        self.detector = detector
        self.image_manager = image_manager

        # Track observations
        self.observations: dict[str, VisionObservation] = {}

    async def initialize(self) -> bool:
        """Initialize the vision service."""
        self.logger.info("Initializing VisionService")

        try:
            detector_ok = await self.detector.initialize()
            if not detector_ok:
                self.logger.error("Detector initialization failed")
                return False

            self.logger.info("VisionService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"VisionService initialization failed: {e}")
            return False

    async def process_inspection_result(
        self,
        asset_id: str,
        client_detection: DetectionResult | None = None,
        image_path: Path | None = None,
        vehicle_state: dict | None = None,
    ) -> VisionObservation:
        """Process inspection result with detailed server-side analysis.

        Args:
            asset_id (str): Asset being inspected.
            client_detection (DetectionResult | None): Optional client-side detection result,
                typically a :class:`vision.data_models.DetectionResult`.
            image_path (Path | None): Optional image path for analysis
                (a :class:`pathlib.Path`).
            vehicle_state (dict | None): Optional vehicle state at time of capture.

        Returns:
            VisionObservation: Analysis results and any anomaly created.
        """
        observation_id = str(uuid.uuid4())

        self.logger.info(
            f"Processing inspection result for asset {asset_id} (observation: {observation_id})"
        )

        # Perform detailed server-side analysis
        if image_path and image_path.exists():
            detection = await self.detector.analyze_image(
                image_path=image_path, client_detection=client_detection
            )
        elif client_detection:
            # No image available - use client detection
            detection = client_detection
        else:
            # Simulate detection
            detection = await self.simulate_inspection_result(asset_id)

        detection = self._apply_splat_change_detection(detection, image_path)

        # Extract position from vehicle state
        position = None
        altitude_agl = None
        heading_deg = None
        distance_to_asset = None

        if vehicle_state:
            pos_data = vehicle_state.get("position", {})
            if pos_data:
                position = Position(
                    latitude=pos_data.get("latitude", 0.0),
                    longitude=pos_data.get("longitude", 0.0),
                    altitude_msl=pos_data.get("altitude_msl", 0.0),
                )
            altitude_agl = vehicle_state.get("altitude_agl")
            heading_deg = vehicle_state.get("heading_deg")
            distance_to_asset = vehicle_state.get("distance_to_asset")

        detection_payloads = [d.model_dump(mode="json") for d in detection.detections]
        detected_labels = [d.detection_class.value for d in detection.detections]

        # Create observation
        observation = VisionObservation(
            observation_id=observation_id,
            asset_id=asset_id,
            timestamp=datetime.now(),
            image_path=image_path,
            camera_metadata=CameraMetadata(camera_type="simulated"),
            position=position,
            altitude_agl=altitude_agl,
            heading_deg=heading_deg,
            distance_to_asset=distance_to_asset,
            detections=detection_payloads,
            max_confidence=detection.max_confidence,
            max_severity=detection.max_severity,
            defect_detected=len(detection.detected_defects) > 0,
            model_name=detection.model_name,
            inference_time_ms=detection.inference_time_ms,
            processed_on_server=True,
        )

        # Store observation
        self.observations[observation_id] = observation

        if detected_labels:
            defect_labels = [d.detection_class.value for d in detection.detected_defects]
            self.logger.info(
                "Vision detections for asset %s: %s (defects: %s, max_conf=%.2f, max_sev=%.2f)",
                asset_id,
                ", ".join(detected_labels),
                ", ".join(defect_labels) if defect_labels else "none",
                detection.max_confidence,
                detection.max_severity,
            )

        # Check if we should create an anomaly
        if self._should_create_anomaly(detection):
            anomaly = await self._create_anomaly(asset_id, detection, observation)
            if anomaly:
                observation.anomaly_created = True
                observation.anomaly_id = anomaly.anomaly_id
                self.logger.info(f"✨ Anomaly created: {anomaly.anomaly_id} for asset {asset_id}")

        return observation

    def _apply_splat_change_detection(
        self,
        detection: DetectionResult,
        image_path: Path | None,
    ) -> DetectionResult:
        if not self.config.enable_splat_change_detection:
            return detection

        if not image_path:
            return detection

        if not self.splat_scene_provider:
            return detection

        splat_scene = self.splat_scene_provider()
        if not splat_scene:
            return detection

        change_result = detect_splat_change(
            image_path,
            splat_scene,
            SplatChangeConfig(
                enabled=True,
                min_age_s=self.config.splat_change_min_age_s,
                max_age_s=self.config.splat_change_max_age_s,
            ),
        )

        metadata = dict(detection.metadata)
        metadata["splat_change"] = change_result.to_dict()

        detections = list(detection.detections)
        if change_result.available and change_result.change_score >= self.config.splat_change_threshold:
            detections.append(
                Detection(
                    detection_class=DetectionClass.SCENE_CHANGE,
                    confidence=change_result.confidence,
                    severity=change_result.change_score,
                    bounding_box=BoundingBox(
                        x_min=0.0,
                        y_min=0.0,
                        x_max=1.0,
                        y_max=1.0,
                    ),
                    metadata={
                        "source": "splat_change_detection",
                        "scene_path": change_result.scene_path,
                    },
                )
            )

        return DetectionResult(
            timestamp=detection.timestamp,
            detections=detections,
            image_path=detection.image_path,
            model_name=detection.model_name,
            inference_time_ms=detection.inference_time_ms,
            metadata=metadata,
        )

    async def simulate_inspection_result(self, asset_id: str) -> DetectionResult:
        """Simulate inspection result without actual image.

        Used when client doesn't provide image but indicates anomaly detected.

        Args:
            asset_id (str): Asset ID.

        Returns:
            DetectionResult: Simulated detection result
            (:class:`vision.data_models.DetectionResult`).
        """
        self.logger.debug(f"Simulating inspection result for asset {asset_id}")

        # Use detector to simulate
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / f"simulated_{asset_id}.png"
            return await self.detector.analyze_image(fake_path, client_detection=None)

    def _should_create_anomaly(self, detection: DetectionResult) -> bool:
        """Check if detection meets thresholds for anomaly creation.

        Args:
            detection (DetectionResult): Detection result.

        Returns:
            bool: True if anomaly should be created.
        """
        if not detection.detected_defects:
            return False

        # Check thresholds
        meets_confidence = detection.max_confidence >= self.config.confidence_threshold
        meets_severity = detection.max_severity >= self.config.severity_threshold

        return meets_confidence and meets_severity

    async def _create_anomaly(
        self,
        asset_id: str,
        detection: DetectionResult,
        observation: VisionObservation,
    ) -> Anomaly | None:
        """Create anomaly from detection result.

        Includes deduplication - won't create duplicate anomalies for same asset.

        Args:
            asset_id: Asset ID
            detection: Detection result
            observation: Vision observation

        Returns:
            Created Anomaly or None
        """
        # Check for existing anomaly on this asset
        anomaly_asset_ids = self.world_model.get_anomaly_assets()

        if asset_id in anomaly_asset_ids:
            # Anomaly already exists - update severity if worse
            snapshot = self.world_model.get_snapshot()
            existing = next(
                (a for a in snapshot.anomalies if a.asset_id == asset_id),
                None,
            )
            if existing and detection.max_severity > existing.severity:
                self.logger.info(
                    "Updating existing anomaly severity: %.2f -> %.2f",
                    existing.severity,
                    detection.max_severity,
                )
                # Note: WorldModel doesn't have update method, so we just log
            return None

        # Create new anomaly
        defect_types = [d.detection_class.value for d in detection.detected_defects]
        description = f"Vision detected: {', '.join(defect_types)}"

        anomaly = Anomaly(
            anomaly_id=str(uuid.uuid4()),
            asset_id=asset_id,
            detected_at=datetime.now(),
            severity=detection.max_severity,
            description=description,
            position=observation.position,
            acknowledged=False,
            resolved=False,
        )

        # Add to world model
        self.world_model.add_anomaly(anomaly)

        self.logger.info(
            "Created anomaly %s: %s (severity: %.2f)",
            anomaly.anomaly_id,
            description,
            anomaly.severity,
        )

        # Broadcast to Unreal for visualization
        if detection.detected_defects:
            primary_defect = detection.detected_defects[0]
            asyncio.create_task(
                self._broadcast_anomaly_to_unreal(anomaly, primary_defect, observation)
            )

        return anomaly

    async def _broadcast_anomaly_to_unreal(
        self,
        anomaly: Anomaly,
        detection: Detection,
        observation: VisionObservation,
    ) -> None:
        """Broadcast anomaly detection to Unreal for visualization.

        Args:
            anomaly: The created anomaly
            detection: The primary detection that triggered the anomaly
            observation: The vision observation context
        """
        if not self.unreal_manager:
            return

        try:
            from agent.server.unreal_stream import (
                AnomalyDetectionMessage,
                UnrealMessageType,
            )

            # Build the message
            message = AnomalyDetectionMessage(
                anomaly_id=anomaly.anomaly_id,
                asset_id=anomaly.asset_id,
                severity=anomaly.severity,
                defect_type=detection.detection_class.value,
                confidence=detection.confidence,
                description=anomaly.description,
                bbox_x=detection.bounding_box.x if detection.bounding_box else None,
                bbox_y=detection.bounding_box.y if detection.bounding_box else None,
                bbox_width=detection.bounding_box.width if detection.bounding_box else None,
                bbox_height=detection.bounding_box.height if detection.bounding_box else None,
                latitude=observation.position.latitude if observation.position else None,
                longitude=observation.position.longitude if observation.position else None,
                altitude_m=observation.altitude_agl,
                timestamp=anomaly.detected_at.isoformat(),
            )

            # Broadcast to all connected Unreal clients
            await self.unreal_manager.broadcast(
                {
                    "type": UnrealMessageType.ANOMALY_DETECTED.value,
                    **message.model_dump(),
                }
            )

            self.logger.debug(
                "Broadcast anomaly %s to Unreal (asset: %s, defect: %s)",
                anomaly.anomaly_id,
                anomaly.asset_id,
                detection.detection_class.value,
            )

        except Exception as e:
            self.logger.warning("Failed to broadcast anomaly to Unreal: %s", e)

    def get_observations_for_asset(self, asset_id: str) -> list[VisionObservation]:
        """Get all vision observations for an asset.

        Args:
            asset_id: Asset ID

        Returns:
            List of observations
        """
        return [obs for obs in self.observations.values() if obs.asset_id == asset_id]

    def get_recent_observations(self, limit: int = 100) -> list[VisionObservation]:
        """Get most recent observations.

        Args:
            limit: Maximum number to return

        Returns:
            List of recent observations (newest first)
        """
        observations = sorted(
            self.observations.values(),
            key=lambda o: o.timestamp,
            reverse=True,
        )
        return observations[:limit]

    def get_statistics(self) -> dict[str, Any]:
        """Get vision service statistics.

        Returns:
            dict[str, Any]: Statistics dictionary.
        """
        total_observations = len(self.observations)
        defects_detected = sum(1 for obs in self.observations.values() if obs.defect_detected)
        anomalies_created = sum(1 for obs in self.observations.values() if obs.anomaly_created)

        avg_confidence = (
            sum(obs.max_confidence for obs in self.observations.values()) / total_observations
            if total_observations > 0
            else 0.0
        )
        avg_severity = (
            sum(obs.max_severity for obs in self.observations.values()) / total_observations
            if total_observations > 0
            else 0.0
        )

        return {
            "total_observations": total_observations,
            "defects_detected": defects_detected,
            "anomalies_created": anomalies_created,
            "detection_rate": defects_detected / total_observations
            if total_observations > 0
            else 0.0,
            "anomaly_rate": anomalies_created / total_observations
            if total_observations > 0
            else 0.0,
            "average_confidence": round(avg_confidence, 3),
            "average_severity": round(avg_severity, 3),
        }

    def seed_navigation_map(self, assets: list[Any], scenario_id: str) -> dict[str, Any]:
        """Seed a navigation map from known assets for initial avoidance."""
        from agent.server.navigation_map import build_navigation_map

        nav_map = build_navigation_map(assets, scenario_id, source="vision_seed")
        self.logger.info(
            "vision_map_seeded",
            scenario_id=scenario_id,
            obstacle_count=len(nav_map.get("obstacles", [])),
        )
        return nav_map

    async def shutdown(self) -> None:
        """Shutdown the vision service."""
        self.logger.info("Shutting down VisionService")
        await self.detector.shutdown()
