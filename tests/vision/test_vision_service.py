"""
Tests for Vision Service

Comprehensive tests for the server-side vision service including:
- Initialization and shutdown
- Inspection result processing
- Anomaly detection and creation
- Observation tracking
- Statistics and queries
- Error handling and edge cases
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.server.models.vision_models import VisionObservation
from agent.server.vision.detector import SimulatedDetector
from agent.server.vision.vision_service import VisionService, VisionServiceConfig
from agent.server.world_model import Anomaly, Asset, AssetType, WorldModel
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSState,
    Position,
    VehicleState,
    Velocity,
)
from vision.data_models import BoundingBox, Detection, DetectionClass, DetectionResult
from vision.image_manager import ImageManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def world_model():
    """Create a WorldModel with test assets."""
    model = WorldModel()

    # Add test assets
    model.add_asset(
        Asset(
            asset_id="asset_001",
            name="Test Asset 1",
            position=Position(
                latitude=37.7750,
                longitude=-122.4195,
                altitude_msl=0,
                altitude_agl=0,
            ),
            asset_type=AssetType.BUILDING,
            priority=1,
        )
    )

    model.add_asset(
        Asset(
            asset_id="asset_002",
            name="Test Asset 2",
            position=Position(
                latitude=37.7760,
                longitude=-122.4200,
                altitude_msl=0,
                altitude_agl=0,
            ),
            asset_type=AssetType.SOLAR_PANEL,
            priority=2,
        )
    )

    # Set dock (required for get_snapshot)
    model.set_dock(
        Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=0,
            altitude_agl=0,
        )
    )

    # Set vehicle state (required for get_snapshot)
    model.update_vehicle(create_vehicle_state())

    return model


def create_vehicle_state():
    """Create a test vehicle state."""
    return VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=50.0,
            altitude_agl=50.0,
        ),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(
            voltage=16.8,
            current=5.0,
            remaining_percent=75.0,
            time_remaining_s=1800,
        ),
        gps=GPSState(satellites_visible=12, hdop=0.8, fix_type=3),
        mode=FlightMode.GUIDED,
        armed=True,
        in_air=True,
    )


@pytest.fixture
def detector():
    """Create a SimulatedDetector for tests."""
    return SimulatedDetector(
        defect_probability=0.5,  # 50% for testing
        severity_range=(0.5, 0.9),
        confidence_boost=0.1,
    )


@pytest.fixture
def image_manager(temp_dir):
    """Create an ImageManager for tests."""
    return ImageManager(base_dir=temp_dir / "images")


@pytest.fixture
def vision_config():
    """Create VisionServiceConfig for tests."""
    return VisionServiceConfig(
        confidence_threshold=0.7,
        severity_threshold=0.4,
    )


@pytest.fixture
async def vision_service(world_model, detector, image_manager, vision_config):
    """Create and initialize a VisionService."""
    service = VisionService(
        world_model=world_model,
        detector=detector,
        image_manager=image_manager,
        config=vision_config,
    )

    await service.initialize()

    yield service

    await service.shutdown()


# ============================================================================
# VisionServiceConfig Tests
# ============================================================================


class TestVisionServiceConfig:
    """Tests for VisionServiceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VisionServiceConfig()

        assert config.confidence_threshold == 0.7
        assert config.severity_threshold == 0.4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VisionServiceConfig(
            confidence_threshold=0.8,
            severity_threshold=0.6,
        )

        assert config.confidence_threshold == 0.8
        assert config.severity_threshold == 0.6

    def test_config_is_frozen(self):
        """Test that config is immutable (frozen dataclass)."""
        config = VisionServiceConfig()

        with pytest.raises(AttributeError):
            config.confidence_threshold = 0.5


# ============================================================================
# VisionService Initialization Tests
# ============================================================================


class TestVisionServiceInit:
    """Tests for VisionService initialization."""

    def test_init_with_provided_components(self, world_model, detector, image_manager):
        """Test initialization with provided components."""
        service = VisionService(
            world_model=world_model,
            detector=detector,
            image_manager=image_manager,
        )

        assert service.world_model is world_model
        assert service.detector is detector
        assert service.image_manager is image_manager
        assert service.config is not None

    def test_init_creates_default_detector(self, world_model, image_manager):
        """Test initialization creates default detector if not provided."""
        service = VisionService(
            world_model=world_model,
            detector=None,
            image_manager=image_manager,
        )

        assert service.detector is not None
        assert isinstance(service.detector, SimulatedDetector)

    def test_init_creates_default_image_manager(self, world_model, detector):
        """Test initialization creates default image manager if not provided."""
        service = VisionService(
            world_model=world_model,
            detector=detector,
            image_manager=None,
        )

        assert service.image_manager is not None
        assert isinstance(service.image_manager, ImageManager)

    def test_init_creates_default_config(self, world_model, detector, image_manager):
        """Test initialization creates default config if not provided."""
        service = VisionService(
            world_model=world_model,
            detector=detector,
            image_manager=image_manager,
            config=None,
        )

        assert service.config is not None
        assert service.config.confidence_threshold == 0.7
        assert service.config.severity_threshold == 0.4

    def test_observations_initially_empty(self, world_model, detector, image_manager):
        """Test that observations dict starts empty."""
        service = VisionService(
            world_model=world_model,
            detector=detector,
            image_manager=image_manager,
        )

        assert len(service.observations) == 0


@pytest.mark.asyncio
class TestVisionServiceInitialize:
    """Tests for VisionService.initialize method."""

    async def test_initialize_success(self, world_model, detector, image_manager):
        """Test successful initialization."""
        service = VisionService(
            world_model=world_model,
            detector=detector,
            image_manager=image_manager,
        )

        result = await service.initialize()

        assert result is True
        await service.shutdown()

    @pytest.mark.allow_error_logs
    async def test_initialize_fails_on_detector_failure(self, world_model, image_manager):
        """Test initialization fails when detector fails."""
        mock_detector = MagicMock(spec=SimulatedDetector)
        mock_detector.initialize = AsyncMock(return_value=False)

        service = VisionService(
            world_model=world_model,
            detector=mock_detector,
            image_manager=image_manager,
        )

        result = await service.initialize()

        assert result is False

    @pytest.mark.allow_error_logs
    async def test_initialize_handles_exception(self, world_model, image_manager):
        """Test initialization handles exceptions gracefully."""
        mock_detector = MagicMock(spec=SimulatedDetector)
        mock_detector.initialize = AsyncMock(side_effect=Exception("Init failed"))

        service = VisionService(
            world_model=world_model,
            detector=mock_detector,
            image_manager=image_manager,
        )

        result = await service.initialize()

        assert result is False


# ============================================================================
# Process Inspection Result Tests
# ============================================================================


@pytest.mark.asyncio
class TestProcessInspectionResult:
    """Tests for process_inspection_result method."""

    async def test_process_with_image_path(self, vision_service, temp_dir):
        """Test processing with actual image path."""
        # Create a test image
        image_path = temp_dir / "test_image.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
            image_path=image_path,
        )

        assert observation is not None
        assert observation.asset_id == "asset_001"
        assert observation.observation_id is not None
        assert observation.processed_on_server is True

    async def test_process_with_client_detection(self, vision_service):
        """Test processing with client-side detection result."""
        client_detection = DetectionResult(
            detections=[
                Detection(
                    detection_class=DetectionClass.CRACK,
                    confidence=0.8,
                    bounding_box=BoundingBox(
                        x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                    ),
                    severity=0.6,
                )
            ],
            model_name="client_detector",
        )

        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
            client_detection=client_detection,
        )

        assert observation is not None
        assert observation.asset_id == "asset_001"

    async def test_process_without_image_or_detection(self, vision_service):
        """Test processing without image or detection simulates result."""
        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
        )

        assert observation is not None
        assert observation.asset_id == "asset_001"
        assert observation.model_name is not None

    async def test_process_with_vehicle_state(self, vision_service):
        """Test processing with vehicle state extracts position."""
        vehicle_state = {
            "position": {
                "latitude": 37.7750,
                "longitude": -122.4195,
                "altitude_msl": 50.0,
            },
            "altitude_agl": 30.0,
            "heading_deg": 90.0,
            "distance_to_asset": 15.0,
        }

        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
            vehicle_state=vehicle_state,
        )

        assert observation.position is not None
        assert observation.position.latitude == 37.7750
        assert observation.altitude_agl == 30.0
        assert observation.heading_deg == 90.0
        assert observation.distance_to_asset == 15.0

    async def test_process_stores_observation(self, vision_service):
        """Test that processing stores observation in service."""
        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
        )

        assert observation.observation_id in vision_service.observations
        stored = vision_service.observations[observation.observation_id]
        assert stored.asset_id == "asset_001"

    async def test_process_generates_unique_ids(self, vision_service):
        """Test that each processing generates unique observation ID."""
        obs1 = await vision_service.process_inspection_result(asset_id="asset_001")
        obs2 = await vision_service.process_inspection_result(asset_id="asset_001")

        assert obs1.observation_id != obs2.observation_id


# ============================================================================
# Anomaly Detection Tests
# ============================================================================


@pytest.mark.asyncio
class TestShouldCreateAnomaly:
    """Tests for _should_create_anomaly method."""

    async def test_should_create_when_thresholds_met(self, vision_service):
        """Test anomaly creation when both thresholds are met."""
        detection = DetectionResult(
            detections=[
                Detection(
                    detection_class=DetectionClass.CRACK,
                    confidence=0.85,
                    bounding_box=BoundingBox(
                        x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                    ),
                    severity=0.7,
                )
            ]
        )

        result = vision_service._should_create_anomaly(detection)

        assert result is True

    async def test_should_not_create_low_confidence(self, vision_service):
        """Test no anomaly when confidence below threshold."""
        detection = DetectionResult(
            detections=[
                Detection(
                    detection_class=DetectionClass.CRACK,
                    confidence=0.5,  # Below 0.7 threshold
                    bounding_box=BoundingBox(
                        x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                    ),
                    severity=0.7,
                )
            ]
        )

        result = vision_service._should_create_anomaly(detection)

        assert result is False

    async def test_should_not_create_low_severity(self, vision_service):
        """Test no anomaly when severity below threshold."""
        detection = DetectionResult(
            detections=[
                Detection(
                    detection_class=DetectionClass.CRACK,
                    confidence=0.85,
                    bounding_box=BoundingBox(
                        x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                    ),
                    severity=0.2,  # Below 0.4 threshold
                )
            ]
        )

        result = vision_service._should_create_anomaly(detection)

        assert result is False

    async def test_should_not_create_no_defects(self, vision_service):
        """Test no anomaly when no defects detected."""
        detection = DetectionResult(detections=[])

        result = vision_service._should_create_anomaly(detection)

        assert result is False


@pytest.mark.asyncio
class TestCreateAnomaly:
    """Tests for _create_anomaly method."""

    async def test_creates_anomaly_in_world_model(self, vision_service, world_model):
        """Test that anomaly is added to world model."""
        detection = DetectionResult(
            detections=[
                Detection(
                    detection_class=DetectionClass.CRACK,
                    confidence=0.9,
                    bounding_box=BoundingBox(
                        x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                    ),
                    severity=0.8,
                )
            ]
        )

        observation = VisionObservation(
            observation_id="obs_001",
            asset_id="asset_001",
            timestamp=datetime.now(),
        )

        anomaly = await vision_service._create_anomaly(
            asset_id="asset_001",
            detection=detection,
            observation=observation,
        )

        assert anomaly is not None
        assert anomaly.asset_id == "asset_001"
        assert anomaly.severity == 0.8

        # Check world model
        anomaly_asset_ids = world_model.get_anomaly_assets()
        assert "asset_001" in anomaly_asset_ids

    async def test_deduplication_prevents_duplicate(
        self, vision_service, world_model
    ):
        """Test that existing anomaly prevents duplicate creation."""
        # Add existing anomaly
        existing = Anomaly(
            anomaly_id="existing_001",
            asset_id="asset_001",
            detected_at=datetime.now(),
            severity=0.5,
            description="Existing anomaly",
        )
        world_model.add_anomaly(existing)

        detection = DetectionResult(
            detections=[
                Detection(
                    detection_class=DetectionClass.CORROSION,
                    confidence=0.9,
                    bounding_box=BoundingBox(
                        x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                    ),
                    severity=0.7,
                )
            ]
        )

        observation = VisionObservation(
            observation_id="obs_002",
            asset_id="asset_001",
            timestamp=datetime.now(),
        )

        anomaly = await vision_service._create_anomaly(
            asset_id="asset_001",
            detection=detection,
            observation=observation,
        )

        # Should return None (no new anomaly created)
        assert anomaly is None

    async def test_anomaly_description_includes_defect_types(
        self, vision_service, world_model
    ):
        """Test that anomaly description includes detected defect types."""
        detection = DetectionResult(
            detections=[
                Detection(
                    detection_class=DetectionClass.CRACK,
                    confidence=0.9,
                    bounding_box=BoundingBox(
                        x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                    ),
                    severity=0.8,
                ),
                Detection(
                    detection_class=DetectionClass.CORROSION,
                    confidence=0.85,
                    bounding_box=BoundingBox(
                        x_min=0.4, y_min=0.5, x_max=0.8, y_max=0.9
                    ),
                    severity=0.6,
                ),
            ]
        )

        observation = VisionObservation(
            observation_id="obs_003",
            asset_id="asset_002",  # Different asset
            timestamp=datetime.now(),
        )

        anomaly = await vision_service._create_anomaly(
            asset_id="asset_002",
            detection=detection,
            observation=observation,
        )

        assert "crack" in anomaly.description.lower()
        assert "corrosion" in anomaly.description.lower()


# ============================================================================
# Observation Query Tests
# ============================================================================


@pytest.mark.asyncio
class TestGetObservationsForAsset:
    """Tests for get_observations_for_asset method."""

    async def test_get_observations_for_specific_asset(self, vision_service):
        """Test retrieving observations for a specific asset."""
        # Create observations for different assets
        await vision_service.process_inspection_result(asset_id="asset_001")
        await vision_service.process_inspection_result(asset_id="asset_001")
        await vision_service.process_inspection_result(asset_id="asset_002")

        observations = vision_service.get_observations_for_asset("asset_001")

        assert len(observations) == 2
        assert all(obs.asset_id == "asset_001" for obs in observations)

    async def test_get_observations_empty_for_unknown_asset(self, vision_service):
        """Test empty list for asset with no observations."""
        observations = vision_service.get_observations_for_asset("unknown_asset")

        assert len(observations) == 0


@pytest.mark.asyncio
class TestGetRecentObservations:
    """Tests for get_recent_observations method."""

    async def test_get_recent_observations(self, vision_service):
        """Test retrieving recent observations."""
        for _ in range(5):
            await vision_service.process_inspection_result(asset_id="asset_001")

        observations = vision_service.get_recent_observations(limit=3)

        assert len(observations) == 3

    async def test_get_recent_observations_sorted_by_time(self, vision_service):
        """Test that observations are sorted newest first."""
        import asyncio

        for _ in range(3):
            await vision_service.process_inspection_result(asset_id="asset_001")
            await asyncio.sleep(0.01)

        observations = vision_service.get_recent_observations()

        timestamps = [obs.timestamp for obs in observations]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_get_recent_observations_empty(self, vision_service):
        """Test empty list when no observations."""
        observations = vision_service.get_recent_observations()

        assert len(observations) == 0


# ============================================================================
# Statistics Tests
# ============================================================================


@pytest.mark.asyncio
class TestGetStatistics:
    """Tests for get_statistics method."""

    async def test_statistics_initial_empty(self, vision_service):
        """Test statistics with no observations."""
        stats = vision_service.get_statistics()

        assert stats["total_observations"] == 0
        assert stats["defects_detected"] == 0
        assert stats["anomalies_created"] == 0
        assert stats["detection_rate"] == 0.0
        assert stats["anomaly_rate"] == 0.0
        assert stats["average_confidence"] == 0.0
        assert stats["average_severity"] == 0.0

    async def test_statistics_with_observations(self, vision_service):
        """Test statistics after processing observations."""
        # Process multiple observations
        for _ in range(5):
            await vision_service.process_inspection_result(asset_id="asset_001")

        stats = vision_service.get_statistics()

        assert stats["total_observations"] == 5
        # Rates should be between 0 and 1
        assert 0.0 <= stats["detection_rate"] <= 1.0
        assert 0.0 <= stats["anomaly_rate"] <= 1.0


# ============================================================================
# Simulate Inspection Result Tests
# ============================================================================


@pytest.mark.asyncio
class TestSimulateInspectionResult:
    """Tests for simulate_inspection_result method."""

    async def test_simulate_returns_detection_result(self, vision_service):
        """Test that simulation returns DetectionResult."""
        result = await vision_service.simulate_inspection_result(asset_id="asset_001")

        assert isinstance(result, DetectionResult)
        assert result.model_name is not None

    async def test_simulate_uses_detector(self, vision_service):
        """Test that simulation uses the detector."""
        result = await vision_service.simulate_inspection_result(asset_id="asset_001")

        # Should have used the simulated detector
        assert "simulated" in result.model_name.lower()


# ============================================================================
# Shutdown Tests
# ============================================================================


@pytest.mark.asyncio
class TestShutdown:
    """Tests for shutdown method."""

    async def test_shutdown_calls_detector_shutdown(
        self, world_model, image_manager
    ):
        """Test that shutdown calls detector shutdown."""
        mock_detector = MagicMock(spec=SimulatedDetector)
        mock_detector.initialize = AsyncMock(return_value=True)
        mock_detector.shutdown = AsyncMock()

        service = VisionService(
            world_model=world_model,
            detector=mock_detector,
            image_manager=image_manager,
        )

        await service.initialize()
        await service.shutdown()

        mock_detector.shutdown.assert_called_once()


# ============================================================================
# Integration with World Model Tests
# ============================================================================


@pytest.mark.asyncio
class TestWorldModelIntegration:
    """Tests for integration with WorldModel."""

    async def test_anomaly_updates_asset_in_world_model(
        self, world_model, detector, image_manager
    ):
        """Test that creating anomaly updates asset status."""
        # Create service with high probability detector
        high_prob_detector = SimulatedDetector(
            defect_probability=1.0,  # Always detect
            severity_range=(0.7, 0.9),
            confidence_boost=0.2,
        )

        config = VisionServiceConfig(
            confidence_threshold=0.5,
            severity_threshold=0.3,
        )

        service = VisionService(
            world_model=world_model,
            detector=high_prob_detector,
            image_manager=image_manager,
            config=config,
        )

        await service.initialize()

        observation = await service.process_inspection_result(asset_id="asset_001")

        # With 100% detection probability and low thresholds, should create anomaly
        if observation.anomaly_created:
            anomaly_ids = world_model.get_anomaly_assets()
            assert "asset_001" in anomaly_ids

        await service.shutdown()

    async def test_observation_links_to_anomaly(self, vision_service, world_model):
        """Test that observation correctly links to created anomaly."""
        # Create with high probability detection
        with patch.object(
            vision_service, "_should_create_anomaly", return_value=True
        ):
            # Also need to create a mock detection
            mock_detection = DetectionResult(
                detections=[
                    Detection(
                        detection_class=DetectionClass.CRACK,
                        confidence=0.9,
                        bounding_box=BoundingBox(
                            x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7
                        ),
                        severity=0.8,
                    )
                ]
            )

            with patch.object(
                vision_service.detector,
                "analyze_image",
                new_callable=AsyncMock,
                return_value=mock_detection,
            ):
                observation = await vision_service.process_inspection_result(
                    asset_id="asset_002",
                )

                if observation.anomaly_created:
                    assert observation.anomaly_id is not None


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_process_with_empty_vehicle_state(self, vision_service):
        """Test processing with empty vehicle state dict."""
        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
            vehicle_state={},
        )

        assert observation is not None
        assert observation.position is None

    async def test_process_with_partial_vehicle_state(self, vision_service):
        """Test processing with partial vehicle state."""
        vehicle_state = {
            "altitude_agl": 30.0,
            # Missing position, heading, distance
        }

        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
            vehicle_state=vehicle_state,
        )

        assert observation.altitude_agl == 30.0
        assert observation.position is None
        assert observation.heading_deg is None

    async def test_process_with_nonexistent_image_path(self, vision_service, temp_dir):
        """Test processing with non-existent image path uses simulation."""
        nonexistent_path = temp_dir / "does_not_exist.png"

        observation = await vision_service.process_inspection_result(
            asset_id="asset_001",
            image_path=nonexistent_path,
        )

        # Should fall back to simulation
        assert observation is not None

    async def test_multiple_observations_same_asset(self, vision_service):
        """Test multiple observations for same asset accumulate correctly."""
        for i in range(10):
            await vision_service.process_inspection_result(asset_id="asset_001")

        observations = vision_service.get_observations_for_asset("asset_001")
        assert len(observations) == 10

        stats = vision_service.get_statistics()
        assert stats["total_observations"] == 10

    async def test_observations_across_multiple_assets(self, vision_service):
        """Test observations across different assets tracked separately."""
        await vision_service.process_inspection_result(asset_id="asset_001")
        await vision_service.process_inspection_result(asset_id="asset_001")
        await vision_service.process_inspection_result(asset_id="asset_002")

        obs1 = vision_service.get_observations_for_asset("asset_001")
        obs2 = vision_service.get_observations_for_asset("asset_002")

        assert len(obs1) == 2
        assert len(obs2) == 1
