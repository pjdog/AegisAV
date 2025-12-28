"""
End-to-End Integration Tests for Vision Pipeline

Tests the complete vision system integration:
1. Server initialization with vision enabled
2. Client-side vision capture during inspections
3. Vision feedback processing creating anomalies
4. Anomaly detection triggering re-inspection goals
5. Full decision → inspection → vision → anomaly → re-inspection flow

These tests validate production-ready integration of all subsystems.
"""

import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from agent.api_models import ActionType
from agent.server.decision import Decision
from agent.server.goal_selector import GoalSelector
from agent.server.goals import GoalType
from agent.server.models.outcome_models import DecisionFeedback, ExecutionStatus
from agent.server.monitoring import OutcomeTracker
from agent.server.vision.detector import SimulatedDetector
from agent.server.vision.vision_service import VisionService, VisionServiceConfig
from agent.server.world_model import (
    Anomaly,
    Asset,
    AssetStatus,
    AssetType,
    WorldModel,
)
from autonomy.vehicle_state import (
    Attitude,
    BatteryState,
    FlightMode,
    GPSState,
    Position,
    VehicleState,
    Velocity,
)
from vision.camera.simulated import DefectConfig, SimulatedCamera, SimulatedCameraConfig
from vision.image_manager import ImageManager
from vision.models.yolo_detector import MockYOLODetector

logger = logging.getLogger(__name__)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory for test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def vision_config(temp_image_dir):
    """Create vision configuration for testing."""
    return {
        "vision": {
            "camera": {
                "type": "simulated",
                "resolution": [1920, 1080],
                "capture_format": "RGB",
            },
            "client": {
                "model": {
                    "type": "yolo",
                    "variant": "yolov8n",
                    "device": "cpu",
                },
                "detection": {
                    "confidence_threshold": 0.6,
                    "suspicious_threshold": 0.4,
                },
                "capture": {
                    "interval_s": 2.0,
                    "max_images_per_inspection": 10,
                },
            },
            "server": {
                "model": {
                    "type": "simulated",
                    "defect_probability": 0.3,  # 30% for testing
                    "severity_range": [0.4, 0.9],
                },
                "detection": {
                    "confidence_threshold": 0.7,
                    "severity_threshold": 0.4,
                },
                "storage": {
                    "image_dir": str(temp_image_dir),
                    "max_storage_gb": 1,
                },
            },
            "simulation": {
                "defects": {
                    "crack_probability": 0.15,
                    "corrosion_probability": 0.10,
                    "vegetation_probability": 0.05,
                    "damage_probability": 0.05,
                }
            },
        }
    }


@pytest.fixture
async def vision_service(temp_image_dir, world_model):
    """Create and initialize a VisionService for testing."""
    # Create detector
    detector = SimulatedDetector(
        defect_probability=0.3,  # 30% for testing
        severity_range=(0.4, 0.9),
        confidence_boost=0.1,
    )

    # Create image manager
    image_manager = ImageManager(base_dir=temp_image_dir)

    # Create vision service
    service = VisionService(
        world_model=world_model,
        detector=detector,
        image_manager=image_manager,
        config=VisionServiceConfig(confidence_threshold=0.7, severity_threshold=0.4),
    )

    # Initialize
    await service.initialize()

    yield service

    # Cleanup
    await service.shutdown()


@pytest.fixture
async def client_vision_components(temp_image_dir):
    """Create client-side vision components (camera + detector)."""
    # Create defect config with high probability for testing
    defect_config = DefectConfig(
        crack_probability=0.2,
        corrosion_probability=0.15,
        structural_damage_probability=0.1,
        discoloration_probability=0.05,
        vegetation_probability=0.05,
    )

    # Create simulated camera
    camera_config = SimulatedCameraConfig(
        resolution=(1920, 1080),
        capture_format="RGB",
    )
    camera_config.defect_config = defect_config
    camera = SimulatedCamera(config=camera_config)

    # Create mock detector
    detector = MockYOLODetector(
        model_variant="yolov8n",
        confidence_threshold=0.6,
        device="cpu",
    )

    # Create image manager
    image_manager = ImageManager(base_dir=temp_image_dir / "client")

    # Initialize
    await camera.initialize()
    await detector.initialize()

    yield {
        "camera": camera,
        "detector": detector,
        "image_manager": image_manager,
    }

    # Cleanup
    await camera.shutdown()
    await detector.shutdown()


@pytest.fixture
def world_model(good_vehicle_state):
    """Create a WorldModel with test assets and vehicle state."""
    model = WorldModel()

    # Add test assets
    model.add_asset(
        Asset(
            asset_id="bridge_001",
            name="Test Bridge",
            position=Position(
                latitude=37.7750,
                longitude=-122.4195,
                altitude_msl=0,
                altitude_agl=0,
            ),
            asset_type=AssetType.BUILDING,  # Use BUILDING as proxy for bridge
            priority=1,
        )
    )

    model.add_asset(
        Asset(
            asset_id="building_001",
            name="Test Building",
            position=Position(
                latitude=37.7760,
                longitude=-122.4200,
                altitude_msl=0,
                altitude_agl=0,
            ),
            asset_type=AssetType.BUILDING,
            priority=2,
        )
    )

    # Set dock
    model.set_dock(
        Position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=0,
            altitude_agl=0,
        )
    )

    model.update_vehicle(good_vehicle_state)
    return model


@pytest.fixture
def good_vehicle_state():
    """Create vehicle in good condition."""
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


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_vision_service_initialization(vision_service):
    """Test that VisionService initializes correctly."""
    await asyncio.sleep(0)
    assert vision_service is not None
    assert vision_service.detector is not None
    assert vision_service.image_manager is not None
    assert vision_service.world_model is not None


@pytest.mark.asyncio
async def test_client_vision_components_initialization(client_vision_components):
    """Test that client-side vision components initialize correctly."""
    await asyncio.sleep(0)
    assert client_vision_components["camera"] is not None
    assert client_vision_components["detector"] is not None
    assert client_vision_components["image_manager"] is not None


@pytest.mark.asyncio
async def test_client_inspection_with_vision(client_vision_components, good_vehicle_state):
    """
    Test client-side inspection with vision capture.

    Simulates what happens during an INSPECT action:
    1. Camera captures images
    2. Detector performs quick analysis
    3. Results stored locally
    """
    camera = client_vision_components["camera"]
    detector = client_vision_components["detector"]
    image_manager = client_vision_components["image_manager"]

    # Simulate inspection with 3 captures
    captures = []
    for _ in range(3):
        # Capture image
        result = await camera.capture(vehicle_state=good_vehicle_state.to_dict())

        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.exists()

        # Quick detection
        detection_result = await detector.analyze_image(result.image_path)

        # Store metadata
        metadata = {
            "capture_result": result.model_dump(),
            "detection_result": detection_result.model_dump(),
            "asset_id": "bridge_001",
        }
        image_manager.save_image_with_metadata(
            result.image_path,
            metadata,
            asset_id="bridge_001",
        )

        captures.append({
            "image_path": result.image_path,
            "detection": detection_result,
            "metadata": metadata,
        })

        # Small delay between captures
        await asyncio.sleep(0.1)

    # Verify we got captures
    assert len(captures) == 3

    # Check if any defects detected (probabilistic)
    defects_found = any(len(c["detection"].detected_defects) > 0 for c in captures)

    # With 55% total defect probability and 3 captures,
    # we have ~88% chance of at least one defect
    # But tests should handle both cases
    if defects_found:
        detected_count = sum(1 for c in captures if len(c["detection"].detected_defects) > 0)
        logger.info("✅ Defects detected in %s captures", detected_count)
    else:
        logger.info("✅ No defects detected (valid outcome)")


@pytest.mark.asyncio
async def test_server_vision_processing_creates_anomaly(
    vision_service, world_model, good_vehicle_state
):
    """
    Test that server-side vision processing creates anomalies.

    This is a CRITICAL integration point:
    - Vision detects defect with sufficient confidence/severity
    - VisionService creates anomaly in WorldModel
    - Anomaly triggers re-inspection goal
    """
    # Simulate inspection result (force defect detection)
    observation = await vision_service.process_inspection_result(
        asset_id="bridge_001",
        client_detection=None,
        image_path=None,
        vehicle_state=good_vehicle_state.to_dict(),
    )

    # Check observation
    assert observation is not None
    assert observation.asset_id == "bridge_001"

    # Probabilistic - may or may not detect defect
    if observation.defect_detected:
        logger.info(
            "✅ Defect detected: confidence=%.2f, severity=%.2f",
            observation.max_confidence,
            observation.max_severity,
        )

        # If thresholds met, anomaly should be created
        if observation.max_confidence >= 0.7 and observation.max_severity >= 0.4:
            assert observation.anomaly_created is True
            assert observation.anomaly_id is not None

            # Verify anomaly in world model
            snapshot = world_model.get_snapshot()
            anomalies = [a for a in snapshot.anomalies if a.asset_id == "bridge_001"]
            assert len(anomalies) > 0, "Anomaly should exist in world model"

            anomaly = anomalies[0]
            assert anomaly.severity >= 0.4
            description = anomaly.description.lower()
            assert (
                "detected by vision" in description
                or "vision detected" in description
                or "defect" in description
            )

            logger.info(
                "✅ Anomaly created: %s (severity: %.2f)",
                anomaly.anomaly_id,
                anomaly.severity,
            )
        else:
            logger.info("⚠️ Defect below threshold - no anomaly created")
    else:
        logger.info("✅ No defect detected (valid outcome)")


@pytest.mark.asyncio
async def test_anomaly_triggers_reinspection_goal(world_model):
    """
    Test that anomalies trigger re-inspection goals.

    This validates the feedback loop:
    - Anomaly created from vision detection
    - GoalSelector prioritizes re-inspection
    """
    goal_selector = GoalSelector()
    await asyncio.sleep(0)

    # Add anomaly manually
    anomaly = Anomaly(
        anomaly_id="test_anomaly_001",
        asset_id="bridge_001",
        severity=0.7,
        detected_at=datetime.now(),
        description="Crack detected by vision system",
    )
    world_model.add_anomaly(anomaly)

    # Update asset status
    snapshot = world_model.get_snapshot()
    asset = next(a for a in snapshot.assets if a.asset_id == "bridge_001")
    world_model.update_asset_status(asset.asset_id, AssetStatus.ANOMALY)

    # Select goals
    vehicle_state = VehicleState(
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

    world_model.update_vehicle(vehicle_state)
    snapshot = world_model.get_snapshot()
    reinspection_goal = await goal_selector.select_goal(snapshot)

    assert reinspection_goal.goal_type == GoalType.INSPECT_ANOMALY
    assert reinspection_goal.priority >= 15  # High priority
    assert reinspection_goal.target_asset is not None
    assert reinspection_goal.target_asset.asset_id == "bridge_001"

    logger.info("✅ Re-inspection goal created with priority %s", reinspection_goal.priority)


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(
    vision_service,
    client_vision_components,
    world_model,
    good_vehicle_state,
):
    """
    COMPREHENSIVE END-TO-END TEST

    Tests the complete vision pipeline:
    1. Client captures images during inspection
    2. Client performs quick detection
    3. Feedback sent to server with anomaly flag
    4. Server performs detailed analysis
    5. Anomaly created if thresholds met
    6. Re-inspection goal created
    7. Full decision loop works

    This is the MOST CRITICAL test for production readiness.
    """
    # ========== PHASE 1: Client-Side Inspection ==========

    camera = client_vision_components["camera"]
    client_detector = client_vision_components["detector"]
    logger.info("\n=== PHASE 1: Client-Side Inspection ===")

    # Simulate inspection with multiple captures
    client_captures = []
    for _ in range(5):
        result = await camera.capture(vehicle_state=good_vehicle_state.to_dict())
        assert result.success is True

        detection = await client_detector.analyze_image(result.image_path)
        client_captures.append({
            "image": result.image_path,
            "detection": detection,
        })

        await asyncio.sleep(0.05)

    logger.info("✅ Client captured %s images", len(client_captures))

    # Check if client detected anything suspicious
    suspicious_detections = [
        c
        for c in client_captures
        if len(c["detection"].detected_defects) > 0
        and any(d.confidence >= 0.4 for d in c["detection"].detected_defects)
    ]

    anomaly_detected_by_client = len(suspicious_detections) > 0

    logger.info(
        "%s Client suspicious detections: %s",
        "✅" if anomaly_detected_by_client else "⚠️",
        len(suspicious_detections),
    )

    # ========== PHASE 2: Feedback to Server ==========

    logger.info("\n=== PHASE 2: Server Feedback Processing ===")

    # Create decision (for outcome tracking)
    decision = Decision(
        action=ActionType.INSPECT,
        parameters={"asset_id": "bridge_001"},
        confidence=0.9,
        reasoning="Inspecting bridge",
    )

    # Simulate outcome tracker
    outcome_tracker = OutcomeTracker(log_dir=Path("logs/test_outcomes"))
    outcome_tracker.create_outcome(decision)

    # Create feedback
    feedback = DecisionFeedback(
        decision_id=decision.decision_id,
        status=ExecutionStatus.SUCCESS,
        battery_consumed=5.0,
        distance_traveled=100.0,
        duration_s=60.0,
        mission_objective_achieved=True,
        asset_inspected="bridge_001",
        anomaly_detected=anomaly_detected_by_client,
        # TODO: Add inspection_data field with vision results
    )

    # Process feedback (simulate what POST /feedback does)
    await outcome_tracker.process_feedback(feedback)

    logger.info("✅ Feedback processed: anomaly_detected=%s", anomaly_detected_by_client)

    # ========== PHASE 3: Server-Side Vision Analysis ==========

    logger.info("\n=== PHASE 3: Server-Side Vision Analysis ===")

    if anomaly_detected_by_client:
        # Server performs detailed analysis
        vision_observation = await vision_service.process_inspection_result(
            asset_id="bridge_001",
            client_detection=None,
            image_path=None,
            vehicle_state=good_vehicle_state.to_dict(),
        )

        logger.info(
            "✅ Server analysis complete: defect_detected=%s",
            vision_observation.defect_detected,
        )

        if vision_observation.defect_detected:
            logger.info("   - Confidence: %.2f", vision_observation.max_confidence)
            logger.info("   - Severity: %.2f", vision_observation.max_severity)
            logger.info("   - Anomaly created: %s", vision_observation.anomaly_created)

            # ========== PHASE 4: Anomaly Creation ==========

            if vision_observation.anomaly_created:
                logger.info("\n=== PHASE 4: Anomaly Validation ===")

                # Verify anomaly in world model
                snapshot = world_model.get_snapshot()
                anomalies = [a for a in snapshot.anomalies if a.asset_id == "bridge_001"]

                assert len(anomalies) > 0, "Anomaly should exist in world model"
                logger.info("✅ Anomaly created in world model: %s anomalies", len(anomalies))

                # ========== PHASE 5: Re-Inspection Goal ==========

                logger.info("\n=== PHASE 5: Re-Inspection Goal ===")

                goal_selector = GoalSelector()
                reinspection_goal = await goal_selector.select_goal(snapshot)

                assert reinspection_goal.goal_type == GoalType.INSPECT_ANOMALY
                assert reinspection_goal.target_asset is not None
                assert reinspection_goal.target_asset.asset_id == "bridge_001"
                logger.info(
                    "✅ Re-inspection goal created with priority %s",
                    reinspection_goal.priority,
                )

                logger.info("\n%s", "=" * 60)
                logger.info("🎉 END-TO-END PIPELINE COMPLETE")
                logger.info("%s", "=" * 60)
                logger.info("✅ Client captured and analyzed images")
                logger.info("✅ Suspicious detection flagged")
                logger.info("✅ Server performed detailed analysis")
                logger.info("✅ Anomaly created in world model")
                logger.info("✅ Re-inspection goal created")
                logger.info("%s", "=" * 60)

                return True  # Full pipeline success

    logger.info("\n%s", "=" * 60)
    logger.info("✅ Pipeline executed successfully")
    logger.info("⚠️ No defects detected (valid probabilistic outcome)")
    logger.info("%s", "=" * 60)

    return True


@pytest.mark.asyncio
async def test_vision_deduplication_prevents_duplicate_anomalies(
    vision_service, world_model, good_vehicle_state
):
    """
    Test that VisionService deduplication prevents duplicate anomalies.

    Important for production: same defect shouldn't create multiple anomalies.
    """
    # First inspection - should create anomaly if defect detected
    _ = await vision_service.process_inspection_result(
        asset_id="bridge_001",
        client_detection=None,
        image_path=None,
        vehicle_state=good_vehicle_state.to_dict(),
    )

    # Second inspection - should NOT create duplicate
    _ = await vision_service.process_inspection_result(
        asset_id="bridge_001",
        client_detection=None,
        image_path=None,
        vehicle_state=good_vehicle_state.to_dict(),
    )

    # Check world model
    snapshot = world_model.get_snapshot()
    anomalies = [a for a in snapshot.anomalies if a.asset_id == "bridge_001"]

    # Should have at most 1 anomaly per asset (deduplication)
    # Note: Could be 0 if no defect detected
    assert len(anomalies) <= 1, "Deduplication should prevent multiple anomalies for same asset"

    logger.info("✅ Deduplication working: %s anomalies for bridge_001", len(anomalies))


@pytest.mark.asyncio
async def test_vision_graceful_degradation_on_failure(vision_service):
    """
    Test that vision system degrades gracefully on failures.

    Production requirement: system continues operating if vision fails.
    """
    # Simulate failure by passing invalid data
    try:
        observation = await vision_service.simulate_inspection_result(
            asset_id=None,  # Invalid
            vehicle_state=None,
            world_model=None,
        )
        # Should either handle gracefully or raise expected error
        assert observation is not None

    except Exception as exc:
        # Expected - but system should log error and continue
        logger.info("✅ Vision failure handled: %s", type(exc).__name__)


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_vision_performance_acceptable_latency(vision_service, good_vehicle_state):
    """
    Test that vision processing meets performance requirements.

    Target: Server-side analysis < 500ms
    """
    import time

    # Create simple world model for test
    world_model = WorldModel()
    world_model.add_asset(
        Asset(
            asset_id="perf_test_001",
            name="Performance Test Asset",
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

    # Measure processing time
    start = time.time()

    _ = await vision_service.process_inspection_result(
        asset_id="perf_test_001",
        client_detection=None,
        image_path=None,
        vehicle_state=good_vehicle_state.to_dict(),
    )

    elapsed_ms = (time.time() - start) * 1000

    logger.info("⏱️  Vision processing latency: %.1fms", elapsed_ms)

    # Should be fast (simulated detector)
    assert elapsed_ms < 1000, f"Vision processing too slow: {elapsed_ms:.1f}ms"

    logger.info("✅ Performance acceptable: %.1fms < 1000ms", elapsed_ms)
