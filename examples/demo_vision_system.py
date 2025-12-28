#!/usr/bin/env python
"""
Computer Vision System Demo

Demonstrates the complete vision pipeline:
1. Simulated camera captures images with defect injection
2. Client-side quick detection
3. Server-side detailed analysis
4. Anomaly creation in world model
5. Statistics and visualization

This is a standalone demo showing impressive CV capabilities.
"""

import asyncio
import logging
from pathlib import Path

import yaml

from agent.client.vision_client import VisionClient
from agent.server.vision.vision_service import VisionService, VisionServiceConfig
from agent.server.world_model import WorldModel
from vision.camera.simulated import DefectConfig, SimulatedCamera, SimulatedCameraConfig

# Setup logging with colors
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


async def demo_vision_pipeline():
    """
    Run complete vision pipeline demonstration.
    """
    logger.info("\n%s", "=" * 80)
    logger.info("🎥 AEGISAV COMPUTER VISION SYSTEM DEMONSTRATION")
    logger.info("%s\n", "=" * 80)

    # Load configuration
    config_path = Path("configs/vision_config.yaml")
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            vision_config = yaml.safe_load(f) or {}
            logger.info("✅ Loaded configuration from %s", config_path)
    else:
        vision_config = {}
        logger.warning("⚠️  Using default configuration")

    logger.info("\n📦 Initializing vision components...")

    # Create simulated camera with defect injection
    defect_config = DefectConfig(
        crack_probability=0.15,
        corrosion_probability=0.10,
        structural_damage_probability=0.05,
        severity_min=0.4,
        severity_max=0.9,
    )

    camera_config = SimulatedCameraConfig(
        output_dir=Path("data/vision/demo"),
        defect_config=defect_config,
        save_images=True,
    )

    camera = SimulatedCamera(config=camera_config)

    # Create vision client
    vision_client = VisionClient(
        camera=camera,
        capture_interval_s=1.0,  # Fast for demo
        max_captures_per_inspection=5,
        enabled=True,
    )

    # Create world model and vision service
    world_model = WorldModel()
    server_detection = vision_config.get("vision", {}).get("server", {}).get("detection", {})
    vision_service = VisionService(
        world_model=world_model,
        config=VisionServiceConfig(
            confidence_threshold=server_detection.get("confidence_threshold", 0.6),
            severity_threshold=server_detection.get("severity_threshold", 0.3),
        ),
    )

    # Initialize
    logger.info("\n🔧 Initializing vision client...")
    await vision_client.initialize()

    logger.info("🔧 Initializing vision service...")
    await vision_service.initialize()

    logger.info("\n✅ All components initialized!\n")

    # Simulate inspections of multiple assets
    assets = [
        ("tower_001", "Power transmission tower"),
        ("pole_042", "Distribution pole"),
        ("tower_015", "Substation tower"),
        ("pole_088", "Rural distribution pole"),
        ("tower_023", "High-voltage transmission tower"),
    ]

    logger.info("%s", "=" * 80)
    logger.info("🚁 STARTING SIMULATED INSPECTIONS")
    logger.info("%s\n", "=" * 80)

    for asset_id, description in assets:
        logger.info("\n%s", "─" * 80)
        logger.info("📍 Inspecting: %s (%s)", asset_id, description)
        logger.info("%s\n", "─" * 80)

        # Client-side: Capture and quick detection
        inspection_results = await vision_client.capture_during_inspection(
            asset_id=asset_id,
            duration_s=5.0,  # 5 second inspection
            vehicle_state_fn=lambda asset_id=asset_id: {
                "position": {
                    "latitude": 37.7749 + (hash(asset_id) % 100) / 10000,
                    "longitude": -122.4194 + (hash(asset_id) % 100) / 10000,
                    "altitude_msl": 150.0,
                },
                "altitude_agl": 20.0,
                "heading_deg": 45.0,
            },
        )

        # Display client-side results
        logger.info("\n📸 Client captured %s images", len(inspection_results.captures))
        logger.info("🔍 Quick detections: %s", len(inspection_results.detections))
        logger.info("⚠️  Defects found: %s", inspection_results.defects_detected)
        if inspection_results.defects_detected > 0:
            logger.info("📊 Max confidence: %.2f", inspection_results.max_confidence)
            logger.info("📊 Max severity: %.2f", inspection_results.max_severity)
            logger.info(
                "🔔 Needs server analysis: %s",
                inspection_results.needs_server_analysis(),
            )

        # Server-side: Detailed analysis if defects detected
        if inspection_results.defects_detected > 0:
            logger.info("\n🖥️  Sending to server for detailed analysis...")

            observation = await vision_service.process_inspection_result(
                asset_id=asset_id,
                client_detection=inspection_results.detections[0]
                if inspection_results.detections
                else None,
                image_path=inspection_results.best_detection_image,
                vehicle_state={
                    "position": {
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "altitude_msl": 150.0,
                    },
                    "altitude_agl": 20.0,
                },
            )

            if observation.anomaly_created:
                logger.info("\n✨ ANOMALY CREATED!")
                logger.info("   ID: %s", observation.anomaly_id)
                logger.info("   Confidence: %.2f", observation.max_confidence)
                logger.info("   Severity: %.2f", observation.max_severity)
            else:
                logger.info("\n✓ Analysis complete (no anomaly threshold met)")

        else:
            logger.info("\n✓ No defects detected - asset OK")

        # Small delay between inspections
        await asyncio.sleep(0.5)

    # Display final statistics
    logger.info("\n\n%s", "=" * 80)
    logger.info("📊 VISION SYSTEM STATISTICS")
    logger.info("%s\n", "=" * 80)

    stats = vision_service.get_statistics()
    logger.info("Total observations:     %s", stats["total_observations"])
    logger.info("Defects detected:       %s", stats["defects_detected"])
    logger.info("Anomalies created:      %s", stats["anomalies_created"])
    logger.info("Detection rate:         %.1f%%", stats["detection_rate"] * 100)
    logger.info("Anomaly rate:           %.1f%%", stats["anomaly_rate"] * 100)
    logger.info("Average confidence:     %.3f", stats["average_confidence"])
    logger.info("Average severity:       %.3f", stats["average_severity"])

    # Display world model anomalies
    logger.info("\n%s", "=" * 80)
    logger.info("🗺️  WORLD MODEL ANOMALIES")
    logger.info("%s\n", "=" * 80)

    snapshot = world_model.get_snapshot()
    if snapshot.anomalies:
        for i, anomaly in enumerate(snapshot.anomalies, 1):
            logger.info("%s. Asset: %s", i, anomaly.asset_id)
            logger.info("   Severity: %.2f", anomaly.severity)
            logger.info("   Description: %s", anomaly.description)
            logger.info(
                "   Detected: %s",
                anomaly.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
            logger.info("")
    else:
        logger.info("No anomalies detected - all assets OK ✅")

    # Storage statistics
    logger.info("\n%s", "=" * 80)
    logger.info("💾 STORAGE STATISTICS")
    logger.info("%s\n", "=" * 80)

    storage_stats = vision_client.image_manager.get_storage_usage()
    logger.info("Total images:     %s", storage_stats["total_images"])
    logger.info("Storage used:     %.3f GB", storage_stats["total_size_gb"])
    logger.info("Storage limit:    %.1f GB", storage_stats["max_storage_gb"])
    logger.info("Usage:            %.1f%%", storage_stats["usage_percent"])

    # Cleanup
    logger.info("\n%s", "=" * 80)
    logger.info("🧹 CLEANUP")
    logger.info("%s\n", "=" * 80)

    await vision_client.shutdown()
    await vision_service.shutdown()

    logger.info("✅ Vision system demonstration complete!\n")
    logger.info("📂 Images saved to: data/vision/demo/")
    logger.info("🔍 Check the images to see simulated defects\n")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_vision_pipeline())
