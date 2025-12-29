"""
AegisAV Integrated Vision System Demo

Visual demonstration of the complete vision pipeline for video recording.
Generates:
- Annotated images with bounding boxes
- Real-time statistics dashboard
- Timeline visualization
- HTML report for browser viewing

Perfect for creating demo videos showing the system in action.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agent.server.goal_selector import GoalSelector
from agent.server.goals import GoalType
from agent.server.vision.detector import SimulatedDetector
from agent.server.vision.vision_service import VisionService, VisionServiceConfig
from agent.server.world_model import Asset, AssetType, WorldModel
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


class VisualDemo:
    """Visual demonstration controller."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.annotated_dir = output_dir / "annotated"
        self.annotated_dir.mkdir(exist_ok=True)

        self.reports_dir = output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Statistics tracking
        self.timeline = []
        self.inspection_count = 0
        self.defect_count = 0
        self.anomaly_count = 0
        self.reinspection_count = 0

    def log_event(self, event_type: str, message: str, data: dict | None = None):
        """Log an event to the timeline."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "data": data or {},
        }
        self.timeline.append(event)
        logger.info("[%s] %s", event_type.upper(), message)

    def create_annotated_image(
        self,
        image_path: Path,
        detections: list,
        asset_id: str,
        inspection_num: int,
    ) -> Path:
        """
        Create annotated image with bounding boxes and labels.

        Args:
            image_path: Original image path
            detections: List of Detection objects
            asset_id: Asset being inspected
            inspection_num: Inspection number

        Returns:
            Path to annotated image
        """
        # For demo purposes, create a simple colored image with annotations
        img_width, img_height = 1920, 1080
        bg_color = (40, 20, 20) if detections else (20, 40, 20)

        if image_path.exists():
            try:
                img = Image.open(image_path).convert("RGB")
            except OSError:
                img = Image.new("RGB", (img_width, img_height), bg_color)
        else:
            img = Image.new("RGB", (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)

        # Try to load a font, fallback to default
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 48
            )
            label_font = ImageFont.truetype(
                "/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 32
            )
            info_font = ImageFont.truetype(
                "/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 24
            )
        except OSError:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Draw header
        header_text = f"INSPECTION #{inspection_num} - {asset_id.upper()}"
        draw.text((50, 30), header_text, fill=(255, 255, 255), font=title_font)

        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((50, 100), f"Timestamp: {timestamp}", fill=(200, 200, 200), font=info_font)

        # Draw detection status
        if len(detections) > 0:
            status_text = f"⚠️ {len(detections)} DEFECT(S) DETECTED"
            status_color = (255, 100, 100)
        else:
            status_text = "✓ NO DEFECTS DETECTED"
            status_color = (100, 255, 100)

        draw.text((50, 150), status_text, fill=status_color, font=title_font)

        # Draw bounding boxes for detections
        y_offset = 250
        for i, detection in enumerate(detections):
            bbox = detection.bounding_box

            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox.x_min * img_width)
            y1 = int(bbox.y_min * img_height)
            x2 = int(bbox.x_max * img_width)
            y2 = int(bbox.y_max * img_height)

            # Draw bounding box
            box_color = self._get_detection_color(detection.severity)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=5)

            # Draw label background
            label_text = f"{detection.detection_class.value.upper()}"
            label_bg = [x1, y1 - 40, x1 + 300, y1]
            draw.rectangle(label_bg, fill=box_color)

            # Draw label text
            draw.text((x1 + 10, y1 - 35), label_text, fill=(255, 255, 255), font=label_font)

            # Draw detection info on the side
            info_y = y_offset + (i * 100)
            info_lines = [
                f"Detection {i + 1}: {detection.detection_class.value}",
                f"Confidence: {detection.confidence:.1%}",
                f"Severity: {detection.severity:.1%}",
            ]

            for j, line in enumerate(info_lines):
                draw.text((50, info_y + (j * 30)), line, fill=(255, 255, 255), font=info_font)

        # Draw footer with system info
        footer_y = img_height - 100
        draw.text(
            (50, footer_y),
            f"AegisAV Vision System | Model: SimulatedDetector | Resolution: {img_width}x{img_height}",
            fill=(150, 150, 150),
            font=info_font,
        )

        # Save annotated image
        output_path = self.annotated_dir / f"inspection_{inspection_num:03d}_{asset_id}.png"
        img.save(output_path)

        return output_path

    def _get_detection_color(self, severity: float) -> tuple:
        """Get color based on severity."""
        if severity >= 0.7:
            return (255, 50, 50)  # Red - high severity
        elif severity >= 0.4:
            return (255, 150, 50)  # Orange - medium severity
        else:
            return (255, 255, 50)  # Yellow - low severity

    def generate_html_report(self):
        """Generate an HTML report for browser viewing."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>AegisAV Vision System Demo Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        h1 {
            text-align: center;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #aaa;
            margin-bottom: 40px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .stat-value {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 1.1em;
            color: #ccc;
        }
        .timeline {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 40px;
            max-height: 400px;
            overflow-y: auto;
        }
        .timeline-event {
            background: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #4CAF50;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 4px;
        }
        .timeline-event.defect {
            border-left-color: #ff9800;
        }
        .timeline-event.anomaly {
            border-left-color: #f44336;
        }
        .timeline-event.reinspection {
            border-left-color: #2196F3;
        }
        .event-type {
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
            color: #4CAF50;
        }
        .event-type.defect { color: #ff9800; }
        .event-type.anomaly { color: #f44336; }
        .event-type.reinspection { color: #2196F3; }
        .event-message {
            margin: 5px 0;
        }
        .event-time {
            font-size: 0.85em;
            color: #888;
        }
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .image-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .image-card img {
            width: 100%;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .image-label {
            font-size: 0.9em;
            color: #ccc;
        }
        .success { color: #4CAF50; }
        .warning { color: #ff9800; }
        .error { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 AegisAV Vision System Demo</h1>
        <div class="subtitle">Integrated Computer Vision Pipeline - Live Demonstration</div>

        <h2>📊 Mission Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Inspections</div>
                <div class="stat-value">{{inspections}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Defects Detected</div>
                <div class="stat-value warning">{{defects}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Anomalies Created</div>
                <div class="stat-value error">{{anomalies}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Re-Inspections</div>
                <div class="stat-value">{{reinspections}}</div>
            </div>
        </div>

        <h2>📅 Event Timeline</h2>
        <div class="timeline">
            {{timeline_events}}
        </div>

        <h2>🖼️ Annotated Images</h2>
        <div class="image-gallery">
            {{image_gallery}}
        </div>

        <div style="text-align: center; margin-top: 40px; color: #888;">
            <p>Generated: {{timestamp}}</p>
            <p>AegisAV Autonomous Infrastructure Monitoring System</p>
        </div>
    </div>
</body>
</html>
"""

        # Generate timeline events HTML
        timeline_html = ""
        for event in self.timeline:
            event_type = event["type"]
            timeline_html += f"""
            <div class="timeline-event {event_type}">
                <div class="event-type {event_type}">{event_type}</div>
                <div class="event-message">{event["message"]}</div>
                <div class="event-time">{event["timestamp"]}</div>
            </div>
            """

        # Generate image gallery HTML
        gallery_html = ""
        for img_path in sorted(self.annotated_dir.glob("*.png")):
            rel_path = img_path.relative_to(self.reports_dir.parent)
            gallery_html += f"""
            <div class="image-card">
                <img src="../{rel_path}" alt="{img_path.stem}">
                <div class="image-label">{img_path.stem.replace("_", " ").title()}</div>
            </div>
            """

        # Replace placeholders
        html = html.replace("{{inspections}}", str(self.inspection_count))
        html = html.replace("{{defects}}", str(self.defect_count))
        html = html.replace("{{anomalies}}", str(self.anomaly_count))
        html = html.replace("{{reinspections}}", str(self.reinspection_count))
        html = html.replace("{{timeline_events}}", timeline_html)
        html = html.replace("{{image_gallery}}", gallery_html)
        html = html.replace("{{timestamp}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Save HTML report
        report_path = self.reports_dir / "demo_report.html"
        with open(report_path, "w") as f:
            f.write(html)

        return report_path


async def run_visual_demo():
    """Run the visual demonstration."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("\n%s", "=" * 80)
    logger.info("🎥 AegisAV INTEGRATED VISION SYSTEM - VISUAL DEMO")
    logger.info("%s", "=" * 80)
    logger.info("\nThis demo will:")
    logger.info("  1. Simulate multiple asset inspections with vision capture")
    logger.info("  2. Detect defects using computer vision")
    logger.info("  3. Create anomalies when thresholds are met")
    logger.info("  4. Trigger re-inspection goals")
    logger.info("  5. Generate annotated images and HTML report for video recording")
    logger.info("\n%s\n", "=" * 80)

    # Setup
    output_dir = Path("data/vision/demo_visual")
    demo = VisualDemo(output_dir)

    demo.log_event("system", "🚀 Initializing AegisAV Vision System...")

    # Create world model with test assets
    world_model = WorldModel()

    assets = [
        Asset(
            asset_id="solar_array_001",
            name="Solar Panel Array A",
            position=Position(
                latitude=37.7750, longitude=-122.4195, altitude_msl=0, altitude_agl=0
            ),
            asset_type=AssetType.SOLAR_PANEL,
            priority=1,
        ),
        Asset(
            asset_id="wind_turbine_001",
            name="Wind Turbine #1",
            position=Position(
                latitude=37.7760, longitude=-122.4200, altitude_msl=0, altitude_agl=0
            ),
            asset_type=AssetType.WIND_TURBINE,
            priority=1,
        ),
        Asset(
            asset_id="substation_001",
            name="Main Substation",
            position=Position(
                latitude=37.7770, longitude=-122.4205, altitude_msl=0, altitude_agl=0
            ),
            asset_type=AssetType.SUBSTATION,
            priority=2,
        ),
    ]

    for asset in assets:
        world_model.add_asset(asset)
        demo.log_event("system", f"✓ Added asset: {asset.name} ({asset.asset_id})")

    # Initialize dock position
    dock_position = Position(latitude=37.7749, longitude=-122.4194, altitude_msl=0, altitude_agl=0)
    world_model.set_dock(dock_position)

    # Initialize vehicle state for goal selection
    vehicle_state = VehicleState(
        timestamp=datetime.now(),
        position=Position(latitude=37.7749, longitude=-122.4194, altitude_msl=50, altitude_agl=50),
        velocity=Velocity(north=0.0, east=0.0, down=0.0),
        attitude=Attitude(roll=0.0, pitch=0.0, yaw=0.0),
        battery=BatteryState(voltage=22.5, current=5.0, remaining_percent=85.0),
        mode=FlightMode.GUIDED,
        armed=True,
        gps=GPSState(fix_type=3, satellites_visible=12, hdop=1.0, vdop=1.0),
        in_air=True,
    )
    world_model.update_vehicle(vehicle_state)

    # Setup vision components
    demo.log_event("system", "🔧 Initializing vision components...")

    # Client-side camera with high defect probability for demo
    defect_config = DefectConfig(
        crack_probability=0.25,
        corrosion_probability=0.20,
        structural_damage_probability=0.15,
        discoloration_probability=0.10,
        vegetation_probability=0.10,
    )

    camera_config = SimulatedCameraConfig(
        resolution=(1920, 1080),
        capture_format="RGB",
        output_dir=demo.images_dir,
    )
    camera_config.defect_config = defect_config
    camera = SimulatedCamera(config=camera_config)

    detector = MockYOLODetector(model_variant="yolov8n", confidence_threshold=0.5, device="cpu")

    # Server-side vision service
    server_detector = SimulatedDetector(defect_probability=0.4, severity_range=(0.4, 0.9))

    image_manager = ImageManager(base_dir=demo.images_dir)
    vision_service = VisionService(
        world_model=world_model,
        detector=server_detector,
        image_manager=image_manager,
        config=VisionServiceConfig(confidence_threshold=0.7, severity_threshold=0.4),
    )

    await camera.initialize()
    await detector.initialize()
    await vision_service.initialize()

    demo.log_event("system", "✅ All systems initialized and ready")

    # Create vehicle state
    vehicle_state = VehicleState(
        timestamp=datetime.now(),
        position=Position(
            latitude=37.7749, longitude=-122.4194, altitude_msl=50.0, altitude_agl=50.0
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

    # Run inspection missions
    demo.log_event("system", "\n🚁 Starting inspection missions...\n")

    goal_selector = GoalSelector()

    for inspection_num, asset in enumerate(assets, 1):
        demo.log_event("inspection", f"📍 Approaching {asset.name} ({asset.asset_id})")
        demo.inspection_count += 1

        # Capture images
        demo.log_event("inspection", "📸 Capturing images during orbit...")

        captures = []
        for _ in range(3):
            result = await camera.capture(vehicle_state=vehicle_state.to_dict())

            if result.success:
                # Quick client-side detection
                detection_result = await detector.analyze_image(result.image_path)
                captures.append({
                    "path": result.image_path,
                    "detection": detection_result,
                })

                if len(detection_result.detected_defects) > 0:
                    demo.log_event(
                        "defect",
                        f"⚠️ Client detected {len(detection_result.detected_defects)} potential defect(s) "
                        f"(confidence: {detection_result.max_confidence:.1%})",
                    )
                    demo.defect_count += 1

            await asyncio.sleep(0.1)

        # Get best capture for annotation
        best_capture = max(
            captures,
            key=lambda c: len(c["detection"].detected_defects),
        )

        # Create annotated image
        annotated_path = demo.create_annotated_image(
            best_capture["path"],
            best_capture["detection"].detected_defects,
            asset.asset_id,
            inspection_num,
        )

        demo.log_event(
            "inspection",
            f"🖼️ Created annotated image: {annotated_path.name}",
        )

        # Simulate feedback to server
        anomaly_detected = len(best_capture["detection"].detected_defects) > 0

        if anomaly_detected:
            demo.log_event(
                "inspection",
                "📤 Sending feedback to server (anomaly flag: TRUE)",
            )

            # Server-side detailed analysis
            observation = await vision_service.process_inspection_result(
                asset_id=asset.asset_id,
                client_detection=best_capture["detection"],
                vehicle_state=vehicle_state.to_dict(),
            )

            if observation.defect_detected:
                # Get primary detection type from detections list
                primary_detection = "unknown"
                if observation.detections and len(observation.detections) > 0:
                    primary_detection = observation.detections[0].get("detection_class", "unknown")

                demo.log_event(
                    "defect",
                    f"🔍 Server confirmed defect: {primary_detection} "
                    f"(confidence: {observation.max_confidence:.1%}, severity: {observation.max_severity:.1%})",
                )

                if observation.anomaly_created:
                    demo.anomaly_count += 1
                    demo.log_event(
                        "anomaly",
                        f"🚨 ANOMALY CREATED: {observation.anomaly_id}",
                        {
                            "asset": asset.asset_id,
                            "severity": f"{observation.max_severity:.1%}",
                            "confidence": f"{observation.max_confidence:.1%}",
                        },
                    )

                    # Check for re-inspection goals
                    snapshot = world_model.get_snapshot()
                    goal = await goal_selector.select_goal(snapshot)

                    if goal and goal.goal_type == GoalType.INSPECT_ANOMALY:
                        if goal.target_asset and goal.target_asset.asset_id == asset.asset_id:
                            demo.reinspection_count += 1
                            demo.log_event(
                                "reinspection",
                                f"🔄 RE-INSPECTION GOAL CREATED (priority: {goal.priority})",
                                {"asset": asset.asset_id},
                            )
                else:
                    demo.log_event(
                        "inspection",
                        "INFO: Defect below threshold - no anomaly created",
                    )
        else:
            demo.log_event("inspection", "✅ No defects detected - inspection complete")

        logger.info("")  # Blank line between inspections
        await asyncio.sleep(0.5)

    # Generate final report
    demo.log_event("system", "\n📝 Generating visual report...")

    report_path = demo.generate_html_report()

    # Cleanup
    await camera.shutdown()
    await detector.shutdown()
    await vision_service.shutdown()

    # Final summary
    logger.info("\n%s", "=" * 80)
    logger.info("✅ DEMO COMPLETE!")
    logger.info("%s", "=" * 80)
    logger.info("\n📊 FINAL STATISTICS:")
    logger.info("   • Total Inspections: %s", demo.inspection_count)
    logger.info("   • Defects Detected: %s", demo.defect_count)
    logger.info("   • Anomalies Created: %s", demo.anomaly_count)
    logger.info("   • Re-Inspections Triggered: %s", demo.reinspection_count)
    logger.info("\n📁 OUTPUT:")
    logger.info("   • Annotated Images: %s", demo.annotated_dir)
    logger.info("   • HTML Report: %s", report_path)
    logger.info("   • Full Output: %s", output_dir)
    logger.info("\n🎥 TO VIEW:")
    logger.info("   Open in browser: file://%s", report_path.absolute())
    logger.info("\n%s\n", "=" * 80)


if __name__ == "__main__":
    asyncio.run(run_visual_demo())
