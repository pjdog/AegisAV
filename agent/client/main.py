"""
Agent Client Main Entry Point

Connects to the vehicle via MAVLink and to the agent server via HTTP.
Runs the main control loop: collect state -> send to server -> execute decisions.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from agent.client.action_executor import ActionExecutor, ExecutionState
from agent.client.state_collector import CollectorConfig, StateCollector
from agent.client.vision_client import VisionClient, VisionClientConfig
from autonomy.mavlink_interface import MAVLinkConfig, MAVLinkInterface
from vision.camera.simulated import DefectConfig, SimulatedCamera, SimulatedCameraConfig
from vision.image_manager import ImageManager
from vision.models.yolo_detector import MockYOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class AgentClient:
    """
    Main agent client orchestrating state collection and action execution.

    The client connects to:
    - The vehicle via MAVLink (receives telemetry, sends commands)
    - The agent server via HTTP (sends state, receives decisions)

    It runs a continuous loop:
    1. Collect vehicle state
    2. Send state to agent server
    3. Receive decision
    4. Execute decision
    5. Repeat
    """

    def __init__(
        self,
        mavlink_config: MAVLinkConfig,
        collector_config: CollectorConfig,
        vision_client: VisionClient | None = None,
    ):
        self.mavlink = MAVLinkInterface(mavlink_config)
        self.state_collector = StateCollector(self.mavlink, collector_config)
        self.vision_client = vision_client
        self.action_executor = ActionExecutor(self.mavlink, vision_client=vision_client)

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to vehicle and verify server."""
        # Connect to MAVLink
        logger.info("Connecting to vehicle...")
        if not await self.mavlink.connect():
            logger.error("Failed to connect to vehicle")
            return False

        # Check server health
        logger.info("Checking agent server...")
        if not await self.state_collector.check_server_health():
            logger.warning("Agent server not responding, will retry during operation")

        # Initialize optional client-side vision
        if self.vision_client:
            logger.info("Initializing vision client...")
            vision_ok = await self.vision_client.initialize()
            if not vision_ok:
                logger.warning("Vision client initialization failed; continuing without vision")
                self.vision_client = None
                self.action_executor.vision_client = None

        return True

    async def disconnect(self) -> None:
        """Disconnect from vehicle."""
        if self.vision_client:
            await self.vision_client.shutdown()
        await self.mavlink.disconnect()

    async def run(self) -> None:
        """Run the main agent client loop."""
        if not await self.connect():
            return

        self._running = True
        logger.info("Agent client running")

        try:
            async for decision in self.state_collector.run():
                if not self._running:
                    break

                action = decision.get("action", "none")

                # Execute the decision (including wait/none for outcome tracking)
                result = await self.action_executor.execute(decision)

                if result.state == ExecutionState.FAILED:
                    logger.error("Action failed: %s", result.message)
                elif result.state == ExecutionState.COMPLETED:
                    logger.info("Action completed in %.1fs", result.duration_s)

                # Send feedback to server to close the loop
                feedback = _build_feedback(decision, result, self.action_executor)
                if feedback.get("decision_id") and feedback["decision_id"] != "unknown":
                    await self.state_collector.send_feedback(feedback)

                # Check for shutdown
                if self._shutdown_event.is_set():
                    break

        except asyncio.CancelledError:
            logger.info("Agent client cancelled")
        finally:
            self._running = False
            await self.disconnect()

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()


def load_config(config_path: Path) -> tuple[MAVLinkConfig, CollectorConfig]:
    """Load configuration from YAML file."""
    mavlink_config = MAVLinkConfig()
    collector_config = CollectorConfig()

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # MAVLink config
        mavlink_section = config.get("mavlink", {})
        mavlink_config = MAVLinkConfig(
            connection_string=mavlink_section.get("connection", mavlink_config.connection_string),
            source_system=mavlink_section.get("source_system", mavlink_config.source_system),
            timeout_ms=mavlink_section.get("timeout_ms", mavlink_config.timeout_ms),
        )

        # Client/collector config
        client_section = config.get("client", {})
        server_section = config.get("server", {})
        collector_config = CollectorConfig(
            server_url=client_section.get(
                "server_url",
                f"http://localhost:{server_section.get('port', 8080)}",
            ),
            update_interval_s=client_section.get("update_interval_ms", 100) / 1000,
        )

    return mavlink_config, collector_config


def _build_vision_client(vision_config: dict[str, Any]) -> VisionClient | None:
    vision_section = vision_config.get("vision", {})
    if not isinstance(vision_section, dict) or not vision_section.get("enabled", False):
        return None

    camera_section = vision_section.get("camera", {}) if isinstance(vision_section, dict) else {}
    client_section = vision_section.get("client", {}) if isinstance(vision_section, dict) else {}
    simulation_section = (
        vision_section.get("simulation", {}) if isinstance(vision_section, dict) else {}
    )

    camera_type = camera_section.get("type", "simulated")
    if camera_type != "simulated":
        logger.warning("Unsupported camera type: %s (vision disabled)", camera_type)
        return None

    resolution = camera_section.get("resolution", [1920, 1080])
    if not isinstance(resolution, list) or len(resolution) != 2:
        resolution = [1920, 1080]

    defect_probs = simulation_section.get("defects", {}) if isinstance(simulation_section, dict) else {}
    severity_cfg = simulation_section.get("severity", {}) if isinstance(simulation_section, dict) else {}

    defect_config = DefectConfig(
        crack_probability=float(defect_probs.get("crack_probability", 0.10)),
        corrosion_probability=float(defect_probs.get("corrosion_probability", 0.08)),
        structural_damage_probability=float(defect_probs.get("structural_damage_probability", 0.03)),
        discoloration_probability=float(defect_probs.get("discoloration_probability", 0.05)),
        vegetation_probability=float(defect_probs.get("vegetation_probability", 0.05)),
        damage_probability=float(defect_probs.get("damage_probability", 0.03)),
        severity_min=float(severity_cfg.get("min", 0.3)),
        severity_max=float(severity_cfg.get("max", 0.9)),
    )

    camera = SimulatedCamera(
        config=SimulatedCameraConfig(
            resolution=(int(resolution[0]), int(resolution[1])),
            capture_format=str(camera_section.get("capture_format", "RGB")),
            output_dir=camera_section.get("output_dir", "data/vision/captures"),
            defect_config=defect_config,
            save_images=bool(camera_section.get("save_images", True)),
        )
    )

    detection_section = client_section.get("detection", {}) if isinstance(client_section, dict) else {}
    model_section = client_section.get("model", {}) if isinstance(client_section, dict) else {}
    capture_section = client_section.get("capture", {}) if isinstance(client_section, dict) else {}

    detector = MockYOLODetector(
        model_variant=str(model_section.get("variant", "yolov8n")),
        confidence_threshold=float(detection_section.get("confidence_threshold", 0.4)),
        device=str(model_section.get("device", "cpu")),
    )

    vision_client_config = VisionClientConfig(
        capture_interval_s=float(capture_section.get("interval_s", 2.0)),
        max_captures_per_inspection=int(capture_section.get("max_images_per_inspection", 10)),
        enabled=True,
    )

    image_manager = ImageManager(base_dir=camera_section.get("output_dir", "data/vision/captures"))
    return VisionClient(
        camera=camera,
        detector=detector,
        image_manager=image_manager,
        config=vision_client_config,
    )


def _build_feedback(
    decision: dict[str, Any],
    result: Any,
    action_executor: ActionExecutor,
) -> dict[str, Any]:
    # Map execution result to server-side ExecutionStatus enum values
    status = "failed"
    if result.state == ExecutionState.COMPLETED:
        status = "success"
    elif result.state == ExecutionState.ABORTED:
        status = "aborted"

    decision_id = decision.get("decision_id", "unknown")
    action = decision.get("action", "none")
    parameters = decision.get("parameters", {}) if isinstance(decision.get("parameters"), dict) else {}

    feedback: dict[str, Any] = {
        "decision_id": decision_id,
        "status": status,
        "duration_s": getattr(result, "duration_s", None),
        "mission_objective_achieved": status == "success" and action != "abort",
        "asset_inspected": parameters.get("asset_id") if action == "inspect" else None,
        "anomaly_detected": False,
        "errors": [result.message] if result.state == ExecutionState.FAILED and result.message else [],
        "notes": result.message or None,
    }

    # Attach inspection vision payload if available
    if action == "inspect":
        inspection_results = action_executor.get_last_inspection_results()
        asset_id = parameters.get("asset_id")
        if inspection_results and (asset_id is None or inspection_results.asset_id == asset_id):
            defect_detections = [d for d in inspection_results.detections if d.detected_defects]
            best_detection = None
            if defect_detections:
                best_detection = max(defect_detections, key=lambda d: d.max_confidence)
            elif inspection_results.detections:
                best_detection = max(inspection_results.detections, key=lambda d: d.max_confidence)

            best_image_path = best_detection.image_path if best_detection else None
            best_capture = None
            if best_image_path:
                best_capture = next(
                    (c for c in inspection_results.captures if c.image_path == best_image_path),
                    None,
                )

            vehicle_state = None
            if best_capture and isinstance(best_capture.metadata, dict):
                vehicle_state = best_capture.metadata.get("vehicle_state")

            anomaly_detected = inspection_results.defects_detected > 0
            feedback["anomaly_detected"] = anomaly_detected
            feedback["inspection_data"] = {
                "asset_id": inspection_results.asset_id,
                "vehicle_state": vehicle_state,
                "vision": {
                    "summary": inspection_results.to_dict(),
                    "best_image_path": str(best_image_path) if best_image_path else None,
                    "best_detection": (
                        best_detection.model_dump(mode="json") if best_detection else None
                    ),
                    "vehicle_state": vehicle_state,
                },
            }

    return feedback


async def async_main(args: argparse.Namespace) -> None:
    """Async main entry point."""
    # Load config
    config_path = Path(args.config)
    mavlink_config, collector_config = load_config(config_path)

    # Override with CLI args
    if args.connection:
        mavlink_config.connection_string = args.connection
    if args.server:
        collector_config.server_url = args.server

    # Create client
    vision_client = None
    vision_config_path = config_path.parent / "vision_config.yaml"
    if vision_config_path.exists():
        with open(vision_config_path, encoding="utf-8") as f:
            vision_config = yaml.safe_load(f) or {}
        if isinstance(vision_config, dict):
            vision_client = _build_vision_client(vision_config)

    client = AgentClient(mavlink_config, collector_config, vision_client=vision_client)

    # Set up signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, client.stop)

    # Run
    await client.run()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AegisAV Agent Client")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/agent_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--connection",
        type=str,
        help="MAVLink connection string (overrides config)",
    )
    parser.add_argument(
        "--server",
        type=str,
        help="Agent server URL (overrides config)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
