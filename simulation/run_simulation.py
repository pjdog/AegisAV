#!/usr/bin/env python3
"""AegisAV Full Simulation Runner

Runs the complete high-fidelity simulation:
1. ArduPilot SITL - Real flight controller
2. AirSim - Photorealistic rendering
3. AegisAV Agent - AI decision making
4. Web Dashboard - Real-time monitoring

Usage:
    python simulation/run_simulation.py --airsim --sitl
    python simulation/run_simulation.py --sitl-only  # No rendering
    python simulation/run_simulation.py --mock       # Pure software sim
"""

import argparse
import asyncio
import logging
import os
import subprocess  # noqa: S404
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomy.vehicle_state import Position
from simulation.sitl_manager import SITLConfig, SITLFrame, SITLManager, SITLVehicle

try:
    from simulation.airsim_bridge import AirSimBridge, AirSimCameraConfig

    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False

from autonomy.mavlink_interface import MAVLinkInterface
from vision.models.yolo_detector import MockYOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Simulation")


class FullSimulation:
    """Orchestrates the complete AegisAV simulation.

    Manages all components:
    - ArduPilot SITL (flight control)
    - AirSim (rendering + physics)
    - AegisAV Server (decision making)
    - Vision Pipeline (detection)
    """

    def __init__(
        self,
        use_airsim: bool = True,
        use_sitl: bool = True,
        ardupilot_path: Path | None = None,
        home_position: tuple[float, float, float] = (37.7749, -122.4194, 0.0),
    ) -> None:
        """Initialize simulation components."""
        self.use_airsim = use_airsim and AIRSIM_AVAILABLE
        self.use_sitl = use_sitl

        # SITL configuration
        self.sitl_config = SITLConfig(
            ardupilot_path=ardupilot_path or Path.home() / "ardupilot",
            vehicle=SITLVehicle.COPTER,
            frame=SITLFrame.AIRSIM_COPTER if self.use_airsim else SITLFrame.QUAD,
            home_lat=home_position[0],
            home_lon=home_position[1],
            home_alt=home_position[2],
            console=True,
            map_display=True,
        )

        # Component references
        self.sitl_manager: SITLManager | None = None
        self.airsim_bridge: AirSimBridge | None = None
        self.mavlink: MAVLinkInterface | None = None
        self.server_process: subprocess.Popen | None = None
        self.detector: MockYOLODetector | None = None

        # State
        self.running = False
        self.mission_active = False

    async def start_all(self) -> bool:
        """Start all simulation components."""
        logger.info("=" * 70)
        logger.info("  AEGIS AV FULL SIMULATION")
        logger.info("=" * 70)
        logger.info("")

        # 1. Start SITL
        if self.use_sitl:
            logger.info("[1/4] Starting ArduPilot SITL...")
            self.sitl_manager = SITLManager(self.sitl_config)

            if not await self.sitl_manager.start(timeout=60.0):
                logger.error("Failed to start SITL")
                return False
            logger.info("  ✓ SITL running")
        else:
            logger.info("[1/4] SITL disabled (mock mode)")

        # 2. Connect to AirSim
        if self.use_airsim:
            logger.info("[2/4] Connecting to AirSim...")
            self.airsim_bridge = AirSimBridge(
                AirSimCameraConfig(output_dir=Path("data/vision/simulation"), save_images=True)
            )

            if not await self.airsim_bridge.connect():
                logger.warning("AirSim not available - using mock vision")
                self.airsim_bridge = None
            else:
                logger.info("  ✓ AirSim connected")
        else:
            logger.info("[2/4] AirSim disabled")

        # 3. Connect MAVLink
        if self.use_sitl and self.sitl_manager:
            logger.info("[3/4] Connecting MAVLink...")
            self.mavlink = MAVLinkInterface()

            connected = await self.mavlink.connect(self.sitl_manager.mavlink_connection_string)
            if not connected:
                logger.error("Failed to connect MAVLink")
                return False
            logger.info("  ✓ MAVLink connected")
        else:
            logger.info("[3/4] MAVLink disabled (mock mode)")

        # 4. Start AegisAV server
        logger.info("[4/4] Starting AegisAV server...")
        self.server_process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "uvicorn",
                "agent.server.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--log-level",
                "warning",
            ],
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )
        await asyncio.sleep(3)  # Wait for server startup
        logger.info("  ✓ Server running on http://localhost:8000")

        # Initialize vision detector
        self.detector = MockYOLODetector(model_variant="yolov8n", confidence_threshold=0.4)
        await self.detector.initialize()

        self.running = True

        logger.info("")
        logger.info("=" * 70)
        logger.info("  ALL SYSTEMS READY")
        logger.info("=" * 70)
        logger.info("")
        logger.info("  Dashboard: http://localhost:8000/dashboard")
        if self.use_sitl:
            logger.info(f"  MAVLink:   {self.sitl_manager.mavlink_connection_string}")
        if self.use_airsim and self.airsim_bridge:
            logger.info("  AirSim:    Connected (Unreal Engine)")
        logger.info("")

        return True

    async def run_inspection_mission(self, waypoints: list[tuple[float, float, float]]) -> None:
        """Run an inspection mission.

        Args:
            waypoints: List of (lat, lon, alt) waypoints to inspect
        """
        if not self.running:
            logger.error("Simulation not running")
            return

        logger.info("Starting inspection mission...")
        self.mission_active = True

        for i, (lat, lon, alt) in enumerate(waypoints):
            if not self.mission_active:
                break

            logger.info(f"\n--- Waypoint {i + 1}/{len(waypoints)} ---")
            logger.info(f"Target: {lat:.6f}, {lon:.6f}, {alt:.1f}m")

            # Navigate to waypoint
            if self.mavlink:
                await self.mavlink.goto(lat, lon, alt)

                # Wait to arrive
                while True:
                    state = await self.mavlink.get_vehicle_state()
                    if state:
                        dist = state.position.distance_to(
                            Position(latitude=lat, longitude=lon, altitude_msl=alt)
                        )
                        if dist < 5.0:  # Within 5m
                            break
                    await asyncio.sleep(0.5)

            # Capture and analyze image
            if self.airsim_bridge:
                logger.info("Capturing image from AirSim...")
                capture = await self.airsim_bridge.capture_frame(
                    metadata={"waypoint": i, "position": (lat, lon, alt)}
                )

                if capture.success:
                    logger.info(f"Image captured: {capture.image_path}")

                    # Run detection
                    detection = await self.detector.analyze_image(capture.image_path)

                    if detection.detected_any:
                        for det in detection.detections:
                            logger.warning(
                                f"  DETECTED: {det.detection_class.value} "
                                f"(conf: {det.confidence:.1%})"
                            )
                    else:
                        logger.info("  No defects detected")
            else:
                logger.info("(No AirSim - skipping image capture)")

            await asyncio.sleep(2)

        self.mission_active = False
        logger.info("\nMission complete!")

    async def stop_all(self) -> None:
        """Stop all simulation components."""
        logger.info("Stopping simulation...")

        self.running = False
        self.mission_active = False

        # Stop detector
        if self.detector:
            await self.detector.shutdown()

        # Disconnect MAVLink
        if self.mavlink:
            await self.mavlink.disconnect()

        # Disconnect AirSim
        if self.airsim_bridge:
            await self.airsim_bridge.disconnect()

        # Stop SITL
        if self.sitl_manager:
            await self.sitl_manager.stop()

        # Stop server
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)

        logger.info("Simulation stopped")


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AegisAV Full Simulation")
    parser.add_argument("--airsim", action="store_true", help="Enable AirSim rendering")
    parser.add_argument("--sitl", action="store_true", help="Enable ArduPilot SITL")
    parser.add_argument("--mock", action="store_true", help="Mock mode (no SITL/AirSim)")
    parser.add_argument("--ardupilot", type=Path, help="Path to ArduPilot installation")

    args = parser.parse_args()

    # Default to full simulation if no flags specified
    if not args.airsim and not args.sitl and not args.mock:
        args.sitl = True
        args.airsim = True

    sim = FullSimulation(
        use_airsim=args.airsim and not args.mock,
        use_sitl=args.sitl and not args.mock,
        ardupilot_path=args.ardupilot,
    )

    try:
        if not await sim.start_all():
            logger.error("Failed to start simulation")
            return 1

        # Demo waypoints (solar farm inspection)
        waypoints = [
            (37.7750, -122.4195, 50.0),  # Solar Array A
            (37.7760, -122.4200, 60.0),  # Wind Turbine
            (37.7770, -122.4205, 40.0),  # Substation
        ]

        await sim.run_inspection_mission(waypoints)

        # Keep running for dashboard access
        logger.info("\nSimulation complete. Dashboard still available.")
        logger.info("Press Ctrl+C to exit...")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    finally:
        await sim.stop_all()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
