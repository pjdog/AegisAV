"""ArduPilot SITL Manager for AegisAV

Manages ArduPilot Software-In-The-Loop simulation for rock-solid flight control.
This runs the EXACT same flight controller code that runs on real Pixhawk hardware.
"""

import asyncio
import logging
import os
import signal
import socket
import subprocess  # noqa: S404
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class SITLVehicle(Enum):
    """Supported SITL vehicle types."""

    COPTER = "ArduCopter"
    PLANE = "ArduPlane"
    ROVER = "Rover"


class SITLFrame(Enum):
    """SITL frame types for different simulators."""

    # Standalone (built-in physics)
    QUAD = "quad"
    HEXA = "hexa"
    OCTA = "octa"
    TRI = "tri"
    Y6 = "y6"

    # AirSim integration
    AIRSIM_COPTER = "airsim-copter"

    # Gazebo integration
    GAZEBO_IRIS = "gazebo-iris"

    # JSBSim integration
    JSBSIM = "jsbsim"


@dataclass
class SITLConfig:
    """Configuration for ArduPilot SITL."""

    # ArduPilot installation
    ardupilot_path: Path = field(default_factory=lambda: Path.home() / "ardupilot")

    # Vehicle configuration
    vehicle: SITLVehicle = SITLVehicle.COPTER
    frame: SITLFrame = SITLFrame.AIRSIM_COPTER

    # Network configuration
    sitl_host: str = "127.0.0.1"
    sitl_port: int = 5760  # SITL native port
    mavlink_port: int = 14550  # MAVLink output port
    mavlink_port2: int = 14551  # Secondary MAVLink port

    # Startup options
    speedup: float = 1.0  # Simulation speed (1.0 = realtime)
    home_lat: float = 37.7749  # Starting latitude
    home_lon: float = -122.4194  # Starting longitude
    home_alt: float = 0.0  # Starting altitude (m)
    home_heading: float = 0.0  # Starting heading (degrees)

    # Console/map options
    console: bool = True  # Show MAVProxy console
    map_display: bool = True  # Show map display

    # Logging
    log_dir: Path = field(default_factory=lambda: Path("logs/sitl"))


class SITLManager:
    """Manager for ArduPilot SITL simulation.

    Handles starting, stopping, and monitoring the SITL process.
    Designed for integration with AirSim for rendering.

    Example:
        config = SITLConfig(
            frame=SITLFrame.AIRSIM_COPTER,
            home_lat=37.7749,
            home_lon=-122.4194
        )
        manager = SITLManager(config)

        # Start SITL
        await manager.start()

        # ... run mission ...

        # Stop SITL
        await manager.stop()
    """

    def __init__(self, config: SITLConfig | None = None) -> None:
        """Initialize SITL manager."""
        self.config = config or SITLConfig()
        self.process: subprocess.Popen | None = None
        self.running = False

        # Validate ArduPilot path
        self._validate_ardupilot_installation()

        # Ensure log directory exists
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"SITLManager initialized (vehicle: {self.config.vehicle.value}, "
            f"frame: {self.config.frame.value})"
        )

    def _validate_ardupilot_installation(self) -> None:
        """Validate ArduPilot installation exists."""
        sitl_script = self.config.ardupilot_path / "Tools" / "autotest" / "sim_vehicle.py"

        if not sitl_script.exists():
            logger.warning(
                f"ArduPilot not found at {self.config.ardupilot_path}. "
                "SITL will not be available until installed. "
                "See simulation/README.md for installation instructions."
            )

    def _build_sitl_command(self) -> list[str]:
        """Build the SITL startup command."""
        sim_vehicle = self.config.ardupilot_path / "Tools" / "autotest" / "sim_vehicle.py"

        cmd = [
            "python3",
            str(sim_vehicle),
            "-v",
            self.config.vehicle.value,
            "-f",
            self.config.frame.value,
            "--speedup",
            str(self.config.speedup),
            "-l",
            f"{self.config.home_lat},{self.config.home_lon},{self.config.home_alt},{self.config.home_heading}",
            "--out",
            f"udp:{self.config.sitl_host}:{self.config.mavlink_port}",
            "--out",
            f"udp:{self.config.sitl_host}:{self.config.mavlink_port2}",
        ]

        if self.config.console:
            cmd.append("--console")

        if self.config.map_display:
            cmd.append("--map")

        return cmd

    async def start(self, timeout: float = 60.0) -> bool:
        """Start ArduPilot SITL.

        Args:
            timeout: Maximum time to wait for SITL to be ready

        Returns:
            True if SITL started successfully
        """
        if self.running:
            logger.warning("SITL already running")
            return True

        sim_vehicle = self.config.ardupilot_path / "Tools" / "autotest" / "sim_vehicle.py"
        if not sim_vehicle.exists():
            logger.error(f"sim_vehicle.py not found at {sim_vehicle}")
            return False

        try:
            logger.info("Starting ArduPilot SITL...")
            logger.info(f"  Vehicle: {self.config.vehicle.value}")
            logger.info(f"  Frame: {self.config.frame.value}")
            logger.info(f"  Home: {self.config.home_lat}, {self.config.home_lon}")

            cmd = self._build_sitl_command()
            logger.debug(f"Command: {' '.join(cmd)}")

            # Start process
            self.process = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.config.ardupilot_path / self.config.vehicle.value),
                env={**os.environ, "HOME": str(Path.home())},
                preexec_fn=os.setsid,  # Create new process group for clean shutdown
            )

            # Wait for SITL to be ready
            ready = await self._wait_for_ready(timeout)

            if ready:
                self.running = True
                logger.info(f"SITL started successfully (PID: {self.process.pid})")
                logger.info(
                    f"  MAVLink available at: udp:{self.config.sitl_host}:{self.config.mavlink_port}"
                )
                return True
            else:
                logger.error("SITL failed to start within timeout")
                await self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start SITL: {e}")
            return False

    async def _wait_for_ready(self, timeout: float) -> bool:
        """Wait for SITL to be ready to accept connections."""
        start_time = time.time()
        check_interval = 1.0

        while time.time() - start_time < timeout:
            # Check if process died
            if self.process and self.process.poll() is not None:
                logger.error(f"SITL process died with code: {self.process.returncode}")
                return False

            # Try to connect to MAVLink port
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(1.0)
                sock.connect((self.config.sitl_host, self.config.mavlink_port))
                sock.close()

                # Give it a moment more to fully initialize
                await asyncio.sleep(2.0)
                return True

            except OSError:
                await asyncio.sleep(check_interval)

        return False

    async def stop(self) -> None:
        """Stop ArduPilot SITL."""
        if not self.process:
            self.running = False
            return

        try:
            logger.info("Stopping SITL...")

            # Send SIGTERM to process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()

            logger.info("SITL stopped")

        except Exception as e:
            logger.error(f"Error stopping SITL: {e}")

        finally:
            self.process = None
            self.running = False

    async def restart(self) -> bool:
        """Restart SITL."""
        await self.stop()
        await asyncio.sleep(2.0)
        return await self.start()

    def is_running(self) -> bool:
        """Check if SITL is running."""
        if not self.process:
            return False
        return self.process.poll() is None

    @property
    def mavlink_connection_string(self) -> str:
        """Get MAVLink connection string for pymavlink."""
        return f"udp:{self.config.sitl_host}:{self.config.mavlink_port}"


class SITLEnvironment:
    """Complete SITL + AirSim environment manager.

    Coordinates starting both SITL and AirSim together.
    """

    def __init__(
        self, sitl_config: SITLConfig | None = None, airsim_binary: Path | None = None
    ) -> None:
        """Initialize environment."""
        self.sitl_manager = SITLManager(sitl_config)
        self.airsim_binary = airsim_binary
        self.airsim_process: subprocess.Popen | None = None

    async def start(self) -> bool:
        """Start both SITL and AirSim."""
        # Start AirSim first (it provides the physics)
        if self.airsim_binary:
            if not await self._start_airsim():
                return False

        # Then start SITL
        if not await self.sitl_manager.start():
            await self.stop()
            return False

        return True

    async def _start_airsim(self) -> bool:
        """Start AirSim binary."""
        if not self.airsim_binary or not self.airsim_binary.exists():
            logger.warning("AirSim binary not specified or not found")
            return True  # Continue without AirSim binary (might be started separately)

        try:
            logger.info(f"Starting AirSim: {self.airsim_binary}")

            self.airsim_process = subprocess.Popen(  # noqa: S603
                [str(self.airsim_binary), "-ResX=1920", "-ResY=1080", "-windowed"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Wait for AirSim to initialize
            await asyncio.sleep(10.0)

            if self.airsim_process.poll() is not None:
                logger.error("AirSim failed to start")
                return False

            logger.info("AirSim started")
            return True

        except Exception as e:
            logger.error(f"Failed to start AirSim: {e}")
            return False

    async def stop(self) -> None:
        """Stop both SITL and AirSim."""
        await self.sitl_manager.stop()

        if self.airsim_process:
            self.airsim_process.terminate()
            try:
                self.airsim_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.airsim_process.kill()
            self.airsim_process = None

    def is_running(self) -> bool:
        """Check if environment is running."""
        sitl_ok = self.sitl_manager.is_running()

        if self.airsim_process:
            airsim_ok = self.airsim_process.poll() is None
            return sitl_ok and airsim_ok

        return sitl_ok
