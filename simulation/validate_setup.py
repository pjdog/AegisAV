#!/usr/bin/env python3
"""AegisAV Simulation Setup Validator

Validates that all simulation components are properly configured:
- AirSim connection and camera capture
- ArduPilot SITL installation
- Python dependencies
- Directory structure
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from simulation.airsim_bridge import AIRSIM_AVAILABLE, AirSimBridge, AirSimCameraConfig
except ImportError:
    AIRSIM_AVAILABLE = False
    AirSimBridge = None
    AirSimCameraConfig = None

logger = logging.getLogger(__name__)


def check_mark(passed: bool) -> str:
    """Return check or X mark."""
    return "\033[92m✓\033[0m" if passed else "\033[91m✗\033[0m"


async def validate_airsim() -> bool:
    """Test AirSim connection and camera capture."""
    logger.info("")
    logger.info("[1/5] Testing AirSim connection...")

    try:
        if not AIRSIM_AVAILABLE or AirSimBridge is None or AirSimCameraConfig is None:
            logger.info("  %s airsim package not installed", check_mark(False))
            logger.info("      Install with: pip install airsim")
            return False

        logger.info("  %s airsim package available", check_mark(True))

        # Try to connect
        bridge = AirSimBridge(
            AirSimCameraConfig(output_dir=Path("data/validation"), save_images=True)
        )

        if await bridge.connect():
            logger.info("  %s Connected to AirSim", check_mark(True))

            # Test camera capture
            result = await bridge.capture_frame({"validation": True})
            if result.success:
                logger.info(
                    "  %s Camera capture successful: %s",
                    check_mark(True),
                    result.image_path,
                )

                # Check image dimensions
                if result.metadata.get("resolution"):
                    w, h = result.metadata["resolution"]
                    logger.info("      Resolution: %sx%s", w, h)
            else:
                logger.info("  %s Camera capture failed", check_mark(False))
                return False

            # Get vehicle state
            state = await bridge.get_vehicle_state()
            if state:
                logger.info("  %s Vehicle state available", check_mark(True))
                logger.info(
                    "      Position: (%.4f, %.4f)",
                    state.position.latitude,
                    state.position.longitude,
                )
                logger.info("      Altitude: %.1fm", state.position.altitude_msl)
            else:
                logger.info("  %s Vehicle state unavailable", check_mark(False))

            await bridge.disconnect()
            return True
        else:
            logger.info("  %s AirSim connection failed", check_mark(False))
            logger.info("      Make sure Unreal Engine with AirSim is running!")
            return False

    except ImportError as e:
        logger.info("  %s Import error: %s", check_mark(False), e)
        return False
    except Exception as e:
        logger.info("  %s Error: %s", check_mark(False), e)
        return False


def validate_ardupilot() -> bool:
    """Check ArduPilot SITL installation."""
    logger.info("")
    logger.info("[2/5] Checking ArduPilot SITL...")

    ardupilot_path = Path.home() / "ardupilot"
    sim_vehicle = ardupilot_path / "Tools" / "autotest" / "sim_vehicle.py"

    if ardupilot_path.exists():
        logger.info("  %s ArduPilot directory exists: %s", check_mark(True), ardupilot_path)
    else:
        logger.info("  %s ArduPilot not found at %s", check_mark(False), ardupilot_path)
        logger.info("      Run: ./simulation/setup_desktop.sh")
        return False

    if sim_vehicle.exists():
        logger.info("  %s sim_vehicle.py found", check_mark(True))
    else:
        logger.info("  %s sim_vehicle.py not found", check_mark(False))
        logger.info("      ArduPilot may not be fully installed")
        return False

    # Check for ArduCopter binary
    copter_binary = ardupilot_path / "build" / "sitl" / "bin" / "arducopter"
    if copter_binary.exists():
        logger.info("  %s ArduCopter binary built", check_mark(True))
    else:
        logger.info("  %s ArduCopter binary not found", check_mark(False))
        logger.info("      Run: cd ~/ardupilot && ./waf configure --board sitl && ./waf copter")
        return False

    return True


def validate_dependencies() -> bool:
    """Check Python dependencies."""
    logger.info("")
    logger.info("[3/5] Checking Python dependencies...")

    required = [
        ("airsim", "airsim"),
        ("pymavlink", "pymavlink"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("pydantic", "pydantic"),
        ("fastapi", "fastapi"),
    ]

    all_good = True
    for module, package in required:
        try:
            __import__(module)
            logger.info("  %s %s", check_mark(True), package)
        except ImportError:
            logger.info(
                "  %s %s - install with: pip install %s",
                check_mark(False),
                package,
                package,
            )
            all_good = False

    return all_good


def validate_directories() -> bool:
    """Check required directories exist."""
    logger.info("")
    logger.info("[4/5] Checking directory structure...")

    project_root = Path(__file__).parent.parent
    required_dirs = [
        "simulation",
        "agent/server",
        "agent/client",
        "autonomy",
        "vision",
        "data",
        "logs",
    ]

    all_good = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            logger.info("  %s %s/", check_mark(True), dir_path)
        else:
            logger.info("  %s %s/ - missing", check_mark(False), dir_path)
            all_good = False

    # Create data directories if missing
    data_dirs = [
        "data/vision/airsim",
        "data/vision/validation",
        "data/telemetry",
        "logs/sitl",
    ]

    for dir_path in data_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info("  %s Created %s/", check_mark(True), dir_path)

    return all_good


def validate_airsim_settings() -> bool:
    """Check AirSim settings file."""
    logger.info("")
    logger.info("[5/5] Checking AirSim settings...")

    settings_path = Path.home() / "Documents" / "AirSim" / "settings.json"

    if settings_path.exists():
        logger.info("  %s settings.json exists", check_mark(True))

        # Check content
        try:
            with open(settings_path) as f:
                settings = json.load(f)

            if settings.get("SimMode") == "Multirotor":
                logger.info("  %s SimMode: Multirotor", check_mark(True))
            else:
                logger.info("  %s SimMode should be 'Multirotor'", check_mark(False))

            vehicles = settings.get("Vehicles", {})
            if vehicles:
                for name, config in vehicles.items():
                    vtype = config.get("VehicleType", "Unknown")
                    logger.info(
                        "  %s Vehicle '%s': %s",
                        check_mark(True),
                        name,
                        vtype,
                    )
            else:
                logger.info("  %s No vehicles configured", check_mark(False))

            return True

        except json.JSONDecodeError as e:
            logger.info("  %s Invalid JSON: %s", check_mark(False), e)
            return False

    else:
        logger.info("  %s settings.json not found", check_mark(False))
        logger.info("      Expected at: %s", settings_path)
        logger.info("      Run: ./simulation/setup_desktop.sh")
        return False


async def main() -> int:
    """Run all validation checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("  AegisAV Simulation Setup Validator")
    logger.info("=" * 60)

    results = {
        "AirSim": False,
        "ArduPilot": False,
        "Dependencies": False,
        "Directories": False,
        "Settings": False,
    }

    # Run checks
    results["Dependencies"] = validate_dependencies()
    results["Directories"] = validate_directories()
    results["ArduPilot"] = validate_ardupilot()
    results["Settings"] = validate_airsim_settings()

    # AirSim requires the simulator to be running
    logger.info("")
    logger.info("-" * 60)
    logger.info("Note: AirSim validation requires Unreal Engine to be running.")
    logger.info("If Unreal is not running, AirSim test will be skipped.")
    logger.info("-" * 60)

    try:
        results["AirSim"] = await asyncio.wait_for(validate_airsim(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.info("  %s AirSim connection timed out", check_mark(False))
        logger.info("      Unreal Engine may not be running")
    except Exception as e:
        logger.info("  %s AirSim error: %s", check_mark(False), e)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Validation Summary")
    logger.info("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = check_mark(passed)
        logger.info("  %s %s", status, name)
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\033[92m  All checks passed! Ready for simulation.\033[0m")
    else:
        logger.info("\033[93m  Some checks failed. See above for details.\033[0m")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start Unreal Engine with AirSim")
    logger.info("  2. Start ArduPilot SITL: sim_vehicle.py -v ArduCopter -f airsim-copter")
    logger.info("  3. Run: python simulation/run_simulation.py --airsim --sitl")
    logger.info("  4. Open: http://localhost:8000/dashboard")
    logger.info("")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
