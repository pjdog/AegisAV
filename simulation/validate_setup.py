#!/usr/bin/env python3
"""AegisAV Simulation Setup Validator

Validates that all simulation components are properly configured:
- AirSim connection and camera capture
- ArduPilot SITL installation
- Python dependencies
- Directory structure
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_mark(passed: bool) -> str:
    """Return check or X mark."""
    return "\033[92m✓\033[0m" if passed else "\033[91m✗\033[0m"


async def validate_airsim() -> bool:
    """Test AirSim connection and camera capture."""
    print("\n[1/5] Testing AirSim connection...")

    try:
        from simulation.airsim_bridge import AIRSIM_AVAILABLE, AirSimBridge, AirSimCameraConfig

        if not AIRSIM_AVAILABLE:
            print(f"  {check_mark(False)} airsim package not installed")
            print("      Install with: pip install airsim")
            return False

        print(f"  {check_mark(True)} airsim package available")

        # Try to connect
        bridge = AirSimBridge(AirSimCameraConfig(
            output_dir=Path("data/validation"),
            save_images=True
        ))

        if await bridge.connect():
            print(f"  {check_mark(True)} Connected to AirSim")

            # Test camera capture
            result = await bridge.capture_frame({"validation": True})
            if result.success:
                print(f"  {check_mark(True)} Camera capture successful: {result.image_path}")

                # Check image dimensions
                if result.metadata.get("resolution"):
                    w, h = result.metadata["resolution"]
                    print(f"      Resolution: {w}x{h}")
            else:
                print(f"  {check_mark(False)} Camera capture failed")
                return False

            # Get vehicle state
            state = await bridge.get_vehicle_state()
            if state:
                print(f"  {check_mark(True)} Vehicle state available")
                print(f"      Position: ({state.position.latitude:.4f}, {state.position.longitude:.4f})")
                print(f"      Altitude: {state.position.altitude_msl:.1f}m")
            else:
                print(f"  {check_mark(False)} Vehicle state unavailable")

            await bridge.disconnect()
            return True
        else:
            print(f"  {check_mark(False)} AirSim connection failed")
            print("      Make sure Unreal Engine with AirSim is running!")
            return False

    except ImportError as e:
        print(f"  {check_mark(False)} Import error: {e}")
        return False
    except Exception as e:
        print(f"  {check_mark(False)} Error: {e}")
        return False


def validate_ardupilot() -> bool:
    """Check ArduPilot SITL installation."""
    print("\n[2/5] Checking ArduPilot SITL...")

    ardupilot_path = Path.home() / "ardupilot"
    sim_vehicle = ardupilot_path / "Tools" / "autotest" / "sim_vehicle.py"

    if ardupilot_path.exists():
        print(f"  {check_mark(True)} ArduPilot directory exists: {ardupilot_path}")
    else:
        print(f"  {check_mark(False)} ArduPilot not found at {ardupilot_path}")
        print("      Run: ./simulation/setup_desktop.sh")
        return False

    if sim_vehicle.exists():
        print(f"  {check_mark(True)} sim_vehicle.py found")
    else:
        print(f"  {check_mark(False)} sim_vehicle.py not found")
        print("      ArduPilot may not be fully installed")
        return False

    # Check for ArduCopter binary
    copter_binary = ardupilot_path / "build" / "sitl" / "bin" / "arducopter"
    if copter_binary.exists():
        print(f"  {check_mark(True)} ArduCopter binary built")
    else:
        print(f"  {check_mark(False)} ArduCopter binary not found")
        print("      Run: cd ~/ardupilot && ./waf configure --board sitl && ./waf copter")
        return False

    return True


def validate_dependencies() -> bool:
    """Check Python dependencies."""
    print("\n[3/5] Checking Python dependencies...")

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
            print(f"  {check_mark(True)} {package}")
        except ImportError:
            print(f"  {check_mark(False)} {package} - install with: pip install {package}")
            all_good = False

    return all_good


def validate_directories() -> bool:
    """Check required directories exist."""
    print("\n[4/5] Checking directory structure...")

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
            print(f"  {check_mark(True)} {dir_path}/")
        else:
            print(f"  {check_mark(False)} {dir_path}/ - missing")
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
            print(f"  {check_mark(True)} Created {dir_path}/")

    return all_good


def validate_airsim_settings() -> bool:
    """Check AirSim settings file."""
    print("\n[5/5] Checking AirSim settings...")

    settings_path = Path.home() / "Documents" / "AirSim" / "settings.json"

    if settings_path.exists():
        print(f"  {check_mark(True)} settings.json exists")

        # Check content
        import json
        try:
            with open(settings_path) as f:
                settings = json.load(f)

            if settings.get("SimMode") == "Multirotor":
                print(f"  {check_mark(True)} SimMode: Multirotor")
            else:
                print(f"  {check_mark(False)} SimMode should be 'Multirotor'")

            vehicles = settings.get("Vehicles", {})
            if vehicles:
                for name, config in vehicles.items():
                    vtype = config.get("VehicleType", "Unknown")
                    print(f"  {check_mark(True)} Vehicle '{name}': {vtype}")
            else:
                print(f"  {check_mark(False)} No vehicles configured")

            return True

        except json.JSONDecodeError as e:
            print(f"  {check_mark(False)} Invalid JSON: {e}")
            return False

    else:
        print(f"  {check_mark(False)} settings.json not found")
        print(f"      Expected at: {settings_path}")
        print("      Run: ./simulation/setup_desktop.sh")
        return False


async def main():
    """Run all validation checks."""
    print("=" * 60)
    print("  AegisAV Simulation Setup Validator")
    print("=" * 60)

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
    print("\n" + "-" * 60)
    print("Note: AirSim validation requires Unreal Engine to be running.")
    print("If Unreal is not running, AirSim test will be skipped.")
    print("-" * 60)

    try:
        results["AirSim"] = await asyncio.wait_for(
            validate_airsim(),
            timeout=10.0
        )
    except asyncio.TimeoutError:
        print(f"  {check_mark(False)} AirSim connection timed out")
        print("      Unreal Engine may not be running")
    except Exception as e:
        print(f"  {check_mark(False)} AirSim error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  Validation Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = check_mark(passed)
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("\033[92m  All checks passed! Ready for simulation.\033[0m")
    else:
        print("\033[93m  Some checks failed. See above for details.\033[0m")

    print()
    print("Next steps:")
    print("  1. Start Unreal Engine with AirSim")
    print("  2. Start ArduPilot SITL: sim_vehicle.py -v ArduCopter -f airsim-copter")
    print("  3. Run: python simulation/run_simulation.py --airsim --sitl")
    print("  4. Open: http://localhost:8000/dashboard")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
