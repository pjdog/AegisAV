"""
AegisAV High-Fidelity Simulation Package

Integrates with:
- AirSim (Unreal Engine) for photorealistic rendering
- ArduPilot SITL for rock-solid flight control
"""

from simulation.airsim_bridge import AirSimBridge, AirSimCameraConfig
from simulation.sitl_manager import SITLManager

__all__ = [
    "AirSimBridge",
    "AirSimCameraConfig",
    "SITLManager",
]
