"""Lightweight Physics Simulation for AegisAV.

A standalone drone simulation with realistic physics that runs without
Unreal Engine. Includes:
- Quadrotor dynamics with thrust, drag, gravity
- IMU simulation with noise
- Battery drain modeling
- Wind disturbances
- Web-based 3D visualizer

Usage:
    from simulation.lightweight import LightweightSim, DroneConfig

    sim = LightweightSim()
    sim.add_drone("drone_001", DroneConfig())
    await sim.start()

    # Get state (same API as AirSim bridge)
    state = sim.get_vehicle_state("drone_001")

    # Or use the drop-in replacement for AirSim:
    from simulation.lightweight import LightweightBridge

    bridge = LightweightBridge()
    await bridge.connect()
    state = await bridge.get_vehicle_state()
"""

from simulation.lightweight.bridge import LightweightBridge, LightweightCameraConfig
from simulation.lightweight.physics import (
    DroneConfig,
    DronePhysics,
    DroneState,
    EnvironmentConfig,
    IMUReading,
    MotorCommand,
    SimpleFlightController,
    WindModel,
)
from simulation.lightweight.simulator import LightweightSim

__all__ = [
    "DroneConfig",
    "DronePhysics",
    "DroneState",
    "EnvironmentConfig",
    "IMUReading",
    "LightweightBridge",
    "LightweightCameraConfig",
    "LightweightSim",
    "MotorCommand",
    "SimpleFlightController",
    "WindModel",
]
