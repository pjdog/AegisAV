# AegisAV High-Fidelity Simulation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UNREAL ENGINE + AirSim                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Photorealistic 3D Environment                                      │    │
│  │  • Infrastructure assets (solar farms, wind turbines, substations)  │    │
│  │  • Dynamic weather, lighting, time of day                           │    │
│  │  • Terrain, vegetation, buildings                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  AirSim Plugin                                                       │    │
│  │  • Drone physics simulation                                          │    │
│  │  • Camera sensors (RGB, Depth, Segmentation)                         │    │
│  │  • IMU, GPS, Barometer simulation                                    │    │
│  │  • MAVLink/ArduPilot integration                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      MAVLink Protocol        │
                    │   (UDP ports 14550/14551)    │
                    └──────────────┬──────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────────┐
│                         ArduPilot SITL                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Real Flight Controller Firmware                                     │    │
│  │  • ArduCopter (same code as real Pixhawk)                           │    │
│  │  • Mission execution, failsafes, flight modes                       │    │
│  │  • EKF state estimation                                              │    │
│  │  • PID control loops                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      MAVLink Protocol        │
                    │   (TCP/UDP to AegisAV)       │
                    └──────────────┬──────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────────┐
│                         AegisAV Agent                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │  Decision Engine │  │  Vision Pipeline │  │  MAVLink Interface       │   │
│  │  • LLM reasoning │  │  • YOLO detection│  │  • State collection      │   │
│  │  • Multi-critic  │  │  • Anomaly det.  │  │  • Command execution     │   │
│  │  • Goal selector │  │  • Image analysis│  │  • Mission primitives    │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  AirSim Bridge                                                        │   │
│  │  • Camera feed capture (RGB frames from Unreal)                       │   │
│  │  • Pose/telemetry sync                                                │   │
│  │  • Environment control (weather, time)                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Web Dashboard                                                        │   │
│  │  • Real-time mission monitoring                                       │   │
│  │  • Live camera feed display                                           │   │
│  │  • Detection overlays                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Unreal Engine + AirSim (Rendering & Physics)
- **Version**: Unreal Engine 4.27 (AirSim compatible)
- **Purpose**: Photorealistic environment rendering, camera simulation
- **Output**: RGB camera frames, depth maps, segmentation masks

### 2. ArduPilot SITL (Flight Control)
- **Version**: ArduCopter 4.4+ (latest stable)
- **Purpose**: Real flight controller firmware in simulation
- **Why**: Same code runs on actual Pixhawk hardware - rock solid

### 3. AegisAV Agent (Intelligence)
- **Purpose**: High-level decision making, vision processing
- **Integration**: MAVLink for flight control, AirSim API for camera

## Setup Instructions

### Desktop Requirements
- **GPU**: AMD 6950XT ✓ (Excellent)
- **CPU**: Ryzen ✓
- **RAM**: 36GB ✓
- **OS**: Windows 11 (recommended for Unreal) or Ubuntu 20.04+
- **Storage**: 100GB+ free (Unreal projects are large)

### Installation Steps

#### Step 1: Install Unreal Engine 4.27
```bash
# Via Epic Games Launcher (Windows) or build from source (Linux)
# AirSim requires UE 4.27 specifically
```

#### Step 2: Install AirSim
```bash
# Clone AirSim
git clone https://github.com/Microsoft/AirSim.git
cd AirSim

# Build (Windows)
build.cmd

# Build (Linux)
./setup.sh
./build.sh
```

#### Step 3: Install ArduPilot SITL
```bash
# Clone ArduPilot
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive

# Install dependencies
Tools/environment_install/install-prereqs-ubuntu.sh -y

# Build SITL
./waf configure --board sitl
./waf copter
```

#### Step 4: Configure AirSim for ArduPilot
Create `~/Documents/AirSim/settings.json`:
```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockType": "SteppableClock",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "ArduCopter",
      "UseSerial": false,
      "LocalHostIp": "127.0.0.1",
      "UdpIp": "127.0.0.1",
      "UdpPort": 9003,
      "SitlPort": 5760,
      "ControlPort": 14550,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 1920,
              "Height": 1080,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.5, "Y": 0, "Z": -0.5,
          "Pitch": -15, "Roll": 0, "Yaw": 0
        }
      },
      "X": 0, "Y": 0, "Z": 0
    }
  }
}
```

#### Step 5: Install AegisAV Simulation Dependencies
```bash
cd /path/to/AegisAV
pip install airsim msgpack-rpc-python
```

## Running the Full Simulation

### Terminal 1: Start ArduPilot SITL
```bash
cd ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter --console --map
```

### Terminal 2: Start Unreal/AirSim
```bash
# Launch your AirSim Unreal project
# Or use pre-built AirSim binaries
./AirSimNH.sh -ResX=1920 -ResY=1080 -windowed
```
On Windows (with the installer), run `start_airsim.bat` from the project root.

### Terminal 3: Start AegisAV
```bash
cd AegisAV
python3 run_simulation.py --airsim --sitl
```

### Terminal 4: Open Dashboard
```
http://localhost:8000/dashboard
```

For AirSim/Scenario environment synchronization details, see `docs/AIRSIM_SCENARIO_SYNC.md`.

## Environment Assets

### Infrastructure Scenes (To Build)
1. **Solar Farm** - Large PV array with varied panel conditions
2. **Wind Farm** - Multiple turbines at different heights
3. **Substation** - Electrical infrastructure with transformers
4. **Power Lines** - Transmission corridor inspection
5. **Industrial Facility** - Rooftop equipment inspection

### Defect Injection
- Place 3D defect meshes on assets (cracks, corrosion textures)
- Vary severity via material parameters
- Randomize placement for training data generation

## Data Flow

```
AirSim Camera → PNG/JPEG frames → AegisAV Vision Pipeline
     ↓                                      ↓
  1920x1080                           YOLO Detection
  60 FPS                                    ↓
                                    Anomaly Detection
                                           ↓
                                    Decision Engine
                                           ↓
                                    MAVLink Commands
                                           ↓
                                    ArduPilot SITL
                                           ↓
                                    AirSim Physics
```

## Next Steps

1. [ ] Set up AirSim + ArduPilot SITL on desktop
2. [ ] Create basic infrastructure environment in Unreal
3. [ ] Implement AirSim camera bridge in AegisAV
4. [ ] Test flight + vision integration
5. [ ] Build production-quality asset scenes
