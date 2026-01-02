# AegisAV Unreal Engine Setup Guide

Complete guide for setting up high-fidelity infrastructure inspection environments in Unreal Engine with Cosys-AirSim integration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Unreal Engine Installation](#unreal-engine-installation)
3. [Cosys-AirSim Plugin Setup](#cosys-airsim-plugin-setup)
4. [Project Creation](#project-creation)
5. [Building Infrastructure Environments](#building-infrastructure-environments)
6. [Defect Injection System](#defect-injection-system)
7. [Camera & Lighting Setup](#camera--lighting-setup)
8. [ArduPilot Integration](#ardupilot-integration)
9. [Testing & Validation](#testing--validation)

---

## Prerequisites

### Hardware Requirements (Your Desktop)
- **GPU**: AMD 6950XT (Excellent - 16GB VRAM)
- **CPU**: Ryzen (Modern 8+ core recommended)
- **RAM**: 36GB (Excellent)
- **Storage**: 150GB+ free (SSD strongly recommended)

### Software Requirements
- Ubuntu 22.04 LTS (or Windows 11)
- Unreal Engine 5.5 (Recommended for Cosys-AirSim v3.3) or UE 5.2 (LTS)
- Visual Studio 2022/2026 (Windows) or clang (Linux)
- Git, CMake, Python 3.10+

---

## Unreal Engine Installation

### Linux Installation (Recommended for Your Setup)

```bash
# 1. Register for Epic Games account and link to GitHub
# Visit: https://www.unrealengine.com/en-US/ue-on-github

# 2. Clone Unreal Engine 5.5 (recommended for Cosys-AirSim)
cd ~
git clone -b 5.5 git@github.com:EpicGames/UnrealEngine.git
cd UnrealEngine

# 3. Run setup scripts
./Setup.sh
./GenerateProjectFiles.sh

# 4. Build Unreal Engine (this takes 1-2 hours)
make

# 5. Verify installation
./Engine/Binaries/Linux/UnrealEditor
```

### AMD GPU Optimization

Create `~/.config/UnrealEngine/5.5/Engine.ini`:

```ini
[/Script/Engine.RendererSettings]
r.DefaultFeature.AntiAliasing=2
r.VSync=0
r.AllowOcclusionQueries=1
r.MinScreenRadiusForLights=0.03
r.MinScreenRadiusForDepthPrepass=0.03
r.PrecomputedVisibilityWarning=0
r.Shadow.RadiusThreshold=0.01

[SystemSettings]
r.ViewDistanceScale=1.0
r.PostProcessAAQuality=4
r.DetailMode=2
r.MaterialQualityLevel=1

# AMD-specific optimizations
r.GPUCrashDebugging=0
r.Vulkan.EnableValidation=0
```

---

## Cosys-AirSim Plugin Setup

Cosys-AirSim is the actively maintained fork of Microsoft AirSim with UE 5.5 support.

### Clone and Build Cosys-AirSim

```bash
# 1. Clone Cosys-AirSim
cd ~
git clone https://github.com/Cosys-Lab/Cosys-AirSim.git
cd Cosys-AirSim

# 2. Run setup (installs dependencies)
./setup.sh

# 3. Build Cosys-AirSim
./build.sh

# 4. Copy plugin to Unreal Engine
# This step is done per-project (see Project Creation)
```

**Or download pre-built binaries from:**
https://github.com/Cosys-Lab/Cosys-AirSim/releases

### Cosys-AirSim Settings Configuration

Create `~/Documents/AirSim/settings.json`:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockType": "SteppableClock",
  "LocalHostIp": "127.0.0.1",
  "RecordUIVisible": false,
  "LogMessagesVisible": false,
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "AegisInspector": {
      "VehicleType": "ArduCopter",
      "UseSerial": false,
      "LocalHostIp": "127.0.0.1",
      "UdpIp": "127.0.0.1",
      "UdpPort": 9003,
      "SitlPort": 5760,
      "ControlPort": 14550,
      "AutoCreate": true,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 1920,
              "Height": 1080,
              "FOV_Degrees": 90,
              "TargetGamma": 1.5
            },
            {
              "ImageType": 5,
              "Width": 1920,
              "Height": 1080,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.5, "Y": 0, "Z": -0.3,
          "Pitch": -15, "Roll": 0, "Yaw": 0
        },
        "bottom_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 1920,
              "Height": 1080,
              "FOV_Degrees": 90
            }
          ],
          "X": 0, "Y": 0, "Z": 0.5,
          "Pitch": -90, "Roll": 0, "Yaw": 0
        },
        "thermal": {
          "CaptureSettings": [
            {
              "ImageType": 7,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 60
            }
          ],
          "X": 0.5, "Y": 0, "Z": -0.2,
          "Pitch": -15, "Roll": 0, "Yaw": 0
        }
      },
      "X": 0, "Y": 0, "Z": 0,
      "Pitch": 0, "Roll": 0, "Yaw": 0
    }
  },
  "SubWindows": [
    {"WindowID": 0, "CameraName": "front_center", "ImageType": 0, "VehicleName": "AegisInspector"},
    {"WindowID": 1, "CameraName": "bottom_center", "ImageType": 0, "VehicleName": "AegisInspector"},
    {"WindowID": 2, "CameraName": "thermal", "ImageType": 7, "VehicleName": "AegisInspector"}
  ],
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 1920,
        "Height": 1080,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "MotionBlurAmount": 0
      }
    ]
  },
  "OriginGeopoint": {
    "Latitude": 37.7749,
    "Longitude": -122.4194,
    "Altitude": 0
  }
}
```

---

## Project Creation

### Step 1: Create New Project

```bash
# Launch Unreal Editor
cd ~/UnrealEngine
./Engine/Binaries/Linux/UnrealEditor

# In Editor:
# 1. File → New Project
# 2. Select "Blank" template
# 3. Project Settings:
#    - Blueprint (not C++)
#    - Maximum Quality
#    - Raytracing: OFF (for AMD)
#    - Starter Content: YES
# 4. Name: "AegisAVSim"
# 5. Location: ~/UnrealProjects/
```

### Step 2: Add Cosys-AirSim Plugin

```bash
# Copy Cosys-AirSim plugin to project
mkdir -p ~/UnrealProjects/AegisAVSim/Plugins
cp -r ~/Cosys-AirSim/Unreal/Plugins/AirSim ~/UnrealProjects/AegisAVSim/Plugins/

# Edit project settings
cat >> ~/UnrealProjects/AegisAVSim/Config/DefaultGame.ini << 'EOF'

[/Script/EngineSettings.GameMapsSettings]
EditorStartupMap=/Game/Maps/SolarFarm
GameDefaultMap=/Game/Maps/SolarFarm

[/Script/AirSim.AirSimSettings]
SettingsFilePath=
EOF
```

### Step 3: Project Structure

```
AegisAVSim/
├── Config/
│   ├── DefaultEngine.ini
│   ├── DefaultGame.ini
│   └── DefaultInput.ini
├── Content/
│   ├── Maps/
│   │   ├── SolarFarm.umap          # Sprint 1
│   │   ├── WindFarm.umap           # Sprint 2
│   │   ├── Substation.umap         # Sprint 2
│   │   └── PowerLines.umap         # Sprint 3
│   ├── Meshes/
│   │   ├── SolarPanels/
│   │   ├── WindTurbines/
│   │   ├── ElectricalEquipment/
│   │   └── Infrastructure/
│   ├── Materials/
│   │   ├── Base/
│   │   ├── Defects/
│   │   └── Weather/
│   ├── Blueprints/
│   │   ├── Actors/
│   │   ├── Defects/
│   │   └── Systems/
│   └── Textures/
│       ├── Defects/
│       └── Environment/
├── Plugins/
│   └── AirSim/
└── AegisAVSim.uproject
```

---

## Building Infrastructure Environments

### Solar Farm Environment (Sprint 1)

#### Level Layout

```
┌────────────────────────────────────────────────────────────────────┐
│                         SOLAR FARM LAYOUT                          │
│                                                                    │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│   │Array│ │Array│ │Array│ │Array│ │Array│ │Array│ │Array│ │Array│ │
│   │ A1  │ │ A2  │ │ A3  │ │ A4  │ │ A5  │ │ A6  │ │ A7  │ │ A8  │ │
│   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │
│   ═══════════════════════════════════════════════════════════════  │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│   │Array│ │Array│ │Array│ │Array│ │Array│ │Array│ │Array│ │Array│ │
│   │ B1  │ │ B2  │ │ B3  │ │ B4  │ │ B5  │ │ B6  │ │ B7  │ │ B8  │ │
│   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │
│   ═══════════════════════════════════════════════════════════════  │
│                                                                    │
│        ┌────────────────┐              ┌─────────────────────┐     │
│        │   Inverter     │              │    Transformer      │     │
│        │   Building     │              │    Station          │     │
│        └────────────────┘              └─────────────────────┘     │
│                                                                    │
│   [═══] Access Roads    [───] Perimeter Fence    [H] Helipad      │
└────────────────────────────────────────────────────────────────────┘
```

#### Solar Panel Blueprint (BP_SolarPanel)

Create in: `Content/Blueprints/Actors/BP_SolarPanel.uasset`

```
Blueprint Components:
├── Root (Scene)
├── PanelFrame (Static Mesh)
│   └── SM_SolarPanel_Frame
├── PanelGlass (Static Mesh)
│   └── SM_SolarPanel_Glass
├── MountingBrackets (Static Mesh)
│   └── SM_MountingBracket
├── DefectOverlay (Decal)
│   └── Dynamic Material Instance
└── InspectionTarget (Box Collision)
    └── Used for drone inspection triggers

Variables:
- PanelID: String
- ArrayID: String
- HasDefect: Boolean
- DefectType: Enum (None, Crack, HotSpot, Debris, Damage)
- DefectSeverity: Float (0-1)
- LastInspectionDate: DateTime
```

#### Solar Panel Materials

**M_SolarPanel_Base** (Master Material):
```
Material Inputs:
├── BaseColor
│   ├── Panel: Dark Blue (#1a237e)
│   └── Frame: Silver (#b0bec5)
├── Metallic: 0.8
├── Roughness: 0.2
├── Normal: Subtle texture variation
└── Emissive: None (or slight for power indicator)

Material Parameters (exposed):
├── DefectMask: Texture2D
├── DefectIntensity: Scalar (0-1)
├── DirtAmount: Scalar (0-1)
├── CrackOffset: Vector2D
└── HotSpotCenter: Vector2D
```

**M_SolarPanel_Cracked** (Material Instance):
```
Parent: M_SolarPanel_Base
Parameters:
├── DefectMask: T_Crack_01
├── DefectIntensity: 0.8
└── CrackOffset: (0.3, 0.5)
```

**M_SolarPanel_HotSpot** (Material Instance):
```
Parent: M_SolarPanel_Base
Parameters:
├── DefectMask: T_HotSpot_01
├── HotSpotCenter: (0.5, 0.5)
└── Additional emissive for thermal camera
```

### Wind Turbine Environment (Sprint 2)

#### Turbine Blueprint (BP_WindTurbine)

```
Blueprint Components:
├── Root (Scene)
├── Tower (Static Mesh)
│   └── SM_Tower_Section (instanced 3-5x)
├── Nacelle (Static Mesh)
│   └── SM_Nacelle
├── Hub (Static Mesh)
│   └── SM_Hub
├── Blades (Skeletal Mesh - for rotation)
│   ├── SM_Blade_01
│   ├── SM_Blade_02
│   └── SM_Blade_03
├── DefectMarkers (Instance Static Mesh)
│   └── For runtime defect placement
└── InspectionZones (Box Collision array)
    ├── Zone_Tower_Base
    ├── Zone_Tower_Mid
    ├── Zone_Nacelle
    └── Zone_Blade_Tips

Variables:
- TurbineID: String
- RotationSpeed: Float (RPM)
- IsOperational: Boolean
- BladeDefects: Map<Int, DefectInfo>
- NacelleDefects: Array<DefectInfo>
- TowerDefects: Array<DefectInfo>

Functions:
- StartRotation()
- StopRotation()
- SetDefect(Location, Type, Severity)
- GetInspectionPoints() -> Array<Vector>
```

#### Blade Defect Types

```cpp
UENUM(BlueprintType)
enum class EBladeDefect : uint8
{
    None,
    LeadingEdgeErosion,     // Texture roughness increase
    CrackSurface,           // Decal overlay
    CrackStructural,        // Mesh deformation + decal
    LightningDamage,        // Char marks + deformation
    IceAccumulation,        // Additional mesh + material
    BirdStrike,             // Impact crater decal
    Delamination            // Peeling texture effect
};
```

### Electrical Substation (Sprint 2)

#### Key Components

```
Substation Layout:
├── Transformers (3-5 units)
│   ├── BP_Transformer_Large
│   └── BP_Transformer_Medium
├── Circuit Breakers
│   └── BP_CircuitBreaker
├── Disconnect Switches
│   └── BP_DisconnectSwitch
├── Bus Bars
│   └── BP_BusBar (stretched mesh)
├── Insulators
│   └── BP_Insulator (ceramic material)
├── Control Building
│   └── SM_ControlBuilding
└── Fencing
    └── BP_SecurityFence (modular)
```

---

## Defect Injection System

### Runtime Defect Spawner (Blueprint)

Create: `Content/Blueprints/Systems/BP_DefectManager.uasset`

```
BP_DefectManager:

Functions:
├── SpawnDefect(TargetActor, DefectType, Severity, Location)
│   └── Attaches decal/mesh, updates material parameters
├── ClearDefect(DefectID)
│   └── Removes defect marker
├── RandomizeDefects(Seed, Probability)
│   └── For training data generation
├── GetAllDefects() -> Array<DefectInfo>
│   └── For ground truth export
└── ExportDefectManifest(FilePath)
    └── JSON export for vision training

Events:
├── OnDefectSpawned(DefectInfo)
├── OnDefectCleared(DefectID)
└── OnInspectionTriggered(ActorRef, CameraRef)
```

### Defect Decal Materials

```
Content/Materials/Defects/
├── M_Decal_Crack_Master
│   ├── MI_Crack_Fine
│   ├── MI_Crack_Medium
│   └── MI_Crack_Severe
├── M_Decal_Corrosion_Master
│   ├── MI_Corrosion_Light
│   ├── MI_Corrosion_Heavy
│   └── MI_Corrosion_Rust
├── M_Decal_OilLeak
├── M_Decal_BirdDropping
├── M_Decal_Debris
└── M_Decal_HeatDamage
```

### Defect Texture Atlas

Create high-resolution (2K) texture atlases for each defect type:

```
Content/Textures/Defects/
├── T_Cracks_Atlas_2K.png
│   └── 4x4 grid of crack variations
├── T_Corrosion_Atlas_2K.png
│   └── Progressive rust stages
├── T_Debris_Atlas_2K.png
│   └── Leaves, dirt, bird droppings
├── T_Damage_Atlas_2K.png
│   └── Impact marks, burns, chips
└── T_HotSpot_Atlas_2K.png
    └── Thermal anomaly patterns
```

---

## Camera & Lighting Setup

### Drone Camera Configuration

The camera setup is handled by AirSim, but ensure the level has proper lighting:

```
Directional Light (Sun):
├── Intensity: 10 lux
├── Source Angle: 0.5
├── Cast Shadows: True
├── Dynamic Shadow Distance: 20000
└── Cascaded Shadow Maps: 4

Sky Light:
├── Intensity: 1.0
├── Recapture: Real Time
└── Cubemap Resolution: 256

Exponential Height Fog:
├── Fog Density: 0.002
├── Start Distance: 5000
└── Fog Cutoff Distance: 100000

Post Process Volume (Unbound):
├── Auto Exposure: Off (for consistent captures)
├── Exposure Compensation: 0
├── Bloom: Intensity 0.5
├── Ambient Occlusion: Intensity 0.8
└── Motion Blur: Off
```

### Time of Day System

Create: `Content/Blueprints/Systems/BP_TimeOfDay.uasset`

```
Functions:
├── SetTimeOfDay(Hour: Float)
│   └── Rotates directional light
├── SetWeather(WeatherType: Enum)
│   └── Adjusts fog, clouds, particles
└── GetCurrentLighting() -> LightingInfo

Timeline:
├── 0-6: Night (low intensity, cool tone)
├── 6-8: Sunrise (warm, long shadows)
├── 8-16: Day (high intensity, neutral)
├── 16-18: Sunset (warm, long shadows)
└── 18-24: Night
```

---

## ArduPilot Integration

### Launch Sequence

```bash
# Terminal 1: Start Unreal/Cosys-AirSim
cd ~/UnrealProjects/AegisAVSim
~/UnrealEngine/Engine/Binaries/Linux/UnrealEditor \
    AegisAVSim.uproject \
    -game \
    -ResX=1920 -ResY=1080 \
    -windowed \
    -log

# Terminal 2: Start ArduPilot SITL
cd ~/ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py \
    -v ArduCopter \
    -f airsim-copter \
    --console \
    --map \
    -l 37.7749,-122.4194,0,0

# Terminal 3: Start AegisAV Agent
cd ~/code/AegisAV
python simulation/run_simulation.py --airsim --sitl

# Terminal 4: Open Dashboard
xdg-open http://localhost:8000/dashboard
```

### MAVLink Parameters for Inspection

```python
# Set in ArduPilot for stable hovering during inspection
WPNAV_SPEED = 500       # cm/s - slow for inspection
WPNAV_RADIUS = 200      # cm - tight waypoint radius
WPNAV_SPEED_DN = 150    # cm/s - slow descent
WPNAV_SPEED_UP = 250    # cm/s - moderate climb
WPNAV_ACCEL = 100       # cm/s/s - smooth acceleration

# Loiter tuning for stable hover
LOIT_SPEED = 500        # cm/s
LOIT_ACC_MAX = 250      # cm/s/s
LOIT_BRK_ACCEL = 250    # cm/s/s
LOIT_BRK_DELAY = 1.0    # seconds

# Camera gimbal (if using)
MNT_TYPE = 1            # Servo gimbal
MNT_ANGMIN_TIL = -9000  # -90 degrees
MNT_ANGMAX_TIL = 0      # 0 degrees (forward)
```

---

## Testing & Validation

### Validation Checklist

```
□ Cosys-AirSim connects and shows drone in Unreal viewport
□ ArduPilot SITL connects to Cosys-AirSim (check console)
□ Drone arms and takes off when commanded
□ Camera captures are saved to data/vision/airsim/
□ Defects are visible in captured images
□ GPS coordinates match expected origin
□ MAVLink telemetry flows to AegisAV agent
□ Dashboard shows real-time vehicle state
□ Weather changes affect rendered images
□ Time of day changes lighting correctly
```

### Quick Validation Script

Create: `simulation/validate_setup.py`

```python
#!/usr/bin/env python3
"""Validate Cosys-AirSim + ArduPilot integration."""

import asyncio
import sys
sys.path.insert(0, '.')

from simulation.airsim_bridge import AirSimBridge, AirSimCameraConfig
from simulation.sitl_manager import SITLManager, SITLConfig
from pathlib import Path

async def validate():
    print("=" * 60)
    print("  AegisAV Simulation Validation")
    print("=" * 60)

    # Test Cosys-AirSim connection
    print("\n[1] Testing Cosys-AirSim connection...")
    bridge = AirSimBridge(AirSimCameraConfig(
        output_dir=Path("data/validation")
    ))

    if await bridge.connect():
        print("  ✓ Cosys-AirSim connected")

        # Capture test frame
        result = await bridge.capture_frame({"test": True})
        if result.success:
            print(f"  ✓ Camera capture works: {result.image_path}")
        else:
            print(f"  ✗ Camera capture failed")

        # Get vehicle state
        state = await bridge.get_vehicle_state()
        if state:
            print(f"  ✓ Vehicle state: {state.position}")
        else:
            print("  ✗ Vehicle state failed")

        await bridge.disconnect()
    else:
        print("  ✗ Cosys-AirSim connection failed")
        print("    Make sure Unreal/Cosys-AirSim is running!")
        return False

    # Test SITL (if installed)
    print("\n[2] Testing ArduPilot SITL...")
    sitl = SITLManager(SITLConfig())

    if sitl.config.ardupilot_path.exists():
        print(f"  ✓ ArduPilot found at {sitl.config.ardupilot_path}")
    else:
        print(f"  ✗ ArduPilot not found at {sitl.config.ardupilot_path}")
        print("    Run: simulation/setup_desktop.sh")
        return False

    print("\n" + "=" * 60)
    print("  Validation Complete!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = asyncio.run(validate())
    sys.exit(0 if success else 1)
```

---

## Asset Sources

### Free Assets (Recommended Starting Point)

| Asset | Source | License |
|-------|--------|---------|
| Solar Panels | [Sketchfab - Solar Panel](https://sketchfab.com/search?q=solar+panel&type=models) | CC-BY |
| Wind Turbine | [Sketchfab - Wind Turbine](https://sketchfab.com/search?q=wind+turbine&type=models) | CC-BY |
| Power Lines | Quixel Megascans (Free with UE) | Epic License |
| Terrain | Quixel Megascans | Epic License |
| Vegetation | UE Starter Content | Epic License |

### Marketplace Assets (Quality Upgrade)

| Asset | Price | URL |
|-------|-------|-----|
| Industrial Structures | $50 | [Marketplace](https://www.unrealengine.com/marketplace) |
| Power Plant Pack | $35 | [Marketplace](https://www.unrealengine.com/marketplace) |
| Solar Panel Array | $25 | [Marketplace](https://www.unrealengine.com/marketplace) |
| Electrical Equipment | $40 | [Marketplace](https://www.unrealengine.com/marketplace) |

---

## Quick Start (Minimal Setup)

If you want to test immediately without building custom environments:

```bash
# 1. Download pre-built Cosys-AirSim environment
# Visit: https://github.com/Cosys-Lab/Cosys-AirSim/releases
# Download the latest Blocks environment for your UE version

# 2. Extract and run
cd ~/Cosys-AirSim
./Blocks.sh -ResX=1920 -ResY=1080 -windowed

# 3. In another terminal, start SITL
cd ~/ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter --console --map

# 4. Run AegisAV
cd ~/code/AegisAV
python simulation/run_simulation.py --airsim --sitl
```

This uses a pre-built Blocks environment. It won't have infrastructure assets but allows immediate flight testing while you build custom environments.

---

## Next Steps

1. **Sprint 1**: Build Solar Farm environment with basic defects
2. **Sprint 2**: Add Wind Turbines and Substation
3. **Sprint 3**: Power Line corridor
4. **Sprint 4**: Polish, weather effects, and training data generation

For questions or issues, check the logs:
- AirSim: `~/Documents/AirSim/AirSim.log`
- ArduPilot: Console output
- AegisAV: `logs/` directory
