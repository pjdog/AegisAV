<div align="center">

# AegisAV

### Autonomous Infrastructure Inspection with Explainable AI

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PydanticAI](https://img.shields.io/badge/Engine-PydanticAI-00ff9d?style=for-the-badge)](https://ai.pydantic.dev/)
[![ArduPilot](https://img.shields.io/badge/Flight-ArduPilot%20SITL-FFB800?style=for-the-badge)](https://ardupilot.org/)
[![AirSim](https://img.shields.io/badge/Rendering-AirSim%20%2B%20Unreal-0E1128?style=for-the-badge)](https://microsoft.github.io/AirSim/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**AI-powered autonomous drone system for infrastructure monitoring**

[Quick Start](#-quick-start) | [Simulation](#-high-fidelity-simulation) | [Architecture](#-architecture) | [Vision System](#-vision-pipeline) | [Dashboard](#-dashboard)

</div>

---

## What is AegisAV?

**AegisAV** is an autonomous drone system designed for **infrastructure inspection** - solar farms, wind turbines, power lines, and electrical substations. It combines:

- **LLM-Powered Decision Making** - Goals are selected and prioritized by AI agents
- **Multi-Critic Safety Validation** - Every action is validated by safety, efficiency, and goal-alignment critics
- **Computer Vision Pipeline** - Defect detection with YOLO-based object detection
- **Rock-Solid Flight Control** - Built on ArduPilot, the same autopilot used in production drones
- **High-Fidelity Simulation** - Photorealistic rendering with Unreal Engine + AirSim

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Autonomous Inspection** | AI selects which assets to inspect based on priority and conditions |
| **Defect Detection** | Computer vision identifies cracks, corrosion, hot spots, and damage |
| **Explainable AI** | Every decision includes human-readable reasoning |
| **Real-Time Dashboard** | Monitor vehicle state, detections, and AI reasoning live |
| **Production Flight Controller** | Uses ArduPilot SITL - same code that runs on real Pixhawk hardware |
| **Photorealistic Simulation** | Unreal Engine rendering with AirSim physics |

---

## Quick Start

### Option 1: Demo Mode (No External Dependencies)

```bash
# Clone and setup
git clone https://github.com/pjdog/AegisAV.git && cd AegisAV
uv sync

# Run the integrated demo
uv run python examples/demo_integrated_vision.py
```

This runs a complete inspection simulation with:
- Simulated camera generating realistic images
- Vision pipeline detecting defects
- AI agent making inspection decisions
- Dashboard available at http://localhost:8000/dashboard

### Option 2: Full Simulation (Recommended)

For the complete high-fidelity experience with photorealistic rendering:

```bash
# On your desktop machine (requires GPU)
cd simulation
chmod +x setup_desktop.sh
./setup_desktop.sh

# Then run:
python simulation/run_simulation.py --airsim --sitl
```

See [Simulation Setup](#-high-fidelity-simulation) for full details.

---

## High-Fidelity Simulation

AegisAV includes a complete simulation stack for realistic drone inspection:

```
┌─────────────────────────────────────────────────────────────────────┐
│              UNREAL ENGINE + AIRSIM                                 │
│         Photorealistic 3D environments (Solar farms, turbines)     │
│         Real-time camera feeds at 1920x1080                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ARDUPILOT SITL                                         │
│         Real flight controller code (same as Pixhawk hardware)     │
│         MAVLink protocol, GPS, attitude control                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              AEGISAV AGENT                                          │
│         AI decision making, vision pipeline, anomaly detection     │
│         FastAPI server with real-time dashboard                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1070 / RX 5700 | RTX 3080 / RX 6950XT |
| RAM | 16GB | 32GB+ |
| Storage | 100GB SSD | 200GB NVMe |
| CPU | 6-core | 8+ core |

### Simulation Documentation

| Document | Description |
|----------|-------------|
| [simulation/README.md](simulation/README.md) | Architecture overview and quick start |
| [simulation/UNREAL_SETUP_GUIDE.md](simulation/UNREAL_SETUP_GUIDE.md) | Detailed Unreal Engine setup |
| [simulation/ENVIRONMENTS.md](simulation/ENVIRONMENTS.md) | Infrastructure environment planning |

### Quick Simulation Setup

```bash
# 1. Run the setup script (installs ArduPilot, AirSim, dependencies)
./simulation/setup_desktop.sh

# 2. Validate your setup
python simulation/validate_setup.py

# 3. Start the simulation
#    Terminal 1: Start Unreal/AirSim
#    Terminal 2: Start ArduPilot SITL
#    Terminal 3: Run AegisAV
python simulation/run_simulation.py --airsim --sitl

# 4. Open dashboard
open http://localhost:8000/dashboard
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                       DECISION LAYER                               │
│                  (Agent Server - PydanticAI)                       │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│   │ World Model │  │Goal Selector│  │    Multi-Critic         │   │
│   │  (Assets,   │  │    (LLM)    │  │ Safety│Efficiency│Goal  │   │
│   │  Vehicle)   │  │             │  │ Critic│  Critic  │Align │   │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
         │                   │                      │
         ▼                   ▼                      ▼
┌───────────────────────────────────────────────────────────────────┐
│                       VISION LAYER                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│   │   Camera    │  │    YOLO     │  │    Anomaly Detection    │   │
│   │   Capture   │→ │  Detector   │→ │   & Classification      │   │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│   │   Action    │  │    State    │  │   Mission Primitives    │   │
│   │  Executor   │  │  Collector  │  │  (orbit, goto, land)    │   │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                      CONTROL LAYER                                 │
│                (ArduPilot SITL / Pixhawk Hardware)                │
│            Stabilization • Navigation • Sensor Fusion             │
└───────────────────────────────────────────────────────────────────┘
```

---

## Vision Pipeline

The vision system detects infrastructure defects in real-time:

### Supported Defect Types

| Category | Defects Detected |
|----------|------------------|
| **Solar Panels** | Cracks, hot spots, debris, soiling, cell damage |
| **Wind Turbines** | Blade erosion, lightning damage, ice accumulation |
| **Power Lines** | Damaged conductors, vegetation encroachment, insulator damage |
| **Substations** | Corrosion, oil leaks, thermal anomalies |

### Vision Architecture

```
Camera Frame (1920x1080)
        │
        ▼
┌─────────────────┐
│  Preprocessing  │  Resize, normalize, color correction
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  YOLO Detector  │  Object detection + classification
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Anomaly Filter  │  Severity scoring, false positive reduction
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ World Model     │  Update asset states, trigger re-inspection
└─────────────────┘
```

### Running Vision Demo

```bash
# Run vision-only demo
uv run python examples/demo_integrated_vision.py

# Output includes:
# - Detected defects with confidence scores
# - Anomaly classifications
# - Asset status updates
```

---

## Dashboard

The Aegis Onyx Dashboard provides real-time monitoring:

| Panel | Description |
|-------|-------------|
| **Vehicle State** | Position, altitude, battery, flight mode |
| **Radar View** | Spatial visualization of vehicle and assets |
| **Reasoning Feed** | Live AI decision explanations |
| **Detection Log** | Vision pipeline detections |
| **Mission Status** | Current goal and progress |

Access at: `http://localhost:8000/dashboard`

---

## Project Structure

```
AegisAV/
├── agent/
│   ├── server/              # Decision layer (FastAPI + PydanticAI)
│   │   ├── critics/         # Safety, efficiency, goal-alignment validators
│   │   ├── vision/          # Vision service integration
│   │   └── monitoring/      # Cost tracking, explanations
│   └── client/              # Execution layer (action executor)
├── autonomy/                # Vehicle interface (MAVLink)
├── vision/                  # Computer vision pipeline
│   ├── models/              # YOLO detector, anomaly classifier
│   └── camera/              # Camera capture (simulated + real)
├── simulation/              # High-fidelity simulation
│   ├── airsim_bridge.py     # Unreal Engine camera integration
│   ├── sitl_manager.py      # ArduPilot SITL management
│   └── run_simulation.py    # Unified simulation runner
├── frontend/                # Aegis Onyx dashboard (Vite + React)
├── examples/                # Demo scripts
├── configs/                 # YAML configurations
└── tests/                   # Comprehensive test suite
```

---

## Development

### Prerequisites

- **Python 3.12+**
- **uv** - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 18+** - For dashboard development

### Install & Build

```bash
# Sync Python environment
uv sync

# Build dashboard
cd frontend && npm install && npm run build && cd ..
```

### Run Tests

```bash
# Full test suite (152 tests)
uv run pytest

# With coverage
uv run pytest --cov=agent --cov=autonomy --cov=vision

# Specific modules
uv run pytest tests/test_advanced_decision.py -v
```

### Start Development Server

```bash
# Start the agent server
uv run uvicorn agent.server.main:app --reload --port 8000

# In another terminal, run a demo
uv run python examples/demo_integrated_vision.py
```

---

## Configuration

Key configuration files:

| File | Purpose |
|------|---------|
| `configs/agent_config.yaml` | Agent behavior, thresholds |
| `configs/vision_config.yaml` | Vision pipeline settings |
| `simulation/settings.json` | AirSim camera and vehicle config |

Environment variables:

```bash
export OPENAI_API_KEY=your-key    # For LLM-powered goal selection
export AEGIS_LOG_LEVEL=DEBUG      # Logging verbosity
export AEGIS_MOCK_LLM=true        # Use rule-based fallback
```

---

## Roadmap

- [x] Core decision engine with multi-critic validation
- [x] Vision pipeline with defect detection
- [x] ArduPilot SITL integration
- [x] AirSim camera bridge
- [x] Real-time dashboard
- [ ] Custom Unreal infrastructure environments (Sprint 1-2)
- [ ] YOLO model training on infrastructure defects
- [ ] Hardware deployment guide
- [ ] Multi-vehicle coordination

---

## License

**MIT License** - Free for research, education, and development.

> **Disclaimer**: This software is for simulation and research only. Not certified for actual flight operations.

---

<div align="center">

Built for the future of autonomous infrastructure monitoring.

</div>
