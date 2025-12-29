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


---

## Key Features

| Feature | Description |
|---------|-------------|
| **Autonomous Inspection** | AI selects which assets to inspect based on priority and conditions |
| **Defect Detection** | Computer vision identifies cracks, corrosion, hot spots, and damage |
| **Explainable AI** | Every decision includes human-readable reasoning and risk assessment |
| **Multi-Agent Critics** | 3-layer safety validation (Safety, Efficiency, Goal) for every action |
| **Cost Awareness** | Real-time LLM token tracking and budget enforcement |
| **Edge Intelligence** | Configurable policies for bandwidth management and anomaly gating |
| **Risk Evaluation** | **[NEW]** Framework to quantify mission risk based on battery, weather, and GPS factors |
| **Behavioral Validation** | **[NEW]** Integration tests ensuring correct multi-drone decision logic in complex scenarios |
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

# Run the integrated demo (uses development config by default)
uv run python examples/demo_integrated_vision.py
```

This runs a complete inspection simulation with:
- Simulated camera generating realistic images
- Vision pipeline detecting defects
- AI agent making inspection decisions
- Dashboard available at http://localhost:8000/dashboard

### Option 2: Full Simulation (Recommended)

1. **Setup Simulation**: Follow the [Simulation Setup](#-high-fidelity-simulation) guide.
2. **Configure**: Copy `configs/aegis_config.development.yaml` to `configs/aegis_config.yaml`.
3. **Run**:

```bash
# Terminal 1: Start Unreal/AirSim (see verification guide)
# Terminal 2: Start ArduPilot SITL
# Terminal 3: Run AegisAV
uv run python simulation/run_simulation.py --airsim --sitl
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
│   │   Action    │  │    State    │  │   Edge Policy Engine    │   │
│   │  Executor   │  │  Collector  │  │  (Bandwidth/Gating)     │   │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                      CONTROL LAYER                                 │
│                (ArduPilot SITL / Pixhawk Hardware)                │
│            Stabilization • Navigation • Sensor Fusion             │
│                        (MAVLink)                                  │
└───────────────────────────────────────────────────────────────────┘
```

---

## Configuration & Edge Policy

AegisAV features a unified configuration system (`ConfigManager`) supporting:
- **YAML Configs**: `configs/aegis_config.yaml` (overrides defaults)
- **Environment Variables**: `AEGIS_VISION_ENABLED=true`
- **Runtime Updates**: Change settings via Dashboard UI or API

### Edge Computing Policies
Optimize bandwidth and processing by configuring the client's behavior:
- **Anomaly Gating**: Only transmit images when defects exceed specific confidence/severity thresholds.
- **Capture Cadence**: Dynamically adjust frame rates based on mission phase.
- **Cost Budgeting**: LLM calls are tracked and capped daily (default $1.00/day).

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

---

## Dashboard

The Aegis Onyx Dashboard provides real-time monitoring and configuration:

| Panel | Description |
|-------|-------------|
| **Vehicle State** | Position, altitude, battery, flight mode |
| **Radar View** | Spatial visualization of vehicle and assets |
| **Reasoning Feed** | Live AI decision explanations and critic feedback |
| **Settings** | **[NEW]** Runtime configuration of thresholds and policies |
| **Mission Status** | Current goal and progress |

Access at: `http://localhost:8000/dashboard`

---

## Project Structure

```
AegisAV/
├── agent/
│   ├── server/              # Decision layer
│   │   ├── content/         # ConfigManager and settings
│   │   ├── critics/         # Multi-agent validation system
│   │   ├── monitoring/      # Cost tracking, outcome logging
│   │   └── vision/          # Vision service integration
│   └── client/              # Execution layer & edge policies
├── autonomy/                # Vehicle interface (MAVLink)
├── vision/                  # Computer vision pipeline
├── simulation/              # High-fidelity simulation
├── frontend/                # Aegis Onyx dashboard (Vite + React)
├── configs/                 # Configuration templates
└── tests/                   # Comprehensive test suite (150+ tests)
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
# Full test suite
uv run pytest

# With coverage
uv run pytest --cov=agent --cov=autonomy --cov=vision

# Specific modules
uv run pytest tests/test_config_manager.py -v
```

---

## Roadmap

See [plan.md](plan.md) for the detailed project roadmap.

- [x] **Phase 1**: Foundation (Architecture, Simulation, Vision)
- [x] **Phase 2**: Multi-Agent Validation (Critics, Safety, Integration)
- [ ] **Phase 3**: Intelligence & Production (Optimization, Learning)
    - [x] System Configuration & UI
    - [x] Edge Policies & Cost Tracking
    - [x] Risk Evaluation & Behavioral Tests
    - [ ] Hybrid LLM Decision Engine
    - [ ] Advanced Explanation Agent

---

## License

**MIT License** - Free for research, education, and development.

> **Disclaimer**: This software is for simulation and research only. Not certified for actual flight operations.

---

<div align="center">

Built for the future of autonomous infrastructure monitoring.

</div>

