<div align="center">

# AegisAV

### Autonomous infrastructure inspection with explainable AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PydanticAI](https://img.shields.io/badge/Engine-PydanticAI-00ff9d?style=for-the-badge)](https://ai.pydantic.dev/)
[![ArduPilot](https://img.shields.io/badge/Flight-ArduPilot%20SITL-FFB800?style=for-the-badge)](https://ardupilot.org/)
[![Cosys-AirSim](https://img.shields.io/badge/Rendering-Cosys--AirSim%20%2B%20UE5.5-0E1128?style=for-the-badge)](https://cosys-lab.github.io/Cosys-AirSim/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**AI-powered autonomous drone system for infrastructure monitoring**

[Quick Start](#-quick-start) | [Simulation](#-simulation) | [Architecture](#-architecture) | [Mapping](#-mapping--slam) | [Dashboard](#-dashboard)

</div>

---

## What is AegisAV?

**AegisAV** is an autonomous drone inspection stack for assets like solar farms, wind turbines, substations, and power lines. It integrates AI decision-making, risk-aware validation, computer vision, and high-fidelity simulation.

## Core Capabilities

| Area | Highlights |
|------|------------|
| **Autonomous Decisioning** | Goal selection, prioritization, and mission management (LLM-assisted when enabled) |
| **Safety & Oversight** | Multi-critic validation (safety, efficiency, goal alignment) with explainable outcomes |
| **Vision Pipeline** | Defect detection, inspection observations, and anomaly creation |
| **Mapping** | SLAM preflight, navigation map fusion, and splat training workflows |
| **Simulation** | Multi-drone scenarios with Unreal Engine and Cosys-AirSim |
| **Observability** | Live dashboard, reasoning feed, telemetry, outcomes, and logs |

---

## Quick Start

### Option 1: Vision Demo (no external simulation)

```bash
# From repo root
uv sync
uv run python examples/demo_integrated_vision.py
```

Outputs:
- `data/vision/demo_visual/`
- `data/vision/demo_visual/reports/demo_report.html`

### Option 2: Lightweight Simulation

```bash
uv run python -m simulation.lightweight.server
```

- Visualizer: `http://localhost:8081/viz/`
- API docs: `http://localhost:8081/docs`

Optional: run the agent server alongside it:

```bash
uv run python -m agent.server.main
```

### Option 3: Full Simulation

1. Configure: copy `configs/aegis_config.development.yaml` to `configs/aegis_config.yaml`.
2. Start Unreal/Cosys-AirSim and ArduPilot SITL.
3. Run the simulation:

```bash
uv run python simulation/run_simulation.py --airsim --sitl
```

On Windows, use `scripts/start_server.bat` and `start_airsim.bat` after running `INSTALL.bat`.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                       DECISION LAYER                               │
│                  (Agent Server - PydanticAI)                       │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│   │ World Model │  │Goal Selector│  │    Multi-Critic         │   │
│   │  (Assets,   │  │ (Rules/LLM) │  │ Safety│Efficiency│Goal  │   │
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
│                        MAPPING LAYER                               │
│   SLAM Preflight → Map Fusion → Navigation Map → Splat Training     │
└───────────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                               │
│   Action Executor • Telemetry • Edge Policies • Autopilot (MAVLink) │
└───────────────────────────────────────────────────────────────────┘
```

---

## Mapping & SLAM

AegisAV supports mapping-first missions:

- **Preflight SLAM**: captures a short mapping pass before mission start.
- **Navigation map fusion**: converts point clouds to obstacle maps.
- **Splat training**: optional scene reconstruction for overlay and change detection.

Relevant configuration lives in `configs/aegis_config.yaml` under `mapping`.

---

## Dashboard

The dashboard provides:
- Live telemetry and drone state
- Reasoning feed and critic verdicts
- Vision detections and recent captures
- Mission progress, risks, and configuration updates

Default entry point: `http://localhost:8080/dashboard`.

---

## Configuration

- Primary config: `configs/aegis_config.yaml` (copy from `configs/aegis_config.development.yaml`).
- Environment overrides: `AEGIS_*` variables (see `agent/server/config_manager.py`).
- Vision config: `configs/vision_config.yaml`.

---

## Development

### Prerequisites

- Python 3.10+
- uv (recommended)
- Node.js 18+ (dashboard)

### Install & Build

```bash
uv sync
cd frontend && npm install && npm run build && cd ..
```

### Format

```bash
uv run ruff format .
```

### Tests

```bash
uv run pytest
```

---

## Project Structure

```
AegisAV/
├── agent/                  # Decision layer, APIs, vision service
├── autonomy/               # MAVLink and flight control interfaces
├── vision/                 # Vision pipeline and models
├── mapping/                # SLAM, point clouds, splat training
├── simulation/             # Scenarios and AirSim bridges
├── frontend/               # Dashboard (Vite + React)
├── configs/                # Configuration templates
└── tests/                  # Test suite
```

---

## License

**MIT License** - Free for research and development.

> Disclaimer: This software is for simulation and research only. Not certified for actual flight operations.

<div align="center">

Built for the future of autonomous infrastructure monitoring.

</div>
