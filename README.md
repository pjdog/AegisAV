<div align="center">

# ⚔️ AegisAV

### Agentic Supervisory Autonomy for Next-Gen Aerial Intelligence

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PydanticAI](https://img.shields.io/badge/Engine-PydanticAI-00ff9d?style=for-the-badge)](https://ai.pydantic.dev/)
[![Logfire](https://img.shields.io/badge/Observability-Logfire-FF6B35?style=for-the-badge)](https://pydantic.dev/logfire)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![SITL Compatible](https://img.shields.io/badge/Simulation-ArduPilot%20SITL-FFB800?style=for-the-badge)](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Empowering Autonomous Infrastructure & Aerial Monitoring with Explainable AI**

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Dashboard](#-aegis-onyx-dashboard) • [Development](#%EF%B8%8F-developer-setup)

</div>

---

## 🎯 What is AegisAV?

**AegisAV** is a cutting-edge autonomy framework that layers **LLM-powered agentic decision-making** over classical flight control. It demonstrates a **"Supervisor" architecture** where a high-level AI brain manages:

- 🧠 **Mission Objectives** — Dynamic goal selection and prioritization
- ⚠️ **Risk-Aware Planning** — Multi-factor safety evaluation
- 🌍 **Situational Awareness** — Real-time world model maintenance

...while delegating stabilization to battle-tested autopilots like **ArduPilot** and **PX4**.

---

## 💎 Aegis Onyx Dashboard

| Feature | Description |
|---------|-------------|
| 🖤 **Onyx Visuals** | Deep-mode interface with glassmorphism and Cyber Blue accents |
| 📡 **Spatial Awareness** | Real-time "Radar" view of vehicle and assets |
| 🔍 **Explainable AI** | Live Reasoning Feed exposing the "why" behind every decision |
| 🤖 **Agentic Toggle** | Switch between LLM planning and rule-based logic in real-time |

Access at `http://localhost:8080/dashboard` after launch.

---

## 🧠 Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    DECISION LAYER                             │
│              (Agent Server • PydanticAI • Logfire)            │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐   │
│   │  World Model  │ │ Goal Selector │ │  Risk Evaluator   │   │
│   │   (Pydantic)  │ │     (LLM)     │ │  (Multi-Critic)   │   │
│   └───────────────┘ └───────────────┘ └───────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                            │
│                      (Agent Client)                           │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐   │
│   │    Action     │ │     State     │ │ Mission Primitives│   │
│   │   Executor    │ │   Collector   │ │ (orbit,goto,land) │   │
│   └───────────────┘ └───────────────┘ └───────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                     CONTROL LAYER                             │
│               (ArduPilot SITL / PX4 / Hardware)               │
│           Physics • Stabilization • Sensor Fusion             │
└───────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

The easiest way to experience AegisAV is via Docker Compose:

```bash
# Clone the repository
git clone https://github.com/pjdog/AegisAV.git && cd AegisAV

# (Optional) Set your OpenAI API key for full LLM reasoning
export OPENAI_API_KEY=your-key-here

# Launch the complete stack
docker compose up
```

> 💡 **No API key?** AegisAV gracefully falls back to rule-based autonomy.

- 🌐 **Dashboard**: http://localhost:8080/dashboard
- 📊 **API Docs**: http://localhost:8000/docs

---

## 🛠️ Developer Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for ultra-fast Python environment management.

### Prerequisites

- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js** — [nodejs.org](https://nodejs.org/)
- **ArduPilot SITL** — [Setup Guide](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)

### Install & Build

```bash
# Sync Python environment
uv sync

# Build the Onyx Dashboard
cd frontend && npm install && npm run build && cd ..
```

### Run a Mission

```bash
# Terminal 1: Start realistic SITL simulation
./scripts/run_sim.sh --realistic

# Terminal 2: Launch the Agent Server
uv run aegis-server

# Terminal 3: Execute a demo mission
uv run aegis-demo --scenario anomaly
```

---

## 📊 Observability & Tracing

| Capability | Description |
|------------|-------------|
| 💭 **Live Reasoning** | Real-time LLM logic streamed to the dashboard |
| 📝 **Structured Logs** | Full system logs via the integrated terminal |
| 🔥 **Deep Tracing** | Production-grade observability with [Logfire](https://pydantic.dev/logfire) |

---

## 🧪 Testing

```bash
# Run the full test suite
uv run pytest

# Run with coverage
uv run pytest --cov=agent --cov=autonomy

# Run specific test modules
uv run pytest tests/test_advanced_decision.py -v
```

---

## 📁 Project Structure

```
AegisAV/
├── agent/
│   ├── server/          # Decision layer (PydanticAI agents, critics)
│   └── client/          # Execution layer (action executor, state collector)
├── autonomy/            # Vehicle interface (MAVLink, mission primitives)
├── frontend/            # Aegis Onyx dashboard (Vite + React)
├── configs/             # YAML configurations (thresholds, agent settings)
├── scripts/             # SITL launcher, simulation scenarios
└── tests/               # Comprehensive test suite
```

---

## 📜 License

**MIT License** — Free for research, competitions, and architectural demonstrations.

> ⚠️ **Disclaimer**: This software is not flight-certified. Intended for simulation and research only.

---

<div align="center">

Made with 🤖 + ☕ for the future of autonomous systems.

</div>
