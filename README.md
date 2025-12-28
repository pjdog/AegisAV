# AegisAV: Agentic Supervisory Autonomy

**Empowering Autonomous Infrastructure & Aerial Monitoring with Explainable AI**

`AegisAV` is a state-of-the-art autonomy framework that layers **LLM-powered agentic decision-making** over classical flight control. It demonstrates a "Supervisor" architecture where a high-level brain manages mission objectives, risk-aware planning, and situational awareness while delegating stabilization to proven autopilots (ArduPilot/PX4).

![Aegis Onyx Dashboard Concept](https://img.shields.io/badge/UI-Aegis%20Onyx-00f2ff?style=for-the-badge)
![Engine-PydanticAI](https://img.shields.io/badge/Engine-PydanticAI-00ff9d?style=for-the-badge)
![SITL-Ready](https://img.shields.io/badge/Simulation-SITL%20Ready-ffb800?style=for-the-badge)

---

## 💎 The "Aegis Onyx" Experience

The mission dashboard provides a premium, high-fidelity monitoring environment:
- **Onyx Visuals**: A deep-mode interface with glassmorphism and Cyber Blue highlights.
- **Spatial Situational Awareness**: A real-time "Radar" view of the vehicle and assets.
- **Explainable AI (XAI)**: A live **Reasoning Feed** that exposes the "why" behind every autonomous decision.
- **Agentic Orchestration**: Toggle between autonomous LLM agentic planning and rule-based reactive logic in real-time.

---

## 🧠 Core Architecture

AegisAV follows a clean, three-layer separation:

1.  **Decision Layer (Agent Server)**:
    - Powered by **PydanticAI** & **Logfire**.
    - Maintains a high-fidelity **World Model**.
    - Evaluates complex mission risks and selects goals dynamically.
2.  **Execution Layer (Agent Client)**:
    - Translates agent goals into flight primitives.
    - Manages the MAVLink handshake and state collection.
3.  **Control Layer (SITL / Hardware)**:
    - ArduPilot/PX4 manages physics and stabilization.

---

## 🚀 Quick Start (Dockerized)

The easiest way to experience AegisAV is via Docker.

```bash
# Clone and enter
git clone https://github.com/pjdog/AegisAV.git && cd AegisAV

# Export keys (optional, defaults to 'mock')
export OPENAI_API_KEY=your-key-here

# Launch the whole stack
docker compose up
```

Access the **Aegis Onyx Dashboard** at: `http://localhost:8080/dashboard`

---

## 🛠️ Developer Setup (Local)

This project uses `uv` for ultra-fast Python environment management.

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv)
- Node.js & npm (for dashboard)
- ArduPilot SITL (for simulation)

### 2. Install & Build
```bash
# Sync Python environment
uv sync

# Build the Onyx Dashboard
cd frontend && npm install && npm run build && cd ..
```

### 3. Run Mission
```bash
# Start realistic SITL
./scripts/run_sim.sh --realistic

# Start Agent Server
uv run aegis-server

# Launch Mission Client
uv run aegis-demo --scenario anomaly
```

---

## 📊 Observability & Tracing

AegisAV is built for high-reliability operations:
- **Live Reasoning**: Real-time LLM logic streamed to the UI.
- **Structured Logs**: Full system logs available via the dashboard terminal.
- **Deep Tracing**: Integrated with **Logfire** for production-grade agentic observability.

---

## 📜 License & Use
AegisAV is provided under the MIT License. It is intended for research, competitions, and architectural demonstrations in autonomous systems.

**Disclaimer**: This software is not flight-certified and is intended for simulation and research only.
