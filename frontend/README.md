# Aegis Onyx Interface 🛡️

The **Aegis Onyx Interface** is the primary command and control dashboard for the AegisAV autonomous drone system. It provides real-time telemetry, mission decision awareness, and high-fidelity visualizations of the drone's cognitive state.

![Aegis Shield](../frontend/public/aegis_logo.svg)

## Features

- **Mission Autonomy Monitor**: Watch the LLM-driven decision engine make choices in real-time.
- **SITL Integration**: Connects directly to the PX4/ArduPilot Simulation via the Agent Server.
- **Edge Budgeting**: Configure simulated edge compute constraints (power, bandwidth, latency).
- **Spatial View**: Live 2D radar view of the drone and detected assets.

## Visual Identity

The dashboard follows the **[AegisAV Style Guide](../docs/STYLE_GUIDE.md)**, utilizing the "Dark Mode IO" aesthetic:
- **Palette**: Void Black (`#09090B`) & Aegis Cyan (`#06b6d4`).
- **Typography**: Inter (Headers) & JetBrains Mono (Data).

## Development

This is a **Vite + React + TypeScript** application.

### Prerequisites
- Node.js 18+
- The Agent Server running on `http://localhost:8080` (see [Main README](../README.md))

### Quick Start
```bash
cd frontend
npm install
npm run dev
```
The interface will be available at `http://localhost:5173`. It proxies API requests to the Python server.

### Production Build
```bash
npm run build
```
Build artifacts are output to `dist/`, which the Python server can serve statically at `/dashboard`.

## Links
- [Main Project README](../README.md)
- [Unreal Engine Demo Plan](../docs/plans/unreal_demo_integration.md)
