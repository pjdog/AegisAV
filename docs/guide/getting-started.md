# Getting Started

This guide walks through a basic local setup and a minimal run.

## Prerequisites

- Python 3.10+
- Node.js 18+ (for dashboard build)
- AirSim (optional, for Unreal rendering)

## Install

From the project root:

```bash
python scripts/installer/setup_gui.py
```

The installer writes configuration to `configs/aegis_config.yaml` and creates
`scripts/start_server.*`.

## Run the Server

```bash
python -m agent.server.main
```

Default dashboard:

```
http://localhost:8080/dashboard
```

Overlay:

```
http://localhost:8080/overlay
```

## Start a Scenario

From the overlay or dashboard, select a scenario and click Start. The overlay
renders only after it receives scenario or decision events.

## AirSim (Optional)

If you want Unreal rendering:

1) Start AirSim on Windows with `start_airsim.bat`.
2) Ensure `simulation.airsim_host` points to the host running AirSim.

See `AIRSIM_SCENARIO_SYNC.md` for environment sync details.
