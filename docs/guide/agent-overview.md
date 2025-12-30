# Agent Overview

The AegisAV agent server receives vehicle state, evaluates risk, selects goals,
and emits decisions. It also streams thinking and telemetry events for the
dashboard and overlay.

## Major Components

- World model: Tracks assets, drones, and environment state.
- Risk evaluator: Computes risk level and factors from the world snapshot.
- Goal selector: Chooses what the agent should do next.
- Critics: Validate or block decisions (safety, efficiency, goal alignment).
- Decision logger: Writes decisions and telemetry to logs/storage.

## Inputs

- Vehicle state updates via `POST /state`
- Scenario runner updates (simulated state and events)
- Configuration and edge compute overrides

## Outputs

- Decision responses to clients
- WebSocket events for overlay (`/ws/unreal`) and dashboard (`/ws`)
- Logs and persistence (optional Redis)
