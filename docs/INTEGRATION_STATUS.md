# Integration Status (Audit)

This document summarizes what is wired end-to-end and what still needs connection
work across the agent server, clients, lightweight simulation, and dashboard.

## Connected

- Async decision pipeline is implemented on the agent server and client:
  `/state/async` + `/decisions/next` with `async_mode` enabled in
  `configs/agent_config.yaml`.
- Edge compute profiles are exposed via `GET /api/config/edge` and the main
  dashboard can switch profiles through `frontend/src/components/Dashboard.tsx`.
- Telemetry and decisions are logged to JSONL under `logs/` and persisted to
  Redis when `redis.enabled=true` in `configs/aegis_config.yaml`.
- Lightweight sim exposes aggregated log endpoints at
  `simulation/lightweight/server.py` (`/api/logs` and `/api/logs/sources`).

## Not Yet Connected

- API key auth is not wired to the main dashboard:
  `frontend/src/components/Dashboard.tsx` and
  `frontend/src/components/SettingsPanel.tsx` do not send `X-API-Key`, so
  enabling auth breaks UI requests.
- Lightweight visualizer has an API key input but does not use it for requests
  (`simulation/lightweight/visualizer/index.html`).
- Lightweight sim "live mode" only talks to the Unreal WebSocket
  (`/ws/unreal`), which is a visualization stream. Telemetry sent there does
  not feed the decision engine, and the agent server does not emit `decision`
  messages on that channel. This means no real decision loop is connected.
  Files: `simulation/lightweight/server.py`, `agent/server/unreal_stream.py`,
  `agent/server/main.py`.
- Log aggregation in the lightweight sim is not exposed in any UI. The main
  dashboard reads `/api/logs` from the agent server only (no source selection
  or log file access).
- "System logs" in the lightweight sim are limited to the simulator process
  log file (`logs/lightweight_system.log`), not host OS logs.
- Pathfinding tools (Dijkstra, Markov, neural ranker) exist only as LLM tools
  in `agent/server/advanced_decision.py`. There is no API or UI for users to
  select or invoke these algorithms directly.
- Micro-agent overrides in the dashboard are not consumed by the client
  execution layer (`agent/client/edge_policy.py` does not read
  `micro_agent` settings).
- Redis persistence is not surfaced in the dashboard; no UI views use the
  storage endpoints (`/api/storage/*`), and the lightweight sim does not use
  Redis for its own state/logging.

## Known Build Issue

- `frontend/src/components/Dashboard.tsx` contains an `import` statement inside
  the component body, which will fail TypeScript builds. This should be moved
  to the top-level import block.

