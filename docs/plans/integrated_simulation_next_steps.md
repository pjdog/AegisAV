# Integrated Simulation Next Steps (Server + Feedback)

This document plans the next implementation work needed to reach a believable, end-to-end “agentic drone alert” simulation: scenarios → agent decisions → client execution → feedback → dashboard visibility, with configurable on-drone (“edge”) compute tiers.

## Current State (Already In Place)

- Edge compute profiles are defined and configurable via server API (`agent/edge_config.py`, `GET/POST /api/config/edge` in `agent/server/main.py`).
- The client polls edge config and adapts cadence/latency + feedback payload shape (`agent/client/*`).
- The dashboard UI can switch the edge profile (`frontend/src/components/Dashboard.tsx`).
- Scenario catalog + runner endpoints exist under `/api/dashboard/*` (`agent/server/dashboard.py`, `agent/server/scenarios.py`, `agent/server/scenario_runner.py`).
- Server accepts `/feedback` and already processes outcomes + optional vision inspection data (`agent/server/main.py`).

## Known Blocker (Must Fix First)

**FastAPI/Starlette `TestClient` hangs in this environment (Python 3.14 + anyio stack).** Any sync-style test client usage is unreliable here.

### Decision

- Standardize API tests on `httpx.AsyncClient` + `httpx.ASGITransport`, and make route tests `async def`.
- Avoid `fastapi.testclient.TestClient` entirely.

## Work Plan (Execution Order)

### 0) Stabilize Server/API Tests

**Goal:** Make the server test suite deterministic and non-hanging.

- Refactor tests that use `TestClient` (notably `tests/test_dashboard.py` and `tests/test_server_integration.py`) to async httpx.
- Add/standardize a reusable async client fixture (likely in `tests/conftest.py`) so tests stay small.
- Ensure any endpoints exercised by tests are `async def` and do not force sync-to-threadpool execution.

**Acceptance:** `pytest -q` completes without timeouts/hangs.

### 1) Harden Scenario Runner (Simulation Core)

**Goal:** The runner becomes the “simulation clock” that drives believable state changes and produces logs suitable for the dashboard.

- Determinism controls:
  - Add an explicit `seed` input (or documented default) so tests can replay runs.
  - Make time progression predictable (time scale + max duration + monotonic timestamps).
- Runner lifecycle:
  - Start/stop idempotence, cancellation safety, and clear “already running” behavior.
  - Clean summary outputs (counts, anomalies triggered, battery consumption, abort reason).
- Logging:
  - Ensure decisions and key world events are written to JSONL consistently (so `/api/dashboard/run/{id}` is meaningful).

**Acceptance:** `tests/test_scenario_runner.py` covers lifecycle + deterministic behavior; dashboard endpoints return consistent runner status/decisions.

### 2) Close the Feedback Loop End-to-End

**Goal:** Feedback becomes visible and queryable (not just logged), so the dashboard can show “what happened” after decisions.

- Persist feedback/outcomes (by `run_id` / `scenario_id` / `decision_id`) via the existing persistence layer.
- Add read APIs (server) for:
  - recent feedback entries
  - anomalies created / resolved
  - per-run outcome summary
- Wire dashboard UI panels to visualize:
  - recent feedback timeline
  - anomalies table (severity/confidence/asset)
  - decision → outcome linkage (“decision succeeded/failed, cost, latency”)

**Acceptance:** A short scenario run produces (a) decision log, (b) feedback events, and (c) dashboard-visible outcome summaries.

### 3) Expand “Edge Agent” Realism (Compute Tiers on Drone)

**Goal:** A reasonable approximation of what can run on bolt-on drone hardware, beyond just latency and gating.

Add optional knobs (all configurable in the UI, with presets from the profile):

- **Perception quality degradation:** lower-res capture, frame dropping, confidence noise, missed detections under tight budgets.
- **Bandwidth + latency model:** uplink delay, payload size caps, dropped messages/backpressure.
- **Energy cost model:** battery burn per frame/inference + per uplink; optional thermal throttling reducing throughput over time.
- **On-drone “micro-agent” behaviors (lightweight):**
  - burst capture when anomaly suspected
  - cache-and-forward when uplink is poor
  - local “abort / RTL” safety guard when battery/telemetry is critical

Frontend plan:

- Keep the **profile dropdown** as the fast path.
- Add an **“Advanced” overrides panel**:
  - start from selected preset
  - allow editing key knobs
  - show an at-a-glance “compute + bandwidth” budget indicator

**Acceptance:** Switching profiles and/or overrides visibly changes cadence, detection rate, feedback payload size, and overall outcome metrics in the same scenario.

### 4) End-to-End Simulation Harness

**Goal:** One command runs a full loop (server + simulated drones + dashboard) for demo and regression testing.

- Add a small runner script (e.g. `scripts/run_simulation.py`) that:
  - starts the server in-process (or assumes it’s running)
  - spins up N simulated drone clients
  - selects a scenario, runs for X seconds, then stops
- Provide a “smoke” test that runs a tiny scenario quickly and asserts:
  - runner started
  - at least one decision logged
  - at least one feedback entry recorded

**Acceptance:** A deterministic smoke run completes quickly and exercises the full pipeline.

## Test Strategy (How We’ll Verify Each Phase)

- **Unit tests:** config validation, deep-merge semantics, edge gating and payload shaping.
- **API tests (async httpx):** scenario listing/detail, runner start/stop/status/decisions, feedback read APIs.
- **Scenario-run smoke tests:** short duration + fixed seed; assert stable, minimal invariants (don’t overfit exact numbers).

## Notes on Edge Hardware “Reasonableness”

The existing profiles (`fc_only`, `mcu_*`, `sbc_*`, `jetson_full`) are a good “bolt-on drone compute” ladder. The next realism improvements should focus on *tradeoffs* (latency vs quality vs energy vs bandwidth) rather than trying to fully emulate real embedded stacks.

