# Mapping Completion Plan (Agents A/B/C)

Goal: Close remaining gaps so the mapping pipeline is end to end:
capture -> SLAM -> splat -> fusion -> planning -> UI -> validation.

Scope: Simulation-first, with hooks for real-world sensors.

Status key:
- [DONE] already wired in app
- [IN PROGRESS] partially wired or in flight
- [TODO] not yet implemented

---

## Agent A Plan Split (A1 + A2)

### Agent A1 (planner wiring + map usage in autonomy)

Phase 2: Map fusion quality
- [DONE] Multi-tile occupancy output in `mapping/map_fusion.py`.
- [DONE] GPS bounds emitted when georef is available.
- [DONE] Tune fusion parameters for dense vs sparse scenes (adaptive extraction + occupancy thresholding).

Phase 3: Planner integration
- [DONE] Preserve mission obstacles when applying navigation maps.
- [DONE] Apply navigation maps on mission planner updates.
- [DONE] Apply navigation maps to flight controller path planner when no mission planner is present.
- [DONE] Wire map updates into the runtime pipeline where flight controller is used.
- [DONE] Add replanning triggers on map updates and map staleness.
- [DONE] Define fallback behavior when map is stale or low quality.

### Agent A2 (SLAM backend + validation + real sensors)

Phase 1: SLAM backend (core)
- [DONE] Integrate ORB-SLAM3 or VINS-Fusion as a selectable backend in `mapping/slam_runner.py`.
- [DONE] Emit real `pose_graph.json`, `slam_status.json`, and `map_points.ply` when external backend outputs are present (telemetry fallback configurable).
- [DONE] Capture real tracking metrics (loop closures, reprojection error, drift) into status when backend metrics are available.

Phase 6: Validation and benchmarks
- [DONE] Hook `mapping/benchmark_runner.py` into `scripts/run_map_benchmarks.py`.
- [DONE] Emit baseline vs map-driven metrics (collision risk, replans, time-to-goal).
- [DONE] Add regression capture sequences and a pass/fail report.

Phase 7: Real sensor path (optional)
- [DONE] Wire `mapping/real_capture.py` into server config and runtime.
- [DONE] Add runtime calibration loader and validation checks.

---

## Agent B Plan (splat pipeline + map metadata + API wiring)

Phase 1: Splat training
- [DONE] Wire a real 3DGS trainer into `mapping/splat_trainer.py` (Nerfstudio or gsplat).
- [DONE] Store artifacts via `mapping/splat_storage.py` with versioned `scene_<run_id>/vN`.
- [DONE] Emit `scene.json`, `metadata.json`, `preview.ply` with PSNR/SSIM and gaussian count.

Phase 2: Map update + safety gates
- [DONE] Map update gate wired into `mapping/map_update.py`.
- [DONE] Map update error telemetry surfaced via `/api/navigation/map/status`.
- [DONE] Pose-graph summary and loop-closure rate exposed in `/api/slam/status`.

Phase 3: Planning proxy from splats
- [DONE] Planning proxy generation integrated into map update loop.
- [DONE] Proxy lookup via `/api/navigation/splat/proxy/{run_id}`.
- [DONE] Add config toggles for proxy regeneration cadence and max points.

Phase 4: Scenario scoping
- [DONE] Ensure per-scenario `run_id` namespaces for SLAM/splat directories.
- [DONE] Allow selecting map by scenario in `/api/navigation/map/status` and preview.

Phase 5: Dashboard wiring
- [DONE] Add map selector in dashboard (latest per scenario / per run).
- [DONE] Add occupancy preview heatmap and proxy status badge.

Phase 6: Map health endpoint
- [DONE] Extend `/api/navigation/map/health` with gate history and proxy health.

---

## Agent C Plan (capture/replay + artifacts + tests)

Phase 1: Capture + replay
- [DONE] Add dataset manifest generation during preflight capture.
- [DONE] Keep `mapping/capture_replay.py` compatible with SLAM backend inputs.
- [DONE] Add capture integrity checks (timestamps, intrinsics, depth availability).

Phase 2: Artifact storage
- [DONE] Versioned fused map storage in `mapping/map_storage.py`.
- [DONE] Add cleanup policy for old artifacts (age or size-based).

Phase 3: Tests
- [DONE] Safety gate tests in `tests/test_mapping_safety_gates.py`.
- [DONE] Map fusion output tests (bounds, metadata completeness, obstacles).
- [DONE] Splat proxy output tests (map schema + obstacle count sanity).
- [DONE] API response tests for `/api/navigation/map/status` and `/api/navigation/splat/proxy/{run_id}`.

Phase 4: Preflight flow robustness
- [DONE] Add preflight status events to overlay/dashboard.
- [DONE] Add timeouts and recovery steps for mapping failures.

---

## Dependencies and sequencing

1) Real SLAM backend must land before meaningful splat outputs and accurate map fusion metrics.
2) Safety gates should remain in place before planner consumes map data.
3) Dashboard work depends on stable map outputs and proxy generation.

---

## Suggested execution order

1) Agent A Phase 1 (real SLAM backend)
2) Agent B Phase 1 (real splat training)
3) Agent A Phase 2/3 (fusion + planner wiring)
4) Agent C Phase 1/3 (capture + tests)
5) Agent B Phase 5/6 (dashboard + map health)
6) Agent A Phase 6/7 (benchmarks + real sensors)
