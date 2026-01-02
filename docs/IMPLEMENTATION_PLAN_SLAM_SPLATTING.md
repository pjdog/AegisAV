# Implementation Plan: SLAM + Gaussian Splatting Mapping for Agent Planning

Objective: Build an active mapping pipeline that fuses SLAM (geometry + pose) with Gaussian splatting
for dense scene reconstruction, and outputs a planning-grade map that agents can consume.

Scope: AirSim-first (simulation), with clear hooks for real-world sensors later.

---

## Status Snapshot (Current Repo)

- Simulation pipeline is wired end-to-end (capture/replay, SLAM runner, splat storage, fusion, map update loop, dashboard endpoints).
- External SLAM backends are supported via `slam_backend` + env-configured commands; real outputs depend on those backends.
- Benchmarks and regressions are runnable via `scripts/run_map_benchmarks.py` and `scripts/run_map_regressions.py`.
- Real sensor capture is available when enabled via `/api/navigation/real_capture`.

---

## Phase 0: Requirements, Interfaces, and Success Metrics

Worker A
- Write requirements doc with map outputs (occupancy grid, obstacle list, costmap, metadata).
- Decide map cadence and staleness budgets (update Hz, max map_age_s for planning).
- Define acceptance metrics (pose drift, obstacle recall/precision, collision rate reduction).
- Specify coordinate frames and transforms (NED primary, ENU/GPS conversions).
- Add config defaults for map cadence and map_age thresholds in agent configs.

Worker B
- Draft API contracts for map outputs and updates:
  - `GET /api/navigation/map/latest`
  - `GET /api/navigation/obstacles`
  - `GET /api/navigation/map/metadata`
- Define JSON schema for obstacle entries and map tiles/voxels.
- Identify existing server state integration points (`server_state.navigation_map`).

Deliverables
- Requirements doc and API schema draft.
- Agreed success metrics and performance targets.

---

## Phase 1: Data Capture and Synchronization Pipeline

Worker A
- Add capture mode in AirSim bridge to record RGB, depth, IMU, and pose.
- Emit time-synchronized bundles with `frame_id`, timestamps, intrinsics, camera pose, telemetry:
  - `frames/{timestamp_ns}.png`
  - `frames/{timestamp_ns}.json`
  - `frames/{timestamp_ns}_depth.npy` (optional)
- Provide a capture CLI (frames, interval, output dir, depth/IMU toggles).
- Validate camera intrinsics/extrinsics via `simGetCameraInfo` and log missing fields.
- Verify capture output consistency on at least one scenario sequence.

Worker B
- Implement keyframe selection rules (velocity threshold, rotation delta, time interval).
- Implement a replay tool to feed recorded captures to SLAM/splat pipelines.
- Add dataset manifest generation (sequence index, sensor metadata).

Deliverables
- `data/maps/sequence_YYYYMMDD/` with synchronized capture bundles.
- Replay tool that can stream recorded data at real-time or faster.

---

## Phase 2: SLAM Integration (Pose + Sparse Geometry)

Worker A
- Select SLAM stack (ORB-SLAM3 for RGB-D, VINS-Fusion for RGB+IMU).
- Implement `slam_runner.py` to ingest capture bundles and select keyframes.
- Define pose-graph schema: per-frame pose, intrinsics, image/depth paths, keyframes list.
- Export `pose_graph.json`, `slam_status.json`, and sparse `map_points.ply`.
- Add CLI knobs for backend selection and keyframe thresholds.
- Verify runner on a captured sequence and document output locations.

Worker B
- Integrate SLAM output with AegisAV runtime:
  - Store latest pose graph summary in `server_state`.
  - Add endpoint `GET /api/slam/status`.
- Add drift detection metrics (loop closure rate, reprojection error).

Deliverables
- `slam_runner.py` with CLI args for input dataset and output folder.
- SLAM output artifact format defined and versioned.

---

## Phase 3: Gaussian Splatting Reconstruction

Worker A
- Select Gaussian splatting implementation (offline first, Nerfstudio 3DGS).
- Implement `splat_trainer.py` to ingest pose graph + keyframes and emit artifacts.
- Define splat scene descriptor (`scene.json`) with run metadata and frame list.
- Generate `preview.ply` point cloud for quick visualization in planning/debug.
- Add CLI hooks for backend name and preview density controls.

Worker B
- Implement splat artifact storage with versioning:
  - `splats/scene_{run_id}/`
  - `splats/scene_{run_id}/preview.ply`
- Add CLI to convert splats to a coarse planning proxy.

Deliverables
- Splat training pipeline and stored artifacts.
- Preview mesh/point cloud for planning use.

---

## Phase 4: Map Fusion and Planning-Grade Representation

Worker A
- Implement map fusion pipeline that ingests pose graph + preview point cloud.
- Convert fused geometry into occupancy grid (2D) and optional voxel map (3D).
- Track map bounds, resolution, and quality score in map metadata.
- Add a map update loop (configurable frequency and max map_age_s).
- Output `navigation_map` structure for server_state consumption.

Worker B
- Implement obstacle extraction from fused map:
  - Cluster point cloud into obstacles.
  - Assign height and radius from geometry density.
- Provide map metadata (resolution, bounds, timestamp).

Deliverables
- `navigation_map` format populated from fused map.
- Obstacle list pushed into `server_state.navigation_map`.

---

## Phase 5: Agent Integration and Decision Hooks

Worker A
- Wire planner to consume `server_state.navigation_map` outputs.
- Add obstacle-aware path planning and costmap-aware route selection.
- Add map validity checks (age, quality score, bounds coverage).
- Implement fallback when map is stale (reuse last map or minimal avoidance).

Worker B
- Add new decision metadata entries for map usage:
  - `map_version`, `obstacle_count`, `map_age_s`.
- Emit events/logs for map update and map-based decisions.

Deliverables
- Agents can query and use new map outputs in planning.
- Decision logs indicate map usage.

---

## Phase 6: Dashboard + Monitoring

Worker A
- Add dashboard UI section for map status (timestamp, obstacle count, quality score).
- Add map preview panel for occupancy grid/preview point cloud.
- Include latest detection feed near map status to confirm vision activity.

Worker B
- Add endpoints for map summary:
  - `GET /api/navigation/map/status`
  - `GET /api/navigation/map/preview`
- Add logs/telemetry for map update errors.

Deliverables
- Dashboard shows live map status and detection feed.
- Operators can confirm map pipeline activity.

---

## Phase 7: Validation and Safety Gates

Worker A
- Create simulation benchmark runner (baseline vs map-driven) with fixed seeds.
- Collect metrics: collision rate, path length, time-to-goal, replans per minute.
- Emit CSV/JSON reports and store in `metrics/` with run metadata.
- Add regression capture sequences for SLAM/splat stability checks.

Worker B
- Implement safety gating:
  - Reject map updates if confidence < threshold.
  - Freeze planner if map inconsistent or stale.
- Add automated tests for map generation pipeline outputs.

Deliverables
- Report on collision reduction.
- Safety gating in place for live runs.

---

## Phase 8: Real-World Sensorization (Optional Next Phase)

Worker A
- Extend capture pipeline to real camera + IMU.
- Add calibration procedures and validation checks.

Worker B
- Extend SLAM and splat pipelines to real-world sensor data.
- Update planning integration for real-world coordinate systems.

Deliverables
- Field-ready capture and mapping pipeline.
