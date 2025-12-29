# Edge Compute Profiles (Simulation Plan)

This plan defines a configurable “edge agent” simulation that approximates what can realistically run on different drone-mounted compute tiers. The goal is to let the dashboard switch profiles and have the client’s behavior change in a believable way (cadence, latency, anomaly gating, and uplink payload size).

## What The “Edge Agent” Controls

The profile drives four knobs:

1) **Vision enablement + capture cadence**
- `vision_enabled`
- `capture_interval_s`
- `max_captures_per_inspection`

2) **Compute budget / inference latency**
- `simulated_inference_delay_ms` (adds per-capture delay in the client capture loop)

3) **Local anomaly gating**
- `anomaly_gate.mode`: `any` | `n_of_m` | `severity_override`
- Thresholds: `min_confidence`, `min_severity`, plus optional `n/m` and `min_severity_override`

4) **Uplink shaping (feedback payload size)**
- `uplink.summary_only` (summary-only vs include `best_detection`)
- `uplink.send_images` + `uplink.max_images` (include/omit image evidence pointers)

## Profiles (Reasonable Drone-Mount Approximation)

These map to realistic “bolt-on” tiers for medium consumer/prosumer drones:

- `fc_only` (Flight Controller only)
  - Typical hardware: Pixhawk/ArduPilot FC class MCU, no vision.
  - Intended behavior: no captures, no anomaly detection, minimal uplink.

- `mcu_heuristic` (MCU + heuristics)
  - Typical hardware: STM32/ESP32 class MCU, simple heuristics (motion/blur/edge/brightness checks).
  - Intended behavior: very sparse capture, very strict thresholds, summary-only, no images.

- `mcu_tiny_cnn` (MCU + tiny CNN)
  - Typical hardware: MCU + TFLite Micro / CMSIS-NN (tiny CNN), very limited throughput.
  - Intended behavior: small number of captures, noticeable latency, summary payload, images off.

- `sbc_cpu` (SBC CPU only)
  - Typical hardware: Raspberry Pi / similar ARM SBC, CPU inference.
  - Intended behavior: moderate latency, N-of-M gating to reduce false positives, limited image evidence.

- `sbc_accel` (SBC + accelerator)
  - Typical hardware: SBC + Coral EdgeTPU / NPU / small accelerator.
  - Intended behavior: low latency, allow “severity override” fast-trigger, more evidence.

- `jetson_full` (Jetson-class)
  - Typical hardware: Jetson Nano/Orin class GPU device.
  - Intended behavior: fastest cadence, more captures, more evidence, least restrictive gating.

The opinionated defaults live in `agent/edge_config.py`.

## Implementation Plan (Server → Client → UI)

1) **Shared config model + defaults**
- Add an enum for profile names and a Pydantic config model for all knobs.
- Provide default configs per profile.
- Provide a deep-merge update helper so the UI can patch individual fields without re-sending everything.

2) **Server config endpoints**
- `GET /api/config/edge` returns the current config and a list of profiles.
- `POST /api/config/edge` accepts partial updates (including profile switches).

3) **Client policy application**
- Poll the server for edge config (interval-based).
- Apply cadence + latency to the vision capture loop.
- Apply client-side anomaly gating to determine `anomaly_detected`.
- Shape `inspection_data` according to uplink settings.

4) **Frontend controls**
- Dashboard loads `GET /api/config/edge` and renders a profile dropdown.
- On selection, `POST /api/config/edge` with `{ "profile": "..." }`.

## Test Plan

**Unit tests (fast, deterministic)**
- Config validation:
  - `n_of_m` requires `n,m` and `n<=m`
  - `severity_override` requires `min_severity_override`
  - uplink requires `max_images=0` when `send_images=false`
- Update semantics:
  - profile switches reset to defaults
  - nested updates deep-merge (don’t clobber sibling fields)
- Client policy:
  - applying edge config changes vision client cadence and threshold
  - anomaly gating matches mode expectations
  - feedback payload includes/omits `best_detection` and `best_image_path` per profile

Concrete coverage is in `tests/test_edge_compute_profiles.py`.

**Integration tests (optional / slower)**
- Start server + client; switch profiles from the dashboard; assert:
  - client adapts cadence/latency within one poll interval
  - server receives feedback with correct `inspection_data` shape
  - anomaly events match the selected gate thresholds
