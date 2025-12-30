# Edge Compute Configuration

Edge compute profiles simulate different on-drone compute tiers. The server
exposes:

- `GET /api/config/edge` - current edge config + available profiles
- `POST /api/config/edge` - update profile or overrides

Profiles are defined in `agent/edge_config.py`.

## Profiles

- `fc_only`
- `mcu_heuristic`
- `mcu_tiny_cnn`
- `sbc_cpu`
- `sbc_accel`
- `jetson_full`

## Common Tunables

- `vision_enabled`
- `capture_interval_s`
- `max_captures_per_inspection`
- `simulated_inference_delay_ms`
- `client_confidence_threshold`

## Anomaly Gating

Controls how local detections become anomalies:

- `mode`: `any`, `n_of_m`, `severity_override`
- `min_confidence`, `min_severity`
- `n`, `m` (for `n_of_m`)
- `min_severity_override`

## Uplink Shaping

- `summary_only`
- `send_images`
- `max_images`
- `uplink_delay_ms`
- `max_payload_bytes`
- `drop_probability`

## Perception Degradation

- `resolution_scale`
- `frame_drop_probability`
- `confidence_noise_std`
- `missed_detection_probability`

## Energy Model

- `capture_cost_percent`
- `inference_cost_percent`
- `uplink_cost_per_kb`
- `idle_drain_per_second`
- `thermal_throttle_after_s`
- `thermal_throttle_factor`

## Micro-Agent Behaviors

- `burst_capture_on_anomaly`
- `burst_capture_count`
- `cache_and_forward`
- `cache_max_items`
- `local_abort_battery_threshold`
- `local_abort_on_critical`
- `priority_weighted_capture`
