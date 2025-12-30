# AirSim Scenario Sync Notes

Scenario definitions drive the **logical simulation** (drones, assets, events, weather).
AirSim renders the **Unreal scene**. Only a subset of scenario fields can be mapped to
AirSim at runtime.

## What Can Be Applied to AirSim

These environment fields are now applied when a scenario runs:

- `precipitation` -> AirSim weather (rain/snow/fog/dust)
- `visibility_m` -> AirSim fog intensity (lower visibility = more fog)
- `is_daylight` -> AirSim time of day (day vs night)
- `wind_speed_ms` + `wind_direction_deg` -> AirSim wind vector (if supported)

This mapping is applied at scenario start and updated on each tick when values change.

## What Gets Loaded into AirSim

When a scenario starts, the server attempts to sync it into AirSim:

- The **primary drone** (first drone in the scenario) is teleported to NED `(0, 0, -altitude)` to align
  the AirSim vehicle with the scenario start state.
- The **environment mapping** (weather + time of day + wind) is applied immediately after sync.
- If AirSim is not connected yet, the server retries and applies the sync once the bridge comes online.

Manual re-sync is available via:

- `POST /api/airsim/scene/sync` (uses the currently running scenario)
- `POST /api/airsim/scene/sync?scenario_id=normal_ops_001` (explicit scenario)

## What Cannot Be Auto-Applied (Yet)

- **Scene geometry and assets** (solar panels, turbines, power lines) are part of the
  Unreal level and are not spawned dynamically by the scenario runner.
- **Anomalies** in scenarios do not automatically add meshes or decals in the Unreal scene.
- **Multi-drone scenes**: only the first scenario drone is mapped to the AirSim vehicle name.

If the scenario's assets don't match the AirSim world, you must:
1) Use an Unreal level that already contains matching assets, or
2) Extend the Unreal project to spawn assets programmatically.

## Troubleshooting

If weather/time-of-day changes do not show up:
- Confirm the AirSim bridge is connected (`/api/airsim/status`).
- Make sure AirSim weather is enabled (the bridge calls `simEnableWeather(True)`).
- Verify the scenario is actually running (`/api/scenarios/status`).
- If AirSim logs `BP_Sky_Sphere` warnings, ensure the Unreal level includes a `BP_Sky_Sphere`
  actor or use a time-of-day system compatible with AirSim.

If the drone does not move or reset when a scenario starts:
- Check `vehicles` in `/api/airsim/status` to confirm the vehicle name (the server will auto-select
  the first AirSim vehicle if `Drone1` is not present).
