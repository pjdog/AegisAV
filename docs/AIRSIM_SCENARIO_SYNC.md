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

## Multi-Drone Support

The system now supports **multi-drone scenarios** with automatic vehicle mapping:

### Architecture

1. **MultiVehicleManager** (`simulation/multi_vehicle_manager.py`)
   - Discovers available AirSim vehicles
   - Maps scenario `drone_id` to AirSim `vehicle_name`
   - Provides independent control of each vehicle

2. **DroneCoordinator** (`simulation/drone_coordinator.py`)
   - Loads scenarios and extracts drone definitions
   - Auto-assigns scenario drones to AirSim vehicles
   - Syncs positions/states between scenario engine and AirSim
   - Handles coordinate conversion (geo -> NED)

### Configuration

In `configs/aegis_config.yaml`:

```yaml
simulation:
  airsim_enabled: true
  airsim_vehicle_name: Drone1  # Primary/legacy vehicle

  # Multi-drone vehicle mapping (scenario drone_id -> AirSim vehicle name)
  airsim_vehicle_mapping:
    alpha: Drone1
    bravo: Drone2
    charlie: Drone3

  max_drones: 4
```

### AirSim settings.json

Ensure your AirSim `settings.json` defines multiple vehicles:

```json
{
  "SeeDocsAt": "https://cosys-lab.github.io/settings/",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": 0
    },
    "Drone2": {
      "VehicleType": "SimpleFlight",
      "X": 5, "Y": 0, "Z": 0
    },
    "Drone3": {
      "VehicleType": "SimpleFlight",
      "X": 10, "Y": 0, "Z": 0
    }
  }
}
```

### Usage Example

```python
from simulation.drone_coordinator import get_drone_coordinator
from agent.server.scenarios import get_scenario

# Get coordinator and connect
coordinator = get_drone_coordinator(host="127.0.0.1")
await coordinator.connect()

# Load a multi-drone scenario
scenario = get_scenario("normal_ops_001")  # Has 3 drones
await coordinator.load_scenario(scenario)

# Position drones at scenario start locations
await coordinator.setup_initial_positions()

# Control individual drones
await coordinator.command_takeoff("alpha", altitude=10)
await coordinator.command_takeoff("bravo", altitude=15)
await coordinator.command_takeoff("charlie", altitude=20)

# Move to positions
await coordinator.command_move_to("alpha", lat=37.776, lon=-122.418, altitude_agl=25)

# Fleet commands
await coordinator.land_all()
```

## What Cannot Be Auto-Applied (Yet)

- **Scene geometry and assets** (solar panels, turbines, power lines) are part of the
  Unreal level and are not spawned dynamically by the scenario runner.
- **Anomalies** in scenarios do not automatically add meshes or decals in the Unreal scene.

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

If multi-drone scenarios only control one drone:
- Check that `settings.json` defines multiple vehicles in the `Vehicles` section.
- Verify the vehicle mapping in `aegis_config.yaml` matches scenario `drone_id` values.
- Use `GET /api/airsim/status` to see discovered vehicles.
- Check logs for "Drone assignments" messages during scenario load.
