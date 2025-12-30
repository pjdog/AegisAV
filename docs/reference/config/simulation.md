# Simulation Configuration

Location: `configs/aegis_config.yaml` under `simulation`.

Fields:

- `enabled` (bool) - Master simulation toggle.
- `airsim_enabled` (bool) - Enable AirSim bridge.
- `airsim_host` (string) - AirSim host/IP. Use Windows host IP when server runs in WSL.
- `airsim_vehicle_name` (string) - AirSim vehicle name, default `Drone1`.
- `sitl_enabled` (bool) - Enable ArduPilot SITL integration.
- `ardupilot_path` (string) - Path to ArduPilot repo.
- `sitl_speedup` (float) - Speed multiplier.
- `home_latitude` / `home_longitude` / `home_altitude` (float) - Home position.

Environment overrides:

- `AEGIS_SIM_ENABLED`
- `AEGIS_AIRSIM_ENABLED`
- `AEGIS_AIRSIM_HOST`
- `AEGIS_SITL_ENABLED`
- `AEGIS_ARDUPILOT_PATH`
