# Configuration Overview

AegisAV reads configuration from:

1) `configs/aegis_config.yaml`
2) Optional profiles like `configs/aegis_config.real_sensor.yaml`
3) Environment variables (`AEGIS_*`)
4) Defaults in `agent/server/config_manager.py`

Environment variables override the config file.

## Config Sections

- `server` - host/port and logging
- `redis` - persistence settings
- `auth` - API key and rate limits
- `vision` - detector configuration
- `simulation` - AirSim and SITL integration
- `agent` - decision model and thresholds
- `dashboard` - UI refresh and theme
- `mapping` - SLAM/splat pipeline configuration

## Environment Variable Mapping

Common overrides:

- `AEGIS_HOST`, `AEGIS_PORT`, `AEGIS_LOG_LEVEL`
- `AEGIS_SIM_ENABLED`, `AEGIS_AIRSIM_ENABLED`, `AEGIS_AIRSIM_HOST`
- `AEGIS_SITL_ENABLED`, `AEGIS_ARDUPILOT_PATH`
- `AEGIS_VISION_ENABLED`, `AEGIS_VISION_MODEL`, `AEGIS_VISION_DEVICE`
- `AEGIS_USE_LLM`, `AEGIS_LLM_PROVIDER`, `AEGIS_LLM_MODEL`
- `AEGIS_REDIS_ENABLED`, `AEGIS_REDIS_HOST`, `AEGIS_REDIS_PORT`

See the per-section references for details.
