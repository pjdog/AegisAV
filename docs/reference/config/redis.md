# Redis Configuration

Location: `configs/aegis_config.yaml` under `redis`.

Fields:

- `enabled` (bool)
- `host` (string)
- `port` (int)
- `db` (int)
- `password` (string or null)
- `telemetry_ttl_hours` (int)
- `detection_ttl_days` (int)
- `anomaly_ttl_days` (int)
- `mission_ttl_days` (int)

Environment overrides:

- `AEGIS_REDIS_ENABLED`
- `AEGIS_REDIS_HOST`
- `AEGIS_REDIS_PORT`
- `AEGIS_REDIS_PASSWORD`
