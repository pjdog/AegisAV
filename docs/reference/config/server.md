# Server Configuration

Location: `configs/aegis_config.yaml` under `server`.

Fields:

- `host` (string, default `0.0.0.0`) - Bind address.
- `port` (int, default `8080`) - Server port.
- `log_level` (string, default `INFO`) - Logging level.
- `cors_origins` (list of string, default `["*"]`) - CORS allow list.

Environment overrides:

- `AEGIS_HOST`
- `AEGIS_PORT`
- `AEGIS_LOG_LEVEL`
