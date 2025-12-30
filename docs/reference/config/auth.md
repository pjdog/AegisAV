# Auth Configuration

Location: `configs/aegis_config.yaml` under `auth`.

Fields:

- `enabled` (bool) - Enable API key auth.
- `api_key` (string or null) - API key for protected endpoints.
- `rate_limit_per_minute` (int)
- `public_endpoints` (list of string)

Environment overrides:

- `AEGIS_AUTH_ENABLED`
- `AEGIS_API_KEY`
