# Agent Configuration

Location: `configs/aegis_config.yaml` under `agent`.

Fields:

- `use_llm` (bool) - Enable LLM-based goal selection.
- `llm_model` (string) - Model name.
- `llm_provider` (string) - Provider key.
- `llm_api_key_env` (string) - Env var name for provider API key.
- `llm_api_key` (string) - Provider API key (prefer env vars).
- `llm_base_url_env` (string) - Env var name for base URL.
- `llm_base_url` (string) - Base URL override.
- `battery_warning_percent` (float)
- `battery_critical_percent` (float)
- `wind_warning_ms` (float)
- `wind_abort_ms` (float)
- `decision_interval_seconds` (float)
- `max_decisions_per_mission` (int)

Environment overrides:

- `AEGIS_USE_LLM`
- `AEGIS_LLM_PROVIDER`
- `AEGIS_LLM_MODEL`
- `AEGIS_LLM_API_KEY_ENV`
- `AEGIS_LLM_API_KEY`
- `AEGIS_LLM_BASE_URL_ENV`
- `AEGIS_LLM_BASE_URL`
