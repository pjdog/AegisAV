"""LLM configuration helpers.

Centralizes provider/model resolution and applies API key/base URL settings
from Aegis configuration into environment variables expected by providers.
"""

from __future__ import annotations

import os

from agent.server.config_manager import AegisConfig, get_config_manager

_DEFAULT_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "custom": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
}

_DEFAULT_BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
    "custom": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "groq": "GROQ_BASE_URL",
    "mistral": "MISTRAL_BASE_URL",
    "cohere": "COHERE_BASE_URL",
    "google": "GOOGLE_BASE_URL",
    "azure": "AZURE_OPENAI_ENDPOINT",
}

_PROVIDER_ALIAS = {
    "openrouter": "openai",
    "custom": "openai",
}

_OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _normalize_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    provider = provider.strip()
    return provider.lower() or None


def _provider_from_model(model: str, fallback: str | None) -> str | None:
    if ":" in model:
        return _normalize_provider(model.split(":", maxsplit=1)[0])
    return _normalize_provider(fallback)


def resolve_llm_provider(model: str, provider: str | None) -> str:
    """Resolve the provider name from a model string and fallback."""
    return _provider_from_model(model, provider) or "openai"


def resolve_llm_model(model: str, provider: str | None) -> str:
    """Resolve a provider-prefixed LLM model string."""
    model = model.strip()
    if ":" in model:
        prefix, rest = model.split(":", maxsplit=1)
        normalized = _normalize_provider(prefix)
        alias = _PROVIDER_ALIAS.get(normalized or "", normalized)
        return f"{alias}:{rest}"
    resolved_provider = _normalize_provider(provider) or "openai"
    resolved_provider = _PROVIDER_ALIAS.get(resolved_provider, resolved_provider)
    return f"{resolved_provider}:{model}"


def get_llm_env_keys(
    provider: str | None,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
) -> set[str]:
    """Return env var names that affect LLM configuration for a provider."""
    normalized = _normalize_provider(provider) or "openai"
    alias = _PROVIDER_ALIAS.get(normalized, normalized)
    keys: set[str] = set()
    for env_key in (api_key_env, base_url_env):
        if env_key:
            keys.add(env_key)
    for env_key in (_DEFAULT_API_KEY_ENV.get(normalized), _DEFAULT_API_KEY_ENV.get(alias)):
        if env_key:
            keys.add(env_key)
    for env_key in (_DEFAULT_BASE_URL_ENV.get(normalized), _DEFAULT_BASE_URL_ENV.get(alias)):
        if env_key:
            keys.add(env_key)
    if normalized == "openrouter":
        keys.update(
            {
                _DEFAULT_API_KEY_ENV["openai"],
                _DEFAULT_BASE_URL_ENV["openai"],
                "OPENROUTER_API_KEY",
                "OPENROUTER_BASE_URL",
            }
        )
    return keys


def apply_llm_settings(config: AegisConfig | None = None) -> dict[str, str | None]:
    """Apply LLM settings to environment variables used by providers."""
    config = config or get_config_manager().config
    provider = _provider_from_model(config.agent.llm_model, config.agent.llm_provider)
    alias_provider = _PROVIDER_ALIAS.get(provider or "", provider)

    api_key_env = (
        config.agent.llm_api_key_env
        or _DEFAULT_API_KEY_ENV.get(provider or "")
        or _DEFAULT_API_KEY_ENV.get(alias_provider or "")
    )
    base_url_env = (
        config.agent.llm_base_url_env
        or _DEFAULT_BASE_URL_ENV.get(provider or "")
        or _DEFAULT_BASE_URL_ENV.get(alias_provider or "")
    )

    if config.agent.llm_api_key and api_key_env and api_key_env not in os.environ:
        os.environ[api_key_env] = config.agent.llm_api_key

    if config.agent.llm_base_url and base_url_env and base_url_env not in os.environ:
        os.environ[base_url_env] = config.agent.llm_base_url

    if provider == "openrouter":
        openai_api_key_env = _DEFAULT_API_KEY_ENV["openai"]
        openai_base_url_env = _DEFAULT_BASE_URL_ENV["openai"]
        openrouter_key = (
            config.agent.llm_api_key
            or (api_key_env and os.environ.get(api_key_env))
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if openrouter_key and openai_api_key_env not in os.environ:
            os.environ[openai_api_key_env] = openrouter_key
        base_url = (
            config.agent.llm_base_url
            or (base_url_env and os.environ.get(base_url_env))
            or _OPENROUTER_DEFAULT_BASE_URL
        )
        if base_url and openai_base_url_env not in os.environ:
            os.environ[openai_base_url_env] = base_url

    return {
        "provider": provider,
        "api_key_env": api_key_env,
        "base_url_env": base_url_env,
    }


def llm_credentials_present(config: AegisConfig | None = None) -> bool:
    """Check if any LLM credentials are available."""
    config = config or get_config_manager().config
    provider = _provider_from_model(config.agent.llm_model, config.agent.llm_provider)
    api_key_env = config.agent.llm_api_key_env or _DEFAULT_API_KEY_ENV.get(provider or "")

    if config.agent.llm_api_key:
        return True
    if provider == "openrouter":
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
            return True
    if api_key_env and os.environ.get(api_key_env):
        return True
    return False


def get_default_llm_model() -> str:
    """Return the resolved default model from configuration."""
    config = get_config_manager().config
    apply_llm_settings(config)
    return resolve_llm_model(config.agent.llm_model, config.agent.llm_provider)
