"""Configuration and LLM API routes."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from pydantic import ValidationError

from agent.edge_config import apply_edge_compute_update, available_edge_profiles
from agent.server.config_manager import get_config_manager
from agent.server.deps import auth_handler
from agent.server.llm_config import (
    apply_llm_settings,
    get_llm_env_keys,
    llm_credentials_present,
    resolve_llm_model,
    resolve_llm_provider,
)
from agent.server.state import server_state


def _normalize_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def register_config_routes(app: FastAPI) -> None:
    """Register configuration and LLM-related routes."""

    @app.get("/api/config/agent")
    async def get_agent_config() -> dict:
        """Return the current agent orchestration configuration."""
        return {
            "use_advanced_engine": server_state.goal_selector.use_advanced_engine,
            "is_initialized": server_state.goal_selector.advanced_engine is not None,
        }

    @app.post("/api/config/agent")
    async def update_agent_config(config: dict) -> dict:
        """Update agent orchestration configuration."""
        enabled = config.get("use_advanced_engine", True)
        await server_state.goal_selector.orchestrate(enabled)
        return {"status": "success", "use_advanced_engine": enabled}

    @app.get("/api/config/edge")
    async def get_edge_config() -> dict:
        """Return the current edge compute simulation configuration."""
        return {
            "edge_config": server_state.edge_config.model_dump(mode="json"),
            "profiles": available_edge_profiles(),
        }

    @app.post("/api/config/edge")
    async def update_edge_config(config: dict) -> dict:
        """Update edge compute simulation configuration (supports partial updates)."""
        try:
            server_state.edge_config = apply_edge_compute_update(server_state.edge_config, config)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "status": "success",
            "edge_config": server_state.edge_config.model_dump(mode="json"),
        }

    @app.get("/api/config")
    async def get_all_config(_auth: dict = Depends(auth_handler)) -> dict:
        """Get complete configuration."""
        config_manager = get_config_manager()
        return {
            "config": config_manager.get_all(),
            "config_file": str(config_manager.config_file),
            "loaded": config_manager._loaded,
        }

    @app.post("/api/config/save")
    async def save_config(_auth: dict = Depends(auth_handler)) -> dict:
        """Save current configuration to file."""
        config_manager = get_config_manager()
        errors = config_manager.validate()
        if errors:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration validation failed: {', '.join(errors)}",
            )

        success = config_manager.save()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save configuration")

        return {"status": "saved", "path": str(config_manager.config_file)}

    @app.post("/api/config/reload")
    async def reload_config(_auth: dict = Depends(auth_handler)) -> dict:
        """Reload configuration from file."""
        config_manager = get_config_manager()
        config_manager.load()
        return {"status": "reloaded", "config": config_manager.get_all()}

    @app.post("/api/config/reset/{section}")
    async def reset_config_section(section: str, _auth: dict = Depends(auth_handler)) -> dict:
        """Reset a configuration section to defaults."""
        config_manager = get_config_manager()

        success = config_manager.reset_section(section)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown configuration section: {section}",
            )

        return {
            "status": "reset",
            "section": section,
            "config": config_manager.get_section(section),
        }

    @app.post("/api/config/reset")
    async def reset_all_config(_auth: dict = Depends(auth_handler)) -> dict:
        """Reset all configuration to defaults."""
        config_manager = get_config_manager()
        config_manager.reset_all()
        return {"status": "reset", "config": config_manager.get_all()}

    @app.get("/api/config/validate")
    async def validate_config(_auth: dict = Depends(auth_handler)) -> dict:
        """Validate current configuration."""
        config_manager = get_config_manager()
        errors = config_manager.validate()
        return {"valid": len(errors) == 0, "errors": errors}

    @app.post("/api/config/generate-api-key")
    async def generate_api_key(_auth: dict = Depends(auth_handler)) -> dict:
        """Generate a new API key."""
        config_manager = get_config_manager()
        new_key, saved = config_manager.generate_api_key()
        return {
            "status": "generated",
            "api_key": new_key,
            "saved": saved,
            "note": "Saved to config file." if saved else "Use /api/config/save to persist.",
        }

    @app.get("/api/config/env-template")
    async def get_env_template() -> dict:
        """Get environment variable template."""
        config_manager = get_config_manager()
        return {"template": config_manager.export_env_template()}

    @app.get("/api/config/{section}")
    async def get_config_section(section: str, _auth: dict = Depends(auth_handler)) -> dict:
        """Get a specific configuration section."""
        config_manager = get_config_manager()
        section_data = config_manager.get_section(section)

        if section_data is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown configuration section: {section}. "
                    "Valid sections: server, redis, auth, vision, simulation, agent, dashboard"
                ),
            )

        return {"section": section, "config": section_data}

    @app.put("/api/config/{section}")
    async def update_config_section(
        section: str, values: dict, _auth: dict = Depends(auth_handler)
    ) -> dict:
        """Update a configuration section."""
        config_manager = get_config_manager()

        if not hasattr(config_manager.config, section):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown configuration section: {section}",
            )

        success = config_manager.update_section(section, values)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update configuration")

        errors = config_manager.validate()
        if errors:
            config_manager.load()
            raise HTTPException(
                status_code=400,
                detail=f"Configuration validation failed: {', '.join(errors)}",
            )

        config_manager.save()

        return {
            "status": "updated",
            "section": section,
            "config": config_manager.get_section(section),
        }

    @app.post("/api/llm/test")
    async def test_llm_connection(payload: dict, _auth: dict = Depends(auth_handler)) -> dict:
        """Test LLM connectivity using the provided configuration overrides."""
        config = get_config_manager().config
        if hasattr(config, "model_copy"):
            test_config = config.model_copy(deep=True)
        else:
            test_config = config.copy(deep=True)

        agent_config = test_config.agent

        provider = _normalize_optional_string(payload.get("llm_provider"))
        if provider:
            agent_config.llm_provider = provider

        model = _normalize_optional_string(payload.get("llm_model"))
        if model:
            agent_config.llm_model = model

        for key in ("llm_api_key", "llm_api_key_env", "llm_base_url", "llm_base_url_env"):
            if key in payload:
                setattr(agent_config, key, _normalize_optional_string(payload.get(key)))

        resolved_provider = resolve_llm_provider(agent_config.llm_model, agent_config.llm_provider)
        resolved_model = resolve_llm_model(agent_config.llm_model, agent_config.llm_provider)
        env_keys = get_llm_env_keys(
            resolved_provider,
            agent_config.llm_api_key_env,
            agent_config.llm_base_url_env,
        )
        saved_env = {key: os.environ.get(key) for key in env_keys}

        for key in env_keys:
            os.environ.pop(key, None)

        try:
            apply_llm_settings(test_config)
            if not llm_credentials_present(test_config):
                raise HTTPException(
                    status_code=400,
                    detail="No API key found for the selected provider.",
                )

            try:
                from pydantic_ai import Agent
            except Exception as exc:  # pragma: no cover
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM test unavailable: {exc}",
                ) from exc

            agent = Agent(resolved_model)
            result = await agent.run("Reply with the single word READY.")
            message = str(getattr(result, "data", result))
            if not message.strip():
                message = "LLM reachable."
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        return {
            "status": "ok",
            "provider": resolved_provider,
            "model": resolved_model,
            "message": message,
        }
