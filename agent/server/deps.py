"""FastAPI dependency providers used across route modules."""

from agent.server.auth import APIKeyAuth, get_auth_config

# API Authentication
auth_handler = APIKeyAuth(config_provider=get_auth_config)
