"""API Authentication.

Simple API key authentication for the AegisAV server.
Supports both header-based and query parameter authentication.
"""

import logging
import os
import secrets
from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Environment variable for API key
API_KEY_ENV = "AEGIS_API_KEY"
API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_QUERY_NAME = "api_key"


class AuthConfig(BaseModel):
    """Authentication configuration."""

    # API key (if not set, auth is disabled)
    api_key: str | None = None

    # Allow unauthenticated access to certain endpoints
    public_endpoints: list[str] = [
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/",
    ]

    # Enable authentication (can be disabled for development)
    enabled: bool = True

    # Rate limiting (requests per minute per IP)
    rate_limit: int = 100

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Create config from environment variables."""
        api_key = os.environ.get(API_KEY_ENV)

        # If no key is set, disable auth but warn
        if not api_key:
            logger.warning(
                f"No API key set ({API_KEY_ENV} not found). "
                "Authentication is disabled. Set the environment variable for production use."
            )

        return cls(
            api_key=api_key,
            enabled=api_key is not None,
        )


# Security schemes
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY_NAME, auto_error=False)


class APIKeyAuth:
    """API Key authentication handler.

    Validates API keys from headers or query parameters.
    Supports multiple valid keys for key rotation.

    Example:
        # In FastAPI app
        auth = APIKeyAuth(AuthConfig.from_env())

        @app.get("/protected")
        async def protected_endpoint(auth_result: dict = Depends(auth)):
            return {"message": "Authenticated!"}
    """

    def __init__(self, config: AuthConfig | None = None) -> None:
        """Initialize authentication handler.

        Args:
            config: Auth configuration. Uses env vars if None.
        """
        self.config = config or AuthConfig.from_env()
        self._request_counts: dict[str, list[datetime]] = {}
        self.logger = logger

    async def __call__(
        self,
        request: Request,
        api_key_header: str | None = Security(api_key_header),
        api_key_query: str | None = Security(api_key_query),
    ) -> dict:
        """Validate API key from request.

        Args:
            request: FastAPI request
            api_key_header: API key from header
            api_key_query: API key from query parameter

        Returns:
            Auth result dict with authenticated status

        Raises:
            HTTPException: If authentication fails
        """
        # Check if auth is disabled
        if not self.config.enabled:
            return {"authenticated": True, "method": "disabled"}

        # Check if this is a public endpoint
        path = request.url.path
        if any(self._matches_public_endpoint(path, ep) for ep in self.config.public_endpoints):
            return {"authenticated": True, "method": "public"}

        # Also allow static files and dashboard
        if path.startswith("/static") or path.startswith("/dashboard"):
            return {"authenticated": True, "method": "public"}

        # Rate limiting check
        client_ip = self._get_client_ip(request)
        if not self._check_rate_limit(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please slow down.",
            )

        # Get API key from header or query
        api_key = api_key_header or api_key_query

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Provide via X-API-Key header or api_key query parameter.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Validate key
        if not self._validate_key(api_key):
            self.logger.warning(f"Invalid API key attempt from {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return {
            "authenticated": True,
            "method": "api_key",
            "client_ip": client_ip,
        }

    def _validate_key(self, provided_key: str) -> bool:
        """Validate an API key using constant-time comparison.

        Args:
            provided_key: The key to validate

        Returns:
            True if key is valid
        """
        if not self.config.api_key:
            return False

        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(provided_key, self.config.api_key)

    @staticmethod
    def _matches_public_endpoint(path: str, endpoint: str) -> bool:
        """Determine whether a request path matches a configured public endpoint.

        Notes:
        - For "/" we only allow the root path, not every route.
        - For other entries we allow the exact path or any subpath (prefix + "/...").
        """
        if endpoint == "/":
            return path == "/"

        normalized = endpoint.rstrip("/")
        if not normalized:
            return path == "/"
        return path == normalized or path.startswith(f"{normalized}/")

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, handling proxies."""
        # Check for forwarded header (when behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        return request.client.host if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limit.

        Args:
            client_ip: Client IP address

        Returns:
            True if within limit
        """
        now = datetime.now()
        window_start = datetime.now().replace(second=0, microsecond=0)

        # Get or create request list for this IP
        if client_ip not in self._request_counts:
            self._request_counts[client_ip] = []

        # Remove old entries
        self._request_counts[client_ip] = [
            t for t in self._request_counts[client_ip] if t > window_start
        ]

        # Check limit
        if len(self._request_counts[client_ip]) >= self.config.rate_limit:
            return False

        # Add this request
        self._request_counts[client_ip].append(now)
        return True

    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure random API key.

        Returns:
            32-character hex API key
        """
        return secrets.token_hex(32)


def create_auth_dependency(config: AuthConfig | None = None) -> APIKeyAuth:
    """Create an authentication dependency for FastAPI.

    Args:
        config: Auth configuration

    Returns:
        APIKeyAuth instance
    """
    return APIKeyAuth(config)


# Convenience function for requiring authentication
def require_auth(auth_result: Annotated[dict, Depends(APIKeyAuth())]) -> dict:
    """Dependency that requires authentication.

    Use in endpoint definitions:
        @app.get("/protected")
        async def endpoint(auth: dict = Depends(require_auth)):
            ...
    """
    return auth_result


# Optional authentication - doesn't fail if no key provided
def optional_auth(
    api_key_header: str | None = Security(api_key_header),
    api_key_query: str | None = Security(api_key_query),
) -> dict:
    """Optional authentication that doesn't require a key.

    Returns auth info if key provided and valid, otherwise returns
    unauthenticated status without raising an error.
    """
    config = AuthConfig.from_env()

    if not config.enabled:
        return {"authenticated": True, "method": "disabled"}

    api_key = api_key_header or api_key_query

    if not api_key:
        return {"authenticated": False, "method": None}

    if secrets.compare_digest(api_key, config.api_key or ""):
        return {"authenticated": True, "method": "api_key"}

    return {"authenticated": False, "method": None}
