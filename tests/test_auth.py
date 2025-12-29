"""
Tests for API authentication module.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from agent.server.auth import (
    APIKeyAuth,
    AuthConfig,
    create_auth_dependency,
    optional_auth,
)


class TestAuthConfig:
    """Test AuthConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AuthConfig()
        assert config.api_key is None
        assert config.enabled is True
        assert config.rate_limit == 100
        assert "/health" in config.public_endpoints
        assert "/docs" in config.public_endpoints

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AuthConfig(
            api_key="test-key",
            enabled=True,
            rate_limit=50,
            public_endpoints=["/health", "/custom"],
        )
        assert config.api_key == "test-key"
        assert config.rate_limit == 50
        assert "/custom" in config.public_endpoints

    @pytest.mark.allow_error_logs
    def test_from_env_with_key(self):
        """Test creating config from environment with API key."""
        with patch.dict(os.environ, {"AEGIS_API_KEY": "env-test-key"}):
            config = AuthConfig.from_env()
            assert config.api_key == "env-test-key"
            assert config.enabled is True

    @pytest.mark.allow_error_logs
    def test_from_env_without_key(self):
        """Test creating config from environment without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("AEGIS_API_KEY", None)
            config = AuthConfig.from_env()
            assert config.api_key is None
            assert config.enabled is False


class TestAPIKeyAuth:
    """Test APIKeyAuth handler."""

    @pytest.fixture
    def auth_config(self):
        """Create test auth configuration."""
        return AuthConfig(
            api_key="test-api-key-12345",
            enabled=True,
            rate_limit=10,
            public_endpoints=["/health", "/docs"],
        )

    @pytest.fixture
    def auth_handler(self, auth_config):
        """Create auth handler with test config."""
        return APIKeyAuth(auth_config)

    @pytest.fixture
    def app_with_auth(self, auth_handler):
        """Create FastAPI app with auth endpoints."""
        app = FastAPI()

        @app.get("/protected")
        async def protected(auth: dict = pytest.importorskip("fastapi").Depends(auth_handler)):
            return {"authenticated": auth["authenticated"]}

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    def test_auth_disabled(self):
        """Test that disabled auth allows all requests."""
        config = AuthConfig(enabled=False)
        handler = APIKeyAuth(config)

        request = MagicMock()
        request.url.path = "/protected"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            handler(request, None, None)
        )
        assert result["authenticated"] is True
        assert result["method"] == "disabled"

    def test_public_endpoint(self, auth_handler):
        """Test that public endpoints don't require auth."""
        request = MagicMock()
        request.url.path = "/health"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            auth_handler(request, None, None)
        )
        assert result["authenticated"] is True
        assert result["method"] == "public"

    def test_static_endpoint_public(self, auth_handler):
        """Test that static endpoints are public."""
        request = MagicMock()
        request.url.path = "/static/main.js"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            auth_handler(request, None, None)
        )
        assert result["authenticated"] is True
        assert result["method"] == "public"

    def test_dashboard_endpoint_public(self, auth_handler):
        """Test that dashboard endpoints are public."""
        request = MagicMock()
        request.url.path = "/dashboard"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            auth_handler(request, None, None)
        )
        assert result["authenticated"] is True
        assert result["method"] == "public"

    def test_valid_api_key_header(self, auth_handler):
        """Test authentication with valid API key in header."""
        request = MagicMock()
        request.url.path = "/protected"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            auth_handler(request, "test-api-key-12345", None)
        )
        assert result["authenticated"] is True
        assert result["method"] == "api_key"

    def test_valid_api_key_query(self, auth_handler):
        """Test authentication with valid API key in query parameter."""
        request = MagicMock()
        request.url.path = "/protected"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            auth_handler(request, None, "test-api-key-12345")
        )
        assert result["authenticated"] is True
        assert result["method"] == "api_key"

    @pytest.mark.allow_error_logs
    def test_invalid_api_key(self, auth_handler):
        """Test authentication with invalid API key."""
        request = MagicMock()
        request.url.path = "/protected"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                auth_handler(request, "invalid-key", None)
            )
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    def test_missing_api_key(self, auth_handler):
        """Test authentication with missing API key."""
        request = MagicMock()
        request.url.path = "/protected"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                auth_handler(request, None, None)
            )
        assert exc_info.value.status_code == 401
        assert "API key required" in exc_info.value.detail

    def test_rate_limiting(self, auth_config):
        """Test rate limiting blocks excessive requests."""
        # Create handler with very low rate limit
        config = AuthConfig(
            api_key="test-key",
            enabled=True,
            rate_limit=2,  # Only allow 2 requests per minute
        )
        handler = APIKeyAuth(config)

        request = MagicMock()
        request.url.path = "/protected"
        request.client.host = "127.0.0.1"
        request.headers = {}

        import asyncio

        # First 2 requests should succeed
        for _ in range(2):
            result = asyncio.get_event_loop().run_until_complete(
                handler(request, "test-key", None)
            )
            assert result["authenticated"] is True

        # 3rd request should be rate limited
        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                handler(request, "test-key", None)
            )
        assert exc_info.value.status_code == 429
        assert "Rate limit" in exc_info.value.detail

    def test_get_client_ip_direct(self, auth_handler):
        """Test getting client IP directly."""
        request = MagicMock()
        request.headers = {}
        request.client.host = "192.168.1.100"

        ip = auth_handler._get_client_ip(request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_forwarded(self, auth_handler):
        """Test getting client IP from X-Forwarded-For header."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        request.client.host = "127.0.0.1"

        ip = auth_handler._get_client_ip(request)
        assert ip == "10.0.0.1"

    def test_get_client_ip_real_ip(self, auth_handler):
        """Test getting client IP from X-Real-IP header."""
        request = MagicMock()
        request.headers = {"X-Real-IP": "10.0.0.2"}
        request.client.host = "127.0.0.1"

        ip = auth_handler._get_client_ip(request)
        assert ip == "10.0.0.2"

    def test_generate_api_key(self):
        """Test API key generation."""
        key = APIKeyAuth.generate_api_key()
        assert len(key) == 64  # 32 bytes hex = 64 chars
        assert all(c in "0123456789abcdef" for c in key)

        # Keys should be unique
        key2 = APIKeyAuth.generate_api_key()
        assert key != key2


class TestOptionalAuth:
    """Test optional_auth function."""

    @pytest.mark.allow_error_logs
    def test_optional_auth_disabled(self):
        """Test optional auth when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AEGIS_API_KEY", None)
            result = optional_auth(None, None)
            # When auth is disabled, should return authenticated
            assert result["authenticated"] is True
            assert result["method"] == "disabled"

    def test_optional_auth_no_key(self):
        """Test optional auth when no key provided."""
        with patch.dict(os.environ, {"AEGIS_API_KEY": "test-key"}):
            result = optional_auth(None, None)
            assert result["authenticated"] is False
            assert result["method"] is None

    def test_optional_auth_valid_key(self):
        """Test optional auth with valid key."""
        with patch.dict(os.environ, {"AEGIS_API_KEY": "test-key"}):
            result = optional_auth("test-key", None)
            assert result["authenticated"] is True
            assert result["method"] == "api_key"

    def test_optional_auth_invalid_key(self):
        """Test optional auth with invalid key."""
        with patch.dict(os.environ, {"AEGIS_API_KEY": "test-key"}):
            result = optional_auth("wrong-key", None)
            assert result["authenticated"] is False
            assert result["method"] is None


class TestCreateAuthDependency:
    """Test create_auth_dependency function."""

    def test_creates_auth_handler(self):
        """Test that create_auth_dependency creates an auth handler."""
        config = AuthConfig(api_key="test-key", enabled=True)
        handler = create_auth_dependency(config)

        assert isinstance(handler, APIKeyAuth)
        assert handler.config.api_key == "test-key"

    @pytest.mark.allow_error_logs
    def test_creates_with_defaults(self):
        """Test that create_auth_dependency uses defaults when no config."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AEGIS_API_KEY", None)
            handler = create_auth_dependency()
            assert isinstance(handler, APIKeyAuth)
