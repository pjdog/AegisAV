"""
Tests for the unified configuration manager.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from agent.server.config_manager import (
    AegisConfig,
    AgentSettings,
    AuthSettings,
    ConfigManager,
    DashboardSettings,
    DEFAULT_SERVER_PORT,
    RedisSettings,
    ServerSettings,
    SimulationSettings,
    VisionSettings,
)


class TestSettingsModels:
    """Test individual settings models."""

    def test_server_settings_defaults(self):
        """Test ServerSettings has correct defaults."""
        settings = ServerSettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == DEFAULT_SERVER_PORT
        assert settings.log_level == "INFO"
        assert settings.cors_origins == ["*"]

    def test_redis_settings_defaults(self):
        """Test RedisSettings has correct defaults."""
        settings = RedisSettings()
        assert settings.enabled is True
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.db == 0
        assert settings.password is None
        assert settings.telemetry_ttl_hours == 1
        assert settings.detection_ttl_days == 7
        assert settings.anomaly_ttl_days == 30
        assert settings.mission_ttl_days == 90

    def test_auth_settings_defaults(self):
        """Test AuthSettings has correct defaults."""
        settings = AuthSettings()
        assert settings.enabled is False
        assert settings.api_key is None
        assert settings.rate_limit_per_minute == 100
        assert "/health" in settings.public_endpoints
        assert "/docs" in settings.public_endpoints

    def test_vision_settings_defaults(self):
        """Test VisionSettings has correct defaults."""
        settings = VisionSettings()
        assert settings.enabled is True
        assert settings.use_real_detector is False
        assert settings.model_path == "yolov8n.pt"
        assert settings.device == "auto"
        assert settings.confidence_threshold == 0.5
        assert settings.iou_threshold == 0.45
        assert settings.image_size == 640
        assert settings.camera_resolution == (1920, 1080)
        assert settings.save_images is True

    def test_vision_settings_validation(self):
        """Test VisionSettings validates thresholds."""
        # Valid thresholds
        settings = VisionSettings(confidence_threshold=0.7, iou_threshold=0.5)
        assert settings.confidence_threshold == 0.7
        assert settings.iou_threshold == 0.5

        # Invalid thresholds should raise
        with pytest.raises(ValueError):
            VisionSettings(confidence_threshold=1.5)

        with pytest.raises(ValueError):
            VisionSettings(iou_threshold=-0.1)

    def test_simulation_settings_defaults(self):
        """Test SimulationSettings has correct defaults."""
        settings = SimulationSettings()
        assert settings.enabled is False
        assert settings.airsim_enabled is False
        assert settings.airsim_host == "127.0.0.1"
        assert settings.airsim_vehicle_name == "Drone1"
        assert settings.sitl_enabled is False
        assert settings.ardupilot_path == "~/ardupilot"
        assert settings.sitl_speedup == 1.0
        assert settings.home_latitude == 37.7749
        assert settings.home_longitude == -122.4194

    def test_agent_settings_defaults(self):
        """Test AgentSettings has correct defaults."""
        settings = AgentSettings()
        assert settings.use_llm is True
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.battery_warning_percent == 30.0
        assert settings.battery_critical_percent == 15.0
        assert settings.wind_warning_ms == 8.0
        assert settings.wind_abort_ms == 12.0
        assert settings.decision_interval_seconds == 1.0
        assert settings.max_decisions_per_mission == 1000

    def test_dashboard_settings_defaults(self):
        """Test DashboardSettings has correct defaults."""
        settings = DashboardSettings()
        assert settings.refresh_rate_ms == 1000
        assert settings.map_provider == "openstreetmap"
        assert settings.show_telemetry is True
        assert settings.show_vision is True
        assert settings.show_reasoning is True
        assert settings.theme == "dark"


class TestAegisConfig:
    """Test the master AegisConfig model."""

    def test_aegis_config_defaults(self):
        """Test AegisConfig has all sections with defaults."""
        config = AegisConfig()

        assert config.config_version == "1.0.0"
        assert config.last_modified is not None

        # Check all sections exist
        assert isinstance(config.server, ServerSettings)
        assert isinstance(config.redis, RedisSettings)
        assert isinstance(config.auth, AuthSettings)
        assert isinstance(config.vision, VisionSettings)
        assert isinstance(config.simulation, SimulationSettings)
        assert isinstance(config.agent, AgentSettings)
        assert isinstance(config.dashboard, DashboardSettings)

    def test_aegis_config_custom_values(self):
        """Test AegisConfig with custom section values."""
        config = AegisConfig(
            server=ServerSettings(port=9000),
            redis=RedisSettings(host="redis.local"),
            auth=AuthSettings(enabled=True),
        )

        assert config.server.port == 9000
        assert config.redis.host == "redis.local"
        assert config.auth.enabled is True


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_config_manager_initialization(self):
        """Test ConfigManager initializes with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)

            assert manager.config_dir == Path(tmpdir)
            assert manager.config_file == Path(tmpdir) / "aegis_config.yaml"
            assert manager._loaded is False
            assert isinstance(manager.config, AegisConfig)

    def test_config_manager_load_creates_defaults(self):
        """Test load() works with no existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            config = manager.load()

            assert manager._loaded is True
            assert isinstance(config, AegisConfig)
            assert config.server.port == DEFAULT_SERVER_PORT

    def test_config_manager_save_and_load(self):
        """Test save() and load() round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and modify config using update_section API
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            # Use update_section which is the proper API
            manager.update_section("server", {"port": 9999})
            manager.update_section("redis", {"host": "custom-redis"})

            # Verify changes took effect
            assert manager.config.server.port == 9999
            assert manager.config.redis.host == "custom-redis"

            # Save
            success = manager.save()
            assert success is True
            assert manager.config_file.exists()

            # Load in new manager
            manager2 = ConfigManager(config_dir=tmpdir)
            config2 = manager2.load()

            assert config2.server.port == 9999
            assert config2.redis.host == "custom-redis"

    def test_config_manager_update_section(self):
        """Test update_section() modifies config correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            # Update redis section
            success = manager.update_section(
                "redis",
                {
                    "host": "new-host",
                    "port": 6380,
                },
            )

            assert success is True
            assert manager.config.redis.host == "new-host"
            assert manager.config.redis.port == 6380
            # Other values should remain
            assert manager.config.redis.enabled is True

    @pytest.mark.allow_error_logs
    def test_config_manager_update_section_invalid(self):
        """Test update_section() fails for invalid section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            success = manager.update_section("nonexistent", {"key": "value"})
            assert success is False

    def test_config_manager_get_section(self):
        """Test get_section() returns correct dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            section = manager.get_section("server")
            assert isinstance(section, dict)
            assert section["host"] == "0.0.0.0"
            assert section["port"] == DEFAULT_SERVER_PORT

            # Invalid section returns None
            assert manager.get_section("invalid") is None

    def test_config_manager_get_all(self):
        """Test get_all() returns complete config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            all_config = manager.get_all()
            assert isinstance(all_config, dict)
            assert "server" in all_config
            assert "redis" in all_config
            assert "auth" in all_config
            assert "vision" in all_config
            assert "simulation" in all_config
            assert "agent" in all_config
            assert "dashboard" in all_config
            assert "config_version" in all_config

    def test_config_manager_reset_section(self):
        """Test reset_section() restores defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            # Modify and then reset
            manager.config.server.port = 9999
            assert manager.config.server.port == 9999

            success = manager.reset_section("server")
            assert success is True
            assert manager.config.server.port == DEFAULT_SERVER_PORT

    def test_config_manager_reset_all(self):
        """Test reset_all() restores all defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            # Modify multiple sections
            manager.config.server.port = 9999
            manager.config.redis.host = "custom"
            manager.config.auth.enabled = True

            manager.reset_all()

            assert manager.config.server.port == DEFAULT_SERVER_PORT
            assert manager.config.redis.host == "localhost"
            assert manager.config.auth.enabled is False

    def test_config_manager_validate(self):
        """Test validate() returns errors for invalid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            # Valid config should have no errors
            errors = manager.validate()
            assert errors == []

            # Invalid redis port
            manager.config.redis.port = 99999
            errors = manager.validate()
            assert len(errors) > 0
            assert any("port" in e.lower() for e in errors)

    def test_config_manager_generate_api_key(self):
        """Test generate_api_key() creates valid key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            assert manager.config.auth.api_key is None

            new_key, saved = manager.generate_api_key()

            assert new_key is not None
            assert len(new_key) == 64  # 32 bytes hex = 64 chars
            assert manager.config.auth.api_key == new_key
            assert manager.config.auth.enabled is True
            assert saved is True
            assert manager.config_file.exists()

    def test_config_manager_export_env_template(self):
        """Test export_env_template() generates valid template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=tmpdir)
            manager.load()

            template = manager.export_env_template()

            assert isinstance(template, str)
            assert "AEGIS_HOST" in template
            assert "AEGIS_PORT" in template
            assert "AEGIS_REDIS_HOST" in template
            assert "AEGIS_VISION_ENABLED" in template
            assert "OPENAI_API_KEY" in template


class TestConfigManagerEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_env_override_server_port(self):
        """Test AEGIS_PORT overrides config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AEGIS_PORT"] = "9001"
            try:
                manager = ConfigManager(config_dir=tmpdir)
                manager.load()

                assert manager.config.server.port == 9001
            finally:
                del os.environ["AEGIS_PORT"]

    def test_env_override_redis_host(self):
        """Test AEGIS_REDIS_HOST overrides config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AEGIS_REDIS_HOST"] = "redis.example.com"
            try:
                manager = ConfigManager(config_dir=tmpdir)
                manager.load()

                assert manager.config.redis.host == "redis.example.com"
            finally:
                del os.environ["AEGIS_REDIS_HOST"]

    def test_env_override_boolean(self):
        """Test boolean env var parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AEGIS_VISION_ENABLED"] = "false"
            try:
                manager = ConfigManager(config_dir=tmpdir)
                manager.load()

                assert manager.config.vision.enabled is False
            finally:
                del os.environ["AEGIS_VISION_ENABLED"]

    def test_env_override_api_key(self):
        """Test AEGIS_API_KEY sets key and enables auth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AEGIS_API_KEY"] = "test-api-key-12345"
            try:
                manager = ConfigManager(config_dir=tmpdir)
                manager.load()

                assert manager.config.auth.api_key == "test-api-key-12345"
                assert manager.config.auth.enabled is True
            finally:
                del os.environ["AEGIS_API_KEY"]

    def test_env_override_auth_disabled(self):
        """Test AEGIS_AUTH_ENABLED=false overrides API key enablement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["AEGIS_API_KEY"] = "test-api-key-12345"
            os.environ["AEGIS_AUTH_ENABLED"] = "false"
            try:
                manager = ConfigManager(config_dir=tmpdir)
                manager.load()

                assert manager.config.auth.api_key == "test-api-key-12345"
                assert manager.config.auth.enabled is False
            finally:
                del os.environ["AEGIS_API_KEY"]
                del os.environ["AEGIS_AUTH_ENABLED"]


class TestConfigManagerYAMLLoading:
    """Test YAML file loading and parsing."""

    def test_load_from_yaml_file(self):
        """Test loading config from existing YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "aegis_config.yaml"

            # Write custom config
            config_data = {
                "server": {"port": 8888, "host": "127.0.0.1"},
                "redis": {"enabled": False, "host": "custom-redis"},
                "vision": {"confidence_threshold": 0.8},
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            manager = ConfigManager(config_dir=tmpdir)
            config = manager.load()

            assert config.server.port == 8888
            assert config.server.host == "127.0.0.1"
            assert config.redis.enabled is False
            assert config.redis.host == "custom-redis"
            assert config.vision.confidence_threshold == 0.8

    @pytest.mark.allow_error_logs
    def test_load_handles_malformed_yaml(self):
        """Test loading handles malformed YAML gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "aegis_config.yaml"

            # Write invalid YAML
            with open(config_file, "w") as f:
                f.write("invalid: yaml: content: [")

            manager = ConfigManager(config_dir=tmpdir)
            # Should not raise, should use defaults
            config = manager.load()

            # Should have defaults
            assert config.server.port == DEFAULT_SERVER_PORT

    def test_load_handles_partial_config(self):
        """Test loading handles partial config with missing sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "aegis_config.yaml"

            # Write partial config (only server section)
            config_data = {
                "server": {"port": 7777},
            }
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            manager = ConfigManager(config_dir=tmpdir)
            config = manager.load()

            # Should have custom server port
            assert config.server.port == 7777
            # Other sections should have defaults
            assert config.redis.port == 6379
            assert config.vision.enabled is True
