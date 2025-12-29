"""
Unified Configuration Manager

Provides centralized configuration management for AegisAV.
Supports loading from YAML, environment variables, and runtime updates via API.
"""

import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RedisSettings(BaseModel):
    """Redis connection settings."""

    enabled: bool = Field(default=True, description="Enable Redis persistence")
    host: str = Field(default="localhost", description="Redis server host")
    port: int = Field(default=6379, description="Redis server port")
    db: int = Field(default=0, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password (optional)")

    # TTL settings
    telemetry_ttl_hours: int = Field(default=1, description="Hours to keep telemetry")
    detection_ttl_days: int = Field(default=7, description="Days to keep detections")
    anomaly_ttl_days: int = Field(default=30, description="Days to keep anomalies")
    mission_ttl_days: int = Field(default=90, description="Days to keep missions")


class AuthSettings(BaseModel):
    """Authentication settings."""

    enabled: bool = Field(default=False, description="Enable API key authentication")
    api_key: str | None = Field(default=None, description="API key (auto-generated if empty)")
    rate_limit_per_minute: int = Field(default=100, description="Max requests per minute per IP")

    # Public endpoints that don't require auth
    public_endpoints: list[str] = Field(
        default=["/health", "/docs", "/openapi.json", "/redoc", "/", "/dashboard", "/static"],
        description="Endpoints that don't require authentication",
    )


class VisionSettings(BaseModel):
    """Vision pipeline settings."""

    enabled: bool = Field(default=True, description="Enable vision system")

    # Detector settings
    use_real_detector: bool = Field(default=False, description="Use real YOLO (requires GPU)")
    model_path: str = Field(default="yolov8n.pt", description="YOLO model path or name")
    device: str = Field(default="auto", description="Device: auto, cpu, cuda, cuda:0")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=640, description="Input image size")

    # Camera settings
    camera_resolution: tuple[int, int] = Field(default=(1920, 1080))
    save_images: bool = Field(default=True, description="Save captured images")
    image_output_dir: str = Field(default="data/vision/captures")


class SimulationSettings(BaseModel):
    """Simulation settings."""

    enabled: bool = Field(default=False, description="Enable simulation mode")

    # AirSim settings
    airsim_enabled: bool = Field(default=False, description="Enable AirSim integration")
    airsim_host: str = Field(default="127.0.0.1")
    airsim_vehicle_name: str = Field(default="Drone1")

    # SITL settings
    sitl_enabled: bool = Field(default=False, description="Enable ArduPilot SITL")
    ardupilot_path: str = Field(default="~/ardupilot")
    sitl_speedup: float = Field(default=1.0, description="Simulation speed multiplier")

    # Home position
    home_latitude: float = Field(default=37.7749)
    home_longitude: float = Field(default=-122.4194)
    home_altitude: float = Field(default=0.0)


class AgentSettings(BaseModel):
    """Agent decision-making settings."""

    use_llm: bool = Field(default=True, description="Enable LLM-based goal selection")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model to use")

    # Battery thresholds
    battery_warning_percent: float = Field(default=30.0)
    battery_critical_percent: float = Field(default=15.0)

    # Wind thresholds
    wind_warning_ms: float = Field(default=8.0)
    wind_abort_ms: float = Field(default=12.0)

    # Decision intervals
    decision_interval_seconds: float = Field(default=1.0)
    max_decisions_per_mission: int = Field(default=1000)


class DashboardSettings(BaseModel):
    """Dashboard/UI settings."""

    refresh_rate_ms: int = Field(default=1000, description="Dashboard refresh rate")
    map_provider: str = Field(default="openstreetmap", description="Map tile provider")
    show_telemetry: bool = Field(default=True)
    show_vision: bool = Field(default=True)
    show_reasoning: bool = Field(default=True)
    theme: str = Field(default="dark", description="UI theme: dark, light")


class ServerSettings(BaseModel):
    """Server settings."""

    host: str = Field(default="0.0.0.0", description="Bind address (0.0.0.0 for all interfaces)")  # noqa: S104
    port: int = Field(default=8080)
    log_level: str = Field(default="INFO", description="Logging level")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")


class AegisConfig(BaseModel):
    """Complete AegisAV configuration."""

    # Meta
    config_version: str = Field(default="1.0.0")
    last_modified: datetime = Field(default_factory=datetime.now)

    # Sections
    server: ServerSettings = Field(default_factory=ServerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)


class ConfigManager:
    """
    Centralized configuration manager.

    Handles loading, saving, and runtime updates of all configurations.
    Supports YAML files, environment variables, and API updates.

    Example:
        manager = ConfigManager()
        manager.load()

        # Get config
        redis_host = manager.config.redis.host

        # Update config
        manager.update_section("redis", {"host": "redis.example.com"})
        manager.save()
    """

    def __init__(self, config_dir: Path | str | None = None):
        """
        Initialize config manager.

        Args:
            config_dir: Directory for config files. Defaults to project configs/
        """
        if config_dir is None:
            # Default to project root configs/
            config_dir = Path(__file__).parent.parent.parent / "configs"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / "aegis_config.yaml"
        self.config = AegisConfig()
        self._loaded = False

        self.logger = logger

    def load(self) -> AegisConfig:
        """
        Load configuration from file and environment.

        Priority (highest to lowest):
        1. Environment variables
        2. Config file
        3. Defaults

        Returns:
            Loaded configuration
        """
        # Start with defaults
        self.config = AegisConfig()

        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                self._apply_dict(data)
                self.logger.info(f"Loaded config from {self.config_file}")

            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")

        # Override with environment variables
        self._apply_environment()

        # Auto-generate API key if auth enabled but no key set
        if self.config.auth.enabled and not self.config.auth.api_key:
            self.config.auth.api_key = secrets.token_hex(32)
            self.logger.info("Auto-generated API key")

        self._loaded = True
        return self.config

    def save(self) -> bool:
        """
        Save current configuration to file.

        Returns:
            True if successful
        """
        try:
            self.config.last_modified = datetime.now()

            # Convert to dict, handling special types
            data = self._config_to_dict(self.config)

            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Saved config to {self.config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False

    def update_section(self, section: str, values: dict) -> bool:
        """
        Update a configuration section.

        Args:
            section: Section name (redis, auth, vision, etc.)
            values: Dict of values to update

        Returns:
            True if successful
        """
        if not hasattr(self.config, section):
            self.logger.error(f"Unknown config section: {section}")
            return False

        try:
            current = getattr(self.config, section)
            current_dict = current.model_dump()
            current_dict.update(values)

            # Create new section with updated values
            section_class = type(current)
            new_section = section_class(**current_dict)
            setattr(self.config, section, new_section)

            self.logger.info(f"Updated config section: {section}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update config section {section}: {e}")
            return False

    def get_section(self, section: str) -> dict | None:
        """Get a configuration section as dict."""
        if hasattr(self.config, section):
            return getattr(self.config, section).model_dump()
        return None

    def get_all(self) -> dict:
        """Get complete configuration as dict."""
        return self._config_to_dict(self.config)

    def reset_section(self, section: str) -> bool:
        """Reset a section to defaults."""
        section_classes = {
            "server": ServerSettings,
            "redis": RedisSettings,
            "auth": AuthSettings,
            "vision": VisionSettings,
            "simulation": SimulationSettings,
            "agent": AgentSettings,
            "dashboard": DashboardSettings,
        }

        if section in section_classes:
            setattr(self.config, section, section_classes[section]())
            return True
        return False

    def reset_all(self) -> None:
        """Reset all configuration to defaults."""
        self.config = AegisConfig()

    def validate(self) -> list[str]:
        """
        Validate current configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Redis validation
        if self.config.redis.enabled:
            if self.config.redis.port < 1 or self.config.redis.port > 65535:
                errors.append("Redis port must be between 1 and 65535")

        # Vision validation
        if self.config.vision.enabled:
            if self.config.vision.confidence_threshold < 0 or self.config.vision.confidence_threshold > 1:
                errors.append("Vision confidence threshold must be between 0 and 1")

        # Auth validation
        if self.config.auth.enabled and not self.config.auth.api_key:
            errors.append("API key required when auth is enabled")

        # Simulation validation
        if self.config.simulation.sitl_enabled:
            ardupilot_path = Path(self.config.simulation.ardupilot_path).expanduser()
            if not ardupilot_path.exists():
                errors.append(f"ArduPilot path does not exist: {ardupilot_path}")

        return errors

    def generate_api_key(self) -> str:
        """Generate a new API key and save it."""
        new_key = secrets.token_hex(32)
        self.config.auth.api_key = new_key
        self.config.auth.enabled = True
        return new_key

    def _apply_dict(self, data: dict) -> None:
        """Apply a dict to the current config."""
        for section, values in data.items():
            if hasattr(self.config, section) and isinstance(values, dict):
                self.update_section(section, values)

    def _apply_environment(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            # Redis
            "AEGIS_REDIS_HOST": ("redis", "host"),
            "AEGIS_REDIS_PORT": ("redis", "port", int),
            "AEGIS_REDIS_PASSWORD": ("redis", "password"),
            "AEGIS_REDIS_ENABLED": ("redis", "enabled", self._parse_bool),

            # Auth
            "AEGIS_API_KEY": ("auth", "api_key"),
            "AEGIS_AUTH_ENABLED": ("auth", "enabled", self._parse_bool),

            # Vision
            "AEGIS_VISION_ENABLED": ("vision", "enabled", self._parse_bool),
            "AEGIS_VISION_MODEL": ("vision", "model_path"),
            "AEGIS_VISION_DEVICE": ("vision", "device"),
            "AEGIS_VISION_REAL_DETECTOR": ("vision", "use_real_detector", self._parse_bool),

            # Simulation
            "AEGIS_SIM_ENABLED": ("simulation", "enabled", self._parse_bool),
            "AEGIS_AIRSIM_ENABLED": ("simulation", "airsim_enabled", self._parse_bool),
            "AEGIS_SITL_ENABLED": ("simulation", "sitl_enabled", self._parse_bool),
            "AEGIS_ARDUPILOT_PATH": ("simulation", "ardupilot_path"),

            # Server
            "AEGIS_HOST": ("server", "host"),
            "AEGIS_PORT": ("server", "port", int),
            "AEGIS_LOG_LEVEL": ("server", "log_level"),

            # Agent
            "AEGIS_USE_LLM": ("agent", "use_llm", self._parse_bool),
            "AEGIS_LLM_MODEL": ("agent", "llm_model"),
            "OPENAI_API_KEY": None,  # Special handling - just check if set
        }

        for env_var, mapping in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None and mapping is not None:
                section, key = mapping[0], mapping[1]
                converter = mapping[2] if len(mapping) > 2 else str

                try:
                    converted_value = converter(value)
                    current = getattr(self.config, section).model_dump()
                    current[key] = converted_value
                    self.update_section(section, current)
                except Exception as e:
                    self.logger.warning(f"Failed to apply env var {env_var}: {e}")

    def _parse_bool(self, value: str) -> bool:
        """Parse boolean from string."""
        return value.lower() in ("true", "1", "yes", "on", "enabled")

    def _config_to_dict(self, config: AegisConfig) -> dict:
        """Convert config to serializable dict."""
        data = {}

        for field_name in type(config).model_fields:
            value = getattr(config, field_name)
            if isinstance(value, BaseModel):
                # Convert model to dict and sanitize for YAML
                data[field_name] = self._sanitize_for_yaml(value.model_dump())
            elif isinstance(value, datetime):
                data[field_name] = value.isoformat()
            else:
                data[field_name] = self._sanitize_for_yaml(value)

        return data

    def _sanitize_for_yaml(self, obj: Any) -> Any:
        """Recursively convert tuples to lists for YAML serialization."""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_yaml(item) for item in obj]
        return obj

    def export_env_template(self) -> str:
        """Generate environment variable template."""
        template = """# AegisAV Environment Configuration
# Copy this to .env and customize

# Server
AEGIS_HOST=0.0.0.0
AEGIS_PORT=8080
AEGIS_LOG_LEVEL=INFO

# Authentication
AEGIS_AUTH_ENABLED=false
# AEGIS_API_KEY=your-api-key-here

# Redis Persistence
AEGIS_REDIS_ENABLED=true
AEGIS_REDIS_HOST=localhost
AEGIS_REDIS_PORT=6379
# AEGIS_REDIS_PASSWORD=

# Vision System
AEGIS_VISION_ENABLED=true
AEGIS_VISION_REAL_DETECTOR=false
AEGIS_VISION_MODEL=yolov8n.pt
AEGIS_VISION_DEVICE=auto

# Simulation
AEGIS_SIM_ENABLED=false
AEGIS_AIRSIM_ENABLED=false
AEGIS_SITL_ENABLED=false
AEGIS_ARDUPILOT_PATH=~/ardupilot

# LLM
AEGIS_USE_LLM=true
AEGIS_LLM_MODEL=gpt-4o-mini
# OPENAI_API_KEY=your-openai-key
"""
        return template


# Global config manager instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load()
    return _config_manager


def get_config() -> AegisConfig:
    """Get the current configuration."""
    return get_config_manager().config
