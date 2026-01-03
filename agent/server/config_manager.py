"""Unified Configuration Manager.

Provides centralized configuration management for AegisAV.
Supports loading from YAML, environment variables, and runtime updates via API.
"""

import logging
import os
import secrets
from datetime import datetime
from pathlib import Path, PureWindowsPath

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Global Constants - Single source of truth for ports and URLs
# =============================================================================
DEFAULT_SERVER_PORT = 8090
DEFAULT_WEBSOCKET_PATH = "/ws/unreal"


def _is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def _normalize_wsl_unc_path(path_str: str) -> str | None:
    if not _is_wsl():
        return None
    lower = path_str.lower()
    if not (lower.startswith("\\\\wsl.localhost\\") or lower.startswith("\\\\wsl$\\")):
        return None
    win_path = PureWindowsPath(path_str)
    if len(win_path.parts) < 2:
        return None
    linux_path = Path("/").joinpath(*win_path.parts[1:])
    return str(linux_path)

def get_default_server_url(host: str = "localhost", port: int | None = None) -> str:
    """Get the default server URL."""
    return f"http://{host}:{port or DEFAULT_SERVER_PORT}"

def get_default_websocket_url(host: str = "localhost", port: int | None = None) -> str:
    """Get the default WebSocket URL for Unreal connections."""
    return f"ws://{host}:{port or DEFAULT_SERVER_PORT}{DEFAULT_WEBSOCKET_PATH}"


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

    enabled: bool = Field(default=True, description="Enable simulation mode")

    # AirSim settings
    airsim_enabled: bool = Field(default=True, description="Enable AirSim integration")
    airsim_host: str = Field(default="127.0.0.1")
    airsim_vehicle_name: str = Field(default="Drone1", description="Primary vehicle (legacy)")
    airsim_vehicle_type: str = Field(
        default="SimpleFlight",
        description="Vehicle type for AirSim settings.json (SimpleFlight, ArduCopter)",
    )

    # Multi-drone vehicle mapping: scenario drone_id -> AirSim vehicle name
    # Example: {"alpha": "Drone1", "bravo": "Drone2", "charlie": "Drone3"}
    # If empty, uses auto-mapping based on available vehicles
    airsim_vehicle_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of scenario drone_id to AirSim vehicle names",
    )

    # Maximum number of drones to support (for auto-spawning if needed)
    max_drones: int = Field(default=4, description="Maximum concurrent drones")

    # AirSim settings.json management
    airsim_settings_path: str | None = Field(
        default=None,
        description="Path to AirSim settings.json (optional)",
    )
    airsim_auto_update_settings: bool = Field(
        default=True,
        description="Auto-update AirSim settings.json when config or scenarios change",
    )
    airsim_auto_restart_on_scenario_change: bool = Field(
        default=True,
        description="Auto-restart AirSim when scenario change updates settings.json",
    )
    require_fleet_for_multi_drone: bool = Field(
        default=True,
        description="Require multi-drone fleet bridge to start multi-drone scenarios",
    )

    # SITL settings
    sitl_enabled: bool = Field(default=False, description="Enable ArduPilot SITL")
    sitl_multi_vehicle: bool = Field(
        default=False,
        description="Enable multi-vehicle SITL (generate multiple ArduCopter entries)",
    )
    ardupilot_path: str = Field(default="~/ardupilot")
    sitl_speedup: float = Field(default=1.0, description="Simulation speed multiplier")

    # Home position
    home_latitude: float = Field(default=37.7749)
    home_longitude: float = Field(default=-122.4194)
    home_altitude: float = Field(default=0.0)

    # Battery simulation for AirSim telemetry
    battery_sim_enabled: bool = Field(default=True)
    battery_initial_percent: float = Field(default=100.0)
    battery_min_percent: float = Field(default=5.0)
    battery_max_percent: float = Field(default=100.0)
    battery_drain_hover_percent_per_min: float = Field(default=6.0)
    battery_drain_move_percent_per_m: float = Field(default=0.015)
    battery_charge_percent_per_min: float = Field(default=25.0)
    battery_aggressive_multiplier: float = Field(default=1.3)
    battery_low_speed_threshold_ms: float = Field(default=0.5)


class AgentSettings(BaseModel):
    """Agent decision-making settings."""

    use_llm: bool = Field(default=True, description="Enable LLM-based goal selection")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    llm_provider: str = Field(default="openai", description="LLM provider prefix")
    llm_api_key_env: str | None = Field(
        default=None,
        description="Environment variable name containing provider API key",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="Provider API key (stored in config; prefer env variables)",
    )
    llm_base_url_env: str | None = Field(
        default=None,
        description="Environment variable name for provider base URL",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Provider base URL override (stored in config; optional)",
    )

    # Battery thresholds
    battery_warning_percent: float = Field(default=30.0)
    battery_critical_percent: float = Field(default=15.0)

    # Wind thresholds
    wind_warning_ms: float = Field(default=8.0)
    wind_abort_ms: float = Field(default=12.0)

    # Decision intervals
    decision_interval_seconds: float = Field(default=1.0)
    max_decisions_per_mission: int = Field(default=1000)


class MissionSuccessWeights(BaseModel):
    """Weights for mission success scoring."""

    coverage: float = Field(default=0.30, ge=0.0)
    anomaly: float = Field(default=0.25, ge=0.0)
    decision_quality: float = Field(default=0.20, ge=0.0)
    execution: float = Field(default=0.15, ge=0.0)
    resource_use: float = Field(default=0.10, ge=0.0)


class MissionSuccessThresholds(BaseModel):
    """Grade thresholds for mission success scoring."""

    excellent: float = Field(default=85.0, ge=0.0, le=100.0)
    good: float = Field(default=70.0, ge=0.0, le=100.0)
    fair: float = Field(default=55.0, ge=0.0, le=100.0)


class DashboardSettings(BaseModel):
    """Dashboard/UI settings."""

    refresh_rate_ms: int = Field(default=1000, description="Dashboard refresh rate")
    map_provider: str = Field(default="openstreetmap", description="Map tile provider")
    show_telemetry: bool = Field(default=True)
    show_vision: bool = Field(default=True)
    show_reasoning: bool = Field(default=True)
    theme: str = Field(default="dark", description="UI theme: dark, light")
    mission_success_weights: MissionSuccessWeights = Field(
        default_factory=MissionSuccessWeights,
        description="Weights for mission success score components",
    )
    mission_success_thresholds: MissionSuccessThresholds = Field(
        default_factory=MissionSuccessThresholds,
        description="Grade thresholds for mission success score",
    )


class MappingSettings(BaseModel):
    """Mapping pipeline settings."""

    enabled: bool = Field(default=True, description="Enable SLAM/splat map updates")
    update_interval_s: float = Field(default=2.0, description="Map update loop interval")
    prefer_splat: bool = Field(default=False, description="Prefer splat preview over SLAM points")
    max_map_age_s: float = Field(default=60.0, description="Max age before map considered stale")
    min_quality_score: float = Field(default=0.3, description="Minimum quality score to accept map")
    slam_dir: str = Field(default="data/slam_runs")
    splat_dir: str = Field(default="data/splats")
    map_resolution_m: float = Field(default=2.0)
    tile_size_cells: int = Field(default=120, description="Occupancy tile size in cells")
    voxel_size_m: float | None = Field(default=None)
    max_points: int = Field(default=200000)
    min_points: int = Field(default=50)
    fused_map_dir: str = Field(default="data/maps/fused", description="Fused map artifact storage")
    fused_map_max_versions: int | None = Field(default=None, description="Max fused map versions per map")
    fused_map_max_age_days: int | None = Field(default=None, description="Max age in days for fused maps")
    fused_map_keep_last: int = Field(default=3, description="Keep last N fused map versions")
    slam_backend: str = Field(
        default="telemetry",
        description="SLAM backend: telemetry, orb_slam3, vins_fusion",
    )
    slam_allow_fallback: bool = Field(
        default=True,
        description="Allow telemetry fallback when external SLAM backends fail",
    )

    # Proxy regeneration settings (Agent B Phase 3)
    proxy_regeneration_enabled: bool = Field(default=True, description="Enable planning proxy regeneration")
    proxy_regeneration_cadence_s: float = Field(default=30.0, description="Min interval between proxy regenerations")
    proxy_max_points: int = Field(default=100000, description="Max points for planning proxy")
    proxy_force_regenerate: bool = Field(default=False, description="Force regenerate proxy on every update")

    # Splat training settings
    splat_backend: str = Field(default="stub", description="Splat training backend: stub, gsplat, nerfstudio")
    splat_iterations: int = Field(default=7000, description="Training iterations for real backends")
    splat_auto_train: bool = Field(default=False, description="Auto-train splat after SLAM capture")
    reset_on_scenario_start: bool = Field(
        default=True,
        description="Clear mapping artifacts and in-memory map state when a scenario starts",
    )

    # Preflight mapping pass
    preflight_enabled: bool = Field(default=True, description="Run preflight map pass before targets")
    preflight_altitude_agl: float = Field(default=20.0, description="Preflight mapping altitude AGL")
    preflight_step_m: float = Field(default=15.0, description="Step size along preflight path")
    preflight_velocity_ms: float = Field(default=4.0, description="Preflight mapping velocity")
    preflight_capture_interval_s: float = Field(default=0.4, description="Delay between captures")
    preflight_timeout_s: float = Field(default=180.0, description="Preflight mapping timeout")
    preflight_retry_count: int = Field(default=0, description="Preflight mapping retry count")
    preflight_retry_delay_s: float = Field(default=5.0, description="Delay between preflight retries")
    preflight_move_timeout_s: float = Field(default=30.0, description="Timeout per preflight move")
    preflight_recovery_timeout_s: float = Field(default=60.0, description="Timeout for recovery move after failure")
    preflight_max_move_failures: int = Field(
        default=3,
        description="Max move failures before aborting preflight mapping",
    )
    preflight_max_capture_failures: int = Field(
        default=5,
        description="Max capture failures before aborting preflight mapping",
    )
    preflight_autorun: bool = Field(default=False, description="Auto-run preflight mapping on startup")
    preflight_autorun_scenario_id: str | None = Field(
        default=None,
        description="Scenario id to use for autorun preflight mapping",
    )
    preflight_autorun_delay_s: float = Field(
        default=6.0,
        description="Delay before autorun preflight mapping starts",
    )
    preflight_autorun_max_attempts: int = Field(
        default=2,
        description="Max autorun attempts before giving up",
    )
    preflight_autorun_retry_delay_s: float = Field(
        default=15.0,
        description="Delay between autorun retry attempts",
    )

    # Splat proxy generation
    proxy_regen_interval_s: float = Field(default=60.0, description="Min seconds between proxy rebuilds")
    proxy_max_points: int = Field(default=120000, description="Max points for splat planning proxy")

    # Real sensor capture (optional)
    real_capture_enabled: bool = Field(
        default=False,
        description="Enable real sensor capture endpoints",
    )
    real_capture_output_dir: str = Field(
        default="data/maps/real_capture",
        description="Output directory for real sensor capture",
    )
    real_capture_camera_index: int = Field(default=0, description="Default camera index")
    real_capture_width: int = Field(default=1280)
    real_capture_height: int = Field(default=720)
    real_capture_fps: int = Field(default=15)
    real_capture_frames: int = Field(default=120, description="Frames to capture per run")
    real_capture_interval_s: float = Field(default=0.5, description="Interval between frames")
    real_capture_calibration_path: str | None = Field(
        default=None,
        description="Calibration JSON path for real sensor capture",
    )


class ServerSettings(BaseModel):
    """Server settings."""

    host: str = Field(default="0.0.0.0", description="Bind address (0.0.0.0 for all interfaces)")  # noqa: S104
    port: int = Field(default=DEFAULT_SERVER_PORT)
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
    mapping: MappingSettings = Field(default_factory=MappingSettings)


class ConfigManager:
    """Centralized configuration manager.

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

    def __init__(self, config_dir: Path | str | None = None) -> None:
        """Initialize config manager.

        Args:
            config_dir: Directory for config files. Defaults to project configs/
        """
        if config_dir is None:
            # Default to project root configs/
            config_dir = Path(__file__).parent.parent.parent / "configs"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Store project root for resolving relative paths
        self._project_root = self.config_dir.parent

        self.config_file = self.config_dir / "aegis_config.yaml"
        self.config = AegisConfig()
        self._loaded = False

        self.logger = logger

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root

    def resolve_path(self, path_str: str) -> Path:
        """Resolve a path string relative to project root.

        If the path is absolute, returns it as-is.
        If the path is relative, resolves it relative to project root.

        Args:
            path_str: Path string from configuration

        Returns:
            Resolved absolute path
        """
        normalized = _normalize_wsl_unc_path(path_str)
        if normalized:
            return Path(normalized)
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self._project_root / path

    def load(self) -> AegisConfig:
        """Load configuration from file and environment.

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
            self.save()

        self._loaded = True
        return self.config

    def save(self) -> bool:
        """Save current configuration to file.

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
        """Update a configuration section.

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
        """Validate current configuration.

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
            if (
                self.config.vision.confidence_threshold < 0
                or self.config.vision.confidence_threshold > 1
            ):
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

    def generate_api_key(self) -> tuple[str, bool]:
        """Generate a new API key and save it."""
        new_key = secrets.token_hex(32)
        self.config.auth.api_key = new_key
        self.config.auth.enabled = True
        saved = self.save()
        if not saved:
            self.logger.warning("Generated API key but failed to save config")
        return new_key, saved

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
            "AEGIS_AIRSIM_HOST": ("simulation", "airsim_host"),
            "AEGIS_AIRSIM_SETTINGS_PATH": ("simulation", "airsim_settings_path"),
            "AEGIS_AIRSIM_AUTO_UPDATE_SETTINGS": (
                "simulation",
                "airsim_auto_update_settings",
                self._parse_bool,
            ),
            "AEGIS_AIRSIM_VEHICLE_TYPE": ("simulation", "airsim_vehicle_type"),
            "AEGIS_AIRSIM_AUTO_RESTART_ON_SCENARIO_CHANGE": (
                "simulation",
                "airsim_auto_restart_on_scenario_change",
                self._parse_bool,
            ),
            "AEGIS_REQUIRE_FLEET_FOR_MULTI_DRONE": (
                "simulation",
                "require_fleet_for_multi_drone",
                self._parse_bool,
            ),
            "AEGIS_SITL_ENABLED": ("simulation", "sitl_enabled", self._parse_bool),
            "AEGIS_SITL_MULTI_VEHICLE": (
                "simulation",
                "sitl_multi_vehicle",
                self._parse_bool,
            ),
            "AEGIS_ARDUPILOT_PATH": ("simulation", "ardupilot_path"),
            # Server
            "AEGIS_HOST": ("server", "host"),
            "AEGIS_PORT": ("server", "port", int),
            "AEGIS_LOG_LEVEL": ("server", "log_level"),
            # Agent
            "AEGIS_USE_LLM": ("agent", "use_llm", self._parse_bool),
            "AEGIS_LLM_MODEL": ("agent", "llm_model"),
            "AEGIS_LLM_PROVIDER": ("agent", "llm_provider"),
            "AEGIS_LLM_API_KEY_ENV": ("agent", "llm_api_key_env"),
            "AEGIS_LLM_API_KEY": ("agent", "llm_api_key"),
            "AEGIS_LLM_BASE_URL_ENV": ("agent", "llm_base_url_env"),
            "AEGIS_LLM_BASE_URL": ("agent", "llm_base_url"),
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

        api_key_env = os.environ.get("AEGIS_API_KEY")
        auth_enabled_env = os.environ.get("AEGIS_AUTH_ENABLED")
        if api_key_env and auth_enabled_env is None:
            self.config.auth.enabled = True

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

    def _sanitize_for_yaml(self, obj: object) -> object:
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
        template = f"""# AegisAV Environment Configuration
# Copy this to .env and customize

# Server
AEGIS_HOST=0.0.0.0
AEGIS_PORT={DEFAULT_SERVER_PORT}
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
AEGIS_AIRSIM_HOST=127.0.0.1
AEGIS_SITL_ENABLED=false
AEGIS_ARDUPILOT_PATH=~/ardupilot

# LLM
AEGIS_USE_LLM=true
AEGIS_LLM_PROVIDER=openai
AEGIS_LLM_MODEL=gpt-4o-mini
# AEGIS_LLM_API_KEY=your-api-key-here
# AEGIS_LLM_API_KEY_ENV=OPENAI_API_KEY
# AEGIS_LLM_BASE_URL=
# AEGIS_LLM_BASE_URL_ENV=OPENAI_BASE_URL
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-claude-key
# OPENROUTER_API_KEY=your-openrouter-key
# GROQ_API_KEY=your-groq-key
# MISTRAL_API_KEY=your-mistral-key
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
