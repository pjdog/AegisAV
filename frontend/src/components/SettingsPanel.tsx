import { useCallback, useEffect, useState } from "react";

type ConfigSection = {
  server: ServerConfig;
  redis: RedisConfig;
  auth: AuthConfig;
  vision: VisionConfig;
  simulation: SimulationConfig;
  agent: AgentConfig;
  dashboard: DashboardConfig;
};

type ServerConfig = {
  host: string;
  port: number;
  log_level: string;
  cors_origins: string[];
};

type RedisConfig = {
  enabled: boolean;
  host: string;
  port: number;
  db: number;
  password: string | null;
  telemetry_ttl_hours: number;
  detection_ttl_days: number;
  anomaly_ttl_days: number;
  mission_ttl_days: number;
};

type AuthConfig = {
  enabled: boolean;
  api_key: string | null;
  rate_limit_per_minute: number;
  public_endpoints: string[];
};

type VisionConfig = {
  enabled: boolean;
  use_real_detector: boolean;
  model_path: string;
  device: string;
  confidence_threshold: number;
  iou_threshold: number;
  image_size: number;
  camera_resolution: [number, number];
  save_images: boolean;
  image_output_dir: string;
};

type SimulationConfig = {
  enabled: boolean;
  airsim_enabled: boolean;
  airsim_host: string;
  airsim_vehicle_name: string;
  sitl_enabled: boolean;
  ardupilot_path: string;
  sitl_speedup: number;
  home_latitude: number;
  home_longitude: number;
  home_altitude: number;
};

type AgentConfig = {
  use_llm: boolean;
  llm_model: string;
  battery_warning_percent: number;
  battery_critical_percent: number;
  wind_warning_ms: number;
  wind_abort_ms: number;
  decision_interval_seconds: number;
  max_decisions_per_mission: number;
};

type DashboardConfig = {
  refresh_rate_ms: number;
  map_provider: string;
  show_telemetry: boolean;
  show_vision: boolean;
  show_reasoning: boolean;
  theme: string;
};

type ValidationResult = {
  valid: boolean;
  errors: string[];
};

const SECTION_TITLES: Record<keyof ConfigSection, string> = {
  server: "Server",
  redis: "Redis Persistence",
  auth: "Authentication",
  vision: "Vision System",
  simulation: "Simulation",
  agent: "Agent Decision-Making",
  dashboard: "Dashboard UI",
};

const ToggleSwitch = ({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) => (
  <label className="toggle-row">
    <span>{label}</span>
    <div className={`toggle ${checked ? "on" : ""}`} onClick={() => onChange(!checked)}>
      <div className="toggle-thumb" />
    </div>
  </label>
);

const TextField = ({
  value,
  onChange,
  label,
  type = "text",
  placeholder,
}: {
  value: string | number;
  onChange: (value: string) => void;
  label: string;
  type?: string;
  placeholder?: string;
}) => (
  <label className="field-row">
    <span>{label}</span>
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  </label>
);

const SelectField = ({
  value,
  onChange,
  label,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
  options: { value: string; label: string }[];
}) => (
  <label className="field-row">
    <span>{label}</span>
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  </label>
);

const SliderField = ({
  value,
  onChange,
  label,
  min,
  max,
  step = 1,
  unit,
}: {
  value: number;
  onChange: (value: number) => void;
  label: string;
  min: number;
  max: number;
  step?: number;
  unit?: string;
}) => (
  <label className="field-row slider">
    <span>{label}</span>
    <div className="slider-container">
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
      />
      <span className="slider-value">
        {value}
        {unit}
      </span>
    </div>
  </label>
);

const SectionForm = ({
  section,
  data,
  onChange,
}: {
  section: keyof ConfigSection;
  data: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) => {
  switch (section) {
    case "server":
      return (
        <>
          <TextField
            label="Host"
            value={data.host as string}
            onChange={(v) => onChange("host", v)}
            placeholder="0.0.0.0"
          />
          <TextField
            label="Port"
            value={data.port as number}
            type="number"
            onChange={(v) => onChange("port", parseInt(v))}
          />
          <SelectField
            label="Log Level"
            value={data.log_level as string}
            onChange={(v) => onChange("log_level", v)}
            options={[
              { value: "DEBUG", label: "Debug" },
              { value: "INFO", label: "Info" },
              { value: "WARNING", label: "Warning" },
              { value: "ERROR", label: "Error" },
            ]}
          />
        </>
      );

    case "redis":
      return (
        <>
          <ToggleSwitch
            label="Enable Redis"
            checked={data.enabled as boolean}
            onChange={(v) => onChange("enabled", v)}
          />
          <TextField
            label="Host"
            value={data.host as string}
            onChange={(v) => onChange("host", v)}
          />
          <TextField
            label="Port"
            value={data.port as number}
            type="number"
            onChange={(v) => onChange("port", parseInt(v))}
          />
          <TextField
            label="Database"
            value={data.db as number}
            type="number"
            onChange={(v) => onChange("db", parseInt(v))}
          />
          <SliderField
            label="Telemetry TTL"
            value={data.telemetry_ttl_hours as number}
            onChange={(v) => onChange("telemetry_ttl_hours", v)}
            min={1}
            max={24}
            unit="h"
          />
          <SliderField
            label="Detection TTL"
            value={data.detection_ttl_days as number}
            onChange={(v) => onChange("detection_ttl_days", v)}
            min={1}
            max={30}
            unit="d"
          />
          <SliderField
            label="Anomaly TTL"
            value={data.anomaly_ttl_days as number}
            onChange={(v) => onChange("anomaly_ttl_days", v)}
            min={1}
            max={90}
            unit="d"
          />
        </>
      );

    case "auth":
      return (
        <>
          <ToggleSwitch
            label="Enable Authentication"
            checked={data.enabled as boolean}
            onChange={(v) => onChange("enabled", v)}
          />
          <SliderField
            label="Rate Limit"
            value={data.rate_limit_per_minute as number}
            onChange={(v) => onChange("rate_limit_per_minute", v)}
            min={10}
            max={500}
            unit="/min"
          />
          <div className="field-row readonly">
            <span>API Key</span>
            <code>{data.api_key ? "••••••••" + (data.api_key as string).slice(-8) : "Not set"}</code>
          </div>
        </>
      );

    case "vision":
      return (
        <>
          <ToggleSwitch
            label="Enable Vision System"
            checked={data.enabled as boolean}
            onChange={(v) => onChange("enabled", v)}
          />
          <ToggleSwitch
            label="Use Real YOLO Detector"
            checked={data.use_real_detector as boolean}
            onChange={(v) => onChange("use_real_detector", v)}
          />
          <TextField
            label="Model Path"
            value={data.model_path as string}
            onChange={(v) => onChange("model_path", v)}
            placeholder="yolov8n.pt"
          />
          <SelectField
            label="Device"
            value={data.device as string}
            onChange={(v) => onChange("device", v)}
            options={[
              { value: "auto", label: "Auto" },
              { value: "cpu", label: "CPU" },
              { value: "cuda", label: "CUDA (GPU)" },
              { value: "cuda:0", label: "CUDA:0" },
            ]}
          />
          <SliderField
            label="Confidence Threshold"
            value={data.confidence_threshold as number}
            onChange={(v) => onChange("confidence_threshold", v)}
            min={0.1}
            max={1.0}
            step={0.05}
          />
          <SliderField
            label="IOU Threshold"
            value={data.iou_threshold as number}
            onChange={(v) => onChange("iou_threshold", v)}
            min={0.1}
            max={1.0}
            step={0.05}
          />
          <ToggleSwitch
            label="Save Captured Images"
            checked={data.save_images as boolean}
            onChange={(v) => onChange("save_images", v)}
          />
        </>
      );

    case "simulation":
      return (
        <>
          <ToggleSwitch
            label="Enable Simulation Mode"
            checked={data.enabled as boolean}
            onChange={(v) => onChange("enabled", v)}
          />
          <ToggleSwitch
            label="Enable AirSim"
            checked={data.airsim_enabled as boolean}
            onChange={(v) => onChange("airsim_enabled", v)}
          />
          <TextField
            label="AirSim Host"
            value={data.airsim_host as string}
            onChange={(v) => onChange("airsim_host", v)}
          />
          <TextField
            label="Vehicle Name"
            value={data.airsim_vehicle_name as string}
            onChange={(v) => onChange("airsim_vehicle_name", v)}
          />
          <ToggleSwitch
            label="Enable ArduPilot SITL"
            checked={data.sitl_enabled as boolean}
            onChange={(v) => onChange("sitl_enabled", v)}
          />
          <TextField
            label="ArduPilot Path"
            value={data.ardupilot_path as string}
            onChange={(v) => onChange("ardupilot_path", v)}
          />
          <SliderField
            label="SITL Speed"
            value={data.sitl_speedup as number}
            onChange={(v) => onChange("sitl_speedup", v)}
            min={0.5}
            max={10}
            step={0.5}
            unit="x"
          />
          <TextField
            label="Home Latitude"
            value={data.home_latitude as number}
            type="number"
            onChange={(v) => onChange("home_latitude", parseFloat(v))}
          />
          <TextField
            label="Home Longitude"
            value={data.home_longitude as number}
            type="number"
            onChange={(v) => onChange("home_longitude", parseFloat(v))}
          />
        </>
      );

    case "agent":
      return (
        <>
          <ToggleSwitch
            label="Enable LLM Decision-Making"
            checked={data.use_llm as boolean}
            onChange={(v) => onChange("use_llm", v)}
          />
          <SelectField
            label="LLM Model"
            value={data.llm_model as string}
            onChange={(v) => onChange("llm_model", v)}
            options={[
              { value: "gpt-4o-mini", label: "GPT-4o Mini" },
              { value: "gpt-4o", label: "GPT-4o" },
              { value: "gpt-4-turbo", label: "GPT-4 Turbo" },
              { value: "claude-3-haiku-20240307", label: "Claude 3 Haiku" },
              { value: "claude-3-sonnet-20240229", label: "Claude 3 Sonnet" },
            ]}
          />
          <SliderField
            label="Battery Warning"
            value={data.battery_warning_percent as number}
            onChange={(v) => onChange("battery_warning_percent", v)}
            min={10}
            max={50}
            unit="%"
          />
          <SliderField
            label="Battery Critical"
            value={data.battery_critical_percent as number}
            onChange={(v) => onChange("battery_critical_percent", v)}
            min={5}
            max={30}
            unit="%"
          />
          <SliderField
            label="Wind Warning"
            value={data.wind_warning_ms as number}
            onChange={(v) => onChange("wind_warning_ms", v)}
            min={3}
            max={15}
            unit="m/s"
          />
          <SliderField
            label="Wind Abort"
            value={data.wind_abort_ms as number}
            onChange={(v) => onChange("wind_abort_ms", v)}
            min={5}
            max={20}
            unit="m/s"
          />
          <SliderField
            label="Decision Interval"
            value={data.decision_interval_seconds as number}
            onChange={(v) => onChange("decision_interval_seconds", v)}
            min={0.1}
            max={5}
            step={0.1}
            unit="s"
          />
        </>
      );

    case "dashboard":
      return (
        <>
          <SliderField
            label="Refresh Rate"
            value={data.refresh_rate_ms as number}
            onChange={(v) => onChange("refresh_rate_ms", v)}
            min={500}
            max={10000}
            step={500}
            unit="ms"
          />
          <SelectField
            label="Map Provider"
            value={data.map_provider as string}
            onChange={(v) => onChange("map_provider", v)}
            options={[
              { value: "openstreetmap", label: "OpenStreetMap" },
              { value: "mapbox", label: "Mapbox" },
              { value: "google", label: "Google Maps" },
            ]}
          />
          <ToggleSwitch
            label="Show Telemetry"
            checked={data.show_telemetry as boolean}
            onChange={(v) => onChange("show_telemetry", v)}
          />
          <ToggleSwitch
            label="Show Vision"
            checked={data.show_vision as boolean}
            onChange={(v) => onChange("show_vision", v)}
          />
          <ToggleSwitch
            label="Show Reasoning"
            checked={data.show_reasoning as boolean}
            onChange={(v) => onChange("show_reasoning", v)}
          />
          <SelectField
            label="Theme"
            value={data.theme as string}
            onChange={(v) => onChange("theme", v)}
            options={[
              { value: "dark", label: "Dark" },
              { value: "light", label: "Light" },
            ]}
          />
        </>
      );

    default:
      return <p>Unknown section</p>;
  }
};

const SettingsPanel = ({ onClose }: { onClose: () => void }) => {
  const [activeSection, setActiveSection] = useState<keyof ConfigSection>("agent");
  const [config, setConfig] = useState<ConfigSection | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const loadConfig = useCallback(async () => {
    try {
      setLoading(true);
      const resp = await fetch("/api/config");
      const data = await resp.json();
      setConfig(data.config);
    } catch {
      setMessage({ type: "error", text: "Failed to load configuration" });
    } finally {
      setLoading(false);
    }
  }, []);

  const validateConfig = useCallback(async () => {
    try {
      const resp = await fetch("/api/config/validate");
      const data = await resp.json();
      setValidation(data);
    } catch {
      setValidation({ valid: false, errors: ["Failed to validate"] });
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  useEffect(() => {
    if (config) {
      validateConfig();
    }
  }, [config, validateConfig]);

  const handleSectionChange = (key: string, value: unknown) => {
    if (!config) return;

    setConfig({
      ...config,
      [activeSection]: {
        ...config[activeSection],
        [key]: value,
      },
    });
  };

  const saveSection = async () => {
    if (!config) return;

    try {
      setSaving(true);
      setMessage(null);

      const resp = await fetch(`/api/config/${activeSection}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config[activeSection]),
      });

      if (resp.ok) {
        setMessage({ type: "success", text: `${SECTION_TITLES[activeSection]} updated` });
        validateConfig();
      } else {
        const error = await resp.json();
        setMessage({ type: "error", text: error.detail || "Update failed" });
      }
    } catch {
      setMessage({ type: "error", text: "Failed to save configuration" });
    } finally {
      setSaving(false);
    }
  };

  const saveToFile = async () => {
    try {
      setSaving(true);
      const resp = await fetch("/api/config/save", { method: "POST" });
      if (resp.ok) {
        setMessage({ type: "success", text: "Configuration saved to file" });
      } else {
        const error = await resp.json();
        setMessage({ type: "error", text: error.detail || "Save failed" });
      }
    } catch {
      setMessage({ type: "error", text: "Failed to save to file" });
    } finally {
      setSaving(false);
    }
  };

  const resetSection = async () => {
    try {
      const resp = await fetch(`/api/config/reset/${activeSection}`, { method: "POST" });
      if (resp.ok) {
        const data = await resp.json();
        setConfig((prev) =>
          prev ? { ...prev, [activeSection]: data.config } : null
        );
        setMessage({ type: "success", text: `${SECTION_TITLES[activeSection]} reset to defaults` });
      }
    } catch {
      setMessage({ type: "error", text: "Failed to reset section" });
    }
  };

  const generateApiKey = async () => {
    try {
      const resp = await fetch("/api/config/generate-api-key", { method: "POST" });
      if (resp.ok) {
        const data = await resp.json();
        const savedSuffix = data.saved ? "saved to config" : "save to config";
        setMessage({
          type: "success",
          text: `New API Key: ${data.api_key.slice(0, 16)}... (${savedSuffix})`,
        });
        loadConfig();
      }
    } catch {
      setMessage({ type: "error", text: "Failed to generate API key" });
    }
  };

  if (loading) {
    return (
      <div className="settings-overlay">
        <div className="settings-panel">
          <div className="settings-loading">Loading configuration...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <header className="settings-header">
          <h2>System Configuration</h2>
          <button className="close-btn" onClick={onClose}>
            ×
          </button>
        </header>

        <nav className="settings-nav">
          {(Object.keys(SECTION_TITLES) as (keyof ConfigSection)[]).map((key) => (
            <button
              key={key}
              className={`nav-item ${activeSection === key ? "active" : ""}`}
              onClick={() => setActiveSection(key)}
            >
              {SECTION_TITLES[key]}
            </button>
          ))}
        </nav>

        <div className="settings-content">
          {message && (
            <div className={`message ${message.type}`}>{message.text}</div>
          )}

          {validation && !validation.valid && (
            <div className="validation-errors">
              {validation.errors.map((err, i) => (
                <div key={i} className="error">{err}</div>
              ))}
            </div>
          )}

          <div className="section-form">
            <h3>{SECTION_TITLES[activeSection]}</h3>
            {config && (
              <SectionForm
                section={activeSection}
                data={config[activeSection] as unknown as Record<string, unknown>}
                onChange={handleSectionChange}
              />
            )}
          </div>
        </div>

        <footer className="settings-footer">
          <div className="footer-left">
            <button className="btn secondary" onClick={resetSection}>
              Reset Section
            </button>
            {activeSection === "auth" && (
              <button className="btn secondary" onClick={generateApiKey}>
                Generate API Key
              </button>
            )}
          </div>
          <div className="footer-right">
            <button className="btn secondary" onClick={saveToFile} disabled={saving}>
              Save to File
            </button>
            <button className="btn primary" onClick={saveSection} disabled={saving}>
              {saving ? "Saving..." : "Apply Changes"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default SettingsPanel;
