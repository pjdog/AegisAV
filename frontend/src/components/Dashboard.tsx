import SettingsPanel from "./SettingsPanel";
import SpatialView from "./SpatialView";
import logo from "../assets/aegis_logo.svg";

type RunSummary = {
  total: number;
  avg_risk: number;
  max_risk: number;
  time_start: string | null;
  time_end: string | null;
};

type SeriesPoint = {
  t: string;
  value: number;
};

type ActionCount = {
  action: string;
  count: number;
};

type SpatialAsset = {
  id: string;
  type: string;
  x: number;
  y: number;
};

type RecentEntry = {
  timestamp: string;
  action: string;
  confidence: number;
  risk_level: string;
  battery_percent: number;
  reason?: string;
  vehicle_state?: {
    armed: boolean;
    mode: string;
  };
  spatial_context?: SpatialAsset[];
};

type LogEntry = {
  timestamp: string;
  level: string;
  name: string;
  message: string;
};

type TelemetryPayload = {
  timestamp?: string;
  battery?: {
    remaining_percent?: number;
    voltage?: number;
    current?: number;
  };
  mode?: string;
  armed?: boolean;
  position?: {
    latitude?: number;
    longitude?: number;
    altitude_msl?: number;
    altitude_agl?: number | null;
  };
};

type TelemetryEntry = {
  vehicle_id: string;
  telemetry: TelemetryPayload;
};

type RunData = {
  run_id: string;
  summary: RunSummary;
  actions: ActionCount[];
  risk_series: SeriesPoint[];
  battery_series: SeriesPoint[];
  recent: RecentEntry[];
};

type PerceptionConfig = {
  resolution_scale: number;
  frame_drop_probability: number;
  confidence_noise_std: number;
  missed_detection_probability: number;
};

type EnergyConfig = {
  capture_cost_percent: number;
  inference_cost_percent: number;
  uplink_cost_per_kb: number;
  idle_drain_per_second: number;
  thermal_throttle_after_s: number;
  thermal_throttle_factor: number;
};

type MicroAgentConfig = {
  burst_capture_on_anomaly: boolean;
  burst_capture_count: number;
  cache_and_forward: boolean;
  cache_max_items: number;
  local_abort_battery_threshold: number;
  local_abort_on_critical: boolean;
  priority_weighted_capture: boolean;
};

type UplinkConfig = {
  summary_only: boolean;
  send_images: boolean;
  max_images: number;
  uplink_delay_ms: number;
  max_payload_bytes: number;
  drop_probability: number;
};

type EdgeComputeConfig = {
  profile: string;
  vision_enabled: boolean;
  capture_interval_s: number;
  max_captures_per_inspection: number;
  simulated_inference_delay_ms: number;
  client_confidence_threshold: number;
  anomaly_gate: {
    mode: string;
    min_confidence: number;
    min_severity: number;
    n?: number | null;
    m?: number | null;
    min_severity_override?: number | null;
  };
  uplink: UplinkConfig;
  perception: PerceptionConfig;
  energy: EnergyConfig;
  micro_agent: MicroAgentConfig;
};

const LogViewer = ({ logs }: { logs: LogEntry[] }) => {
  return (
    <div className="log-card">
      <div className="card-header">
        <h2>Live System Logs</h2>
        <span className="pill">Console</span>
      </div>
      <div className="log-viewer">
        {logs.map((log, i) => (
          <div key={i} className="log-entry">
            <span className="ts">{log.timestamp.split('.')[0]}</span>
            <span className={`level ${log.level}`}>{log.level}</span>
            <span className="msg">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const formatNumber = (value?: number, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
};

const formatTime = (value?: string | null) => {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

const formatCoord = (value?: number) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(5);
};

const batteryTone = (value?: number) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "ok";
  if (value <= 20) return "critical";
  if (value <= 35) return "warn";
  return "ok";
};

const buildPath = (points: SeriesPoint[], width: number, height: number, pad = 16) => {
  if (!points.length) return "";
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const xStep = (width - pad * 2) / Math.max(points.length - 1, 1);

  return points
    .map((point, index) => {
      const x = pad + xStep * index;
      const y = pad + (height - pad * 2) * (1 - (point.value - min) / range);
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
};

const LineChart = ({
  title,
  series,
  accent,
}: {
  title: string;
  series: SeriesPoint[];
  accent: string;
}) => {
  const width = 560;
  const height = 180;
  const path = useMemo(() => buildPath(series, width, height), [series]);

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h2>{title}</h2>
        <span className="chart-tag">{series.length} pts</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <defs>
          <linearGradient id={`${title}-gate`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={accent} stopOpacity="0.2" />
            <stop offset="100%" stopColor={accent} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={path} fill="none" stroke={accent} strokeWidth="2" strokeDasharray={series.length < 2 ? "4" : "0"} />
        {path && (
          <path
            d={`${path} L ${width - 16} ${height - 16} L 16 ${height - 16} Z`}
            fill={`url(#${title}-gate)`}
          />
        )}
      </svg>
    </div>
  );
};

const FleetTelemetry = ({ telemetry }: { telemetry: TelemetryEntry[] }) => {
  return (
    <section className="card fleet-section">
      <div className="card-header">
        <h2>Fleet Telemetry</h2>
        <span className="pill">{telemetry.length} drones</span>
      </div>
      {telemetry.length === 0 ? (
        <p className="note">No telemetry received yet.</p>
      ) : (
        <div className="fleet-grid">
          {telemetry.map((entry) => {
            const payload = entry.telemetry || {};
            const battery = payload.battery?.remaining_percent;
            return (
              <div key={entry.vehicle_id} className="fleet-card">
                <div className="fleet-header">
                  <div>
                    <h3 className="fleet-id">{entry.vehicle_id}</h3>
                    <span className="fleet-tag">{payload.mode ?? "UNKNOWN"}</span>
                  </div>
                  <div className={`fleet-battery ${batteryTone(battery)}`}>
                    {battery !== undefined ? `${battery.toFixed(0)}%` : "--"}
                  </div>
                </div>
                <div className="fleet-metrics">
                  <div className="fleet-metric">
                    <span className="fleet-label">Armed</span>
                    <span className="fleet-value">{payload.armed ? "YES" : "NO"}</span>
                  </div>
                  <div className="fleet-metric">
                    <span className="fleet-label">Altitude</span>
                    <span className="fleet-value">
                      {formatNumber(payload.position?.altitude_msl, 1)}m
                    </span>
                  </div>
                  <div className="fleet-metric">
                    <span className="fleet-label">Latitude</span>
                    <span className="fleet-value">{formatCoord(payload.position?.latitude)}</span>
                  </div>
                  <div className="fleet-metric">
                    <span className="fleet-label">Longitude</span>
                    <span className="fleet-value">{formatCoord(payload.position?.longitude)}</span>
                  </div>
                </div>
                <div className="fleet-footer">
                  <span className="fleet-status">Last update</span>
                  <span className="fleet-status">{formatTime(payload.timestamp)}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};

// Compute budget score (0-100) based on config complexity
const computeBudgetScore = (config: EdgeComputeConfig | null): number => {
  if (!config) return 0;
  const perception = config.perception;
  const energy = config.energy;
  const uplink = config.uplink;

  // Higher resolution = more compute
  const resolutionCost = perception.resolution_scale * 30;
  // More captures = more energy
  const captureCost = config.max_captures_per_inspection * 2;
  // Inference delay indicates complexity
  const inferenceCost = Math.min(config.simulated_inference_delay_ms / 10, 20);
  // Uplink costs
  const uplinkCost = uplink.send_images ? 15 : 5;
  // Energy drain rate
  const energyCost = energy.idle_drain_per_second * 200;

  return Math.min(100, Math.round(resolutionCost + captureCost + inferenceCost + uplinkCost + energyCost));
};

// Bandwidth budget score (0-100) based on uplink config
const bandwidthBudgetScore = (config: EdgeComputeConfig | null): number => {
  if (!config) return 0;
  const uplink = config.uplink;

  // Images use most bandwidth
  const imageCost = uplink.send_images ? uplink.max_images * 20 : 0;
  // Delay indicates bandwidth constraints
  const delayCost = Math.min(uplink.uplink_delay_ms / 50, 30);
  // Drop probability indicates poor link
  const dropCost = uplink.drop_probability * 200;
  // Summary only is low bandwidth
  const summaryCost = uplink.summary_only ? 0 : 20;

  return Math.min(100, Math.round(imageCost + delayCost + dropCost + summaryCost));
};

type AdvancedOverridesProps = {
  config: EdgeComputeConfig | null;
  onUpdate: (updates: Record<string, unknown>) => Promise<void>;
  isExpanded: boolean;
  onToggle: () => void;
};

const AdvancedOverrides = ({ config, onUpdate, isExpanded, onToggle }: AdvancedOverridesProps) => {
  if (!config) return null;

  const computeScore = computeBudgetScore(config);
  const bandwidthScore = bandwidthBudgetScore(config);

  const handleSlider = (section: string, field: string, value: number) => {
    onUpdate({ [section]: { [field]: value } });
  };

  const handleToggle = (section: string, field: string, value: boolean) => {
    onUpdate({ [section]: { [field]: value } });
  };

  return (
    <div className="advanced-overrides">
      <div className="advanced-header" onClick={onToggle}>
        <span className="advanced-title">Advanced Edge Overrides</span>
        <div className="budget-indicators">
          <div className="budget-item">
            <span className="budget-label">Compute</span>
            <div className="budget-bar">
              <div
                className="budget-fill compute"
                style={{ width: `${computeScore}%` }}
              />
            </div>
            <span className="budget-value">{computeScore}%</span>
          </div>
          <div className="budget-item">
            <span className="budget-label">Bandwidth</span>
            <div className="budget-bar">
              <div
                className="budget-fill bandwidth"
                style={{ width: `${bandwidthScore}%` }}
              />
            </div>
            <span className="budget-value">{bandwidthScore}%</span>
          </div>
        </div>
        <span className="advanced-chevron">{isExpanded ? "−" : "+"}</span>
      </div>

      {isExpanded && (
        <div className="advanced-content">
          {/* Perception Quality */}
          <div className="override-section">
            <h4>Perception Quality</h4>
            <div className="override-row">
              <span>Resolution Scale</span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={config.perception.resolution_scale}
                onChange={(e) => handleSlider("perception", "resolution_scale", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.perception.resolution_scale * 100).toFixed(0)}%</span>
            </div>
            <div className="override-row">
              <span>Frame Drop Prob</span>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                value={config.perception.frame_drop_probability}
                onChange={(e) => handleSlider("perception", "frame_drop_probability", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.perception.frame_drop_probability * 100).toFixed(0)}%</span>
            </div>
            <div className="override-row">
              <span>Confidence Noise</span>
              <input
                type="range"
                min="0"
                max="0.3"
                step="0.01"
                value={config.perception.confidence_noise_std}
                onChange={(e) => handleSlider("perception", "confidence_noise_std", parseFloat(e.target.value))}
              />
              <span className="override-value">{config.perception.confidence_noise_std.toFixed(2)}</span>
            </div>
            <div className="override-row">
              <span>Missed Detection</span>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                value={config.perception.missed_detection_probability}
                onChange={(e) => handleSlider("perception", "missed_detection_probability", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.perception.missed_detection_probability * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Energy Costs */}
          <div className="override-section">
            <h4>Energy Model</h4>
            <div className="override-row">
              <span>Capture Cost</span>
              <input
                type="range"
                min="0"
                max="0.2"
                step="0.005"
                value={config.energy.capture_cost_percent}
                onChange={(e) => handleSlider("energy", "capture_cost_percent", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.energy.capture_cost_percent * 100).toFixed(1)}%</span>
            </div>
            <div className="override-row">
              <span>Inference Cost</span>
              <input
                type="range"
                min="0"
                max="0.2"
                step="0.005"
                value={config.energy.inference_cost_percent}
                onChange={(e) => handleSlider("energy", "inference_cost_percent", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.energy.inference_cost_percent * 100).toFixed(1)}%</span>
            </div>
            <div className="override-row">
              <span>Uplink Cost/KB</span>
              <input
                type="range"
                min="0"
                max="0.01"
                step="0.0005"
                value={config.energy.uplink_cost_per_kb}
                onChange={(e) => handleSlider("energy", "uplink_cost_per_kb", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.energy.uplink_cost_per_kb * 1000).toFixed(1)}‰</span>
            </div>
            <div className="override-row">
              <span>Idle Drain/s</span>
              <input
                type="range"
                min="0"
                max="0.1"
                step="0.005"
                value={config.energy.idle_drain_per_second}
                onChange={(e) => handleSlider("energy", "idle_drain_per_second", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.energy.idle_drain_per_second * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Uplink Simulation */}
          <div className="override-section">
            <h4>Uplink Simulation</h4>
            <div className="override-row">
              <span>Uplink Delay</span>
              <input
                type="range"
                min="0"
                max="2000"
                step="50"
                value={config.uplink.uplink_delay_ms}
                onChange={(e) => handleSlider("uplink", "uplink_delay_ms", parseInt(e.target.value))}
              />
              <span className="override-value">{config.uplink.uplink_delay_ms}ms</span>
            </div>
            <div className="override-row">
              <span>Drop Probability</span>
              <input
                type="range"
                min="0"
                max="0.2"
                step="0.01"
                value={config.uplink.drop_probability}
                onChange={(e) => handleSlider("uplink", "drop_probability", parseFloat(e.target.value))}
              />
              <span className="override-value">{(config.uplink.drop_probability * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Micro-Agent Behaviors */}
          <div className="override-section">
            <h4>Micro-Agent Behaviors</h4>
            <div className="override-toggle-row" onClick={() => handleToggle("micro_agent", "burst_capture_on_anomaly", !config.micro_agent.burst_capture_on_anomaly)}>
              <span>Burst Capture on Anomaly</span>
              <div className={`toggle ${config.micro_agent.burst_capture_on_anomaly ? "on" : ""}`}>
                <div className="toggle-thumb" />
              </div>
            </div>
            <div className="override-toggle-row" onClick={() => handleToggle("micro_agent", "cache_and_forward", !config.micro_agent.cache_and_forward)}>
              <span>Cache &amp; Forward</span>
              <div className={`toggle ${config.micro_agent.cache_and_forward ? "on" : ""}`}>
                <div className="toggle-thumb" />
              </div>
            </div>
            <div className="override-toggle-row" onClick={() => handleToggle("micro_agent", "local_abort_on_critical", !config.micro_agent.local_abort_on_critical)}>
              <span>Local Abort on Critical</span>
              <div className={`toggle ${config.micro_agent.local_abort_on_critical ? "on" : ""}`}>
                <div className="toggle-thumb" />
              </div>
            </div>
            <div className="override-toggle-row" onClick={() => handleToggle("micro_agent", "priority_weighted_capture", !config.micro_agent.priority_weighted_capture)}>
              <span>Priority Weighted Capture</span>
              <div className={`toggle ${config.micro_agent.priority_weighted_capture ? "on" : ""}`}>
                <div className="toggle-thumb" />
              </div>
            </div>
            <div className="override-row">
              <span>Abort Battery Threshold</span>
              <input
                type="range"
                min="0"
                max="30"
                step="1"
                value={config.micro_agent.local_abort_battery_threshold}
                onChange={(e) => handleSlider("micro_agent", "local_abort_battery_threshold", parseFloat(e.target.value))}
              />
              <span className="override-value">{config.micro_agent.local_abort_battery_threshold.toFixed(0)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const Dashboard = () => {
  const [runId, setRunId] = useState<string | null>(null);
  const [runData, setRunData] = useState<RunData | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [status, setStatus] = useState<string>("Initializing");
  const [lastUpdated, setLastUpdated] = useState<string>("--");
  const [useAdvancedEngine, setUseAdvancedEngine] = useState<boolean>(true);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [edgeConfig, setEdgeConfig] = useState<EdgeComputeConfig | null>(null);
  const [edgeProfiles, setEdgeProfiles] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [fleetTelemetry, setFleetTelemetry] = useState<TelemetryEntry[]>([]);

  const loadRun = useCallback(async () => {
    try {
      // Load runs and data
      const runsResp = await fetch("/api/dashboard/runs");
      const runs = await runsResp.json();
      if (runs.latest) {
        setRunId(runs.latest);
        const dataResp = await fetch(`/api/dashboard/run/${runs.latest}`);
        const data = (await dataResp.json()) as RunData;
        setRunData(data);
        setStatus("Active");
      } else {
        setStatus("No Runs Found");
      }

      // Load agent config
      const configResp = await fetch("/api/config/agent");
      const configData = await configResp.json();
      setUseAdvancedEngine(configData.use_advanced_engine);

      // Load edge compute config
      const edgeResp = await fetch("/api/config/edge");
      const edgeData = await edgeResp.json();
      setEdgeConfig(edgeData.edge_config ?? null);
      setEdgeProfiles(edgeData.profiles ?? []);

      // Load live logs
      const logsResp = await fetch("/api/logs");
      const logsData = await logsResp.json();
      setLogs(logsData.logs || []);

      const telemetryResp = await fetch("/api/telemetry/latest");
      if (telemetryResp.ok) {
        const telemetryData = await telemetryResp.json();
        setFleetTelemetry(telemetryData.vehicles || []);
      }

      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error("Failed to load dashboard data", error);
      setStatus("Offline");
    }
  }, []);

  const toggleLLM = async () => {
    try {
      const next = !useAdvancedEngine;
      await fetch("/api/config/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_advanced_engine: next }),
      });
      setUseAdvancedEngine(next);
    } catch (error) {
      console.error("Failed to toggle LLM", error);
    }
  };

  const updateEdgeProfile = async (profile: string) => {
    try {
      const resp = await fetch("/api/config/edge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile }),
      });
      const data = await resp.json();
      setEdgeConfig(data.edge_config ?? null);
    } catch (error) {
      console.error("Failed to update edge profile", error);
    }
  };

  const updateEdgeOverrides = async (updates: Record<string, unknown>) => {
    try {
      const resp = await fetch("/api/config/edge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      const data = await resp.json();
      setEdgeConfig(data.edge_config ?? null);
    } catch (error) {
      console.error("Failed to update edge config", error);
    }
  };

  import logo from "../assets/aegis_logo.svg";

  useEffect(() => {
    loadRun();
    const timer = window.setInterval(loadRun, 5000);
    return () => window.clearInterval(timer);
  }, [loadRun]);

  return (
    <main className="dashboard-shell">
      <header className="dashboard-hero">
        <div style={{ marginRight: "24px" }}>
          <img src={logo} alt="AegisAV Shield" width="64" height="64" />
        </div>
        <div>
          <p className="eyebrow">Aegis Onyx Interface</p>
          <h1>Mission Autonomy Monitor</h1>
          <p className="subtitle">
            Observing high-fidelity mission decisions and situational awareness in a realistic SITL environment.
          </p>
        </div>
        <div className="hero-meta">
          <div className="hero-card action" onClick={toggleLLM} style={{ cursor: 'pointer', borderColor: useAdvancedEngine ? 'var(--accent-cyber)' : 'var(--text-muted)' }}>
            <span className="meta-label">Orchestration</span>
            <span className="meta-value">{useAdvancedEngine ? "LLM ACTIVE" : "RULES ONLY"}</span>
          </div>
          <div className="hero-card">
            <span className="meta-label">Edge Compute</span>
            <select
              className="meta-select"
              value={edgeConfig?.profile ?? ""}
              onChange={(e) => updateEdgeProfile(e.target.value)}
              disabled={!edgeProfiles.length}
            >
              {(edgeProfiles.length ? edgeProfiles : [edgeConfig?.profile ?? ""]).map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <span className="meta-sub">
              {edgeConfig
                ? `${edgeConfig.max_captures_per_inspection} caps • ${edgeConfig.simulated_inference_delay_ms}ms`
                : "--"}
            </span>
          </div>
          <div className="hero-card">
            <span className="meta-label">Mission ID</span>
            <span className="meta-value">{runId ?? "--"}</span>
          </div>
          <div className="hero-card">
            <span className="meta-label">Agent State</span>
            <span className="meta-value" style={{ color: status === 'Active' ? 'var(--accent-security)' : 'var(--accent-alert)' }}>
              {status}
            </span>
          </div>
          <div className="hero-card">
            <span className="meta-label">Last Sync</span>
            <span className="meta-value">{lastUpdated}</span>
          </div>
          <button className="settings-btn" onClick={() => setShowSettings(true)} title="Settings">
            ⚙
          </button>
        </div>
      </header>

      {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}

      <AdvancedOverrides
        config={edgeConfig}
        onUpdate={updateEdgeOverrides}
        isExpanded={showAdvanced}
        onToggle={() => setShowAdvanced(!showAdvanced)}
      />

      <section className="grid">
        <article className="card primary">
          <h2>Mission Integrity</h2>
          <p className="stat">{runData?.summary.total ?? "0"}</p>
          <p className="note">Total autonomous decisions evaluated.</p>
        </article>
        <article className="card">
          <h2>Risk Baseline</h2>
          <p className="stat">{formatNumber(runData?.summary.avg_risk, 1)}</p>
          <p className="note">Mean mission risk index.</p>
        </article>
        <article className="card">
          <h2>Critical Threshold</h2>
          <p className="stat" style={{ color: (runData?.summary.max_risk ?? 0) > 5.0 ? 'var(--accent-alert)' : 'var(--text-primary)' }}>
            {formatNumber(runData?.summary.max_risk, 1)}
          </p>
          <p className="note">Maximum risk encountered.</p>
        </article>
        <article className="card">
          <h2>Energy Pulse</h2>
          <p className="stat">{formatNumber(runData?.recent[0]?.battery_percent, 0)}%</p>
          <p className="note">Current primary power state.</p>
        </article>
      </section>

      <FleetTelemetry telemetry={fleetTelemetry} />

      <section className="split">
        <LineChart
          title="Risk Vector"
          series={runData?.risk_series ?? []}
          accent="#00f2ff"
        />
        <LineChart
          title="Energy Decay"
          series={runData?.battery_series ?? []}
          accent="#00ff9d"
        />
      </section>

      <section className="split">
        <SpatialView
          assets={runData?.recent[0]?.spatial_context || []}
        />
        <article className="card">
          <div className="card-header">
            <h2>Reasoning Feed</h2>
            <span className="pill">Live Timeline</span>
          </div>
          <div className="feed">
            {(runData?.recent ?? []).slice(0, 6).map((entry, i) => (
              <div key={i} className="feed-item" style={{ borderColor: i === 0 ? 'var(--accent-cyber)' : 'rgba(255,255,255,0.1)' }}>
                <span className="feed-time">{formatTime(entry.timestamp)}</span>
                <div className="feed-content">
                  <h4>{entry.action}</h4>
                  <p className="feed-reason">{entry.reason || "Autonomous adjustment based on current mission parameters."}</p>
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="grid" style={{ gridTemplateColumns: '1fr' }}>
        <LogViewer logs={logs} />
      </section>
    </main>
  );
};

export default Dashboard;
