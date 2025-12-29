import { useCallback, useEffect, useMemo, useState } from "react";
import SettingsPanel from "./SettingsPanel";
import SpatialView from "./SpatialView";

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

type RunData = {
  run_id: string;
  summary: RunSummary;
  actions: ActionCount[];
  risk_series: SeriesPoint[];
  battery_series: SeriesPoint[];
  recent: RecentEntry[];
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
  uplink: {
    summary_only: boolean;
    send_images: boolean;
    max_images: number;
  };
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

      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
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

  useEffect(() => {
    loadRun();
    const timer = window.setInterval(loadRun, 5000);
    return () => window.clearInterval(timer);
  }, [loadRun]);

  return (
    <main className="dashboard-shell">
      <header className="dashboard-hero">
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
