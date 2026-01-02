import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import SettingsPanel from "./SettingsPanel";
import SpatialView from "./SpatialView";
import DroneCameraPanel from "./DroneCameraPanel";
import SplatMapViewer from "./SplatMapViewer";
import ReasoningFeed, { DecisionEntry } from "./ReasoningFeed";
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

type AirSimStatus = {
  enabled: boolean;
  host: string;
  vehicle_name: string;
  bridge_connected: boolean;
  connecting: boolean;
  launch_supported: boolean;
  last_error: string | null;
};

type VisionDetection = {
  detection_class?: string;
  confidence?: number;
  severity?: number;
};

type VisionObservationSummary = {
  asset_id?: string;
  timestamp?: string;
  observation_id?: string;
  defect_detected?: boolean;
  anomaly_created?: boolean;
  max_confidence?: number;
  max_severity?: number;
  detections?: VisionDetection[];
  defect_classes?: string[];
};

type MissionMetrics = {
  total_assets: number;
  assets_inspected: number;
  inspection_coverage_percent: number;
  assets_in_progress: string[];
  total_anomalies_detected: number;
  total_anomalies_expected?: number;
  anomalies_resolved: number;
  anomalies_pending: number;
  defects_by_type: Record<string, number>;
  decisions_total: number;
  decisions_successful: number;
  success_rate_percent: number;
  decision_quality_percent?: number;
  execution_success_percent?: number;
  anomaly_handling_percent?: number;
  resource_use_percent?: number;
  resource_use_available?: boolean;
  battery_consumed?: number;
  battery_budget?: number;
  mission_success_score?: number;
  mission_success_grade?: string;
  mission_success_components?: Record<string, number>;
  mission_completion_percent: number;
  recent_captures_count: number;
  recent_defects_count: number;
};

type MissionSuccessWeights = {
  coverage: number;
  anomaly: number;
  decision_quality: number;
  execution: number;
  resource_use: number;
};

type MissionSuccessThresholds = {
  excellent: number;
  good: number;
  fair: number;
};

type MissionSuccessConfig = {
  weights: MissionSuccessWeights;
  thresholds: MissionSuccessThresholds;
};

const normalizeMetrics = (data: Partial<MissionMetrics> | null | undefined): MissionMetrics => ({
  total_assets: data?.total_assets ?? 0,
  assets_inspected: data?.assets_inspected ?? 0,
  inspection_coverage_percent: data?.inspection_coverage_percent ?? 0,
  assets_in_progress: data?.assets_in_progress ?? [],
  total_anomalies_detected: data?.total_anomalies_detected ?? 0,
  anomalies_resolved: data?.anomalies_resolved ?? 0,
  anomalies_pending: data?.anomalies_pending ?? 0,
  defects_by_type: data?.defects_by_type ?? {},
  decisions_total: data?.decisions_total ?? 0,
  decisions_successful: data?.decisions_successful ?? 0,
  success_rate_percent: data?.success_rate_percent ?? 0,
  mission_completion_percent: data?.mission_completion_percent ?? 0,
  recent_captures_count: data?.recent_captures_count ?? 0,
  recent_defects_count: data?.recent_defects_count ?? 0,
});

type RecentCapture = {
  observation_id: string;
  asset_id: string;
  asset_name: string;
  timestamp: string;
  image_url: string | null;
  thumbnail_url: string | null;
  defect_detected: boolean;
  detections: Array<{ class: string; confidence: number; severity: number }>;
  detection_count: number;
  max_confidence: number;
  max_severity: number;
  anomaly_created: boolean;
  anomaly_id: string | null;
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

// Success Metrics Section - displays inspection coverage, anomaly detection, success rates
const SuccessMetricsSection = ({ metrics }: { metrics: MissionMetrics | null }) => {
  if (!metrics) {
    return (
      <section className="grid metrics-section">
        <article className="card">
          <h2>Mission Metrics</h2>
          <p className="note">Loading metrics...</p>
        </article>
      </section>
    );
  }

  const safeMetrics = normalizeMetrics(metrics);
  const coveragePercent = safeMetrics.inspection_coverage_percent;
  const coverageColor = coveragePercent >= 80
    ? 'var(--accent-security)'
    : coveragePercent >= 50
      ? 'var(--accent-warning)'
      : 'var(--text-primary)';

  const decisionSuccess = safeMetrics.success_rate_percent;
  const missionSuccess = metrics.mission_success_score ?? decisionSuccess;
  const missionGrade = metrics.mission_success_grade ?? "UNKNOWN";
  const resourceUseAvailable = Boolean(metrics.resource_use_available);
  const resourceUse = metrics.resource_use_percent ?? 0;
  const anomalyHandling = metrics.anomaly_handling_percent ?? 0;

  const missionColor = missionSuccess >= 85
    ? 'var(--accent-security)'
    : missionSuccess >= 70
      ? 'var(--accent-warning)'
      : 'var(--accent-alert)';


  return (
    <section className="grid metrics-section">
      <article className="card primary">
        <h2>Inspection Coverage</h2>
        <p className="stat" style={{ color: coverageColor }}>
          {coveragePercent.toFixed(0)}%
        </p>
        <p className="note">
          {safeMetrics.assets_inspected} / {safeMetrics.total_assets} assets inspected
        </p>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{
              width: `${Math.min(100, coveragePercent)}%`,
              background: 'linear-gradient(90deg, var(--accent-cyber), var(--accent-security))',
            }}
          />
        </div>
      </article>

      <article className="card">
        <h2>Anomaly Detection</h2>
        <p className="stat">{safeMetrics.total_anomalies_detected}</p>
        <p className="note">
          {safeMetrics.anomalies_resolved} resolved · {safeMetrics.anomalies_pending} pending
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '8px' }}>
          {Object.entries(safeMetrics.defects_by_type).slice(0, 4).map(([type, count]) => (
            <span key={type} className="pill defect-pill">
              {type}: {count}
            </span>
          ))}
        </div>
      </article>

      <article className="card">
        <h2>Mission Success</h2>
        <p className="stat" style={{ color: missionColor }}>
          {missionSuccess.toFixed(0)}%
        </p>
        <p className="note">
          {missionGrade} · Decision {decisionSuccess.toFixed(0)}%
        </p>
        <p className="note" style={{ marginTop: '4px' }}>
          Anomaly handling {anomalyHandling.toFixed(0)}% ·
          {resourceUseAvailable ? ` Resource ${resourceUse.toFixed(0)}%` : " Resource --"}
        </p>
      </article>

      <article className="card">
        <h2>Mission Progress</h2>
        <p className="stat">{safeMetrics.mission_completion_percent.toFixed(0)}%</p>
        <p className="note">Overall completion</p>
        {safeMetrics.assets_in_progress.length > 0 && (
          <p className="note" style={{ marginTop: '4px', color: 'var(--accent-cyber)' }}>
            {safeMetrics.assets_in_progress.length} in progress
          </p>
        )}
      </article>
    </section>
  );
};

// Recent Captures Section - displays captured images with detection info
const RecentCapturesSection = ({ captures }: { captures: RecentCapture[] }) => {
  if (captures.length === 0) {
    return (
      <section className="card captures-section">
        <div className="card-header">
          <h2>Recent Captures</h2>
          <span className="pill">0 captures</span>
        </div>
        <p className="note">No captures yet. Start a scenario to begin inspections.</p>
      </section>
    );
  }

  return (
    <section className="card captures-section">
      <div className="card-header">
        <h2>Recent Captures</h2>
        <span className="pill">{captures.length} captures</span>
      </div>
      <div className="captures-grid">
        {captures.map((capture) => (
          <div
            key={capture.observation_id}
            className={`capture-card ${capture.defect_detected ? 'has-defect' : ''}`}
          >
            <div className="capture-thumbnail">
              {capture.thumbnail_url ? (
                <img src={capture.thumbnail_url} alt={`Capture of ${capture.asset_name}`} />
              ) : (
                <div className="capture-placeholder">No Image</div>
              )}
              {capture.defect_detected && (
                <span className="defect-badge">!</span>
              )}
            </div>
            <div className="capture-info">
              <span className="capture-asset">{capture.asset_name}</span>
              <span className="capture-time">{formatTime(capture.timestamp)}</span>
              {capture.detections.length > 0 && (
                <div className="capture-detections">
                  {capture.detections.slice(0, 2).map((det, i) => (
                    <span key={i} className="detection-pill">
                      {det.class} ({(det.confidence * 100).toFixed(0)}%)
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
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

const toPercent = (value: number) => Math.round(value * 100);
const toRatio = (value: number) => value / 100;

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

type MissionSuccessControlsProps = {
  config: MissionSuccessConfig | null;
  onChange: (config: MissionSuccessConfig) => void;
  onApply: () => void;
  onReload: () => void;
  saving: boolean;
  message: string | null;
  isExpanded: boolean;
  onToggle: () => void;
};

const MissionSuccessControls = ({
  config,
  onChange,
  onApply,
  onReload,
  saving,
  message,
  isExpanded,
  onToggle,
}: MissionSuccessControlsProps) => {
  if (!config) return null;

  const weightSum = Object.values(config.weights).reduce((sum, value) => sum + value, 0);
  const sumLabel = `${weightSum.toFixed(0)}%`;
  const sumClass = Math.abs(weightSum - 100) < 1 ? "ok" : "warn";

  const updateWeight = (key: keyof MissionSuccessWeights, value: number) => {
    onChange({
      ...config,
      weights: {
        ...config.weights,
        [key]: value,
      },
    });
  };

  const updateThreshold = (key: keyof MissionSuccessThresholds, value: number) => {
    onChange({
      ...config,
      thresholds: {
        ...config.thresholds,
        [key]: value,
      },
    });
  };

  return (
    <div className="mission-controls">
      <div className="advanced-header" onClick={onToggle}>
        <span className="advanced-title">Mission Success Tuning</span>
        <span className={`weight-sum ${sumClass}`}>Weights {sumLabel}</span>
        <span className="advanced-chevron">{isExpanded ? "−" : "+"}</span>
      </div>

      {isExpanded && (
        <>
          <div className="advanced-content">
            <div className="override-section">
              <h4>Weights</h4>
              <div className="override-row">
                <span>Coverage</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.weights.coverage}
                  onChange={(e) => updateWeight("coverage", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.weights.coverage.toFixed(0)}%</span>
              </div>
              <div className="override-row">
                <span>Anomaly Handling</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.weights.anomaly}
                  onChange={(e) => updateWeight("anomaly", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.weights.anomaly.toFixed(0)}%</span>
              </div>
              <div className="override-row">
                <span>Decision Quality</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.weights.decision_quality}
                  onChange={(e) => updateWeight("decision_quality", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.weights.decision_quality.toFixed(0)}%</span>
              </div>
              <div className="override-row">
                <span>Execution</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.weights.execution}
                  onChange={(e) => updateWeight("execution", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.weights.execution.toFixed(0)}%</span>
              </div>
              <div className="override-row">
                <span>Resource Use</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.weights.resource_use}
                  onChange={(e) => updateWeight("resource_use", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.weights.resource_use.toFixed(0)}%</span>
              </div>
            </div>

            <div className="override-section">
              <h4>Grades</h4>
              <div className="override-row">
                <span>Excellent ≥</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.thresholds.excellent}
                  onChange={(e) => updateThreshold("excellent", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.thresholds.excellent.toFixed(0)}%</span>
              </div>
              <div className="override-row">
                <span>Good ≥</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.thresholds.good}
                  onChange={(e) => updateThreshold("good", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.thresholds.good.toFixed(0)}%</span>
              </div>
              <div className="override-row">
                <span>Fair ≥</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={config.thresholds.fair}
                  onChange={(e) => updateThreshold("fair", parseFloat(e.target.value))}
                />
                <span className="override-value">{config.thresholds.fair.toFixed(0)}%</span>
              </div>
            </div>
          </div>

          <div className="mission-actions">
            <span className="mission-status">{message ?? ""}</span>
            <div className="mission-buttons">
              <button
                className="btn secondary"
                type="button"
                onClick={onReload}
                disabled={saving}
              >
                Reset
              </button>
              <button
                className="btn primary"
                type="button"
                onClick={onApply}
                disabled={saving}
              >
                {saving ? "Applying..." : "Apply"}
              </button>
            </div>
          </div>
        </>
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
  const [airsimStatus, setAirsimStatus] = useState<AirSimStatus | null>(null);
  const [airsimLaunching, setAirsimLaunching] = useState<boolean>(false);
  const [lastVision, setLastVision] = useState<VisionObservationSummary | null>(null);
  const [scenarioDrones, setScenarioDrones] = useState<Array<{
    drone_id: string;
    name: string;
    battery_percent: number;
    state: string;
    is_streaming: boolean;
  }>>([]);
  const [liveRunnerStatus, setLiveRunnerStatus] = useState<{
    is_running: boolean;
    run_id: string | null;
    scenario_id: string | null;
    scenario_name: string | null;
    decision_count: number;
    elapsed_seconds: number;
  } | null>(null);
  const [spatialAssets, setSpatialAssets] = useState<SpatialAsset[]>([]);
  const [missionMetrics, setMissionMetrics] = useState<MissionMetrics | null>(null);
  const [recentCaptures, setRecentCaptures] = useState<RecentCapture[]>([]);
  const [missionConfigDraft, setMissionConfigDraft] = useState<MissionSuccessConfig | null>(null);
  const [missionConfigSaving, setMissionConfigSaving] = useState<boolean>(false);
  const [missionConfigMessage, setMissionConfigMessage] = useState<string | null>(null);
  const [missionConfigExpanded, setMissionConfigExpanded] = useState<boolean>(false);
  // Live decisions from WebSocket
  const [liveDecisions, setLiveDecisions] = useState<DecisionEntry[]>([]);
  const [runnerDecisions, setRunnerDecisions] = useState<DecisionEntry[]>([]);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);

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

      const airsimResp = await fetch("/api/airsim/status");
      if (airsimResp.ok) {
        const airsimData = await airsimResp.json();
        setAirsimStatus(airsimData);
      }

      const visionResp = await fetch("/api/vision/last");
      if (visionResp.ok) {
        const visionData = await visionResp.json();
        setLastVision(visionData?.observation ?? null);
      }

      // Load scenario drones for camera panel
      const scenarioResp = await fetch("/api/scenarios/status");
      if (scenarioResp.ok) {
        const scenarioData = await scenarioResp.json();
        if (scenarioData.running && scenarioData.scenario_id) {
          // Fetch scenario details to get drone list
          const detailResp = await fetch(`/api/scenarios/${scenarioData.scenario_id}`);
          if (detailResp.ok) {
            const detailData = await detailResp.json();
            const drones = (detailData.drones || []).map((d: {
              drone_id: string;
              name: string;
              battery_percent: number;
              state: string;
            }) => ({
              drone_id: d.drone_id,
              name: d.name,
              battery_percent: d.battery_percent,
              state: d.state,
              is_streaming: false,
            }));
            setScenarioDrones(drones);
          }
        } else {
          // No scenario running - use telemetry vehicles as fallback
          const dronesFromTelemetry = fleetTelemetry.map((v) => ({
            drone_id: v.vehicle_id,
            name: v.vehicle_id,
            battery_percent: v.telemetry?.battery?.remaining_percent ?? 0,
            state: v.telemetry?.mode ?? "unknown",
            is_streaming: false,
          }));
          setScenarioDrones(dronesFromTelemetry);
        }
      }

      // Fetch live runner status for real-time decision counts
      const runnerResp = await fetch("/api/dashboard/runner/status");
      if (runnerResp.ok) {
        const runnerData = await runnerResp.json();
        setLiveRunnerStatus({
          is_running: runnerData.is_running ?? false,
          run_id: runnerData.run_id ?? null,
          scenario_id: runnerData.scenario_id ?? null,
          scenario_name: runnerData.scenario_name ?? null,
          decision_count: runnerData.decision_count ?? 0,
          elapsed_seconds: runnerData.elapsed_seconds ?? 0,
        });
        // If runner is active, update status to show it's running
        if (runnerData.is_running) {
          setStatus("Live Scenario");
        }
        if (runnerData.is_running) {
          const runnerDecisionsResp = await fetch("/api/dashboard/runner/decisions?limit=20");
          if (runnerDecisionsResp.ok) {
            const runnerDecisionsData = await runnerDecisionsResp.json();
            const decisions = (runnerDecisionsData.decisions || []).map((entry: any) => ({
              timestamp: entry.timestamp,
              elapsed_s: entry.elapsed_s,
              drone_id: entry.drone_id,
              drone_name: entry.drone_name || entry.drone_id,
              agent_label: entry.agent_label || "Drone AG",
              action: entry.action,
              reason: entry.reason || entry.reasoning,
              confidence: entry.confidence ?? 0,
              risk_level: entry.risk_level || "LOW",
              battery_percent: entry.battery_percent,
              reasoning_context: entry.reasoning_context,
              alternatives: entry.alternatives,
              critic_validation: entry.critic_validation,
              target_asset: entry.target_asset,
            }));
            setRunnerDecisions(decisions);
          } else {
            setRunnerDecisions([]);
          }
        } else {
          setRunnerDecisions([]);
        }
      } else {
        setLiveRunnerStatus(null);
        setRunnerDecisions([]);
      }

      // Fetch real-time spatial context for radar widget
      const spatialResp = await fetch("/api/dashboard/spatial");
      if (spatialResp.ok) {
        const spatialData = await spatialResp.json();
        setSpatialAssets(spatialData.assets ?? []);
      }

      // Fetch mission success metrics
      const metricsResp = await fetch("/api/dashboard/metrics");
      if (metricsResp.ok) {
        const metricsData = await metricsResp.json();
        setMissionMetrics(normalizeMetrics(metricsData));
      }

      // Fetch recent captures for image gallery
      const capturesResp = await fetch("/api/vision/captures?limit=12");
      if (capturesResp.ok) {
        const capturesData = await capturesResp.json();
        setRecentCaptures(capturesData.captures ?? []);
      }

      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error("Failed to load dashboard data", error);
      setStatus("Offline");
    }
  }, []);

  const loadMissionConfig = useCallback(async () => {
    try {
      const resp = await fetch("/api/config/mission-success");
      if (!resp.ok) {
        return;
      }
      const data = (await resp.json()) as MissionSuccessConfig;
      const displayConfig: MissionSuccessConfig = {
        weights: {
          coverage: toPercent(data.weights.coverage),
          anomaly: toPercent(data.weights.anomaly),
          decision_quality: toPercent(data.weights.decision_quality),
          execution: toPercent(data.weights.execution),
          resource_use: toPercent(data.weights.resource_use),
        },
        thresholds: data.thresholds,
      };
      setMissionConfigDraft(displayConfig);
    } catch (error) {
      console.error("Failed to load mission success config", error);
    }
  }, []);

  const applyMissionConfig = async () => {
    if (!missionConfigDraft) return;
    try {
      setMissionConfigSaving(true);
      setMissionConfigMessage(null);
      const payload = {
        weights: {
          coverage: toRatio(missionConfigDraft.weights.coverage),
          anomaly: toRatio(missionConfigDraft.weights.anomaly),
          decision_quality: toRatio(missionConfigDraft.weights.decision_quality),
          execution: toRatio(missionConfigDraft.weights.execution),
          resource_use: toRatio(missionConfigDraft.weights.resource_use),
        },
        thresholds: missionConfigDraft.thresholds,
      };
      const resp = await fetch("/api/config/mission-success", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const error = await resp.json().catch(() => ({}));
        throw new Error(error.detail || "Update failed");
      }
      const updated = (await resp.json()) as {
        mission_success?: MissionSuccessConfig;
        note?: string;
      };
      if (updated.mission_success) {
        const displayConfig: MissionSuccessConfig = {
          weights: {
            coverage: toPercent(updated.mission_success.weights.coverage),
            anomaly: toPercent(updated.mission_success.weights.anomaly),
            decision_quality: toPercent(updated.mission_success.weights.decision_quality),
            execution: toPercent(updated.mission_success.weights.execution),
            resource_use: toPercent(updated.mission_success.weights.resource_use),
          },
          thresholds: updated.mission_success.thresholds,
        };
        setMissionConfigDraft(displayConfig);
      }
      setMissionConfigMessage(updated.note || "Mission success settings updated");
    } catch (error) {
      console.error("Failed to update mission success config", error);
      setMissionConfigMessage("Failed to update mission success settings");
    } finally {
      setMissionConfigSaving(false);
    }
  };

  const launchAirSim = async () => {
    if (airsimLaunching) return;
    if (airsimStatus?.bridge_connected) {
      window.open("/overlay", "_blank", "noopener,noreferrer");
      return;
    }
    if (airsimStatus && !airsimStatus.enabled) {
      setShowSettings(true);
      return;
    }

    window.open("/overlay", "_blank", "noopener,noreferrer");
    try {
      setAirsimLaunching(true);
      const resp = await fetch("/api/airsim/start", { method: "POST" });
      if (!resp.ok) {
        const data = await resp.json();
        console.error("Failed to launch AirSim", data);
      }
    } catch (error) {
      console.error("Failed to launch AirSim", error);
    } finally {
      setAirsimLaunching(false);
      loadRun();
    }
  };

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

  // WebSocket connection for live decision updates
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[Dashboard] WebSocket connected");
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const eventType = data.event_type || data.event;
        const payload = data.data || data.payload || {};
        const eventTimestamp = data.timestamp || payload.timestamp || new Date().toISOString();
        const normalizedEvent = String(eventType || "").toLowerCase();

        // Handle live decision events
        if (normalizedEvent === "server_decision") {
          const entry: DecisionEntry = {
            timestamp: eventTimestamp,
            elapsed_s: payload.elapsed_s,
            drone_id: payload.drone_id,
            drone_name: payload.drone_name || payload.drone_id,
            action: payload.action,
            reason: payload.reasoning || payload.reason,
            confidence: payload.confidence,
            risk_level: payload.risk_level || "LOW",
            battery_percent: payload.battery_percent,
            reasoning_context: payload.reasoning_context,
            alternatives: payload.alternatives,
            critic_validation: payload.critic_validation,
            target_asset: payload.target_asset,
            agent_label: payload.agent_label || "Drone AG",
          };
          setLiveDecisions((prev) => [entry, ...prev].slice(0, 20));
        }

        // Handle critic validation events (update most recent decision)
        if (normalizedEvent === "critic_validation") {
          setLiveDecisions((prev) => {
            if (prev.length === 0) return prev;
            const updated = [...prev];
            if (updated[0].drone_id === payload.drone_id) {
              updated[0] = {
                ...updated[0],
                critic_validation: {
                  approved: payload.approved,
                  escalation: payload.escalation,
                },
              };
            }
            return updated;
          });

          const criticEntry: DecisionEntry = {
            timestamp: eventTimestamp,
            drone_id: payload.drone_id,
            drone_name: payload.drone_name || payload.drone_id,
            action: "critic_validation",
            reason: payload.escalation?.reason
              ? `Critic escalation: ${payload.escalation.reason}`
              : payload.approved
                ? "Critics approved decision"
                : "Critics rejected decision",
            confidence: 1,
            risk_level: payload.risk_level || "LOW",
            critic_validation: {
              approved: payload.approved,
              escalation: payload.escalation,
            },
            agent_label: payload.agent_label || "Orchestration AG",
          };
          setLiveDecisions((prev) => [criticEntry, ...prev].slice(0, 20));
        }
      } catch (e) {
        // Ignore parse errors for non-JSON messages
      }
    };

    ws.onerror = () => {
      console.error("[Dashboard] WebSocket error");
    };

    ws.onclose = () => {
      console.log("[Dashboard] WebSocket closed");
      setWsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    loadRun();
    const timer = window.setInterval(loadRun, 5000);
    return () => window.clearInterval(timer);
  }, [loadRun]);

  useEffect(() => {
    loadMissionConfig();
  }, [loadMissionConfig]);

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
        <div className="hero-controls">
          <div className="hero-meta">
          <div className="hero-card action" onClick={toggleLLM} style={{ cursor: 'pointer', borderColor: useAdvancedEngine ? 'var(--accent-cyber)' : 'var(--text-muted)' }}>
            <span className="meta-label">Orchestration</span>
            <span className="meta-value">{useAdvancedEngine ? "LLM ACTIVE" : "RULES ONLY"}</span>
          </div>
          <div
            className="hero-card action"
            onClick={launchAirSim}
            style={{
              cursor: "pointer",
              borderColor: airsimStatus?.bridge_connected
                ? "var(--accent-security)"
                : airsimStatus?.connecting || airsimLaunching
                  ? "var(--accent-cyber)"
                  : "var(--accent-alert)",
            }}
          >
            <span className="meta-label">AirSim</span>
            <span className="meta-value">
              {!airsimStatus
                ? "UNKNOWN"
                : !airsimStatus.enabled
                  ? "DISABLED"
                  : airsimStatus.bridge_connected
                    ? "CONNECTED"
                    : airsimStatus.connecting || airsimLaunching
                      ? "STARTING"
                      : "OFFLINE"}
            </span>
            <span className="meta-sub">
              {airsimStatus?.last_error
                ? airsimStatus.last_error
                : !airsimStatus
                  ? "Click to launch + open overlay"
                  : !airsimStatus.enabled
                    ? "Enable AirSim in Settings"
                    : !airsimStatus.launch_supported
                      ? "Launch unsupported on this host"
                      : "Click to launch + open overlay"}
            </span>
          </div>
          <div className="hero-card">
            <span className="meta-label">Edge Compute</span>
            <select
              className="meta-select"
              value={edgeConfig?.profile ?? ""}
              onChange={(e) => updateEdgeProfile(e.target.value)}
              disabled={!edgeProfiles.length}
            >
              {(edgeProfiles.length ? edgeProfiles : [edgeConfig?.profile ?? ""]).map((p: string) => (
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
          </div>
          <div className="hero-links">
          <a className="btn secondary hero-link" href="/overlay" target="_blank" rel="noreferrer">
            Overlay
          </a>
          <a className="btn secondary hero-link" href="/dashboard/maps">
            Maps
          </a>
          <a className="btn secondary hero-link" href="/api/scenarios" target="_blank" rel="noreferrer">
            Scenarios
          </a>
          <a className="btn secondary hero-link" href="/api/dashboard/scenarios" target="_blank" rel="noreferrer">
            Scenario Details
          </a>
          <a className="btn secondary hero-link" href="/docs" target="_blank" rel="noreferrer">
            API Docs
          </a>
          <button
            className="btn secondary icon settings-btn"
            onClick={() => setShowSettings(true)}
            title="Settings"
            type="button"
          >
            ⚙
          </button>
        </div>
        </div>
      </header>

      {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}

      <AdvancedOverrides
        config={edgeConfig}
        onUpdate={updateEdgeOverrides}
        isExpanded={showAdvanced}
        onToggle={() => setShowAdvanced(!showAdvanced)}
      />
      <MissionSuccessControls
        config={missionConfigDraft}
        onChange={(next) => setMissionConfigDraft(next)}
        onApply={applyMissionConfig}
        onReload={loadMissionConfig}
        saving={missionConfigSaving}
        message={missionConfigMessage}
        isExpanded={missionConfigExpanded}
        onToggle={() => setMissionConfigExpanded(!missionConfigExpanded)}
      />

      <section className="grid">
        <article className="card primary">
          <h2>Mission Integrity</h2>
          <p className="stat" style={{ color: liveRunnerStatus?.is_running ? 'var(--accent-cyber)' : undefined }}>
            {liveRunnerStatus?.is_running
              ? liveRunnerStatus.decision_count
              : runData?.summary.total ?? "0"}
          </p>
          <p className="note">
            {liveRunnerStatus?.is_running
              ? `Live: ${liveRunnerStatus.scenario_name ?? liveRunnerStatus.scenario_id ?? "Running"}`
              : "Total autonomous decisions evaluated."}
          </p>
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

      {/* Mission Success Metrics */}
      <SuccessMetricsSection metrics={missionMetrics} />

      {/* Recent Captures Gallery */}
      <RecentCapturesSection captures={recentCaptures} />

      {/* Drone Camera Panel - only show when AirSim connected and drones available */}
      {airsimStatus?.bridge_connected && scenarioDrones.length > 0 && (
        <section className="grid" style={{ gridTemplateColumns: '1fr' }}>
          <DroneCameraPanel drones={scenarioDrones} />
        </section>
      )}

      {/* 3D Map Reconstruction - Splat Viewer */}
      <section className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <SplatMapViewer />
        <article className="card">
          <div className="card-header">
            <h2>Vision Detections</h2>
            <span className="pill">Live</span>
          </div>
          <p className="stat small">{lastVision?.asset_id ?? "No captures yet"}</p>
          <p className="note">
            {lastVision?.timestamp
              ? `Last capture at ${formatTime(lastVision.timestamp)}`
              : "Waiting for inspection imagery."}
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginTop: "12px" }}>
            {(lastVision?.detections ?? []).length ? (
              (lastVision?.detections ?? []).map((det, idx) => (
                <span key={`${det.detection_class ?? "det"}-${idx}`} className="pill">
                  {det.detection_class ?? "unknown"}
                </span>
              ))
            ) : (
              <span className="meta-sub">No detections reported.</span>
            )}
          </div>
          <p className="note" style={{ marginTop: "12px" }}>
            Max confidence {formatNumber(lastVision?.max_confidence, 2)} ·
            Max severity {formatNumber(lastVision?.max_severity, 2)} ·
            Anomaly {lastVision?.anomaly_created ? "created" : "not created"}
          </p>
        </article>
      </section>

      <section className="split">
        <LineChart
          title="Risk Vector"
          series={runData?.risk_series ?? []}
          accent="#06b6d4"
        />
        <LineChart
          title="Energy Decay"
          series={runData?.battery_series ?? []}
          accent="#10b981"
        />
      </section>

      <section className="split">
        <SpatialView
          assets={spatialAssets.length > 0 ? spatialAssets : (runData?.recent[0]?.spatial_context || [])}
        />
        <article className="card reasoning-card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>Reasoning Feed</h2>
            <span className="pill" style={{
              background: wsConnected ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255,255,255,0.1)',
              color: wsConnected ? 'var(--accent-security)' : 'var(--text-muted)'
            }}>
              {wsConnected ? 'Live' : 'Polling'}
            </span>
          </div>
          <div className="feed-container">
            <ReasoningFeed
              decisions={
                liveDecisions.length > 0
                  ? liveDecisions
                  : runnerDecisions.length > 0
                    ? runnerDecisions
                    : (runData?.recent ?? []).map((entry) => ({
                        timestamp: entry.timestamp,
                        action: entry.action,
                        reason: entry.reason,
                        confidence: entry.confidence,
                        risk_level: entry.risk_level || "LOW",
                        battery_percent: entry.battery_percent,
                        reasoning_context: (entry as any).reasoning_context,
                        alternatives: (entry as any).alternatives,
                        critic_validation: (entry as any).critic_validation,
                        drone_name: (entry as any).drone_name,
                        drone_id: (entry as any).drone_id,
                        agent_label: (entry as any).agent_label || "Drone AG",
                        elapsed_s: (entry as any).elapsed_s,
                        target_asset: (entry as any).target_asset,
                    }))
              }
            />
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
