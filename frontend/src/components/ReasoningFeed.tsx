/**
 * ReasoningFeed - Rich decision reasoning display component
 *
 * Displays decision entries with:
 * - Action and reason
 * - Reasoning context (battery, assets, weather)
 * - Critic validation verdicts
 * - Alternative actions considered
 */

import React from "react";

interface CriticValidation {
  approved: boolean;
  escalation?: {
    reason: string;
    required_action?: string;
  };
}

interface Alternative {
  action: string;
  rejected: boolean;
  reason: string;
}

interface ReasoningContext {
  available_assets: number;
  pending_inspections: number;
  fleet_in_progress: number;
  fleet_completed: number;
  battery_ok: boolean;
  battery_percent: number;
  weather_ok: boolean;
  wind_speed: number;
}

export interface DecisionEntry {
  timestamp: string;
  elapsed_s?: number;
  drone_id?: string;
  drone_name?: string;
  agent_label?: string;
  action: string;
  reason?: string;
  confidence: number;
  risk_level: string;
  battery_percent?: number;
  reasoning_context?: ReasoningContext;
  alternatives?: Alternative[];
  critic_validation?: CriticValidation;
  target_asset?: {
    asset_id: string;
    name: string;
  };
}

interface Props {
  decisions: DecisionEntry[];
}

const getActionClass = (action: string | undefined | null): string => {
  switch (action || "") {
    case "critic_validation":
    case "orchestration":
      return "action-orchestrator";
    case "inspect_asset":
    case "inspect_anomaly":
      return "action-inspect";
    case "return_low_battery":
    case "return_weather":
    case "return_mission_complete":
      return "action-return";
    case "abort":
      return "action-abort";
    case "wait":
      return "action-wait";
    default:
      return "action-default";
  }
};

const getRiskClass = (level: string | undefined | null): string => {
  switch ((level || "").toUpperCase()) {
    case "CRITICAL":
      return "risk-critical";
    case "HIGH":
      return "risk-high";
    case "MODERATE":
      return "risk-moderate";
    default:
      return "risk-low";
  }
};

const formatAction = (action: string | undefined | null): string => {
  if (!action) return "UNKNOWN";
  return action.replace(/_/g, " ").toUpperCase();
};

export const ReasoningFeed: React.FC<Props> = ({ decisions }) => {
  if (!decisions || decisions.length === 0) {
    return (
      <div className="reasoning-feed-empty">
        <p>No decisions yet. Waiting for scenario to start...</p>
      </div>
    );
  }

  return (
    <div className="reasoning-feed">
      {decisions.map((d, i) => (
        <div key={`${d.timestamp}-${i}`} className="decision-card">
          {/* Header */}
          <div className="decision-header">
            <span className="drone-label">
              {d.agent_label && (d.drone_name || d.drone_id)
                ? `${d.drone_name || d.drone_id} · ${d.agent_label}`
                : d.agent_label || d.drone_name || d.drone_id || "Agent"}
            </span>
            <span className={`risk-badge ${getRiskClass(d.risk_level)}`}>
              {d.risk_level}
            </span>
          </div>

          {/* Action & Reason */}
          <h4 className={`decision-action ${getActionClass(d.action)}`}>
            {formatAction(d.action)}
          </h4>
          {d.reason && <p className="decision-reason">{d.reason}</p>}

          {/* Target asset if present */}
          {d.target_asset && (
            <div className="decision-target">
              Target: {d.target_asset.name || d.target_asset.asset_id}
            </div>
          )}

          {/* Context badges */}
          {d.reasoning_context && (
            <div className="context-badges">
              <span className="context-badge">
                <span className="badge-icon">B</span>
                {d.reasoning_context.battery_percent?.toFixed(0)}%
              </span>
              <span className="context-badge">
                <span className="badge-icon">P</span>
                {d.reasoning_context.pending_inspections} pending
              </span>
              <span className="context-badge">
                <span className="badge-icon">W</span>
                {d.reasoning_context.wind_speed?.toFixed(1)} m/s
              </span>
              <span className="context-badge">
                <span className="badge-icon">F</span>
                {d.reasoning_context.fleet_completed}/
                {d.reasoning_context.fleet_completed +
                  d.reasoning_context.pending_inspections}{" "}
                done
              </span>
            </div>
          )}

          {/* Critic Validation */}
          {d.critic_validation && (
            <div
              className={`critic-verdict ${
                d.critic_validation.approved
                  ? "critic-approved"
                  : "critic-rejected"
              }`}
            >
              <span className="critic-status">
                Critics:{" "}
                {d.critic_validation.approved ? "Approved" : "Rejected"}
              </span>
              {d.critic_validation.escalation && (
                <p className="critic-reason">
                  {d.critic_validation.escalation.reason}
                </p>
              )}
            </div>
          )}

          {/* Alternatives */}
          {d.alternatives && d.alternatives.length > 0 && (
            <details className="alternatives-section">
              <summary className="alternatives-toggle">
                Why not other actions?
              </summary>
              <ul className="alternatives-list">
                {d.alternatives.map((alt, j) => (
                  <li key={j} className="alternative-item">
                    <span className="alt-action">{formatAction(alt.action)}:</span>{" "}
                    {alt.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}

          {/* Timestamp */}
          <div className="decision-timestamp">
            {d.elapsed_s !== undefined && (
              <span className="elapsed-time">{d.elapsed_s.toFixed(0)}s</span>
            )}
            <span className="timestamp-value">
              {new Date(d.timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ReasoningFeed;
