import { useCallback, useEffect, useState } from "react";

import SplatMapViewer from "./SplatMapViewer";

type MapHealth = {
  map_available: boolean;
  map_age_s: number;
  map_quality_score: number;
  map_valid: boolean;
  gate_status: string;
  gate_reason: string;
  last_update: string | null;
};

type FusedMapEntry = {
  map_id: string;
  version: number;
  scenario_id?: string | null;
  source?: string | null;
  obstacle_count?: number;
  tile_count?: number;
  voxel_count?: number;
  stored_at?: string | null;
  generated_at?: string | null;
  metadata?: Record<string, unknown>;
};

const formatAge = (seconds: number): string => {
  if (!isFinite(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
};

const MapPage = () => {
  const [health, setHealth] = useState<MapHealth | null>(null);
  const [fusedMaps, setFusedMaps] = useState<FusedMapEntry[]>([]);
  const [showPreview, setShowPreview] = useState(true);
  const [showFusedList, setShowFusedList] = useState(true);

  const loadMapData = useCallback(async () => {
    try {
      const [healthResp, fusedResp] = await Promise.all([
        fetch("/api/navigation/map/health"),
        fetch("/api/navigation/map/fused"),
      ]);

      if (healthResp.ok) {
        const data = await healthResp.json();
        setHealth(data);
      }

      if (fusedResp.ok) {
        const data = await fusedResp.json();
        setFusedMaps(data.maps || []);
      }
    } catch (error) {
      console.error("Failed to load map data", error);
    }
  }, []);

  useEffect(() => {
    loadMapData();
    const timer = window.setInterval(loadMapData, 12000);
    return () => window.clearInterval(timer);
  }, [loadMapData]);

  return (
    <main className="map-page">
      <header className="map-page-header">
        <div>
          <p className="eyebrow">Aegis Mapping Workspace</p>
          <h1>SLAM + Splat Map Console</h1>
          <p className="subtitle">
            Review fused navigation maps, splat scenes, and map health signals.
          </p>
        </div>
        <div className="map-page-actions">
          <a className="btn secondary hero-link" href="/dashboard">
            Back to Dashboard
          </a>
          <a className="btn secondary hero-link" href="/overlay" target="_blank" rel="noreferrer">
            Overlay
          </a>
        </div>
      </header>

      <section className="map-page-grid">
        <article className="card map-health-card">
          <div className="card-header">
            <h2>Map Health</h2>
            {health?.map_available ? (
              <span className={`pill ${health.map_valid ? "green" : "warning"}`}>
                {health.map_valid ? "VALID" : "STALE"}
              </span>
            ) : (
              <span className="pill">NO MAP</span>
            )}
          </div>
          {health ? (
            <div className="map-health-body">
              <div className="meta-sub">Quality {(health.map_quality_score * 100).toFixed(0)}%</div>
              <div className="meta-sub">Age {formatAge(health.map_age_s)}</div>
              <div className="meta-sub">Gate {health.gate_status}</div>
              <div className="meta-sub">Reason {health.gate_reason}</div>
              <div className="meta-sub">
                Updated {health.last_update ? new Date(health.last_update).toLocaleTimeString() : "--"}
              </div>
            </div>
          ) : (
            <p className="hint">Map health will appear after the first SLAM update.</p>
          )}
        </article>

        <article className="card map-controls-card">
          <div className="card-header">
            <h2>Preview Toggles</h2>
          </div>
          <div className="map-toggle-group">
            <label className="splat-toggle">
              <input
                type="checkbox"
                checked={showPreview}
                onChange={(e) => setShowPreview(e.target.checked)}
              />
              <span>Show Map Preview</span>
            </label>
            <label className="splat-toggle">
              <input
                type="checkbox"
                checked={showFusedList}
                onChange={(e) => setShowFusedList(e.target.checked)}
              />
              <span>Show Fused Map List</span>
            </label>
          </div>
          <p className="hint">
            If no previews appear, run a scenario with preflight mapping enabled.
          </p>
        </article>
      </section>

      {showPreview && (
        <section className="map-preview-section">
          <SplatMapViewer />
        </section>
      )}

      {showFusedList && (
        <section className="map-list-section">
          <article className="card">
            <div className="card-header">
              <h2>Fused Map Runs</h2>
              <span className="pill">{fusedMaps.length}</span>
            </div>
            {fusedMaps.length === 0 ? (
              <p className="hint">No fused maps stored yet.</p>
            ) : (
              <div className="map-list">
                {fusedMaps.map((entry) => (
                  <div key={`${entry.map_id}-v${entry.version}`} className="map-list-item">
                    <div>
                      <div className="map-list-title">
                        {entry.map_id} <span className="map-version">v{entry.version}</span>
                      </div>
                      <div className="map-list-meta">
                        Scenario {entry.scenario_id ?? "--"} • Source {entry.source ?? "--"}
                      </div>
                    </div>
                    <div className="map-list-meta">
                      Obstacles {entry.obstacle_count ?? 0} • Tiles {entry.tile_count ?? 0}
                    </div>
                    <div className="map-list-meta">
                      Stored {entry.stored_at ? new Date(entry.stored_at).toLocaleTimeString() : "--"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </article>
        </section>
      )}
    </main>
  );
};

export default MapPage;
