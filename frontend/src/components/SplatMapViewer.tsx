import { useCallback, useEffect, useMemo, useState } from "react";

type Obstacle = {
  id: string;
  x: number;
  y: number;
  r: number;
  h: number;
  type: string;
  lat?: number;
  lon?: number;
};

type MapPreview = {
  obstacle_count: number;
  obstacles: Obstacle[];
  bounds: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  };
  resolution_m: number;
  heatmap?: {
    width: number;
    height: number;
    values: number[];
    bounds: {
      min_x: number;
      max_x: number;
      min_y: number;
      max_y: number;
    };
  } | null;
};

type SplatPoint = {
  x: number;
  y: number;
  z: number;
  r: number;
  g: number;
  b: number;
};

type SplatScene = {
  run_id: string;
  version: number;
  versions_available: number;
  preview_available: boolean;
  model_available: boolean;
  gaussian_count: number;
  quality: {
    psnr: number;
    ssim: number;
  };
  created_at: string | null;
};

type MapStatus = {
  map_available: boolean;
  map_age_s: number;
  map_version: number;
  obstacle_count: number;
  map_quality_score: number;
  splat_available: boolean;
  last_update: string | null;
  map_update_error?: string | null;
  map_update_error_at?: string | null;
  slam_status?: {
    backend?: string;
    tracking_state?: string;
    pose_confidence?: number;
    loop_closure_count?: number;
    map_point_count?: number;
  } | null;
};

type ProxyHealth = {
  available: boolean;
  path: string | null;
  obstacle_count: number;
  last_updated: string | null;
  age_s: number | null;
};

type MapHealth = {
  map_available: boolean;
  map_quality_score: number;
  gate_status: string;
  gate_reason: string;
  proxy_health: ProxyHealth | null;
};

// Color map for obstacle types
const obstacleColors: Record<string, string> = {
  building: "#ef4444",
  building_tall: "#dc2626",
  tower: "#b91c1c",
  tree: "#22c55e",
  structure_low: "#f97316",
  ground_obstacle: "#eab308",
  house: "#f59e0b",
  warehouse: "#d97706",
  solar_panel: "#3b82f6",
  substation: "#8b5cf6",
  power_line: "#ec4899",
  wind_turbine: "#06b6d4",
  unknown: "#6b7280",
};

const formatAge = (seconds: number): string => {
  if (!isFinite(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
};

const SplatMapViewer = () => {
  const [mapStatus, setMapStatus] = useState<MapStatus | null>(null);
  const [mapPreview, setMapPreview] = useState<MapPreview | null>(null);
  const [mapHealth, setMapHealth] = useState<MapHealth | null>(null);
  const [splatScenes, setSplatScenes] = useState<SplatScene[]>([]);
  const [selectedScene, setSelectedScene] = useState<string | null>(null);
  const [splatPoints, setSplatPoints] = useState<SplatPoint[]>([]);
  const [showSplat, setShowSplat] = useState(true);
  const [showObstacles, setShowObstacles] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [loading, setLoading] = useState(false);

  const size = 400;
  const padding = 40;

  // Fetch map status and preview
  const loadMapData = useCallback(async () => {
    try {
      const [statusResp, previewResp, scenesResp, healthResp] = await Promise.all([
        fetch("/api/navigation/map/status"),
        fetch("/api/navigation/map/preview"),
        fetch("/api/navigation/splat/scenes"),
        fetch("/api/navigation/map/health"),
      ]);

      if (statusResp.ok) {
        const data = await statusResp.json();
        setMapStatus(data);
      }

      if (previewResp.ok) {
        const data = await previewResp.json();
        setMapPreview(data.preview ?? null);
      }

      if (scenesResp.ok) {
        const data = await scenesResp.json();
        const scenes = data.scenes || [];
        setSplatScenes(scenes);
        if (!scenes.length) {
          setSelectedScene(null);
          setSplatPoints([]);
        } else if (!selectedScene || !scenes.some((scene: SplatScene) => scene.run_id === selectedScene)) {
          setSelectedScene(scenes[0].run_id);
        }
      }

      if (healthResp.ok) {
        const data = await healthResp.json();
        setMapHealth(data);
      }
    } catch (error) {
      console.error("Failed to load map data", error);
    }
  }, [selectedScene]);

  // Load splat preview when scene selected
  const loadSplatPreview = useCallback(async (runId: string) => {
    if (!runId) return;
    setLoading(true);
    try {
      const resp = await fetch(`/api/navigation/splat/preview/${runId}`);
      if (resp.ok) {
        const data = await resp.json();
        setSplatPoints(data.points || []);
      }
    } catch (error) {
      console.error("Failed to load splat preview", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMapData();
    const timer = setInterval(loadMapData, 10000);
    return () => clearInterval(timer);
  }, [loadMapData]);

  useEffect(() => {
    if (selectedScene) {
      loadSplatPreview(selectedScene);
    } else {
      setSplatPoints([]);
    }
  }, [selectedScene, loadSplatPreview]);

  // Calculate view transform
  const { transform, scale, center } = useMemo(() => {
    const obstacles = mapPreview?.obstacles || [];
    const bounds = mapPreview?.bounds ?? mapPreview?.heatmap?.bounds;

    // Get bounds from obstacles + splat points
    let minX = bounds?.min_x ?? 0;
    let maxX = bounds?.max_x ?? 0;
    let minY = bounds?.min_y ?? 0;
    let maxY = bounds?.max_y ?? 0;

    if (obstacles.length > 0) {
      for (const obs of obstacles) {
        minX = Math.min(minX, obs.x - obs.r);
        maxX = Math.max(maxX, obs.x + obs.r);
        minY = Math.min(minY, obs.y - obs.r);
        maxY = Math.max(maxY, obs.y + obs.r);
      }
    }

    if (splatPoints.length > 0) {
      for (const pt of splatPoints) {
        minX = Math.min(minX, pt.x);
        maxX = Math.max(maxX, pt.x);
        minY = Math.min(minY, pt.y);
        maxY = Math.max(maxY, pt.y);
      }
    }

    // Add padding
    const rangeX = maxX - minX || 100;
    const rangeY = maxY - minY || 100;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    const viewSize = size - padding * 2;
    const scaleVal = viewSize / Math.max(rangeX, rangeY) * 0.9;

    return {
      transform: { minX, maxX, minY, maxY, rangeX, rangeY },
      scale: scaleVal,
      center: { x: centerX, y: centerY },
    };
  }, [mapPreview, splatPoints, size, padding]);

  const toViewX = (x: number) => size / 2 + (x - center.x) * scale;
  const toViewY = (y: number) => size / 2 - (y - center.y) * scale; // Flip Y for screen coords

  const selectedSceneData = splatScenes.find((s) => s.run_id === selectedScene);
  const hasPreview = (mapPreview?.obstacles?.length ?? 0) > 0;
  const hasSplat = splatPoints.length > 0;
  const hasHeatmap = (mapPreview?.heatmap?.values?.length ?? 0) > 0;
  const mapReady = Boolean(mapStatus?.map_available && (hasPreview || hasSplat || hasHeatmap));
  const mapError = mapStatus?.map_update_error;

  return (
    <div className="splat-map-card">
      <div className="card-header">
        <h2>3D Map Reconstruction</h2>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {mapStatus?.map_available ? (
            <span
              className="pill"
              style={{
                background:
                  mapStatus.map_quality_score > 0.7
                    ? "var(--accent-security)"
                    : mapStatus.map_quality_score > 0.4
                    ? "var(--accent-cyber)"
                    : "var(--accent-alert)",
              }}
            >
              {(mapStatus.map_quality_score * 100).toFixed(0)}% Quality
            </span>
          ) : (
            <span className="pill">No Map</span>
          )}
          {splatScenes.length > 0 && (
            <span className="pill">{splatScenes.length} Splat{splatScenes.length !== 1 ? "s" : ""}</span>
          )}
        </div>
      </div>

      <div className="splat-meta" style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
        <span className="meta-sub">
          Last update {mapStatus?.last_update ? new Date(mapStatus.last_update).toLocaleTimeString() : "--"}
        </span>
        <span className="meta-sub">
          Obstacles {mapStatus?.obstacle_count ?? 0}
        </span>
        <span className="meta-sub">
          Age {mapStatus ? formatAge(mapStatus.map_age_s) : "--"}
        </span>
      </div>

      <div className="splat-controls">
        <label className="splat-toggle">
          <input
            type="checkbox"
            checked={showObstacles}
            onChange={(e) => setShowObstacles(e.target.checked)}
          />
          <span>Obstacles</span>
        </label>
        <label className="splat-toggle">
          <input
            type="checkbox"
            checked={showSplat}
            onChange={(e) => setShowSplat(e.target.checked)}
          />
          <span>Splat Points</span>
        </label>
        <label className="splat-toggle">
          <input
            type="checkbox"
            checked={showHeatmap}
            onChange={(e) => setShowHeatmap(e.target.checked)}
          />
          <span>Heatmap</span>
        </label>
        {splatScenes.length > 0 && (
          <select
            className="splat-select"
            value={selectedScene || ""}
            onChange={(e) => setSelectedScene(e.target.value)}
          >
            {splatScenes.map((scene) => (
              <option key={scene.run_id} value={scene.run_id}>
                {scene.run_id} (v{scene.version})
              </option>
            ))}
          </select>
        )}
      </div>

      {!mapReady && (
        <div className="splat-alert">
          <strong>Map data incomplete.</strong>
          <span className="meta-sub">
            {mapStatus?.map_available ? "No preview/splat geometry yet." : "Map unavailable from SLAM pipeline."}
          </span>
          {mapStatus?.slam_status?.backend && (
            <span className="meta-sub">
              SLAM {mapStatus.slam_status.backend} · {mapStatus.slam_status.tracking_state ?? "unknown"}
            </span>
          )}
          {mapError && (
            <span className="meta-sub">
              Last error {mapError}
              {mapStatus?.map_update_error_at
                ? ` · ${new Date(mapStatus.map_update_error_at).toLocaleTimeString()}`
                : ""}
            </span>
          )}
          {splatScenes.length === 0 && (
            <span className="meta-sub">No splat scenes available yet.</span>
          )}
        </div>
      )}

      {/* Proxy Status Badge */}
      {mapHealth?.proxy_health && (
        <div className="splat-proxy-status" style={{ display: "flex", gap: "8px", marginBottom: "8px" }}>
          <span
            className="pill"
            style={{
              background: mapHealth.proxy_health.available ? "var(--accent-security)" : "var(--accent-alert)",
              fontSize: "10px",
            }}
          >
            Proxy {mapHealth.proxy_health.available ? "OK" : "N/A"}
          </span>
          {mapHealth.proxy_health.available && (
            <>
              <span className="meta-sub" style={{ fontSize: "10px" }}>
                {mapHealth.proxy_health.obstacle_count} obstacles
              </span>
              <span className="meta-sub" style={{ fontSize: "10px" }}>
                Age {mapHealth.proxy_health.age_s != null ? formatAge(mapHealth.proxy_health.age_s) : "--"}
              </span>
            </>
          )}
          <span
            className="pill"
            style={{
              background:
                mapHealth.gate_status === "pass"
                  ? "var(--accent-security)"
                  : mapHealth.gate_status === "warn"
                  ? "var(--accent-cyber)"
                  : "var(--accent-alert)",
              fontSize: "10px",
            }}
          >
            Gate {mapHealth.gate_status.toUpperCase()}
          </span>
        </div>
      )}

      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="splat-svg">
        {/* Background */}
        <rect x="0" y="0" width={size} height={size} fill="rgba(0,0,0,0.3)" />

        {/* Grid */}
        <g className="splat-grid">
          {/* Grid lines every 50m */}
          {[-200, -150, -100, -50, 0, 50, 100, 150, 200].map((val) => (
            <g key={val}>
              <line
                x1={toViewX(val)}
                y1={padding}
                x2={toViewX(val)}
                y2={size - padding}
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="1"
              />
              <line
                x1={padding}
                y1={toViewY(val)}
                x2={size - padding}
                y2={toViewY(val)}
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="1"
              />
            </g>
          ))}
        </g>

        {/* Occupancy Heatmap */}
        {showHeatmap && mapPreview && (
          <g className="occupancy-heatmap">
            {/* Heatmap from tiles or fallback to obstacle density */}
            {(() => {
              const heatmap = mapPreview.heatmap;
              if (heatmap && heatmap.values.length > 0) {
                const { width, height, values } = heatmap;
                const bounds = heatmap.bounds ?? mapPreview.bounds;
                const spanX = bounds.max_x - bounds.min_x || 1;
                const spanY = bounds.max_y - bounds.min_y || 1;
                const cellW = spanX / width;
                const cellH = spanY / height;

                return values.map((value, idx) => {
                  if (value <= 0.01) return null;
                  const col = idx % width;
                  const row = Math.floor(idx / width);
                  const mapX = bounds.min_x + col * cellW;
                  const mapY = bounds.min_y + row * cellH;
                  const intensity = Math.min(1, Math.max(0, value));
                  const color = `rgba(255, ${Math.floor(200 * (1 - intensity))}, 0, ${0.15 + intensity * 0.55})`;
                  return (
                    <rect
                      key={`heatmap-${col}-${row}`}
                      x={toViewX(mapX)}
                      y={toViewY(mapY + cellH)}
                      width={cellW * scale}
                      height={cellH * scale}
                      fill={color}
                    />
                  );
                });
              }

              if (!mapPreview.obstacles || mapPreview.obstacles.length === 0) {
                return null;
              }

              const gridSize = 20;
              const cells: { x: number; y: number; count: number }[] = [];
              const cellMap: Record<string, number> = {};

              for (const obs of mapPreview.obstacles) {
                const cellX = Math.floor(toViewX(obs.x) / gridSize);
                const cellY = Math.floor(toViewY(obs.y) / gridSize);
                const key = `${cellX},${cellY}`;
                cellMap[key] = (cellMap[key] || 0) + 1;
              }

              for (const [key, count] of Object.entries(cellMap)) {
                const [x, y] = key.split(",").map(Number);
                cells.push({ x, y, count });
              }

              const maxCount = Math.max(...cells.map((c) => c.count), 1);

              return cells.map(({ x, y, count }) => {
                const intensity = count / maxCount;
                const color = `rgba(255, ${Math.floor(255 * (1 - intensity))}, 0, ${0.3 + intensity * 0.4})`;
                return (
                  <rect
                    key={`${x},${y}`}
                    x={x * gridSize}
                    y={y * gridSize}
                    width={gridSize}
                    height={gridSize}
                    fill={color}
                  />
                );
              });
            })()}
          </g>
        )}

        {/* Splat points */}
        {showSplat && splatPoints.length > 0 && (
          <g className="splat-points">
            {splatPoints.map((pt, i) => (
              <circle
                key={i}
                cx={toViewX(pt.x)}
                cy={toViewY(pt.y)}
                r="1.5"
                fill={`rgb(${pt.r}, ${pt.g}, ${pt.b})`}
                opacity="0.8"
              />
            ))}
          </g>
        )}

        {/* Obstacles */}
        {showObstacles && mapPreview?.obstacles.map((obs) => (
          <g key={obs.id} className="obstacle">
            <circle
              cx={toViewX(obs.x)}
              cy={toViewY(obs.y)}
              r={Math.max(obs.r * scale, 3)}
              fill={obstacleColors[obs.type] || obstacleColors.unknown}
              fillOpacity="0.6"
              stroke={obstacleColors[obs.type] || obstacleColors.unknown}
              strokeWidth="2"
            />
            {obs.r * scale > 10 && (
              <text
                x={toViewX(obs.x)}
                y={toViewY(obs.y)}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="white"
                fontSize="8"
                fontWeight="bold"
              >
                {obs.type.slice(0, 3).toUpperCase()}
              </text>
            )}
          </g>
        ))}

        {/* Origin marker */}
        <g className="origin-marker">
          <line x1={toViewX(0) - 8} y1={toViewY(0)} x2={toViewX(0) + 8} y2={toViewY(0)} stroke="#06b6d4" strokeWidth="2" />
          <line x1={toViewX(0)} y1={toViewY(0) - 8} x2={toViewX(0)} y2={toViewY(0) + 8} stroke="#06b6d4" strokeWidth="2" />
        </g>

        {/* Scale indicator */}
        <g className="scale-indicator" transform={`translate(${padding}, ${size - 20})`}>
          <line x1="0" y1="0" x2={50 * scale} y2="0" stroke="white" strokeWidth="2" />
          <line x1="0" y1="-4" x2="0" y2="4" stroke="white" strokeWidth="2" />
          <line x1={50 * scale} y1="-4" x2={50 * scale} y2="4" stroke="white" strokeWidth="2" />
          <text x={25 * scale} y="-8" textAnchor="middle" fill="white" fontSize="10">
            50m
          </text>
        </g>

        {/* Loading indicator */}
        {loading && (
          <g transform={`translate(${size / 2}, ${size / 2})`}>
            <circle r="20" fill="rgba(0,0,0,0.7)" />
            <text textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="10">
              Loading...
            </text>
          </g>
        )}
      </svg>

      <div className="splat-stats">
        <div className="splat-stat">
          <span className="stat-label">Obstacles</span>
          <span className="stat-value">{mapPreview?.obstacle_count ?? 0}</span>
        </div>
        <div className="splat-stat">
          <span className="stat-label">Splat Points</span>
          <span className="stat-value">{splatPoints.length.toLocaleString()}</span>
        </div>
        <div className="splat-stat">
          <span className="stat-label">Map Age</span>
          <span className="stat-value">{formatAge(mapStatus?.map_age_s ?? Infinity)}</span>
        </div>
        {selectedSceneData && (
          <div className="splat-stat">
            <span className="stat-label">Gaussians</span>
            <span className="stat-value">{selectedSceneData.gaussian_count.toLocaleString()}</span>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="splat-legend">
        {Object.entries(obstacleColors).slice(0, 6).map(([type, color]) => (
          <div key={type} className="legend-item">
            <span className="legend-color" style={{ background: color }} />
            <span className="legend-label">{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SplatMapViewer;
