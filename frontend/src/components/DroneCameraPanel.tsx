/**
 * DroneCameraPanel - Multi-drone camera view with click-to-follow selection
 *
 * Architecture:
 * - Each drone in the scenario can have its camera streamed
 * - Click on a drone to select it as the "active" view (large panel)
 * - Other drones show as thumbnails
 * - Uses WebSocket for real-time frame updates
 */

import { useCallback, useEffect, useRef, useState } from "react";

// Types
type DroneInfo = {
  drone_id: string;
  name: string;
  battery_percent: number;
  state: string;
  is_streaming: boolean;
  // Per-drone mission metrics (optional - may not be provided)
  inspections_completed?: number;
  anomalies_found?: number;
  inspection_progress_percent?: number;
};

type CameraFrame = {
  drone_id: string;
  sequence: number;
  timestamp_ms: number;
  image_base64: string;
  width: number;
  height: number;
};

type CameraStatus = {
  bridge_connected: boolean;
  active_streams: string[];
  available_vehicles: string[];
};

type Props = {
  /** List of drones in the current scenario */
  drones: DroneInfo[];
  /** Server base URL (defaults to same host) */
  serverUrl?: string;
  /** WebSocket URL for camera frames */
  wsUrl?: string;
  /** Maximum FPS for camera streams */
  maxFps?: number;
};

const DroneCameraPanel = ({
  drones,
  serverUrl = "",
  wsUrl,
  maxFps = 15,
}: Props) => {
  // State
  const [selectedDroneId, setSelectedDroneId] = useState<string | null>(null);
  const [frames, setFrames] = useState<Record<string, CameraFrame>>({});
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);
  const [streamingDrones, setStreamingDrones] = useState<Set<string>>(new Set());
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Auto-select first drone if none selected
  useEffect(() => {
    if (!selectedDroneId && drones.length > 0) {
      setSelectedDroneId(drones[0].drone_id);
    }
  }, [drones, selectedDroneId]);

  // Fetch camera status
  const fetchCameraStatus = useCallback(async () => {
    try {
      const resp = await fetch(`${serverUrl}/api/camera/status`);
      if (resp.ok) {
        const data = await resp.json();
        setCameraStatus(data);
        setStreamingDrones(new Set(data.active_streams || []));
      }
    } catch (e) {
      console.warn("Failed to fetch camera status:", e);
    }
  }, [serverUrl]);

  // Start camera stream for a drone
  const startStream = useCallback(
    async (droneId: string) => {
      try {
        const resp = await fetch(
          `${serverUrl}/api/camera/start_stream?drone_id=${encodeURIComponent(droneId)}&fps=${maxFps}`,
          { method: "POST" }
        );
        if (resp.ok) {
          setStreamingDrones((prev) => new Set([...prev, droneId]));
          setError(null);
        } else {
          const data = await resp.json();
          setError(data.detail || "Failed to start stream");
        }
      } catch (e) {
        setError(`Stream error: ${e}`);
      }
    },
    [serverUrl, maxFps]
  );

  // Stop camera stream for a drone
  const stopStream = useCallback(
    async (droneId: string) => {
      try {
        await fetch(
          `${serverUrl}/api/camera/stop_stream?drone_id=${encodeURIComponent(droneId)}`,
          { method: "POST" }
        );
        setStreamingDrones((prev) => {
          const next = new Set(prev);
          next.delete(droneId);
          return next;
        });
      } catch (e) {
        console.warn("Failed to stop stream:", e);
      }
    },
    [serverUrl]
  );

  // Handle drone selection - start streaming for selected drone
  const handleSelectDrone = useCallback(
    (droneId: string) => {
      setSelectedDroneId(droneId);
      // Auto-start stream for selected drone if not already streaming
      if (!streamingDrones.has(droneId) && cameraStatus?.bridge_connected) {
        startStream(droneId);
      }
    },
    [streamingDrones, cameraStatus, startStream]
  );

  // WebSocket connection for receiving camera frames
  useEffect(() => {
    // Determine WebSocket URL
    const wsBaseUrl =
      wsUrl ||
      (window.location.protocol === "https:" ? "wss://" : "ws://") +
        window.location.host;

    const connect = () => {
      try {
        // Connect to the Unreal stream endpoint (same as overlay)
        const ws = new WebSocket(`${wsBaseUrl}/ws/unreal`);

        ws.onopen = () => {
          console.log("[DroneCameraPanel] WebSocket connected");
          setWsConnected(true);
          setError(null);
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);

            // Handle camera frame messages
            if (msg.type === "camera_frame") {
              const frame: CameraFrame = {
                drone_id: msg.drone_id,
                sequence: msg.sequence,
                timestamp_ms: msg.timestamp_ms,
                image_base64: msg.image_base64,
                width: msg.width || 1280,
                height: msg.height || 720,
              };
              setFrames((prev) => ({
                ...prev,
                [frame.drone_id]: frame,
              }));
            }
          } catch (e) {
            // Ignore parse errors for non-JSON messages
          }
        };

        ws.onerror = () => {
          setError("WebSocket connection error");
        };

        ws.onclose = () => {
          console.log("[DroneCameraPanel] WebSocket disconnected");
          setWsConnected(false);
          // Attempt reconnect after 3 seconds
          reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
        };

        wsRef.current = ws;
      } catch (e) {
        console.error("[DroneCameraPanel] WebSocket connection failed:", e);
        reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
      }
    };

    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [wsUrl]);

  // Poll camera status
  useEffect(() => {
    fetchCameraStatus();
    const interval = setInterval(fetchCameraStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchCameraStatus]);

  // Get selected drone info
  const selectedDrone = drones.find((d) => d.drone_id === selectedDroneId);
  const selectedFrame = selectedDroneId ? frames[selectedDroneId] : null;
  const otherDrones = drones.filter((d) => d.drone_id !== selectedDroneId);

  return (
    <div className="drone-camera-panel">
      {/* Header */}
      <div className="camera-panel-header">
        <h2>Drone Camera Views</h2>
        <div className="camera-status">
          <span
            className={`status-dot ${wsConnected ? "connected" : "disconnected"}`}
          />
          <span>
            {cameraStatus?.bridge_connected ? "AirSim Connected" : "AirSim Offline"}
          </span>
          {streamingDrones.size > 0 && (
            <span className="stream-count">
              {streamingDrones.size} streaming
            </span>
          )}
        </div>
      </div>

      {error && <div className="camera-error">{error}</div>}

      {/* Main View - Selected Drone */}
      <div className="main-camera-view">
        {selectedDrone ? (
          <>
            <div className="camera-overlay">
              <span className="drone-name">{selectedDrone.name}</span>
              <span className="drone-state">{selectedDrone.state}</span>
              <span className="drone-battery">{selectedDrone.battery_percent.toFixed(0)}%</span>
            </div>
            {/* Per-drone mission metrics overlay */}
            {(selectedDrone.inspections_completed !== undefined ||
              selectedDrone.anomalies_found !== undefined) && (
              <div className="camera-metrics-overlay">
                {selectedDrone.inspections_completed !== undefined && (
                  <span className="metric-item inspections">
                    <span className="metric-value">{selectedDrone.inspections_completed}</span>
                    <span className="metric-label">inspections</span>
                  </span>
                )}
                {selectedDrone.anomalies_found !== undefined && (
                  <span className="metric-item anomalies">
                    <span className="metric-value">{selectedDrone.anomalies_found}</span>
                    <span className="metric-label">anomalies</span>
                  </span>
                )}
                {selectedDrone.inspection_progress_percent !== undefined && (
                  <div className="metric-progress">
                    <div
                      className="metric-progress-bar"
                      style={{ width: `${selectedDrone.inspection_progress_percent}%` }}
                    />
                  </div>
                )}
              </div>
            )}
            {selectedFrame ? (
              <img
                src={`data:image/png;base64,${selectedFrame.image_base64}`}
                alt={`${selectedDrone.name} camera`}
                className="camera-image main"
              />
            ) : (
              <div className="camera-placeholder main">
                <span>
                  {streamingDrones.has(selectedDroneId!)
                    ? "Waiting for frames..."
                    : "Click to start stream"}
                </span>
                {!streamingDrones.has(selectedDroneId!) && cameraStatus?.bridge_connected && (
                  <button
                    className="btn stream-btn"
                    onClick={() => startStream(selectedDroneId!)}
                  >
                    Start Stream
                  </button>
                )}
              </div>
            )}
            {streamingDrones.has(selectedDroneId!) && (
              <button
                className="btn stop-btn"
                onClick={() => stopStream(selectedDroneId!)}
              >
                Stop
              </button>
            )}
          </>
        ) : (
          <div className="camera-placeholder main">
            <span>No drone selected</span>
          </div>
        )}
      </div>

      {/* Thumbnail Strip - Other Drones */}
      {otherDrones.length > 0 && (
        <div className="camera-thumbnails">
          {otherDrones.map((drone) => {
            const frame = frames[drone.drone_id];
            const isStreaming = streamingDrones.has(drone.drone_id);

            return (
              <div
                key={drone.drone_id}
                className={`camera-thumbnail ${isStreaming ? "streaming" : ""}`}
                onClick={() => handleSelectDrone(drone.drone_id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    handleSelectDrone(drone.drone_id);
                  }
                }}
              >
                <div className="thumbnail-overlay">
                  <span className="drone-name">{drone.name}</span>
                  {drone.anomalies_found !== undefined && drone.anomalies_found > 0 && (
                    <span className="thumbnail-anomaly-badge">{drone.anomalies_found}</span>
                  )}
                </div>
                {frame ? (
                  <img
                    src={`data:image/png;base64,${frame.image_base64}`}
                    alt={`${drone.name} camera`}
                    className="camera-image thumbnail"
                  />
                ) : (
                  <div className="camera-placeholder thumbnail">
                    <span>{isStreaming ? "..." : "Click"}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Drone List for scenarios with many drones */}
      {drones.length > 4 && (
        <div className="drone-list">
          <h3>All Drones</h3>
          {drones.map((drone) => (
            <div
              key={drone.drone_id}
              className={`drone-list-item ${
                drone.drone_id === selectedDroneId ? "selected" : ""
              } ${streamingDrones.has(drone.drone_id) ? "streaming" : ""}`}
              onClick={() => handleSelectDrone(drone.drone_id)}
              role="button"
              tabIndex={0}
            >
              <div className="drone-list-info">
                <span className="drone-name">{drone.name}</span>
                <span className="drone-state">{drone.state}</span>
                <span className="drone-battery">{drone.battery_percent.toFixed(0)}%</span>
                {streamingDrones.has(drone.drone_id) && (
                  <span className="streaming-indicator">LIVE</span>
                )}
              </div>
              {(drone.inspections_completed !== undefined ||
                drone.anomalies_found !== undefined) && (
                <div className="drone-list-metrics">
                  {drone.inspections_completed !== undefined && (
                    <span className="drone-metric">
                      {drone.inspections_completed} inspections
                    </span>
                  )}
                  {drone.anomalies_found !== undefined && (
                    <span className="drone-metric anomalies">
                      {drone.anomalies_found} anomalies
                    </span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DroneCameraPanel;
