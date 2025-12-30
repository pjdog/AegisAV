/**
 * AegisAV Control Panel Component
 *
 * Unified control panel for scenario selection, edge profile configuration,
 * mode switching (Live/Demo), and playback controls. Used by:
 * - Lightweight Visualizer
 * - Unreal OBS Control Interface
 *
 * Usage:
 *   import { ControlPanel } from '/shared/components/ControlPanel.js';
 *
 *   const panel = new ControlPanel(document.getElementById('controls'), {
 *     onScenarioSelected: (id) => console.log('Scenario:', id),
 *     onModeChanged: (mode) => console.log('Mode:', mode),
 *     onPlayback: (action) => console.log('Playback:', action),
 *   });
 *
 *   // Load available scenarios and profiles
 *   panel.setScenarios(scenarios);
 *   panel.setEdgeProfiles(profiles);
 */

// Inject styles only once
const STYLE_ID = 'aegis-control-panel-styles';

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ============================================
     * CONTROL PANEL COMPONENT STYLES
     * ============================================ */

    .aegis-control-panel {
      font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
      background: linear-gradient(135deg, rgba(24, 24, 27, 0.92), rgba(24, 24, 27, 0.86));
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(39, 39, 42, 0.9);
      border-radius: 14px;
      padding: 16px;
      color: #FAFAFA;
      min-width: 280px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .aegis-cp-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .aegis-cp-title {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .aegis-cp-title-icon {
      font-size: 18px;
    }

    .aegis-cp-title-text {
      font-weight: 600;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #06b6d4;
    }

    .aegis-cp-status {
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: "JetBrains Mono", "Roboto Mono", monospace;
      font-size: 10px;
      text-transform: uppercase;
    }

    .aegis-cp-status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      animation: aegis-pulse 2s infinite;
    }

    .aegis-cp-status-dot.connected {
      background: #10b981;
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
    }

    .aegis-cp-status-dot.disconnected {
      background: #ef4444;
      box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
      animation: none;
    }

    .aegis-cp-status-dot.connecting {
      background: #f59e0b;
      box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
      animation: aegis-blink 1s infinite;
    }

    @keyframes aegis-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    @keyframes aegis-blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.2; }
    }

    /* Section styling */
    .aegis-cp-section {
      margin-bottom: 14px;
    }

    .aegis-cp-section:last-child {
      margin-bottom: 0;
    }

    .aegis-cp-label {
      font-size: 9px;
      color: rgba(255, 255, 255, 0.45);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 6px;
    }

    /* Select/Dropdown */
    .aegis-cp-select {
      width: 100%;
      appearance: none;
      font-family: "JetBrains Mono", "Roboto Mono", monospace;
      font-size: 12px;
      font-weight: 500;
      color: #FAFAFA;
      background: rgba(9, 9, 11, 0.65);
      border: 1px solid rgba(39, 39, 42, 0.9);
      border-radius: 8px;
      padding: 10px 32px 10px 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2352525B' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 10px center;
    }

    .aegis-cp-select:hover {
      border-color: rgba(6, 182, 212, 0.35);
      background-color: rgba(9, 9, 11, 0.8);
    }

    .aegis-cp-select:focus {
      outline: none;
      border-color: #06b6d4;
      box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.25);
    }

    .aegis-cp-select option {
      background: #09090B;
      color: #FAFAFA;
      padding: 8px;
    }

    /* Scenario description */
    .aegis-cp-scenario-desc {
      font-size: 10px;
      color: rgba(255, 255, 255, 0.5);
      margin-top: 6px;
      line-height: 1.4;
      padding: 8px 10px;
      background: rgba(9, 9, 11, 0.4);
      border-radius: 6px;
      border-left: 2px solid #06b6d4;
    }

    /* Mode Toggle */
    .aegis-cp-mode-toggle {
      display: flex;
      gap: 4px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 8px;
      padding: 4px;
    }

    .aegis-cp-mode-btn {
      flex: 1;
      padding: 8px 12px;
      font-family: "JetBrains Mono", "Roboto Mono", monospace;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: rgba(255, 255, 255, 0.5);
      background: transparent;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .aegis-cp-mode-btn:hover {
      color: rgba(255, 255, 255, 0.8);
      background: rgba(255, 255, 255, 0.05);
    }

    .aegis-cp-mode-btn.active {
      color: #06b6d4;
      background: rgba(6, 182, 212, 0.18);
      box-shadow: 0 0 12px rgba(6, 182, 212, 0.35);
    }

    /* Playback Controls */
    .aegis-cp-playback {
      display: flex;
      gap: 8px;
      align-items: center;
    }

    .aegis-cp-playback-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 14px;
    }

    .aegis-cp-playback-btn:hover {
      background: rgba(0, 0, 0, 0.5);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .aegis-cp-playback-btn:active {
      transform: scale(0.95);
    }

    .aegis-cp-playback-btn.primary {
      background: rgba(6, 182, 212, 0.18);
      border-color: rgba(6, 182, 212, 0.45);
    }

    .aegis-cp-playback-btn.primary:hover {
      background: rgba(6, 182, 212, 0.3);
      box-shadow: 0 0 12px rgba(6, 182, 212, 0.35);
    }

    .aegis-cp-playback-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .aegis-cp-playback-time {
      flex: 1;
      font-family: "JetBrains Mono", "Roboto Mono", monospace;
      font-size: 11px;
      color: rgba(255, 255, 255, 0.6);
      text-align: center;
    }

    /* Speed Slider */
    .aegis-cp-speed {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .aegis-cp-speed-slider {
      flex: 1;
      -webkit-appearance: none;
      appearance: none;
      height: 4px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 2px;
      cursor: pointer;
    }

    .aegis-cp-speed-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 14px;
      height: 14px;
      background: #06b6d4;
      border-radius: 50%;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(6, 182, 212, 0.5);
      transition: transform 0.15s ease;
    }

    .aegis-cp-speed-slider::-webkit-slider-thumb:hover {
      transform: scale(1.15);
    }

    .aegis-cp-speed-slider::-moz-range-thumb {
      width: 14px;
      height: 14px;
      background: #06b6d4;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(6, 182, 212, 0.5);
    }

    .aegis-cp-speed-value {
      font-family: "JetBrains Mono", "Roboto Mono", monospace;
      font-size: 11px;
      font-weight: 600;
      color: #06b6d4;
      min-width: 40px;
      text-align: right;
    }

    /* Action Button */
    .aegis-cp-action-btn {
      width: 100%;
      padding: 12px 16px;
      font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #06b6d4;
      background: rgba(6, 182, 212, 0.12);
      border: 1px solid rgba(6, 182, 212, 0.4);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .aegis-cp-action-btn:hover {
      background: rgba(6, 182, 212, 0.2);
      box-shadow: 0 0 20px rgba(6, 182, 212, 0.25);
    }

    .aegis-cp-action-btn:active {
      transform: scale(0.98);
    }

    .aegis-cp-action-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .aegis-cp-action-btn.stop {
      color: #ef4444;
      background: rgba(239, 68, 68, 0.12);
      border-color: rgba(239, 68, 68, 0.4);
    }

    .aegis-cp-action-btn.stop:hover {
      background: rgba(239, 68, 68, 0.2);
      box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    }

    /* Divider */
    .aegis-cp-divider {
      height: 1px;
      background: rgba(255, 255, 255, 0.08);
      margin: 14px 0;
    }

    /* Stats Row */
    .aegis-cp-stats {
      display: flex;
      gap: 12px;
    }

    .aegis-cp-stat {
      flex: 1;
      text-align: center;
      padding: 8px;
      background: rgba(0, 0, 0, 0.2);
      border-radius: 6px;
    }

    .aegis-cp-stat-value {
      font-family: "JetBrains Mono", "Roboto Mono", monospace;
      font-size: 16px;
      font-weight: 600;
      color: #06b6d4;
    }

    .aegis-cp-stat-label {
      font-size: 9px;
      color: rgba(255, 255, 255, 0.4);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-top: 2px;
    }
  `;

  document.head.appendChild(style);
}

// Default scenarios
const DEFAULT_SCENARIOS = [
  {
    id: 'normal_ops_001',
    name: 'Normal Operations',
    description: 'Routine solar panel inspection with minimal anomalies. Good baseline scenario.'
  },
  {
    id: 'battery_cascade_001',
    name: 'Battery Cascade',
    description: 'Multi-drone battery degradation cascade requiring coordinated RTL decisions.'
  },
  {
    id: 'gps_degrade_001',
    name: 'GPS Degradation',
    description: 'Progressive GPS signal loss forcing position estimation and safety protocols.'
  },
  {
    id: 'weather_001',
    name: 'Weather Onset',
    description: 'Sudden weather deterioration requiring mission abort decisions.'
  },
  {
    id: 'sensor_cascade_001',
    name: 'Sensor Cascade',
    description: 'IMU anomalies propagating across fleet, testing cross-drone reasoning.'
  },
  {
    id: 'multi_anom_001',
    name: 'Multi-Anomaly',
    description: 'Simultaneous asset anomalies requiring prioritization decisions.'
  },
  {
    id: 'coord_001',
    name: 'Coordination',
    description: 'Multi-drone coordination with conflicting objectives and resource constraints.'
  }
];

// Default edge profiles
const DEFAULT_PROFILES = [
  { id: 'FC_ONLY', name: 'FC Only', description: 'Flight controller only - no vision' },
  { id: 'MCU_HEURISTIC', name: 'MCU Heuristic', description: 'Basic heuristics on microcontroller' },
  { id: 'MCU_TINY_CNN', name: 'MCU Tiny CNN', description: 'TinyML CNN for basic detection' },
  { id: 'SBC_CPU', name: 'SBC CPU', description: 'Raspberry Pi class CPU inference' },
  { id: 'SBC_ACCEL', name: 'SBC Accelerated', description: 'SBC with Coral/NPU accelerator' },
  { id: 'JETSON_FULL', name: 'Jetson Full', description: 'Full Jetson GPU inference' }
];

/**
 * ControlPanel Component
 */
export class ControlPanel extends EventTarget {
  /**
   * Create a new ControlPanel
   * @param {HTMLElement} container - The container element to render into
   * @param {Object} options - Configuration options
   */
  constructor(container, options = {}) {
    super();
    injectStyles();

    this.container = container;
    this.options = {
      scenarios: DEFAULT_SCENARIOS,
      profiles: DEFAULT_PROFILES,
      showStats: true,
      showPlayback: true,
      ...options
    };

    this.state = {
      connectionStatus: 'disconnected',
      selectedScenario: null,
      selectedProfile: 'SBC_CPU',
      mode: 'live', // 'live' or 'demo'
      isRunning: false,
      isPaused: false,
      speed: 1.0,
      simTime: 0,
      drones: 0,
      assets: 0
    };

    this.render();
    this.attachEventListeners();
  }

  /**
   * Render the control panel HTML
   */
  render() {
    this.container.innerHTML = `
      <div class="aegis-control-panel">
        <div class="aegis-cp-header">
          <div class="aegis-cp-title">
            <span class="aegis-cp-title-icon">⚙️</span>
            <span class="aegis-cp-title-text">Control Panel</span>
          </div>
          <div class="aegis-cp-status">
            <span class="aegis-cp-status-dot disconnected"></span>
            <span class="aegis-cp-status-text">Offline</span>
          </div>
        </div>

        <div class="aegis-cp-section">
          <div class="aegis-cp-label">Scenario</div>
          <select class="aegis-cp-select aegis-cp-scenario-select">
            <option value="">Select scenario...</option>
            ${this.options.scenarios.map(s => `
              <option value="${s.id}">${s.name}</option>
            `).join('')}
          </select>
          <div class="aegis-cp-scenario-desc" style="display: none;"></div>
        </div>

        <div class="aegis-cp-section">
          <div class="aegis-cp-label">Edge Profile</div>
          <select class="aegis-cp-select aegis-cp-profile-select">
            ${this.options.profiles.map(p => `
              <option value="${p.id}" ${p.id === this.state.selectedProfile ? 'selected' : ''}>${p.name}</option>
            `).join('')}
          </select>
        </div>

        <div class="aegis-cp-section">
          <div class="aegis-cp-label">Mode</div>
          <div class="aegis-cp-mode-toggle">
            <button class="aegis-cp-mode-btn active" data-mode="live">Live Agent</button>
            <button class="aegis-cp-mode-btn" data-mode="demo">Demo Playback</button>
          </div>
        </div>

        ${this.options.showPlayback ? `
        <div class="aegis-cp-section aegis-cp-playback-section" style="display: none;">
          <div class="aegis-cp-label">Playback</div>
          <div class="aegis-cp-playback">
            <button class="aegis-cp-playback-btn primary aegis-cp-play-btn" title="Play">▶</button>
            <button class="aegis-cp-playback-btn aegis-cp-pause-btn" title="Pause" disabled>⏸</button>
            <button class="aegis-cp-playback-btn aegis-cp-reset-btn" title="Reset">↺</button>
            <span class="aegis-cp-playback-time">00:00 / 00:00</span>
          </div>
        </div>
        ` : ''}

        <div class="aegis-cp-section">
          <div class="aegis-cp-label">Speed</div>
          <div class="aegis-cp-speed">
            <input type="range" class="aegis-cp-speed-slider" min="0.5" max="5" step="0.5" value="1">
            <span class="aegis-cp-speed-value">1.0x</span>
          </div>
        </div>

        <div class="aegis-cp-divider"></div>

        <div class="aegis-cp-section">
          <button class="aegis-cp-action-btn aegis-cp-start-btn" disabled>Start Scenario</button>
        </div>

        ${this.options.showStats ? `
        <div class="aegis-cp-divider"></div>
        <div class="aegis-cp-stats">
          <div class="aegis-cp-stat">
            <div class="aegis-cp-stat-value aegis-cp-drones-value">0</div>
            <div class="aegis-cp-stat-label">Drones</div>
          </div>
          <div class="aegis-cp-stat">
            <div class="aegis-cp-stat-value aegis-cp-assets-value">0</div>
            <div class="aegis-cp-stat-label">Assets</div>
          </div>
          <div class="aegis-cp-stat">
            <div class="aegis-cp-stat-value aegis-cp-time-value">0s</div>
            <div class="aegis-cp-stat-label">Sim Time</div>
          </div>
        </div>
        ` : ''}
      </div>
    `;

    this.cacheElements();
  }

  /**
   * Cache DOM element references
   */
  cacheElements() {
    this.elements = {
      statusDot: this.container.querySelector('.aegis-cp-status-dot'),
      statusText: this.container.querySelector('.aegis-cp-status-text'),
      scenarioSelect: this.container.querySelector('.aegis-cp-scenario-select'),
      scenarioDesc: this.container.querySelector('.aegis-cp-scenario-desc'),
      profileSelect: this.container.querySelector('.aegis-cp-profile-select'),
      modeButtons: this.container.querySelectorAll('.aegis-cp-mode-btn'),
      playbackSection: this.container.querySelector('.aegis-cp-playback-section'),
      playBtn: this.container.querySelector('.aegis-cp-play-btn'),
      pauseBtn: this.container.querySelector('.aegis-cp-pause-btn'),
      resetBtn: this.container.querySelector('.aegis-cp-reset-btn'),
      playbackTime: this.container.querySelector('.aegis-cp-playback-time'),
      speedSlider: this.container.querySelector('.aegis-cp-speed-slider'),
      speedValue: this.container.querySelector('.aegis-cp-speed-value'),
      startBtn: this.container.querySelector('.aegis-cp-start-btn'),
      dronesValue: this.container.querySelector('.aegis-cp-drones-value'),
      assetsValue: this.container.querySelector('.aegis-cp-assets-value'),
      timeValue: this.container.querySelector('.aegis-cp-time-value')
    };
  }

  /**
   * Attach event listeners
   */
  attachEventListeners() {
    // Scenario selection
    this.elements.scenarioSelect.addEventListener('change', (e) => {
      this.state.selectedScenario = e.target.value || null;
      this.updateScenarioDescription();
      this.updateStartButton();
      this.dispatchEvent(new CustomEvent('scenario-selected', {
        detail: { scenarioId: this.state.selectedScenario }
      }));
    });

    // Profile selection
    this.elements.profileSelect.addEventListener('change', (e) => {
      this.state.selectedProfile = e.target.value;
      this.dispatchEvent(new CustomEvent('profile-selected', {
        detail: { profileId: this.state.selectedProfile }
      }));
    });

    // Mode toggle
    this.elements.modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        if (mode !== this.state.mode) {
          this.state.mode = mode;
          this.updateModeToggle();
          this.updatePlaybackVisibility();
          this.dispatchEvent(new CustomEvent('mode-changed', {
            detail: { mode }
          }));
        }
      });
    });

    // Playback controls
    if (this.elements.playBtn) {
      this.elements.playBtn.addEventListener('click', () => {
        this.dispatchEvent(new CustomEvent('playback', { detail: { action: 'play' } }));
      });
    }

    if (this.elements.pauseBtn) {
      this.elements.pauseBtn.addEventListener('click', () => {
        this.dispatchEvent(new CustomEvent('playback', { detail: { action: 'pause' } }));
      });
    }

    if (this.elements.resetBtn) {
      this.elements.resetBtn.addEventListener('click', () => {
        this.dispatchEvent(new CustomEvent('playback', { detail: { action: 'reset' } }));
      });
    }

    // Speed slider
    this.elements.speedSlider.addEventListener('input', (e) => {
      this.state.speed = parseFloat(e.target.value);
      this.elements.speedValue.textContent = `${this.state.speed.toFixed(1)}x`;
      this.dispatchEvent(new CustomEvent('speed-changed', {
        detail: { speed: this.state.speed }
      }));
    });

    // Start/Stop button
    this.elements.startBtn.addEventListener('click', () => {
      if (this.state.isRunning) {
        this.dispatchEvent(new CustomEvent('stop-scenario'));
      } else {
        this.dispatchEvent(new CustomEvent('start-scenario', {
          detail: {
            scenarioId: this.state.selectedScenario,
            profileId: this.state.selectedProfile,
            mode: this.state.mode,
            speed: this.state.speed
          }
        }));
      }
    });
  }

  /**
   * Update connection status display
   * @param {string} status - 'connected', 'disconnected', or 'connecting'
   */
  setConnectionStatus(status) {
    this.state.connectionStatus = status;

    this.elements.statusDot.className = `aegis-cp-status-dot ${status}`;

    const statusText = {
      connected: 'Online',
      disconnected: 'Offline',
      connecting: 'Connecting...'
    };
    this.elements.statusText.textContent = statusText[status] || status;
  }

  /**
   * Set available scenarios
   * @param {Array} scenarios - Array of scenario objects
   */
  setScenarios(scenarios) {
    this.options.scenarios = scenarios;

    this.elements.scenarioSelect.innerHTML = `
      <option value="">Select scenario...</option>
      ${scenarios.map(s => `
        <option value="${s.id}">${s.name}</option>
      `).join('')}
    `;

    if (this.state.selectedScenario) {
      this.elements.scenarioSelect.value = this.state.selectedScenario;
      this.updateScenarioDescription();
    }
  }

  /**
   * Set available edge profiles
   * @param {Array} profiles - Array of profile objects
   */
  setEdgeProfiles(profiles) {
    this.options.profiles = profiles;

    this.elements.profileSelect.innerHTML = profiles.map(p => `
      <option value="${p.id}" ${p.id === this.state.selectedProfile ? 'selected' : ''}>${p.name}</option>
    `).join('');
  }

  /**
   * Update stats display
   * @param {Object} stats - Stats object with drones, assets, simTime
   */
  updateStats(stats) {
    if (stats.drones !== undefined && this.elements.dronesValue) {
      this.state.drones = stats.drones;
      this.elements.dronesValue.textContent = stats.drones;
    }

    if (stats.assets !== undefined && this.elements.assetsValue) {
      this.state.assets = stats.assets;
      this.elements.assetsValue.textContent = stats.assets;
    }

    if (stats.simTime !== undefined && this.elements.timeValue) {
      this.state.simTime = stats.simTime;
      this.elements.timeValue.textContent = `${Math.floor(stats.simTime)}s`;
    }
  }

  /**
   * Set running state
   * @param {boolean} isRunning - Whether scenario is running
   */
  setRunning(isRunning) {
    this.state.isRunning = isRunning;
    this.updateStartButton();

    // Disable selects while running
    this.elements.scenarioSelect.disabled = isRunning;
    this.elements.profileSelect.disabled = isRunning;
  }

  /**
   * Update playback time display
   * @param {number} current - Current time in seconds
   * @param {number} total - Total duration in seconds
   */
  updatePlaybackTime(current, total) {
    if (this.elements.playbackTime) {
      const formatTime = (s) => {
        const mins = Math.floor(s / 60);
        const secs = Math.floor(s % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
      };

      this.elements.playbackTime.textContent = `${formatTime(current)} / ${formatTime(total)}`;
    }
  }

  /**
   * Set playback state
   * @param {string} state - 'playing', 'paused', or 'stopped'
   */
  setPlaybackState(state) {
    if (!this.elements.playBtn) return;

    this.state.isPaused = state === 'paused';

    if (state === 'playing') {
      this.elements.playBtn.disabled = true;
      this.elements.pauseBtn.disabled = false;
    } else if (state === 'paused') {
      this.elements.playBtn.disabled = false;
      this.elements.pauseBtn.disabled = true;
    } else {
      this.elements.playBtn.disabled = false;
      this.elements.pauseBtn.disabled = true;
    }
  }

  // === Private methods ===

  updateScenarioDescription() {
    const scenario = this.options.scenarios.find(s => s.id === this.state.selectedScenario);

    if (scenario && scenario.description) {
      this.elements.scenarioDesc.textContent = scenario.description;
      this.elements.scenarioDesc.style.display = 'block';
    } else {
      this.elements.scenarioDesc.style.display = 'none';
    }
  }

  updateModeToggle() {
    this.elements.modeButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === this.state.mode);
    });
  }

  updatePlaybackVisibility() {
    if (this.elements.playbackSection) {
      this.elements.playbackSection.style.display = this.state.mode === 'demo' ? 'block' : 'none';
    }
  }

  updateStartButton() {
    const canStart = this.state.selectedScenario && !this.state.isRunning;
    this.elements.startBtn.disabled = !canStart && !this.state.isRunning;

    if (this.state.isRunning) {
      this.elements.startBtn.textContent = 'Stop Scenario';
      this.elements.startBtn.classList.add('stop');
    } else {
      this.elements.startBtn.textContent = 'Start Scenario';
      this.elements.startBtn.classList.remove('stop');
    }
  }

  /**
   * Destroy the component
   */
  destroy() {
    this.container.innerHTML = '';
  }
}

export default ControlPanel;
