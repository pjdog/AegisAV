/**
 * AegisAV ThoughtBubble Component
 *
 * Reusable agent thought bubble for displaying cognitive state, reasoning,
 * critic verdicts, and decisions. Used by:
 * - Lightweight Visualizer
 * - Unreal OBS Overlay
 *
 * Usage:
 *   import { ThoughtBubble } from '/shared/components/ThoughtBubble.js';
 *
 *   const bubble = new ThoughtBubble(document.getElementById('thought-container'));
 *   bubble.startThinking({ droneId: 'drone_001', cognitiveLevel: 'DELIBERATIVE' });
 *   bubble.updateThinking({ considerations: [...], riskScore: 0.3 });
 *   bubble.reportCritic({ criticName: 'safety', verdict: 'approve' });
 *   bubble.completeThinking({ action: 'INSPECT', confidence: 0.92 });
 */

// Inject styles only once
const STYLE_ID = 'aegis-thought-bubble-styles';

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    /* ============================================
     * THOUGHT BUBBLE COMPONENT STYLES
     * ============================================ */

    .aegis-thought-bubble {
      font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(135deg, rgba(18, 18, 22, 0.95), rgba(26, 26, 32, 0.95));
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(0, 242, 255, 0.25);
      border-radius: 14px;
      padding: 16px 18px;
      min-width: 300px;
      max-width: 380px;
      box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.5),
        0 0 24px rgba(0, 242, 255, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.06);
      transform: translateY(10px);
      opacity: 0;
      transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
      pointer-events: none;
      color: #f0f0f5;
    }

    .aegis-thought-bubble.visible {
      transform: translateY(0);
      opacity: 1;
      pointer-events: auto;
    }

    .aegis-thought-bubble.thinking {
      border-color: rgba(255, 184, 0, 0.4);
      box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.5),
        0 0 30px rgba(255, 184, 0, 0.12);
    }

    .aegis-thought-bubble.complete {
      border-color: rgba(0, 255, 157, 0.4);
      box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.5),
        0 0 30px rgba(0, 255, 157, 0.12);
    }

    .aegis-thought-bubble.error {
      border-color: rgba(255, 62, 62, 0.4);
      box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.5),
        0 0 30px rgba(255, 62, 62, 0.12);
    }

    /* Header */
    .aegis-tb-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .aegis-tb-drone {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .aegis-tb-drone-icon {
      font-size: 18px;
    }

    .aegis-tb-drone-name {
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      font-weight: 600;
      font-size: 12px;
      color: #00f2ff;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .aegis-tb-cognitive-badge {
      padding: 4px 10px;
      border-radius: 10px;
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      transition: all 0.25s ease;
    }

    .aegis-tb-cognitive-badge.reactive {
      background: rgba(138, 138, 149, 0.2);
      color: #8a8a95;
      border: 1px solid rgba(138, 138, 149, 0.4);
    }

    .aegis-tb-cognitive-badge.deliberative {
      background: rgba(0, 242, 255, 0.15);
      color: #00f2ff;
      border: 1px solid rgba(0, 242, 255, 0.4);
    }

    .aegis-tb-cognitive-badge.reflective {
      background: rgba(168, 85, 247, 0.15);
      color: #a855f7;
      border: 1px solid rgba(168, 85, 247, 0.4);
    }

    .aegis-tb-cognitive-badge.predictive {
      background: rgba(0, 255, 157, 0.15);
      color: #00ff9d;
      border: 1px solid rgba(0, 255, 157, 0.4);
    }

    /* Situation section */
    .aegis-tb-situation {
      margin-bottom: 12px;
      padding: 10px 12px;
      background: rgba(0, 0, 0, 0.25);
      border-left: 3px solid #00f2ff;
      border-radius: 0 8px 8px 0;
    }

    .aegis-tb-situation-label {
      font-size: 9px;
      color: rgba(255, 255, 255, 0.45);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 4px;
    }

    .aegis-tb-situation-text {
      font-size: 13px;
      font-weight: 500;
      color: #f0f0f5;
      line-height: 1.35;
    }

    .aegis-tb-situation-target {
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      font-size: 11px;
      color: #00f2ff;
      margin-top: 4px;
    }

    /* Risk section */
    .aegis-tb-risk {
      margin-bottom: 12px;
    }

    .aegis-tb-risk-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }

    .aegis-tb-risk-label {
      font-size: 9px;
      color: rgba(255, 255, 255, 0.45);
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .aegis-tb-risk-value {
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      font-size: 11px;
      font-weight: 600;
    }

    .aegis-tb-risk-value.low { color: #00ff9d; }
    .aegis-tb-risk-value.moderate { color: #ffb800; }
    .aegis-tb-risk-value.high { color: #ff8c00; }
    .aegis-tb-risk-value.critical { color: #ff3e3e; }

    .aegis-tb-risk-bar {
      height: 5px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
      overflow: hidden;
    }

    .aegis-tb-risk-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 0.4s ease, background 0.4s ease;
    }

    .aegis-tb-risk-fill.low {
      background: linear-gradient(90deg, #00ff9d, #00d4aa);
    }

    .aegis-tb-risk-fill.moderate {
      background: linear-gradient(90deg, #ffb800, #ff9500);
    }

    .aegis-tb-risk-fill.high {
      background: linear-gradient(90deg, #ff8c00, #ff5500);
    }

    .aegis-tb-risk-fill.critical {
      background: linear-gradient(90deg, #ff3e3e, #ff0000);
      animation: aegis-pulse-critical 1s infinite;
    }

    @keyframes aegis-pulse-critical {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }

    /* Critics section */
    .aegis-tb-critics {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
    }

    .aegis-tb-critic {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      padding: 7px 6px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      transition: all 0.25s ease;
    }

    .aegis-tb-critic.pending {
      background: rgba(255, 255, 255, 0.04);
      border-color: rgba(255, 255, 255, 0.08);
    }

    .aegis-tb-critic.approve {
      background: rgba(0, 255, 157, 0.1);
      border-color: rgba(0, 255, 157, 0.35);
    }

    .aegis-tb-critic.reject {
      background: rgba(255, 62, 62, 0.1);
      border-color: rgba(255, 62, 62, 0.35);
    }

    .aegis-tb-critic-icon {
      font-size: 13px;
    }

    .aegis-tb-critic-name {
      font-size: 8px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: rgba(255, 255, 255, 0.65);
    }

    /* Considerations section */
    .aegis-tb-considerations {
      margin-bottom: 12px;
    }

    .aegis-tb-considerations-label {
      font-size: 9px;
      color: rgba(255, 255, 255, 0.45);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 6px;
    }

    .aegis-tb-considerations-list {
      list-style: none;
      padding: 0;
      margin: 0;
    }

    .aegis-tb-consideration {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 11px;
      color: rgba(255, 255, 255, 0.8);
      line-height: 1.4;
      margin-bottom: 4px;
      opacity: 0;
      transform: translateX(-8px);
      animation: aegis-slide-in 0.25s ease forwards;
    }

    .aegis-tb-consideration::before {
      content: '>';
      color: #00f2ff;
      font-weight: 600;
      flex-shrink: 0;
    }

    @keyframes aegis-slide-in {
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }

    /* Decision section */
    .aegis-tb-decision {
      display: none;
      padding: 12px;
      background: linear-gradient(135deg, rgba(0, 255, 157, 0.12), rgba(0, 200, 157, 0.08));
      border: 1px solid rgba(0, 255, 157, 0.3);
      border-radius: 10px;
      margin-top: 10px;
    }

    .aegis-tb-decision.active {
      display: block;
      animation: aegis-decision-flash 0.4s ease;
    }

    @keyframes aegis-decision-flash {
      0% {
        transform: scale(0.96);
        opacity: 0;
      }
      50% {
        transform: scale(1.01);
      }
      100% {
        transform: scale(1);
        opacity: 1;
      }
    }

    .aegis-tb-decision-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }

    .aegis-tb-decision-icon {
      font-size: 18px;
    }

    .aegis-tb-decision-action {
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      font-size: 14px;
      font-weight: 700;
      color: #00ff9d;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .aegis-tb-decision-confidence {
      margin-left: auto;
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      font-size: 13px;
      font-weight: 600;
      color: #00ff9d;
    }

    .aegis-tb-decision-reasoning {
      font-size: 11px;
      color: rgba(255, 255, 255, 0.75);
      line-height: 1.4;
    }

    /* Empty state */
    .aegis-tb-empty {
      text-align: center;
      padding: 20px;
    }

    .aegis-tb-empty-icon {
      font-size: 28px;
      margin-bottom: 8px;
      opacity: 0.5;
    }

    .aegis-tb-empty-text {
      font-size: 12px;
      color: rgba(255, 255, 255, 0.4);
    }

    /* Connection state */
    .aegis-tb-connecting .aegis-tb-empty-icon {
      animation: aegis-spin 1.5s linear infinite;
    }

    @keyframes aegis-spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  `;

  document.head.appendChild(style);
}

// Action icons mapping
const ACTION_ICONS = {
  'INSPECT': '🔍',
  'ABORT': '🛑',
  'GOTO': '📍',
  'DOCK': '🏠',
  'RTL': '↩️',
  'RETURN': '↩️',
  'WAIT': '⏳',
  'HOVER': '🚁',
  'LAND': '🛬',
  'TAKEOFF': '🛫',
  'default': '✈️'
};

// Goal type display names
const GOAL_DISPLAY = {
  'INSPECT_ASSET': 'Inspect Asset',
  'INSPECT_ANOMALY': 'Inspect Anomaly',
  'RETURN_LOW_BATTERY': 'Return (Low Battery)',
  'RETURN_WEATHER': 'Return (Weather)',
  'RETURN_MISSION_COMPLETE': 'Mission Complete',
  'ABORT': 'Abort Mission',
  'WAIT': 'Waiting',
  'DOCK': 'Return to Dock',
  'TAKEOFF': 'Taking Off',
  'LAND': 'Landing'
};

/**
 * ThoughtBubble Component
 */
export class ThoughtBubble {
  /**
   * Create a new ThoughtBubble
   * @param {HTMLElement} container - The container element to render into
   * @param {Object} options - Configuration options
   * @param {string} options.droneId - Initial drone ID
   * @param {number} options.hideTimeout - Auto-hide timeout in ms (default: 8000)
   */
  constructor(container, options = {}) {
    injectStyles();

    this.container = container;
    this.options = {
      hideTimeout: 8000,
      droneId: 'agent',
      ...options
    };

    this.state = {
      visible: false,
      droneId: this.options.droneId,
      cognitiveLevel: 'deliberative',
      situation: '',
      target: '',
      riskScore: 0,
      riskLevel: 'low',
      critics: {
        safety: 'pending',
        efficiency: 'pending',
        goal_alignment: 'pending'
      },
      considerations: [],
      decision: null
    };

    this.hideTimeoutId = null;
    this.render();
  }

  /**
   * Render the bubble HTML
   */
  render() {
    this.container.innerHTML = `
      <div class="aegis-thought-bubble">
        <div class="aegis-tb-header">
          <div class="aegis-tb-drone">
            <span class="aegis-tb-drone-icon">🚁</span>
            <span class="aegis-tb-drone-name">${this.state.droneId}</span>
          </div>
          <span class="aegis-tb-cognitive-badge deliberative">Deliberative</span>
        </div>

        <div class="aegis-tb-situation">
          <div class="aegis-tb-situation-label">Current Goal</div>
          <div class="aegis-tb-situation-text">Initializing...</div>
          <div class="aegis-tb-situation-target"></div>
        </div>

        <div class="aegis-tb-risk">
          <div class="aegis-tb-risk-header">
            <span class="aegis-tb-risk-label">Risk Level</span>
            <span class="aegis-tb-risk-value low">0%</span>
          </div>
          <div class="aegis-tb-risk-bar">
            <div class="aegis-tb-risk-fill low" style="width: 0%"></div>
          </div>
        </div>

        <div class="aegis-tb-critics">
          <div class="aegis-tb-critic pending" data-critic="safety">
            <span class="aegis-tb-critic-icon">🛡️</span>
            <span class="aegis-tb-critic-name">Safety</span>
          </div>
          <div class="aegis-tb-critic pending" data-critic="efficiency">
            <span class="aegis-tb-critic-icon">⚡</span>
            <span class="aegis-tb-critic-name">Efficiency</span>
          </div>
          <div class="aegis-tb-critic pending" data-critic="goal_alignment">
            <span class="aegis-tb-critic-icon">🎯</span>
            <span class="aegis-tb-critic-name">Goal</span>
          </div>
        </div>

        <div class="aegis-tb-considerations">
          <div class="aegis-tb-considerations-label">Reasoning</div>
          <ul class="aegis-tb-considerations-list">
            <li class="aegis-tb-consideration">Analyzing situation...</li>
          </ul>
        </div>

        <div class="aegis-tb-decision">
          <div class="aegis-tb-decision-header">
            <span class="aegis-tb-decision-icon">✈️</span>
            <span class="aegis-tb-decision-action">DECIDING</span>
            <span class="aegis-tb-decision-confidence">--</span>
          </div>
          <div class="aegis-tb-decision-reasoning"></div>
        </div>
      </div>
    `;

    this.bubble = this.container.querySelector('.aegis-thought-bubble');
    this.elements = {
      droneName: this.container.querySelector('.aegis-tb-drone-name'),
      cognitiveBadge: this.container.querySelector('.aegis-tb-cognitive-badge'),
      situationText: this.container.querySelector('.aegis-tb-situation-text'),
      situationTarget: this.container.querySelector('.aegis-tb-situation-target'),
      riskValue: this.container.querySelector('.aegis-tb-risk-value'),
      riskFill: this.container.querySelector('.aegis-tb-risk-fill'),
      critics: {
        safety: this.container.querySelector('[data-critic="safety"]'),
        efficiency: this.container.querySelector('[data-critic="efficiency"]'),
        goal_alignment: this.container.querySelector('[data-critic="goal_alignment"]')
      },
      considerationsList: this.container.querySelector('.aegis-tb-considerations-list'),
      decision: this.container.querySelector('.aegis-tb-decision'),
      decisionIcon: this.container.querySelector('.aegis-tb-decision-icon'),
      decisionAction: this.container.querySelector('.aegis-tb-decision-action'),
      decisionConfidence: this.container.querySelector('.aegis-tb-decision-confidence'),
      decisionReasoning: this.container.querySelector('.aegis-tb-decision-reasoning')
    };
  }

  /**
   * Handle thinking_start message
   * @param {Object} data - Message data
   */
  startThinking(data) {
    this.state.droneId = data.drone_id || data.droneId || this.state.droneId;
    this.state.cognitiveLevel = (data.cognitive_level || data.cognitiveLevel || 'deliberative').toLowerCase();
    this.state.situation = data.current_goal || data.currentGoal || '';
    this.state.target = data.target_asset || data.targetAsset || '';

    // Reset state
    this.state.critics = {
      safety: 'pending',
      efficiency: 'pending',
      goal_alignment: 'pending'
    };
    this.state.decision = null;

    // Update UI
    this.elements.droneName.textContent = this.state.droneId;
    this.updateCognitiveBadge();
    this.updateSituation();
    this.resetCritics();
    this.elements.decision.classList.remove('active');

    // Show bubble in thinking state
    this.bubble.classList.remove('complete', 'error');
    this.bubble.classList.add('visible', 'thinking');

    this.resetHideTimeout();
  }

  /**
   * Handle thinking_update message
   * @param {Object} data - Message data
   */
  updateThinking(data) {
    // Update risk
    if (data.risk_score !== undefined || data.riskScore !== undefined) {
      this.state.riskScore = data.risk_score ?? data.riskScore;
      this.state.riskLevel = data.risk_level || data.riskLevel || this.getRiskLevel(this.state.riskScore);
      this.updateRisk();
    }

    // Update considerations
    if (data.considerations && data.considerations.length > 0) {
      this.state.considerations = data.considerations;
      this.updateConsiderations();
    }

    // Update situation if provided
    if (data.current_goal || data.currentGoal) {
      this.state.situation = data.current_goal || data.currentGoal;
      this.updateSituation();
    }

    // Ensure visible
    this.bubble.classList.add('visible');
    this.resetHideTimeout();
  }

  /**
   * Handle critic_result message
   * @param {Object} data - Message data
   */
  reportCritic(data) {
    const criticName = data.critic_name || data.criticName;
    const verdict = (data.verdict || '').toLowerCase();

    if (this.state.critics.hasOwnProperty(criticName)) {
      this.state.critics[criticName] = verdict;

      const badge = this.elements.critics[criticName];
      if (badge) {
        badge.classList.remove('pending', 'approve', 'reject');
        badge.classList.add(verdict === 'approve' ? 'approve' : 'reject');
      }
    }

    this.resetHideTimeout();
  }

  /**
   * Handle thinking_complete message
   * @param {Object} data - Message data
   */
  completeThinking(data) {
    const action = data.decision_action || data.action || 'UNKNOWN';
    const confidence = data.decision_confidence || data.confidence || 0;
    const reasoning = data.decision_reasoning || data.reasoning || '';

    this.state.decision = { action, confidence, reasoning };

    // Update decision UI
    this.elements.decisionIcon.textContent = ACTION_ICONS[action] || ACTION_ICONS['default'];
    this.elements.decisionAction.textContent = action;
    this.elements.decisionConfidence.textContent = `${(confidence * 100).toFixed(0)}%`;
    this.elements.decisionReasoning.textContent = reasoning;

    // Show decision
    this.elements.decision.classList.add('active');

    // Update bubble state
    this.bubble.classList.remove('thinking');
    this.bubble.classList.add('complete');

    // Extended timeout after decision
    this.resetHideTimeout(10000);
  }

  /**
   * Handle risk_update message
   * @param {Object} data - Message data
   */
  updateRisk(data) {
    if (data) {
      this.state.riskScore = data.overall_score || data.overallScore || data.risk_score || 0;
      this.state.riskLevel = data.level || data.risk_level || this.getRiskLevel(this.state.riskScore);
    }

    const percentage = Math.round(this.state.riskScore * 100);
    const level = this.state.riskLevel.toLowerCase();

    this.elements.riskValue.textContent = `${percentage}%`;
    this.elements.riskValue.className = `aegis-tb-risk-value ${level}`;

    this.elements.riskFill.style.width = `${percentage}%`;
    this.elements.riskFill.className = `aegis-tb-risk-fill ${level}`;
  }

  /**
   * Show empty state
   */
  setEmpty() {
    this.bubble.classList.remove('visible', 'thinking', 'complete');
  }

  /**
   * Show connecting state
   */
  setConnecting() {
    this.state.situation = 'Connecting to agent...';
    this.updateSituation();
    this.bubble.classList.add('visible', 'aegis-tb-connecting');
    this.bubble.classList.remove('thinking', 'complete');
  }

  /**
   * Show connected state briefly
   */
  setConnected() {
    this.bubble.classList.remove('aegis-tb-connecting');
    this.state.situation = 'Connected - Awaiting agent activity';
    this.updateSituation();
    this.resetHideTimeout(3000);
  }

  /**
   * Show error state
   * @param {string} message - Error message
   */
  setError(message) {
    this.state.situation = message || 'Connection error';
    this.updateSituation();
    this.bubble.classList.remove('thinking', 'complete', 'aegis-tb-connecting');
    this.bubble.classList.add('visible', 'error');
  }

  /**
   * Handle incoming WebSocket message
   * @param {Object} data - Parsed message data
   */
  handleMessage(data) {
    const type = data.type;

    switch (type) {
      case 'thinking_start':
        this.startThinking(data);
        break;
      case 'thinking_update':
        this.updateThinking(data);
        break;
      case 'thinking_complete':
        this.completeThinking(data);
        break;
      case 'critic_result':
        this.reportCritic(data);
        break;
      case 'risk_update':
        this.updateRisk(data);
        break;
      default:
        // Ignore other message types
        break;
    }
  }

  // === Private methods ===

  updateCognitiveBadge() {
    const level = this.state.cognitiveLevel.toLowerCase();
    const displayName = level.charAt(0).toUpperCase() + level.slice(1);

    this.elements.cognitiveBadge.className = `aegis-tb-cognitive-badge ${level}`;
    this.elements.cognitiveBadge.textContent = displayName;
  }

  updateSituation() {
    const displayText = GOAL_DISPLAY[this.state.situation] || this.state.situation || 'Evaluating...';
    this.elements.situationText.textContent = displayText;

    if (this.state.target) {
      this.elements.situationTarget.textContent = `Target: ${this.state.target}`;
      this.elements.situationTarget.style.display = 'block';
    } else {
      this.elements.situationTarget.style.display = 'none';
    }
  }

  updateConsiderations() {
    const list = this.elements.considerationsList;
    list.innerHTML = '';

    this.state.considerations.slice(0, 5).forEach((item, index) => {
      const li = document.createElement('li');
      li.className = 'aegis-tb-consideration';
      li.style.animationDelay = `${index * 0.08}s`;
      li.textContent = item;
      list.appendChild(li);
    });
  }

  resetCritics() {
    Object.values(this.elements.critics).forEach(badge => {
      if (badge) {
        badge.classList.remove('approve', 'reject');
        badge.classList.add('pending');
      }
    });
  }

  getRiskLevel(score) {
    if (score < 0.25) return 'low';
    if (score < 0.5) return 'moderate';
    if (score < 0.75) return 'high';
    return 'critical';
  }

  resetHideTimeout(duration = this.options.hideTimeout) {
    if (this.hideTimeoutId) {
      clearTimeout(this.hideTimeoutId);
    }

    this.hideTimeoutId = setTimeout(() => {
      this.bubble.classList.remove('visible');
    }, duration);
  }

  /**
   * Destroy the component
   */
  destroy() {
    if (this.hideTimeoutId) {
      clearTimeout(this.hideTimeoutId);
    }
    this.container.innerHTML = '';
  }
}

export default ThoughtBubble;
