# AegisAV Reasoning Feed & Multi-Agent Decision System Fix Plan

## Executive Summary

The reasoning feed in the dashboard only displays "wait" messages because the decision-making pipeline has several issues preventing real inspection goals from being selected. This document provides a comprehensive plan to fix the decision system and fully enable the multi-agent orchestration architecture.

**Plan Structure:** Tasks are divided into three parallel workstreams (Agent A, B, C) that can be executed concurrently.

---

## Current Architecture Analysis

### Multi-Agent System (Already Implemented)

The system has a sophisticated multi-agent architecture that is **already implemented** but not fully utilized:

```
                    ┌─────────────────────────────────────┐
                    │         CRITIC ORCHESTRATOR         │
                    │   (agent/server/critics/orchestrator.py)   │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  SAFETY CRITIC  │     │EFFICIENCY CRITIC│     │GOAL ALIGN CRITIC│
    │  (safety_critic │     │(efficiency_critic│    │(goal_alignment_  │
    │      .py)       │     │     .py)        │     │   critic.py)    │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
              │                       │                       │
              └───────────────────────┴───────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   CONSENSUS & VERDICT   │
                         │  (Approve/Reject/Escalate)│
                         └─────────────────────────┘
```

**Authority Models Available:**
1. **ADVISORY** - Critics advise but don't block
2. **BLOCKING** - Any rejection blocks decision
3. **ESCALATION** (default) - Risk-based gates with LLM escalation
4. **HIERARCHICAL** - Full LLM review for all decisions

### Dashboard Streaming (Already Implemented)

Two WebSocket channels exist:
1. **`/ws`** - Generic event broadcasting (decisions, risks, goals)
2. **`/ws/unreal`** - High-frequency visualization with thinking state

---

## Root Cause Analysis: Why Only "WAIT" Appears

| Issue | Location | Problem |
|-------|----------|---------|
| Fleet Over-Filtering | `scenario_runner.py:874-916` | Assets filtered out by stale fleet state |
| Goal Selector Empty | `goal_selector.py:301-324` | No pending assets reach inspection check |
| Critic Blocking | `api_decision.py:211-234` | Critics may silently reject to WAIT |
| Dashboard Missing Data | `dashboard.py:168-215` | Feed not showing all decision context |

---

## Parallel Workstream Assignment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONCURRENT WORKSTREAMS                               │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│      AGENT A        │      AGENT B        │           AGENT C               │
│  Decision Pipeline  │  Multi-Agent Wiring │    Dashboard & Frontend         │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ • Fleet coord fix   │ • Critic integration│ • Reasoning feed UI             │
│ • Goal selector fix │ • Explanation agent │ • WebSocket live updates        │
│ • Decision logging  │ • Authority models  │ • Critic verdict display        │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ Files:              │ Files:              │ Files:                          │
│ - scenario_runner.py│ - scenario_runner.py│ - Dashboard.tsx                 │
│ - goal_selector.py  │ - api_decision.py   │ - dashboard.py                  │
│ - world_model.py    │ - orchestrator.py   │ - events.py                     │
│                     │ - explanation_agent │ - api_telemetry.py              │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
```

---

# AGENT A: Decision Pipeline Fix

**Focus:** Fix the core decision-making to produce real inspection goals instead of WAIT.

**Files:** `scenario_runner.py`, `goal_selector.py`, `world_model.py`

## A.1 [DONE] Add Decision Pipeline Debug Logging

**File:** `agent/server/scenario_runner.py`

**Location:** In `_make_decision()` method

Add comprehensive logging to trace decision flow:

```python
async def _make_decision(self, drone_state: DroneSimState) -> None:
    # ... after getting snapshot ...

    logger.info(f"[DECISION] Drone {drone.name}:")
    logger.info(f"  - Total assets in world: {len(snapshot.assets)}")
    logger.info(f"  - Fleet in-progress: {list(self.run_state.assets_in_progress.keys())}")
    logger.info(f"  - Fleet inspected: {list(self.run_state.assets_inspected)}")
    logger.info(f"  - Available after filter: {len(available_snapshot.assets)}")
    logger.info(f"  - Pending inspections: {[a.asset_id for a in available_snapshot.get_pending_assets()]}")

    # ... after goal selection ...
    logger.info(f"  - Selected: {goal.goal_type.value} - {goal.reason}")
```

## A.2 [DONE] Add Stale Fleet State Cleanup

**File:** `agent/server/scenario_runner.py`

**Location:** New method + call in `_tick()`

```python
def _cleanup_stale_fleet_state(self) -> None:
    """Remove stale entries from fleet coordination state.

    Cleans up assets_in_progress entries where the drone is no longer
    actively targeting that asset (e.g., after abort or return).
    """
    if not self.run_state:
        return

    # Get all assets currently being targeted by active drones
    active_targets = {
        ds.target_asset_id
        for ds in self.run_state.drone_states.values()
        if ds.target_asset_id is not None
    }

    # Find and remove stale entries
    stale = set(self.run_state.assets_in_progress.keys()) - active_targets
    for asset_id in stale:
        self.run_state.assets_in_progress.pop(asset_id, None)
        logger.warning(f"Cleaned stale in-progress asset: {asset_id}")

async def _tick(self) -> None:
    # ADD AT START OF TICK:
    self._cleanup_stale_fleet_state()
    # ... rest of tick ...
```

## A.3 [DONE] Add Asset Filtering Safety Bounds

**File:** `agent/server/scenario_runner.py`

**Location:** In `_filter_available_assets()` method

```python
def _filter_available_assets(self, snapshot: WorldSnapshot, drone_id: str) -> WorldSnapshot:
    # ... existing filtering logic ...

    # NEW: Safety check - don't filter out ALL assets
    if not filtered_assets and snapshot.assets:
        logger.warning(
            f"Fleet filtering removed all {len(snapshot.assets)} assets! "
            f"In-progress: {self.run_state.assets_in_progress}, "
            f"Inspected: {self.run_state.assets_inspected}"
        )
        # Fallback: return assets not yet fully inspected
        filtered_assets = [
            a for a in snapshot.assets
            if a.asset_id not in self.run_state.assets_inspected
        ]
        # If still empty, return all assets (reset cycle)
        if not filtered_assets:
            logger.info("All assets inspected - resetting for new inspection cycle")
            filtered_assets = list(snapshot.assets)

    # ... create new snapshot with filtered_assets ...
```

## A.4 [DONE] Add Goal Selector Logging

**File:** `agent/server/goal_selector.py`

**Location:** In `select_goal()` and check methods

```python
async def select_goal(self, world: WorldSnapshot) -> Goal:
    logger.debug(f"[GOAL SELECT] Vehicle battery: {world.vehicle.battery.remaining_percent:.1f}%")
    logger.debug(f"[GOAL SELECT] Assets in snapshot: {len(world.assets)}")
    logger.debug(f"[GOAL SELECT] Pending assets: {len(world.get_pending_assets())}")
    logger.debug(f"[GOAL SELECT] Anomaly assets: {len(world.get_anomaly_assets())}")

    # ... existing logic ...

def _check_inspections(self, world: WorldSnapshot) -> Goal | None:
    pending_assets = world.get_pending_assets()
    logger.debug(f"[INSPECT CHECK] Found {len(pending_assets)} pending assets")

    if pending_assets:
        asset = pending_assets[0]
        logger.info(f"[INSPECT CHECK] Selecting asset: {asset.asset_id} (priority {asset.priority})")
        # ... return Goal ...
```

## A.5 [DONE] Fix World Model Asset State

**File:** `agent/server/world_model.py`

**Location:** `get_pending_assets()` method

Ensure assets with `next_scheduled=None` are always considered pending:

```python
def get_pending_assets(self) -> list[Asset]:
    """Get assets that need inspection, sorted by priority."""
    pending = []
    for asset in self.assets:
        # Asset needs inspection if:
        # 1. Never scheduled (next_scheduled is None) OR
        # 2. Schedule time has passed
        if asset.next_scheduled is None or datetime.now() >= asset.next_scheduled:
            pending.append(asset)

    logger.debug(f"[WORLD MODEL] {len(pending)}/{len(self.assets)} assets pending")
    return sorted(pending, key=lambda a: a.priority)
```

---

# AGENT B: Multi-Agent Orchestration Wiring

**Focus:** Wire up the existing critic system and explanation agent into the decision flow.

**Files:** `scenario_runner.py`, `api_decision.py`, `critics/orchestrator.py`, `monitoring/explanation_agent.py`

## B.1 [DONE] Integrate Critic Orchestrator in Scenario Runner

**File:** `agent/server/scenario_runner.py`

**Location:** Class init and `_make_decision()` method

```python
from agent.server.critics.orchestrator import CriticOrchestrator, AuthorityModel
from agent.server.models.critic_models import CriticConfig

class ScenarioRunner:
    def __init__(
        self,
        seed: int | None = None,
        tick_interval_s: float = 1.0,
        decision_interval_s: float = 5.0,
        enable_critics: bool = True,  # NEW PARAM
    ) -> None:
        # ... existing init ...

        # Initialize critic orchestrator
        self.enable_critics = enable_critics
        if enable_critics:
            self.critic_orchestrator = CriticOrchestrator(
                authority_model=AuthorityModel.ADVISORY,  # Start advisory, upgrade later
                config=CriticConfig(use_llm=False),  # Fast mode initially
            )
        else:
            self.critic_orchestrator = None
```

## B.2 [DONE] Add Critic Validation to Decision Flow

**File:** `agent/server/scenario_runner.py`

**Location:** In `_make_decision()` after goal selection

```python
async def _make_decision(self, drone_state: DroneSimState) -> None:
    # ... existing goal selection ...

    # Validate with critics (if enabled)
    critic_result = None
    if self.critic_orchestrator and goal.goal_type != GoalType.WAIT:
        try:
            risk_score = self._calculate_risk(drone_state)
            approved, escalation = await self.critic_orchestrator.validate_decision(
                decision=goal,
                world_snapshot=available_snapshot,
                risk_assessment={"score": risk_score, "level": self._risk_level(drone_state)},
            )

            critic_result = {
                "approved": approved,
                "escalation": escalation.to_dict() if escalation else None,
            }

            if not approved:
                logger.warning(
                    f"Critics rejected {goal.goal_type.value} for {drone.name}: "
                    f"{escalation.reason if escalation else 'unknown'}"
                )
                # Optionally modify goal based on escalation
                if escalation and escalation.required_action:
                    logger.info(f"Escalation requires: {escalation.required_action}")
        except Exception as e:
            logger.error(f"Critic validation failed: {e}")

    # ... build decision record ...
    decision_record["critic_validation"] = critic_result
```

## B.3 [DONE] Wire Up Explanation Agent

**File:** `agent/server/scenario_runner.py`

**Location:** Class init and `_make_decision()` method

```python
from agent.server.monitoring.explanation_agent import ExplanationAgent

class ScenarioRunner:
    def __init__(self, ..., enable_explanations: bool = False) -> None:
        # ... existing init ...

        self.enable_explanations = enable_explanations
        if enable_explanations:
            self.explanation_agent = ExplanationAgent()
        else:
            self.explanation_agent = None

    async def _make_decision(self, drone_state: DroneSimState) -> None:
        # ... after goal selection and critic validation ...

        # Generate explanation (if enabled)
        explanation = None
        if self.explanation_agent:
            try:
                explanation = await self.explanation_agent.explain_decision(
                    decision=goal,
                    world_snapshot=available_snapshot,
                    risk_score=self._calculate_risk(drone_state),
                )
                decision_record["explanation"] = {
                    "summary": explanation.summary,
                    "factors": explanation.contributing_factors,
                    "counterfactuals": explanation.counterfactuals,
                }
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")
```

## B.4 [DONE] Add Configurable Authority Model

**File:** `agent/server/scenario_runner.py`

**Location:** New method for runtime configuration

```python
def set_critic_authority(self, authority: str) -> None:
    """Change critic authority model at runtime.

    Args:
        authority: One of 'advisory', 'blocking', 'escalation', 'hierarchical'
    """
    if not self.critic_orchestrator:
        logger.warning("Critics not enabled")
        return

    authority_map = {
        "advisory": AuthorityModel.ADVISORY,
        "blocking": AuthorityModel.BLOCKING,
        "escalation": AuthorityModel.ESCALATION,
        "hierarchical": AuthorityModel.HIERARCHICAL,
    }

    if authority.lower() in authority_map:
        self.critic_orchestrator.authority_model = authority_map[authority.lower()]
        logger.info(f"Critic authority set to: {authority}")
    else:
        logger.error(f"Unknown authority model: {authority}")
```

## B.5 [DONE] Broadcast Critic Events via WebSocket

**File:** `agent/server/scenario_runner.py`

**Location:** In `_make_decision()` after critic validation

```python
from agent.server.state import server_state
from agent.server.events import ServerEvent

async def _make_decision(self, drone_state: DroneSimState) -> None:
    # ... after critic validation ...

    # Broadcast critic validation event
    if critic_result and server_state.connection_manager:
        await server_state.connection_manager.broadcast({
            "event": ServerEvent.CRITIC_VALIDATION.value,
            "payload": {
                "drone_id": drone.drone_id,
                "goal": goal.goal_type.value,
                "approved": critic_result["approved"],
                "escalation": critic_result.get("escalation"),
                "timestamp": self.run_state.current_time.isoformat(),
            }
        })
```

---

# AGENT C: Dashboard & Frontend

**Focus:** Update dashboard UI to display rich reasoning feed with critic verdicts and live updates.

**Files:** `Dashboard.tsx`, `dashboard.py`, `events.py`, `api_telemetry.py`

## C.1 [DONE] Enrich Decision Records in Backend

**File:** `agent/server/scenario_runner.py`

**Location:** In `_make_decision()` when building decision_record

```python
decision_record = {
    "type": "decision",
    "timestamp": self.run_state.current_time.isoformat(),
    "elapsed_s": self.run_state.elapsed_seconds,
    "drone_id": drone.drone_id,
    "drone_name": drone.name,
    "action": goal.goal_type.value,
    "confidence": goal.confidence,
    "reason": goal.reason,
    "priority": goal.priority,
    # ... existing fields ...

    # NEW: Reasoning context
    "reasoning_context": {
        "available_assets": len(available_snapshot.assets),
        "pending_inspections": len(available_snapshot.get_pending_assets()),
        "fleet_in_progress": len(self.run_state.assets_in_progress),
        "fleet_completed": len(self.run_state.assets_inspected),
        "battery_ok": drone.battery_percent > 25,
        "battery_percent": drone.battery_percent,
        "weather_ok": self.run_state.environment.wind_speed_ms < 12,
        "wind_speed": self.run_state.environment.wind_speed_ms,
    },

    # NEW: Alternatives considered
    "alternatives": self._get_alternatives_considered(drone_state, available_snapshot, goal),
}
```

Add helper method:

```python
def _get_alternatives_considered(
    self,
    drone_state: DroneSimState,
    snapshot: WorldSnapshot,
    selected_goal: Goal,
) -> list[dict]:
    """Generate list of alternative actions and why they weren't selected."""
    alternatives = []
    drone = drone_state.drone

    # Check why we didn't inspect
    if selected_goal.goal_type != GoalType.INSPECT_ASSET:
        pending = snapshot.get_pending_assets()
        if not pending:
            alternatives.append({
                "action": "inspect_asset",
                "rejected": True,
                "reason": "No pending assets available",
            })
        elif drone.battery_percent < 25:
            alternatives.append({
                "action": "inspect_asset",
                "rejected": True,
                "reason": f"Battery too low ({drone.battery_percent:.0f}%)",
            })

    # Check why we didn't return
    if selected_goal.goal_type != GoalType.RETURN_LOW_BATTERY:
        if drone.battery_percent > 25:
            alternatives.append({
                "action": "return_low_battery",
                "rejected": True,
                "reason": f"Battery sufficient ({drone.battery_percent:.0f}%)",
            })

    return alternatives
```

## C.2 [DONE] Update Dashboard API for Rich Feed

**File:** `agent/server/dashboard.py`

**Location:** `_recent()` function

```python
def _recent(entries: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    """Return the most recent decision entries with full context."""
    recent_entries = entries[-limit:]
    items = []

    for entry in recent_entries[::-1]:
        item = {
            "timestamp": entry.get("timestamp"),
            "elapsed_s": entry.get("elapsed_s"),
            "drone_id": entry.get("drone_id"),
            "drone_name": entry.get("drone_name"),
            "action": entry.get("action"),
            "reason": entry.get("reason"),
            "confidence": entry.get("confidence"),
            "battery_percent": entry.get("battery_percent"),
            "risk_level": entry.get("risk_level"),

            # NEW: Rich context
            "reasoning_context": entry.get("reasoning_context", {}),
            "alternatives": entry.get("alternatives", []),
            "critic_validation": entry.get("critic_validation"),
            "explanation": entry.get("explanation"),

            # Target info if present
            "target_asset": entry.get("target_asset"),
        }
        items.append(item)

    return items
```

## C.3 [DONE] Add Reasoning Feed Component

**File:** `frontend/src/components/ReasoningFeed.tsx` (NEW FILE)

```tsx
import React from 'react';

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

interface DecisionEntry {
    timestamp: string;
    drone_name: string;
    action: string;
    reason: string;
    confidence: number;
    risk_level: string;
    reasoning_context?: ReasoningContext;
    alternatives?: Alternative[];
    critic_validation?: CriticValidation;
}

interface Props {
    decisions: DecisionEntry[];
}

export const ReasoningFeed: React.FC<Props> = ({ decisions }) => {
    const getActionColor = (action: string) => {
        switch (action) {
            case 'inspect_asset': return 'text-green-400';
            case 'return_low_battery': return 'text-yellow-400';
            case 'abort': return 'text-red-400';
            case 'wait': return 'text-gray-400';
            default: return 'text-blue-400';
        }
    };

    const getRiskColor = (level: string) => {
        switch (level?.toUpperCase()) {
            case 'CRITICAL': return 'bg-red-500';
            case 'HIGH': return 'bg-orange-500';
            case 'MODERATE': return 'bg-yellow-500';
            default: return 'bg-green-500';
        }
    };

    return (
        <div className="reasoning-feed space-y-3">
            {decisions.map((d, i) => (
                <div key={i} className="decision-card bg-gray-800 rounded-lg p-3">
                    {/* Header */}
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-gray-400">{d.drone_name}</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${getRiskColor(d.risk_level)}`}>
                            {d.risk_level}
                        </span>
                    </div>

                    {/* Action & Reason */}
                    <h4 className={`text-lg font-semibold ${getActionColor(d.action)}`}>
                        {d.action.replace(/_/g, ' ').toUpperCase()}
                    </h4>
                    <p className="text-sm text-gray-300 mt-1">{d.reason}</p>

                    {/* Context badges */}
                    {d.reasoning_context && (
                        <div className="flex flex-wrap gap-2 mt-2">
                            <span className="text-xs bg-gray-700 px-2 py-1 rounded">
                                🔋 {d.reasoning_context.battery_percent?.toFixed(0)}%
                            </span>
                            <span className="text-xs bg-gray-700 px-2 py-1 rounded">
                                📦 {d.reasoning_context.pending_inspections} pending
                            </span>
                            <span className="text-xs bg-gray-700 px-2 py-1 rounded">
                                🌬️ {d.reasoning_context.wind_speed?.toFixed(1)} m/s
                            </span>
                        </div>
                    )}

                    {/* Critic Validation */}
                    {d.critic_validation && (
                        <div className={`mt-2 p-2 rounded text-xs ${
                            d.critic_validation.approved ? 'bg-green-900' : 'bg-red-900'
                        }`}>
                            <span className="font-semibold">
                                Critics: {d.critic_validation.approved ? '✓ Approved' : '✗ Rejected'}
                            </span>
                            {d.critic_validation.escalation && (
                                <p className="mt-1">{d.critic_validation.escalation.reason}</p>
                            )}
                        </div>
                    )}

                    {/* Alternatives */}
                    {d.alternatives && d.alternatives.length > 0 && (
                        <details className="mt-2">
                            <summary className="text-xs text-gray-500 cursor-pointer">
                                Why not other actions?
                            </summary>
                            <ul className="text-xs text-gray-400 mt-1 ml-4">
                                {d.alternatives.map((alt, j) => (
                                    <li key={j}>
                                        <span className="text-gray-500">{alt.action}:</span> {alt.reason}
                                    </li>
                                ))}
                            </ul>
                        </details>
                    )}

                    {/* Timestamp */}
                    <div className="text-xs text-gray-500 mt-2">
                        {new Date(d.timestamp).toLocaleTimeString()}
                    </div>
                </div>
            ))}
        </div>
    );
};
```

## C.4 [DONE] Add WebSocket Live Updates

**File:** `frontend/src/components/Dashboard.tsx`

**Location:** Add WebSocket subscription for live decisions

```tsx
import { useEffect, useState, useRef } from 'react';

// In Dashboard component:
const [liveDecisions, setLiveDecisions] = useState<DecisionEntry[]>([]);
const wsRef = useRef<WebSocket | null>(null);

useEffect(() => {
    // Connect to WebSocket for live updates
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
        console.log('Dashboard WebSocket connected');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.event === 'SERVER_DECISION') {
                setLiveDecisions(prev => [data.payload, ...prev].slice(0, 20));
            }

            if (data.event === 'CRITIC_VALIDATION') {
                // Update the most recent decision with critic results
                setLiveDecisions(prev => {
                    if (prev.length === 0) return prev;
                    const updated = [...prev];
                    if (updated[0].drone_id === data.payload.drone_id) {
                        updated[0] = {
                            ...updated[0],
                            critic_validation: {
                                approved: data.payload.approved,
                                escalation: data.payload.escalation,
                            }
                        };
                    }
                    return updated;
                });
            }
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('Dashboard WebSocket closed');
    };

    return () => {
        ws.close();
    };
}, []);

// Use liveDecisions in render, falling back to polled data
const displayDecisions = liveDecisions.length > 0 ? liveDecisions : runData?.recent ?? [];
```

## C.5 [DONE] Update Dashboard Layout for Reasoning Feed

**File:** `frontend/src/components/Dashboard.tsx`

**Location:** Replace existing reasoning feed section

```tsx
import { ReasoningFeed } from './ReasoningFeed';

// In render:
<article className="card reasoning-card">
    <div className="card-header flex justify-between items-center">
        <h2>Reasoning Feed</h2>
        <span className="text-xs text-gray-400">
            {liveDecisions.length > 0 ? '🟢 Live' : '⚪ Polling'}
        </span>
    </div>
    <div className="feed-container max-h-96 overflow-y-auto">
        <ReasoningFeed decisions={displayDecisions} />
    </div>
</article>
```

---

## Dependency Graph

```
AGENT A (Decision Pipeline)          AGENT B (Multi-Agent)           AGENT C (Dashboard)
─────────────────────────           ──────────────────────          ────────────────────
        │                                    │                              │
   A.1 Logging ◄──────────────────────────── │ ─────────────────────────────┤
        │                                    │                              │
   A.2 Stale Cleanup                    B.1 Critic Init                C.1 Enrich Records
        │                                    │                              │
   A.3 Bounds Check                     B.2 Critic Validation ─────────► C.2 Dashboard API
        │                                    │                              │
   A.4 Goal Logging                     B.3 Explanation Agent          C.3 ReasoningFeed.tsx
        │                                    │                              │
   A.5 World Model Fix                  B.4 Authority Config           C.4 WebSocket Live
        │                                    │                              │
        └──────────────────► B.5 Event Broadcast ◄─────────────────────► C.5 Dashboard Layout
```

**Key Dependencies:**
- A.1 can inform debugging for B and C
- B.2 (critic validation) feeds into C.2 (dashboard API)
- B.5 (event broadcast) required for C.4 (WebSocket)
- All A tasks are independent of B and C
- C tasks can proceed with mock data while waiting for A/B

---

## Testing Checklist

### Agent A Tests
- [ ] Logging shows asset counts at each filtering stage
- [ ] Stale cleanup removes orphaned in-progress entries
- [ ] Bounds check prevents filtering all assets
- [ ] Goal selector finds and returns pending assets
- [ ] `inspect_asset` decisions appear in logs

### Agent B Tests
- [ ] Critic orchestrator initializes without errors
- [ ] Critic validation runs on non-WAIT decisions
- [ ] Explanation agent generates summaries
- [ ] Authority model can be changed at runtime
- [ ] CRITIC_VALIDATION events broadcast via WebSocket

### Agent C Tests
- [ ] Dashboard shows reasoning context (battery, assets, weather)
- [ ] Alternatives section expands to show rejected options
- [ ] Critic verdicts display with approve/reject styling
- [ ] WebSocket connects and receives live updates
- [ ] Feed updates in real-time without page refresh

---

## Success Metrics

| Metric | Target | Owner |
|--------|--------|-------|
| Decision diversity | ≥3 types in 5-min run | Agent A |
| Asset coverage | All inspected in mission | Agent A |
| Critic activity | 100% decisions validated | Agent B |
| Feed latency | <500ms to dashboard | Agent C |
| Multi-drone coordination | No asset conflicts | Agent A |

---

## Files Summary by Agent

| Agent | Files to Create/Modify |
|-------|------------------------|
| **A** | `scenario_runner.py`, `goal_selector.py`, `world_model.py` |
| **B** | `scenario_runner.py`, `api_decision.py`, `critics/orchestrator.py` |
| **C** | `Dashboard.tsx`, `ReasoningFeed.tsx` (new), `dashboard.py` |

**Shared file:** `scenario_runner.py` - coordinate changes between A and B
