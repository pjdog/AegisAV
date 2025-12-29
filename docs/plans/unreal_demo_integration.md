# Unreal Engine Demo Integration Plan

## Vision

A 3D Unreal visualization where:
- Multiple drones fly through an environment (farm, industrial site, etc.)
- Each drone has a floating "thought bubble" showing real-time agent reasoning
- Thought bubbles reveal: current goal, risk assessment, critic evaluations, and decisions
- Viewers can see the AI "thinking" as it happens - making autonomy transparent

## Demo Experience (Target: 3-5 minute showcase)

```
[Scene: Industrial facility with solar panels, tanks, and perimeter]

1. LAUNCH (30s)
   - Drones power up, thought bubbles appear: "Initializing... Loading mission..."
   - Goal displays: "INSPECT 12 assets, starting with high-priority solar array"

2. INSPECTION FLIGHT (2m)
   - Drone 1 approaches solar panel
   - Thought bubble: "Approaching asset_001 | Risk: LOW | Battery: 85%"
   - Detection popup: "Anomaly detected: panel crack, severity 0.7"
   - Critic flash: "Safety: APPROVE | Efficiency: APPROVE | Goal: APPROVE"

3. RISK RESPONSE (1m)
   - Weather changes (wind picks up)
   - Thought bubble shifts: "Wind 15kts | Risk: MODERATE | Considering RTL..."
   - Decision: "Reducing altitude, continuing inspection"
   - Critics re-evaluate with concerns visible

4. MULTI-DRONE COORDINATION (1m)
   - Drone 2 enters frame, different thought bubble
   - Shows parallel reasoning: "Drone 1 handling sector A, I'll cover sector B"
   - Battery warning on Drone 1: "Battery 25% | Goal: RETURN_LOW_BATTERY"

5. MISSION COMPLETE (30s)
   - Drones return to dock
   - Summary overlay: "12 assets inspected, 2 anomalies found, 0 safety incidents"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UNREAL ENGINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Drone Actors │  │ ThoughtBubble│  │ Demo Controller      │  │
│  │ (3D models)  │  │ UI Widgets   │  │ (scenario playback)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └────────────┬────┴──────────────────────┘              │
│                      │                                           │
│              ┌───────▼────────┐                                 │
│              │ WebSocket      │                                 │
│              │ Client (C++)   │                                 │
│              └───────┬────────┘                                 │
└──────────────────────┼──────────────────────────────────────────┘
                       │ ws://localhost:8000/ws/unreal
                       │
┌──────────────────────┼──────────────────────────────────────────┐
│                      │         AEGISAV SERVER                    │
│              ┌───────▼────────┐                                 │
│              │ WebSocket      │                                 │
│              │ /ws/unreal     │  Enhanced endpoint for          │
│              └───────┬────────┘  rich visualization data        │
│                      │                                           │
│    ┌─────────────────┼─────────────────┐                        │
│    │                 │                 │                        │
│  ┌─▼──────┐   ┌──────▼─────┐   ┌──────▼──────┐                 │
│  │Decision│   │ Critics    │   │ World Model │                 │
│  │Engine  │   │ Orchestr.  │   │ (positions) │                 │
│  └────────┘   └────────────┘   └─────────────┘                 │
│                                                                  │
│              ┌────────────────┐                                 │
│              │ScenarioRunner  │  Drives simulation              │
│              │(deterministic) │  with callbacks                 │
│              └────────────────┘                                 │
└──────────────────────────────────────────────────────────────────┘
```

## Backend Changes Required

### 1. New WebSocket Endpoint: `/ws/unreal`

Higher-bandwidth stream with visualization-specific data:

```python
# New event types for Unreal
class UnrealEventType(str, Enum):
    DRONE_STATE = "drone_state"           # Position, battery, mode
    THINKING_START = "thinking_start"     # Agent begins reasoning
    THINKING_UPDATE = "thinking_update"   # Intermediate thoughts
    THINKING_COMPLETE = "thinking_complete" # Final decision
    CRITIC_EVALUATION = "critic_eval"     # Individual critic result
    RISK_ASSESSMENT = "risk_assessment"   # Risk factors breakdown
    ANOMALY_DETECTED = "anomaly_detected" # Visual detection event
    GOAL_CHANGED = "goal_changed"         # Mission goal shift
```

### 2. ThinkingUpdate Message Format

```json
{
  "event": "thinking_update",
  "drone_id": "drone_001",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "cognitive_state": {
    "level": "DELIBERATIVE",  // REACTIVE, DELIBERATIVE, REFLECTIVE, PREDICTIVE
    "urgency": "MODERATE",
    "current_goal": "INSPECT_ASSET",
    "target": "solar_panel_007"
  },
  "reasoning": {
    "situation": "Approaching high-priority asset, no anomalies in view",
    "considerations": [
      "Battery at 72% - sufficient for inspection",
      "Wind 8kts from NW - within limits",
      "Asset overdue for inspection by 2 days"
    ],
    "options_evaluated": [
      {"action": "INSPECT", "score": 0.92, "selected": true},
      {"action": "SKIP", "score": 0.15, "selected": false},
      {"action": "RTL", "score": 0.08, "selected": false}
    ]
  },
  "critics": {
    "safety": {"verdict": "APPROVE", "confidence": 0.95, "concerns": []},
    "efficiency": {"verdict": "APPROVE", "confidence": 0.88, "concerns": ["Could optimize route"]},
    "goal_alignment": {"verdict": "APPROVE", "confidence": 0.97, "concerns": []}
  },
  "risk": {
    "overall": 0.25,
    "level": "LOW",
    "factors": {
      "battery": 0.15,
      "weather": 0.20,
      "distance_to_dock": 0.30,
      "vehicle_health": 0.10
    }
  },
  "decision": {
    "action": "INSPECT",
    "confidence": 0.92,
    "reasoning": "High-priority asset inspection with favorable conditions"
  }
}
```

### 3. DroneState Message Format

```json
{
  "event": "drone_state",
  "drone_id": "drone_001",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "position": {
    "lat": 37.7749,
    "lon": -122.4194,
    "alt_msl": 50.0,
    "alt_agl": 30.0,
    "heading": 270.0
  },
  "velocity": {
    "groundspeed": 5.2,
    "vertical": 0.0
  },
  "state": {
    "armed": true,
    "mode": "GUIDED",
    "battery_percent": 72.0,
    "battery_voltage": 22.4
  },
  "current_action": "INSPECT",
  "target_asset": "solar_panel_007"
}
```

### 4. Enhanced ScenarioRunner Callbacks

```python
class ScenarioRunner:
    async def run(self, ...):
        # Existing callbacks
        self.on_decision: Callable[[Decision], Awaitable[None]]
        self.on_event: Callable[[Event], Awaitable[None]]

        # New callbacks for Unreal
        self.on_thinking_start: Callable[[str, CognitiveContext], Awaitable[None]]
        self.on_thinking_update: Callable[[str, ThinkingState], Awaitable[None]]
        self.on_drone_state: Callable[[str, DroneState], Awaitable[None]]
        self.on_critic_result: Callable[[str, CriticResponse], Awaitable[None]]
```

## Unreal Implementation

### 1. WebSocket Client (C++ or Blueprint)

```cpp
// UnrealAegisClient.h
UCLASS()
class UUnrealAegisClient : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable)
    void Connect(const FString& ServerUrl);

    DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnThinkingUpdate,
        FString, DroneId, FAegisThinkingState, State);
    UPROPERTY(BlueprintAssignable)
    FOnThinkingUpdate OnThinkingUpdate;

    DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnDroneState,
        FString, DroneId, FAegisDroneState, State);
    UPROPERTY(BlueprintAssignable)
    FOnDroneState OnDroneState;
};
```

### 2. Thought Bubble Widget

```
ThoughtBubble_WBP (Widget Blueprint)
├── Canvas Panel
│   ├── Background (rounded rect, semi-transparent)
│   ├── Header
│   │   ├── DroneIcon
│   │   ├── DroneId Text
│   │   └── CognitiveLevel Badge (color-coded)
│   ├── Goal Section
│   │   ├── "GOAL:" Label
│   │   └── CurrentGoal Text (INSPECT solar_panel_007)
│   ├── Risk Meter
│   │   ├── Risk Bar (color gradient: green→yellow→red)
│   │   └── Risk Factors (collapsible)
│   ├── Critics Panel
│   │   ├── Safety Icon + Verdict
│   │   ├── Efficiency Icon + Verdict
│   │   └── GoalAlign Icon + Verdict
│   ├── Reasoning Text (scrolling recent thoughts)
│   └── Decision Flash (appears on decision)
│       ├── Action Icon
│       ├── "INSPECT" Text
│       └── Confidence Bar
```

### 3. Visual States

| Cognitive Level | Bubble Color | Animation |
|----------------|--------------|-----------|
| REACTIVE | Blue | Quick pulse |
| DELIBERATIVE | Yellow | Steady glow |
| REFLECTIVE | Purple | Slow wave |
| PREDICTIVE | Cyan | Shimmer |

| Risk Level | Indicator |
|------------|-----------|
| LOW | Green bar, no border |
| MODERATE | Yellow bar, amber border |
| HIGH | Orange bar, pulsing border |
| CRITICAL | Red bar, alarm animation |

| Critic Verdict | Icon |
|----------------|------|
| APPROVE | Green checkmark |
| APPROVE_WITH_CONCERNS | Yellow checkmark |
| ESCALATE | Orange warning |
| REJECT | Red X |

## Implementation Phases

### Phase 1: Backend Streaming (Python) [~4 hours]
- [ ] Add `/ws/unreal` endpoint with enhanced message types
- [ ] Create `ThinkingUpdate` and `DroneState` message schemas
- [ ] Hook into ScenarioRunner to emit thinking events
- [ ] Add cognitive state tracking to decision engine
- [ ] Test with simple WebSocket client

### Phase 2: Unreal Client (C++/BP) [~6 hours]
- [ ] Create WebSocket client plugin/module
- [ ] Parse JSON messages into UE structs
- [ ] Implement event dispatchers for Blueprint binding
- [ ] Test connection and message flow

### Phase 3: Visualization (Blueprints/UMG) [~8 hours]
- [ ] Create ThoughtBubble widget blueprint
- [ ] Build Drone actor with bubble attachment point
- [ ] Implement visual states (colors, animations)
- [ ] Add risk meter and critic verdict icons
- [ ] Create decision flash animation

### Phase 4: Demo Scene [~4 hours]
- [ ] Set up environment (facility, assets to inspect)
- [ ] Place dock location and flight paths
- [ ] Configure 2-3 drones with different starting positions
- [ ] Create demo scenario in ScenarioRunner
- [ ] Record/polish final demo

## Alternative: Web-Based 3D Demo

If Unreal setup is too heavy, consider a **Three.js web demo**:
- Runs in browser alongside existing dashboard
- Same WebSocket data source
- Simpler to iterate on
- Can be embedded in presentations

## Files to Create/Modify

**Backend:**
- `agent/server/unreal_stream.py` (new) - Unreal WebSocket handler
- `agent/server/main.py` - Add `/ws/unreal` route
- `agent/server/scenario_runner.py` - Add thinking callbacks
- `agent/server/models/unreal_models.py` (new) - Message schemas

**Unreal (new project or plugin):**
- `Source/AegisViz/AegisWebSocketClient.cpp`
- `Source/AegisViz/AegisDataTypes.h`
- `Content/Blueprints/BP_DroneActor.uasset`
- `Content/Widgets/WBP_ThoughtBubble.uasset`
- `Content/Maps/DemoFacility.umap`

**Assets:**
- `simulation/assets/logo_shield.svg` - Main Aegis Shield Logo
- `simulation/assets/icon_ai.svg` - AI/Cognitive State Icon
- `simulation/assets/icon_vision.svg` - Computer Vision Icon


## Success Criteria

1. **Real-time sync**: <100ms latency from decision to bubble update
2. **Multi-drone**: 3+ drones with independent thought bubbles
3. **Transparency**: Viewer understands WHY each decision was made
4. **Polish**: Smooth animations, clear typography, no visual glitches
5. **Demo length**: 3-5 minutes of compelling autonomous behavior
