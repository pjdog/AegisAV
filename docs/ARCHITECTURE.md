# AegisAV System Architecture

## Table of Contents
- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [System Layers](#system-layers)
- [Component Details](#component-details)
- [Communication Protocols](#communication-protocols)
- [Data Models](#data-models)
- [Decision Flow](#decision-flow)
- [Deployment Topologies](#deployment-topologies)

---

## Overview

AegisAV implements a **supervisory autonomy architecture** that separates high-level agentic decision-making from low-level flight control. This design enables:

- **Certification-friendly** autonomy (classical control handles safety-critical functions)
- **Flexible deployment** (agent can run onboard, at edge, or in cloud)
- **Explainable decisions** (structured reasoning traces)
- **Graceful degradation** (vehicle remains safe if agent fails)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION LAYER (Agentic)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Agent Server                            │  │
│  │  ┌──────────┐ ┌────────────┐ ┌───────────┐ ┌──────────┐  │  │
│  │  │  World   │ │    Goal    │ │    Risk   │ │ Decision │  │  │
│  │  │  Model   │→│  Selector  │→│ Evaluator │→│  Logger  │  │  │
│  │  └──────────┘ └────────────┘ └───────────┘ └──────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/gRPC
┌────────────────────────────▼────────────────────────────────────┐
│                   EXECUTION LAYER (Translation)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Agent Client                           │  │
│  │  ┌────────────────┐           ┌──────────────────┐       │  │
│  │  │ State Collector │          │  Action Executor  │       │  │
│  │  │  (Telemetry)    │──────────│  (Commands)       │       │  │
│  │  └────────────────┘           └──────────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ MAVLink
┌────────────────────────────▼────────────────────────────────────┐
│                    CONTROL LAYER (Classical)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ArduPilot / PX4                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │  │
│  │  │ Attitude │ │ Position │ │ Failsafe │ │   Mission    │ │  │
│  │  │ Control  │ │ Control  │ │  Logic   │ │   Executor   │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Philosophy

### 1. Supervisory, Not Replacement

The agent does **not** replace the flight controller. Instead, it:
- Observes vehicle state
- Reasons about mission goals
- Issues high-level commands
- Monitors execution and adapts

The flight controller handles:
- Attitude stabilization
- Position holding
- Safety failsafes (geofence, RTL, battery)

### 2. Agentic Decision-Making

The system exhibits agency through:

| Property | Implementation |
|----------|----------------|
| **Goal-Directed** | Selects goals based on mission state, not just waypoint sequences |
| **Adaptive** | Replans when anomalies detected or conditions change |
| **Self-Aware** | Maintains confidence scores; knows when to abort |
| **Explainable** | Logs structured reasoning for every decision |

### 3. Separation of Concerns

```
Heavy computation ───────────────► Agent Server (edge/cloud)
    - World modeling                    
    - Planning/replanning               
    - Risk assessment                   

Light execution ─────────────────► Agent Client (onboard)
    - Telemetry aggregation
    - Command translation
    - Connection management

Safety-critical ─────────────────► ArduPilot (onboard)
    - Stabilization
    - Failsafes
    - Navigation
```

---

## System Layers

### Decision Layer: Agent Server

**Purpose**: High-level reasoning and decision-making

**Components**:

| Component | Responsibility |
|-----------|----------------|
| **World Model** | Maintains unified view of vehicle, assets, environment, dock |
| **Goal Selector** | Prioritizes competing goals based on state and history |
| **Risk Evaluator** | Assesses battery, weather, operational risks |
| **Planner** | Generates mission plans; handles replanning |
| **Decision Logger** | Records decisions with reasoning traces |

**Characteristics**:
- May be compute-intensive (ML inference, optimization)
- Tolerates 100-500ms latency
- Can be deployed remotely

### Execution Layer: Agent Client

**Purpose**: Bridge between agent decisions and flight controller

**Components**:

| Component | Responsibility |
|-----------|----------------|
| **State Collector** | Aggregates MAVLink telemetry into structured state |
| **Action Executor** | Translates actions to MAVLink commands |
| **Connection Manager** | Handles server connection, heartbeats, reconnection |

**Characteristics**:
- Lightweight (can run on companion computer)
- Low latency (< 50ms loop time)
- Must run close to vehicle

### Control Layer: ArduPilot SITL

**Purpose**: Vehicle stabilization and safety

**Capabilities Used**:
- GUIDED mode for waypoint commands
- AUTO mode for mission execution
- RTL (Return to Launch) failsafe
- Battery management
- Geofence

---

## Component Details

### World Model (`agent/server/world_model.py`)

Maintains a fused, consistent view of the operational environment:

```python
class WorldModel:
    """
    Unified state representation for decision-making.
    
    Updates from:
    - Agent client state reports (vehicle telemetry)
    - Asset database (inspection targets)
    - Environment services (weather API, mock in SITL)
    - Anomaly detection (vision pipeline, mock in SITL)
    """
    
    vehicle: VehicleState       # Position, velocity, battery, mode
    assets: List[Asset]         # Inspection targets with status
    dock: DockState             # Dock position, availability, charge
    environment: Environment    # Weather, wind, visibility
    mission: MissionState       # Current mission progress
    anomalies: List[Anomaly]    # Detected issues
    
    def update_vehicle(self, state: VehicleState) -> None: ...
    def get_snapshot(self) -> WorldSnapshot: ...
    def time_since_last_update(self) -> timedelta: ...
```

### Goal Selector (`agent/server/goal_selector.py`)

Chooses the next goal based on current state:

```python
class GoalSelector:
    """
    Selects goals by evaluating current world state against
    mission objectives and operational constraints.
    
    Goal Types:
    - INSPECT: Visit an asset for inspection
    - RETURN: Return to dock for recharge
    - ABORT: Terminate mission due to risk
    - WAIT: Hold position pending condition
    """
    
    def select_goal(self, world: WorldSnapshot) -> Goal:
        """
        Priority logic:
        1. ABORT if risk thresholds exceeded
        2. RETURN if battery critical
        3. INSPECT if assets need attention
        4. RETURN if mission complete
        5. WAIT if no actionable goal
        """
        ...
```

### Risk Evaluator (`agent/server/risk_evaluator.py`)

Assesses operational risks and gates decisions:

```python
class RiskEvaluator:
    """
    Evaluates risk factors and provides go/no-go decisions.
    
    Risk Factors:
    - Battery: Remaining capacity vs return distance
    - Weather: Wind speed, visibility, precipitation
    - Vehicle: Health status, sensor validity
    - Connectivity: Agent server link quality
    """
    
    def evaluate(self, world: WorldSnapshot) -> RiskAssessment:
        """Returns aggregated risk score and component breakdown."""
        ...
    
    def should_abort(self, risk: RiskAssessment) -> bool:
        """Returns True if any risk exceeds abort threshold."""
        ...
```

### MAVLink Interface (`autonomy/mavlink_interface.py`)

Manages communication with the flight controller:

```python
class MAVLinkInterface:
    """
    MAVLink connection and message handling.
    
    Supports:
    - UDP connection to SITL
    - Serial connection to Pixhawk (future)
    - Message subscription and publishing
    - Heartbeat monitoring
    """
    
    async def connect(self, connection_string: str) -> None: ...
    async def send_position_command(self, lat, lon, alt) -> None: ...
    async def set_mode(self, mode: str) -> None: ...
    async def arm(self) -> None: ...
    async def disarm(self) -> None: ...
    
    # Telemetry subscriptions
    def on_position(self, callback: Callable) -> None: ...
    def on_attitude(self, callback: Callable) -> None: ...
    def on_battery(self, callback: Callable) -> None: ...
```

---

## Communication Protocols

### Agent Server ↔ Agent Client

**Protocol**: HTTP/REST (recommended) or gRPC

**Endpoints**:

```
POST /state          # Client sends vehicle state
  Request:  VehicleState
  Response: Decision

GET  /health         # Server health check
  Response: HealthStatus

GET  /config         # Fetch runtime configuration
  Response: AgentConfig
```

**Sequence**:

```
┌──────────┐              ┌──────────┐
│  Client  │              │  Server  │
└────┬─────┘              └────┬─────┘
     │                         │
     │  POST /state            │
     │ ───────────────────────►│
     │                         │ Process state
     │                         │ Update world model
     │                         │ Select goal
     │                         │ Evaluate risk
     │                         │ Generate decision
     │  Decision               │
     │ ◄───────────────────────│
     │                         │
     │  Execute action         │
     ▼                         ▼
```

### Agent Client ↔ ArduPilot

**Protocol**: MAVLink 2.0 over UDP (SITL) or Serial (hardware)

**Key Messages Used**:

| Message | Direction | Purpose |
|---------|-----------|---------|
| `HEARTBEAT` | Both | Connection status |
| `GLOBAL_POSITION_INT` | From AP | Vehicle position |
| `ATTITUDE` | From AP | Vehicle orientation |
| `SYS_STATUS` | From AP | Battery, health |
| `SET_POSITION_TARGET_GLOBAL_INT` | To AP | Position commands |
| `COMMAND_LONG` | To AP | Arm, mode change, etc. |

---

## Data Models

### Core Types

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

@dataclass
class Position:
    latitude: float   # degrees
    longitude: float  # degrees
    altitude: float   # meters MSL
    relative_alt: float  # meters AGL

@dataclass
class Velocity:
    vx: float  # m/s north
    vy: float  # m/s east
    vz: float  # m/s down

@dataclass
class Attitude:
    roll: float   # radians
    pitch: float  # radians
    yaw: float    # radians

@dataclass 
class BatteryState:
    voltage: float           # volts
    current: float           # amps
    remaining_percent: float # 0-100
    time_remaining: int      # seconds estimate

class FlightMode(Enum):
    STABILIZE = "STABILIZE"
    GUIDED = "GUIDED"
    AUTO = "AUTO"
    RTL = "RTL"
    LAND = "LAND"

@dataclass
class VehicleState:
    timestamp: datetime
    position: Position
    velocity: Velocity
    attitude: Attitude
    battery: BatteryState
    mode: FlightMode
    armed: bool
    healthy: bool
```

### Decision Types

```python
class ActionType(Enum):
    GOTO = "goto"           # Fly to position
    INSPECT = "inspect"     # Perform asset inspection
    RETURN = "return"       # Return to dock
    DOCK = "dock"           # Land and dock
    ABORT = "abort"         # Emergency abort
    WAIT = "wait"           # Hold position
    RECHARGE = "recharge"   # Recharge at dock

@dataclass
class Decision:
    action: ActionType
    parameters: dict        # Action-specific params
    confidence: float       # 0.0-1.0
    reasoning: str          # Human-readable explanation
    risk_assessment: dict   # Risk factors considered
    timestamp: datetime

    def to_log_entry(self) -> dict:
        """Returns structured log entry for decision trace."""
        ...
```

---

## Decision Flow

### Main Loop

```python
async def agent_loop():
    """Main decision loop running on agent server."""
    
    while running:
        # 1. Receive state update from client
        state = await receive_state_update()
        
        # 2. Update world model
        world_model.update(state)
        
        # 3. Get world snapshot
        snapshot = world_model.get_snapshot()
        
        # 4. Evaluate risks
        risk = risk_evaluator.evaluate(snapshot)
        
        # 5. Check abort conditions
        if risk_evaluator.should_abort(risk):
            decision = Decision(
                action=ActionType.ABORT,
                reasoning=f"Risk threshold exceeded: {risk.summary}",
                confidence=1.0
            )
        else:
            # 6. Select goal
            goal = goal_selector.select_goal(snapshot)
            
            # 7. Generate plan/action
            decision = planner.plan_for_goal(goal, snapshot)
        
        # 8. Log decision
        decision_logger.log(decision, snapshot, risk)
        
        # 9. Return decision to client
        await send_decision(decision)
```

### Decision Tree (Simplified)

```
                           ┌─────────────────┐
                           │  Receive State  │
                           └────────┬────────┘
                                    │
                           ┌────────▼────────┐
                           │   Update World  │
                           │      Model      │
                           └────────┬────────┘
                                    │
                           ┌────────▼────────┐
                           │  Evaluate Risk  │
                           └────────┬────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
               Risk > ABORT    Risk > WARN    Risk OK
                     │              │              │
              ┌──────▼──────┐      │       ┌──────▼──────┐
              │    ABORT    │      │       │ Select Goal │
              └─────────────┘      │       └──────┬──────┘
                                   │              │
                            ┌──────▼──────┐       │
                            │   Continue  │       │
                            │  with WARN  │       │
                            └─────────────┘       │
                                                  │
                    ┌────────────┬────────────────┼────────────┐
                    │            │                │            │
              Battery Low    Assets Need     Mission Done    No Action
                    │         Inspection          │            │
             ┌──────▼──────┐   ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
             │   RETURN    │   │   INSPECT   │ │   RETURN    │ │    WAIT     │
             └─────────────┘   └─────────────┘ └─────────────┘ └─────────────┘
```

---

## Deployment Topologies

### Development (SITL)

```
┌─────────────────────────────────────────┐
│              Developer Machine           │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  Gazebo  │  │ ArduPilot│  │ Agent  │ │
│  │          │◄─┤   SITL   │◄─┤ Client │ │
│  └──────────┘  └──────────┘  └───┬────┘ │
│                                  │      │
│                           ┌──────▼────┐ │
│                           │   Agent   │ │
│                           │   Server  │ │
│                           └───────────┘ │
└─────────────────────────────────────────┘

All components run locally via localhost connections.
```

### Edge Deployment

```
┌───────────────────────┐      ┌─────────────────────┐
│       Vehicle         │      │    Edge Computer    │
│                       │      │                     │
│  ┌──────────┐         │      │  ┌──────────────┐   │
│  │ Pixhawk  │◄────────┼──────┼──┤ Agent Client │   │
│  │ (ArduPilot)        │ UART │  └───────┬──────┘   │
│  └──────────┘         │      │          │ Local    │
│                       │      │  ┌───────▼──────┐   │
│                       │      │  │ Agent Server │   │
│                       │      │  └──────────────┘   │
└───────────────────────┘      └─────────────────────┘

Agent runs on companion computer (Jetson, Pi, etc.)
```

### Cloud/Ground Station

```
┌───────────────────────┐      ┌─────────────────────┐
│       Vehicle         │      │      Cloud/GCS      │
│                       │      │                     │
│  ┌──────────┐         │      │  ┌──────────────┐   │
│  │ Pixhawk  │◄────────┼──┐   │  │ Agent Server │   │
│  └──────────┘         │  │   │  └───────▲──────┘   │
│       ▲               │  │   │          │          │
│  ┌────┴─────┐         │  │   └──────────┼──────────┘
│  │ Agent    │◄────────┼──┼──────────────┘
│  │ Client   │ LTE/5G  │  │   HTTP over Internet
│  └──────────┘         │  │
└───────────────────────┘  │
                           │ MAVLink over telemetry
```

---

## Configuration

### Agent Configuration (`configs/agent_config.yaml`)

```yaml
agent:
  name: "aegis-primary"
  loop_rate_hz: 10
  
server:
  host: "0.0.0.0"
  port: 8080

mavlink:
  connection: "udp:127.0.0.1:14550"
  timeout_ms: 1000
  
decision:
  confidence_threshold: 0.7
  max_replan_attempts: 3
```

### Risk Thresholds (`configs/risk_thresholds.yaml`)

```yaml
battery:
  warning_percent: 30
  critical_percent: 20
  abort_percent: 15

wind:
  warning_ms: 8
  abort_ms: 12

connectivity:
  heartbeat_timeout_s: 5
  abort_timeout_s: 30
```

---

## Next Steps

See [Implementation Plan](../implementation_plan.md) for phased development approach.
