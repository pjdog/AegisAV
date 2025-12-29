# AegisAV Project Roadmap

## Phase 1: Foundation (✅ Complete)
**Goal:** Establish the core infrastructure for autonomous inspection.
- [x] **Architecture Design**: Decision/Execution/Control layer separation
- [x] **Simulation Stack**: Unreal Engine + AirSim + ArduPilot SITL integration
- [x] **Vision Layer (Basic)**: Simulated camera capture and YOLOv8 integration
- [x] **Dashboard (V1)**: Real-time telemetry and camera feed
- [x] **Execution Engine**: Basic MAVLink primitives (TAKEOFF, GOTO, RTL)

## Phase 2: Multi-Agent Validation (✅ Complete)
**Goal:** Ensure autonomy is safe, explainable, and verified.
- [x] **Multi-Critic Architecture**:
    - `SafetyCritic`: Battery, weather, and geofence validation
    - `EfficiencyCritic`: Resource usage optimization
    - `GoalAlignmentCritic`: Strategic consisteny checks
- [x] **Escalation System**: Advisory → Blocking → Hierarchical review flow
- [x] **Outcome Tracking**: Feedback loop for decision success/failure
- [x] **Integration Testing**: End-to-end pipeline validation (30/30 tests passing)

## Phase 3: Intelligence & Production Readiness (🚧 In Progress)
**Goal:** Enhance decision capabilities and operational robustness.

### 3.1 System Configuration (✅ Complete)
- [x] **Centralized Config**: `ConfigManager` with YAML & Env var support
- [x] **Runtime Settings**: `SettingsPanel` UI for hot-reloading params
- [x] **Environment Profiles**: Dev/Prod presets (`configs/aegis_config.*.yaml`)

### 3.2 Efficiency & Monitoring (✅ Complete)
- [x] **Edge Computing Policy**: Configurable bandwidth & anomaly gating
- [x] **Cost Tracking**: LLM token usage monitoring & budget enforcement
- [x] **Performance Optimization**: Client-side feedback loop improvements

### 3.3 Advanced Reasoning (TODO)
- [ ] **Hybrid Decision Engine**: Route complex decisions to LLM, simple ones to rules
- [ ] **Explanation Agent**: Generate natural language audit trails for post-mission analysis
- [ ] **Learning System**: Update risk models based on `OutcomeTracker` data

### 3.4 Multi-Drone Orchestration (New)
- [ ] **Swarm Server**: Centralized fleet manager for efficient task allocation and spatial deconfliction.
- [ ] **Preconfigured Simulations**:
    - *Solar Farm Sweep*: 2 drones splitting a large field for optimal coverage speed.
    - *Linear Inspection*: 2 drones inspecting power lines in tandem (leapfrog pattern).
    - *Perimeter Defense*: 3 drones coordinating patrol coverage with handover logic.
- [ ] **Hardware Deployment**: Validation on physical Pixhawk/Jetson hardware (moved from 3.4)
- [ ] **Operational Tools**:
    - Mission Replay: Time-scrubbing interface for past missions


## Known Issues / Backlog
- [ ] **Geodetic Transform**: Refine separate `GlobalPosition` vs `LocalPosition` tracking
- [ ] **Vision Model**: Retrain YOLO on specialized infrastructure defect dataset
