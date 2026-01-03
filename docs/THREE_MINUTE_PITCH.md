# AegisAV - 3-Minute Pitch and Differentiators

## Coherent Story (What the App Is)

AegisAV is a full-stack autonomous inspection system for critical
infrastructure (solar, wind, substations, power lines). It combines
realistic flight control, AI reasoning, computer vision, and 3D
reconstruction into one observable loop:

Sensors -> Vision + SLAM/3D reconstruction -> World model -> LLM decision
engine + multi-critic validation -> Flight execution -> Real-time UI +
auditable logs.

This is not a single drone app or a single model. The repository contains
the decision engine (`AegisAV/agent/`), vision pipeline
(`AegisAV/vision/`), mapping and Gaussian splatting (`AegisAV/mapping/`),
hardware-realistic simulation (`AegisAV/simulation/`), and the
dashboard/overlay UI (`AegisAV/frontend/`, `AegisAV/overlay/`).


## 3-Minute Pitch (Narrative)

Today, infrastructure inspections are manual, risky, and slow. AegisAV
turns that into an autonomous, auditable workflow.

AegisAV is an AI-driven drone system that does more than fly - it
understands the mission. It fuses high-fidelity simulation and real
flight control with AI reasoning, computer vision, and 3D mapping. The
drone captures RGB and depth, detects defects, reconstructs the
environment, and then makes decisions with an LLM. Every action is
validated by independent safety, efficiency, and goal-alignment critics.
That means explainable decisions you can trust, not opaque autonomy.

On the mapping side, we go beyond simple point clouds. We fuse SLAM
geometry with Gaussian Splatting to create a denser 3D model and turn it
into a planning proxy for navigation and change detection. That gives us
a live, evolving map and a real-world representation that is usable for
both mission planning and anomaly review.

The system is built to be production-realistic. The flight controller is
ArduPilot SITL (the same firmware used on real hardware), and the
environment is Unreal Engine 5.5 with Cosys-AirSim, including advanced
sensors like depth, segmentation, and LiDAR. Everything you see in the UI
(mission state, reasoning, risk, anomalies) is backed by logs and APIs,
so there is a full audit trail for compliance and operations.

In short: AegisAV is the autonomous inspection stack that pairs
explainable AI with real flight control and high-fidelity reconstruction,
giving operators speed, safety, and confidence that today’s inspection
tools cannot match.


## 5-Slide, 3-Minute Deck Script

### Slide 1 - Autonomous Infrastructure Inspection, You Can Trust (0:00-0:30)
Talk track:
“AegisAV is a full-stack autonomous inspection system for critical
infrastructure. It combines real flight control, AI reasoning, and 3D
reconstruction to cut inspection time and risk, while keeping decisions
explainable and auditable.”

On-slide bullets:
- Autonomous drones for solar, wind, substations, and power lines
- Explainable AI decisions with real-time monitoring
- Production-realistic flight control + high-fidelity simulation

### Slide 2 - End-to-End System, Not Just a Drone (0:30-1:05)
Talk track:
“This is not a single model or a single app. It is an integrated stack:
sensors feed the vision pipeline and SLAM, we build a world model, the AI
decides, safety critics validate, and the execution layer flies.”

On-slide bullets:
- Vision pipeline -> defects and anomalies
- SLAM + 3D reconstruction -> live map
- LLM decision engine -> multi-critic validation
- Flight execution via ArduPilot SITL

### Slide 3 - Why It’s Different (1:05-1:50)
Talk track:
“Most systems stop at detection or waypoint automation. AegisAV creates
an explainable decision loop and a richer 3D environment model. That is
how we move from remote control to trustworthy autonomy.”

On-slide bullets:
- Explainable AI: reasoning feed + risk + critics
- 3D reconstruction: SLAM + Gaussian splats for usable maps
- Safety gating: actions validated by multiple critics
- Auditability: logs + replayable mission history

### Slide 4 - Live Ops + Auditing (1:50-2:25)
Talk track:
“Operators see mission state, live camera, and AI reasoning in one
dashboard. It is not a black box. Every action has a traceable
explanation and safety signal.”

On-slide bullets:
- Real-time dashboard with reasoning feed
- Live telemetry, detections, and map previews
- Decision logs and mission replay

### Slide 5 - ROI and Next Step (2:25-3:00)
Talk track:
“This reduces manual inspection time, expands coverage, and improves
compliance. We are ready to run a pilot on your assets, or plug into
existing workflows.”

On-slide bullets:
- Faster inspections, safer operations
- Better coverage with continuous updates
- Pilot in high-value assets first


## 30-Second Opener

AegisAV is an autonomous inspection stack for critical infrastructure.
It combines real flight control, computer vision, 3D reconstruction, and
explainable AI decisioning. The system does not just detect defects - it
decides what to do next, validates that decision with safety critics, and
logs everything for audit. It is a trustworthy, end-to-end autonomy
platform.

## Agent Architecture + Tools Script (AI Agent Panel)

Let me start with the core of AegisAV: the agent system.

We run a multi-layer decision architecture. At the center is the decision
engine that selects goals based on live mission context - asset priority,
risk, weather, battery, and map confidence. Every action is then validated
by a multi-critic layer: safety, efficiency, and goal alignment. This
ensures the system never ships a decision without explicit checks and
explainable reasoning.

The agent tools are designed to make decisions grounded, not speculative.
Agents can:
- Read the world model (assets, anomalies, telemetry, dock status)
- Query mapping outputs (SLAM and splat-derived map confidence)
- Trigger safe action primitives (takeoff, move, orbit, return)
- Use the vision pipeline for defect confirmation
- Log decisions, reasoning, and outcomes for audit

Methodology is “sense -> model -> decide -> validate -> act -> explain.”
1) We ingest sensor data and telemetry.
2) We update a world model and map confidence.
3) The LLM proposes a goal and candidate action.
4) Critics check the action for safety and mission alignment.
5) The executor performs the action or rejects it with a reason.
6) The UI exposes the reasoning and risk live, and the logs preserve the
   full trace.

This is where AI makes the project truly valuable. Instead of brittle
waypoint automation, the system adapts to live conditions and makes
explainable decisions. It improves coverage, reduces operator load, and
adds a safety envelope that is explicit and auditable. The AI does not
replace engineers - it provides a decision layer that is aware, reasoned,
and constrained by deterministic critics.

So the AI is not a black box - it is a governed decision layer that
expands what autonomous inspection can safely do.


## Technical Differentiators (With Repo Anchors)

- Explainable multi-critic decisioning instead of single-model autonomy:
  `AegisAV/agent/server/critics/`, reasoning feed UI in
  `AegisAV/frontend/src/components/Dashboard.tsx`.
- 3D mapping pipeline that fuses SLAM + Gaussian splatting into planning
  proxies: `AegisAV/mapping/map_update.py`,
  `AegisAV/mapping/splat_trainer.py`, `AegisAV/mapping/splat_proxy.py`.
- Production-realistic flight control: ArduPilot SITL integration in
  `AegisAV/simulation/` and `AegisAV/autonomy/`.
- High-fidelity simulation: Unreal Engine 5.5 + Cosys-AirSim detailed in
  `AegisAV/simulation/README.md`.
- Vision pipeline tailored for infrastructure defects:
  `AegisAV/vision/` and defect taxonomy in `AegisAV/README.md`.
- Edge policies and cost tracking for LLM budget and bandwidth control:
  `AegisAV/agent/server/config_manager.py` and docs in `AegisAV/README.md`.
- Auditability and observability: JSONL logs under `AegisAV/logs/`,
  API endpoints and UI consumption in `AegisAV/agent/server/`.
