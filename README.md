# AegisAV
Enabling truly autonomous infrastructure and farm management using a mix of agentic edge computation as well as base station
# aeigisAv

**Agentic Supervisory Autonomy for Long-Duration Aerial Monitoring**

`aeigisAv` is a simulation-first autonomy framework that demonstrates how **agentic decision-making can supervise classical flight control** to enable long-term, low-touch aerial monitoring missions.

Rather than replacing proven flight controllers, `aeigisAv` layers an **adaptive, explainable, risk-aware agent** on top of existing autopilots (ArduPilot / PX4 via MAVLink), allowing autonomous aerial systems to:

- Monitor infrastructure or agricultural assets over long time horizons
- Autonomously dock and recharge
- Adapt inspection cadence based on detected anomalies
- Replan or abort missions under uncertainty
- Operate with minimal human oversight

This repository contains a **fully hardware-free Software-In-The-Loop (SITL) implementation** demonstrating these capabilities using ArduPilot, Gazebo, and a Python-based agent architecture.

---

## Why aeigisAv?

Most “autonomous” drone systems today are:
- Scripted
- Brittle to environmental change
- Operator-intensive
- Focused on data capture rather than decision-making

`aeigisAv` addresses this gap by introducing **true agency**:
> The system reasons over goals, constraints, and uncertainty — not just waypoints.

The agent:
- Maintains an internal world model
- Chooses between competing goals
- Adapts plans over time
- Explains its decisions
- Knows when *not* to fly

---

## Core Design Principles

### 1. Supervisory Autonomy (Not End-to-End AI)
- Classical control remains responsible for stabilization and safety
- The agent operates at the **mission and decision layer**
- This mirrors real certification-friendly autonomy architectures

### 2. Agentic, Not Scripted
The system is considered *agentic* because it:
- Selects goals dynamically
- Replans in response to outcomes
- Performs self-evaluation and confidence gating
- Logs structured reasoning traces

### 3. Simulation-First, Hardware-Ready
- Entirely runnable with no physical hardware
- Uses real flight software (ArduPilot SITL)
- Agent logic is platform-agnostic and MAVLink-based
- Transfers directly to Pixhawk-class systems later

---

## Example Use Case

**Persistent Infrastructure Monitoring**
1. Drone launches from a dock
2. Inspects assets on a baseline cadence
3. Detects an anomaly at one site
4. Increases revisit frequency and changes vantage points
5. Aborts early if risk thresholds are exceeded
6. Returns to dock and schedules follow-up autonomously

The operator only intervenes on exceptions.

---

## Repository Structure

aeigisAv/
├── sim/ # ArduPilot SITL + Gazebo worlds
├── agent/ # Agentic decision-making logic
├── autonomy/ # MAVLink interface & mission primitives
├── metrics/ # Logging, plots, and comparisons
├── configs/ # Tunable parameters and thresholds
├── scripts/ # Entry points and utilities
└── logs/ # Decision traces and run data


---

## Architecture Overview

┌────────────────────────────┐
│ Agent Server (Heavy) │
│ - World model │
│ - Goal selection │
│ - Risk & confidence logic │
│ - Decision logging │
└────────────┬───────────────┘
│ HTTP / IPC
┌────────────▼───────────────┐
│ Agent Client (Light) │
│ - State collection │
│ - Action execution │
└────────────┬───────────────┘
│ MAVLink
┌────────────▼───────────────┐
│ ArduPilot SITL │
│ - Stabilization │
│ - Failsafes │
│ - Vehicle dynamics │
└────────────────────────────┘


This separation allows the agent to run:
- onboard
- on an edge computer
- on a centralized fleet brain

without changing flight software.

---

## What This Project Demonstrates

✔ True agentic decision-making  
✔ Long-horizon planning  
✔ Risk-aware self-aborts  
✔ Autonomous docking & recharge (simulated)  
✔ Reduced operator cognitive load  
✔ Explainable autonomy via decision logs  

---

## What This Project Does *Not* Attempt

✗ End-to-end learned flight control  
✗ Replacement of certified autopilots  
✗ Hardware-specific tuning  
✗ Vision model training (mocked where needed)

These choices are deliberate and aligned with real-world autonomy deployment constraints.

---

## Running the Simulation (High Level)

```bash
# Start ArduPilot SITL + Gazebo
./scripts/run_sim.sh

# Start the agent server
python agent_server/main.py

# Run the agent client
python scripts/run_agent.py --scenario anomaly

Roadmap (Short-Term)

Multi-asset scheduling

Fleet-level coordination

Communication degradation modeling

Dock reliability modeling

    Replace mock anomalies with vision inputs

License & Use

This project is intended for:

    research

    competitions

    architectural demonstration

See LICENSE for details.
Disclaimer

This software is not flight-certified and is provided for simulation and research purposes only.