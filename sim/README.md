# Simulation Files

This directory contains ArduPilot SITL and Gazebo configuration:

- `launch/` - Gazebo launch files
- `worlds/` - Gazebo world definitions
- `models/` - 3D models for drone and dock

## Setup

See the main README for ArduPilot SITL installation instructions.

## Quick Start

```bash
# Start SITL with defaults
./scripts/run_sim.sh

# Start with 2x speed
./scripts/run_sim.sh --speedup 2
```
