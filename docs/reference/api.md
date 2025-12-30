# API Overview

This page lists the most used endpoints. Refer to `/docs` or `/openapi.json`
for full details.

## Core

- `GET /health`
- `POST /state` - submit vehicle state and receive decision

## Config

- `GET /api/config`
- `PUT /api/config/{section}`
- `POST /api/config/save`
- `POST /api/config/reset/{section}`
- `POST /api/config/generate-api-key`

## Edge Compute

- `GET /api/config/edge`
- `POST /api/config/edge`

## Scenarios

- `GET /api/scenarios`
- `GET /api/scenarios/{scenario_id}`
- `POST /api/scenarios/{scenario_id}/start`
- `POST /api/scenarios/stop`
- `GET /api/scenarios/status`

## AirSim

- `GET /api/airsim/status`
- `POST /api/airsim/start`

## Unreal/Overlay

- `GET /api/unreal/status`
- `WS  /ws/unreal`

## Dashboard

- `GET /api/dashboard/runs`
- `GET /api/dashboard/run/{run_id}`
