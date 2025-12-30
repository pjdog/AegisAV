# Scenarios

Scenarios drive simulated drone state, environment changes, and scripted events.
They are designed to exercise the agent and the overlay.

## Run a Scenario

Use the overlay or API:

```
POST /api/scenarios/{scenario_id}/start
POST /api/scenarios/stop
GET  /api/scenarios/status
```

## What Scenarios Affect

- Drone state updates (battery, GPS health, etc)
- Environment conditions (wind, visibility, precipitation)
- Decision events and overlay output

## What Scenarios Do Not Affect

- Unreal scene geometry and assets are not spawned dynamically.
  You must use an Unreal level that already matches the scenario assets.

See `AIRSIM_SCENARIO_SYNC.md` for environment mapping.
