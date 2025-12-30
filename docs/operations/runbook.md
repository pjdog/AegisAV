# Operations Runbook

## Start Server

```bash
python -m agent.server.main
```

## Verify Services

- Dashboard: `/dashboard`
- Overlay: `/overlay`
- Unreal status: `/api/unreal/status`

## AirSim Checks

- `GET /api/airsim/status`
- Confirm AirSim RPC is reachable on port 41451.

## Scenario Checks

- `GET /api/scenarios`
- `POST /api/scenarios/{id}/start`
- `GET /api/scenarios/status`

## Overlay Blank

If the overlay is blank, verify:

- WebSocket is connected (`/api/unreal/status` shows active connection)
- A scenario is running or `/state` is receiving updates

See `OVERLAY_TROUBLESHOOTING.md` for details.
