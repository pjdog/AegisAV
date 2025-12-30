# Overlay Troubleshooting (Blank Page or No Flight)

This overlay is a passive visualizer. It only renders when it receives
WebSocket events from the server. If it stays empty or nothing flies, it
usually means no events are being emitted.

## How the Overlay Works

- The overlay connects to `ws://<host>:<port>/ws/unreal`.
- It only shows thought bubbles when it receives one of these message types:
  `thinking_start`, `thinking_update`, `thinking_complete`, `critic_result`,
  or `risk_update`.
- Telemetry-only messages (like `airsim_telemetry`) just clear the empty state;
  they do not render a bubble.

## Why the Overlay Is Blank

1) WebSocket not connected
- The top-right badge should read `Live`.
- Check `http://<host>:<port>/api/unreal/status` for active connections.
- Make sure you opened the overlay from the same host/port as the server.

2) No events are being emitted
- The overlay is waiting for telemetry or decisions.
- Start a scenario from the overlay or dashboard (`/api/scenarios/...`).
- Or run a client that posts vehicle state to `/state`.
- AirSim telemetry alone does not generate thought bubbles.

3) AirSim bridge is not connected (if you expect AirSim telemetry)
- Confirm AirSim is running and RPC is reachable on `127.0.0.1:41451`.
- Check server logs for `airsim_bridge_started`.
- If AirSim runs on Windows and the server runs in WSL, set
  `simulation.airsim_host` to the Windows host IP.

## Why Nothing Flies

The overlay does not control flight. Actual movement requires a flight stack:

- **AirSim** (Unreal environment) must be running.
- **ArduPilot SITL** must be running and connected via MAVLink.
- A controller must **arm** and **take off** before `goto` commands work.

The built-in `simulation/run_simulation.py` currently issues `goto` commands
but does not explicitly arm or take off. That means SITL may stay grounded
unless you arm manually or add a takeoff step.

## Quick Fix Checklist

1) Open `http://<host>:<port>/overlay/` and verify the status is `Live`.
2) Click **Start** in the overlay scenario panel to generate events.
3) For real flight:
   - Start AirSim (`start_airsim.bat` on Windows).
   - Start SITL (`sim_vehicle.py -v ArduCopter -f airsim-copter`).
   - Run the simulation runner: `python simulation/run_simulation.py --airsim --sitl`.
4) If flight still does not start, arm and takeoff manually or add those calls
   in `simulation/run_simulation.py` before the waypoint loop.

## Useful Endpoints

- `GET /api/unreal/status` — WebSocket connections and subscriptions
- `GET /api/scenarios` — Available scenario list
- `POST /api/scenarios/{id}/start` — Start a scenario
- `POST /api/scenarios/stop` — Stop a scenario
