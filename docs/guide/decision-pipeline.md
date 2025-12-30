# Decision Pipeline

This is the high-level flow for a single decision cycle.

1) **Receive state** via `POST /state`.
2) **Update world model** with the incoming vehicle snapshot.
3) **Compute risk** based on environment, battery, and health signals.
4) **Select goal** (if any) for the current mission state.
5) **Generate decision** (action + parameters + reasoning).
6) **Critic validation** to approve or block the decision.
7) **Emit events** to dashboard and overlay.
8) **Return response** to the client.

## Overlay Events

The overlay only renders if it receives thinking events:

- `thinking_start`
- `thinking_update`
- `thinking_complete`
- `critic_result`
- `risk_update`

If the overlay is blank, it usually means no state is being posted or no
scenario is running.
