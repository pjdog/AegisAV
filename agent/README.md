# Agent Services

## Overview

The agent package provides:

- **Server**: FastAPI decision engine and dashboard API.
- **Client**: Sends vehicle state updates and receives decisions.

## Agent Server

```bash
uv sync
uv run aegis-server
```

Default address: `http://0.0.0.0:8080`.

Dashboard: `http://<server-host>:8080/dashboard` (requires frontend build).

Decision logs are written under `logs/` for the dashboard.

## Agent Client (Demo)

```bash
uv run aegis-demo --scenario anomaly
```

The client posts state updates to the server via `POST /state`.
