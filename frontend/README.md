# AegisAV Dashboard (React)

## Overview

This is the React + TypeScript frontend for the AegisAV Mission Autonomy Monitor.
It reads data from the agent server via `/api/dashboard/*` and can be served as a static
build from the server at `/dashboard`.

## Prerequisites

- Node.js 18+
- The agent server running on `http://localhost:8080`

## Development

```bash
npm install
npm run dev
```

Vite runs at `http://localhost:5173` and proxies `/api` to the agent server.

## Production Build

```bash
npm install
npm run build
```

The build output is written to `frontend/dist`. The agent server serves it at `/dashboard`.

## Linting

```bash
npm run lint
```

Linting is also enforced via pre-commit.
