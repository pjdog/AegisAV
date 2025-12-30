#!/bin/bash
# Start AegisAV server only

echo "Starting AegisAV Server..."
echo ""

# Load environment
if [ -f "$(dirname "$0")/.env" ]; then
    set -a
    source "$(dirname "$0")/.env"
    set +a
fi

cd "$(dirname "$0")"
AEGIS_PORT="${AEGIS_PORT:-8000}"
echo "==============================================="
echo "AegisAV server URLs:"
echo "  Dashboard:        http://localhost:${AEGIS_PORT}/dashboard"
echo "  Scenarios API:    http://localhost:${AEGIS_PORT}/api/scenarios"
echo "  Scenario start:   http://localhost:${AEGIS_PORT}/api/scenarios/{id}/start"
echo "  Runner start:     http://localhost:${AEGIS_PORT}/api/dashboard/runner/start"
echo "  Overlay:          http://localhost:${AEGIS_PORT}/overlay"
echo "  API docs:         http://localhost:${AEGIS_PORT}/docs"
echo "  Health:           http://localhost:${AEGIS_PORT}/health"
echo "==============================================="
python3 -m uvicorn agent.server.main:app --host ${AEGIS_HOST:-0.0.0.0} --port "${AEGIS_PORT}" --reload
