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
python3 -m uvicorn agent.server.main:app --host ${AEGIS_HOST:-0.0.0.0} --port ${AEGIS_PORT:-8000} --reload
