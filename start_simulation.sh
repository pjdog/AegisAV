#!/bin/bash
# Start complete AegisAV simulation

echo "Starting AegisAV High-Fidelity Simulation..."
echo ""

# Load environment
if [ -f "$(dirname "$0")/.env" ]; then
    set -a
    source "$(dirname "$0")/.env"
    set +a
fi

cd "$(dirname "$0")"
python3 simulation/run_simulation.py --airsim --sitl "$@"
