#!/bin/bash
# AegisAV Simulation Launch Script
# Starts ArduPilot SITL and optionally Gazebo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
VEHICLE_TYPE="${VEHICLE_TYPE:-ArduCopter}"
VEHICLE_FRAME="${VEHICLE_FRAME:-quad}"
HOME_LAT="${HOME_LAT:-47.397742}"
HOME_LON="${HOME_LON:-8.545594}"
HOME_ALT="${HOME_ALT:-488}"
SPEEDUP="${SPEEDUP:-1}"

# Output ports
SITL_PORT="${SITL_PORT:-14550}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_ardupilot() {
    if ! command -v sim_vehicle.py &> /dev/null; then
        log_error "ArduPilot SITL not found!"
        log_error "Please install ArduPilot and add Tools/autotest to your PATH"
        log_error "See: https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html"
        exit 1
    fi
    log_info "ArduPilot SITL found"
}

start_sitl() {
    log_info "Starting ArduPilot SITL..."
    log_info "  Vehicle: $VEHICLE_TYPE ($VEHICLE_FRAME)"
    log_info "  Home: $HOME_LAT, $HOME_LON, $HOME_ALT"
    log_info "  Port: $SITL_PORT"

    # Build home string
    HOME_STR="$HOME_LAT,$HOME_LON,$HOME_ALT,0"

    # Start SITL
    sim_vehicle.py \
        -v "$VEHICLE_TYPE" \
        -f "$VEHICLE_FRAME" \
        --home "$HOME_STR" \
        --speedup "$SPEEDUP" \
        --out "udp:127.0.0.1:$SITL_PORT" \
        --no-mavproxy \
        "$@"
}

show_help() {
    echo "AegisAV Simulation Launch Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help          Show this help message"
    echo "  --check         Check prerequisites only"
    echo "  --speedup N     Set simulation speedup (default: 1)"
    echo ""
    echo "Environment Variables:"
    echo "  VEHICLE_TYPE    ArduPilot vehicle type (default: ArduCopter)"
    echo "  VEHICLE_FRAME   Vehicle frame type (default: quad)"
    echo "  HOME_LAT        Home latitude (default: 47.397742)"
    echo "  HOME_LON        Home longitude (default: 8.545594)"
    echo "  HOME_ALT        Home altitude in meters (default: 488)"
    echo "  SITL_PORT       MAVLink output port (default: 14550)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Start with defaults"
    echo "  $0 --speedup 2          # 2x simulation speed"
    echo "  VEHICLE_FRAME=hexa $0   # Use hexacopter"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --check)
            check_ardupilot
            log_info "All prerequisites satisfied!"
            exit 0
            ;;
        --speedup)
            SPEEDUP=$2
            shift 2
            ;;
        --realistic)
            REALISTIC=true
            shift
            ;;
        *)
            # Pass remaining args to sim_vehicle.py
            break
            ;;
    esac
done

# Prepare additional params
EXTRA_PARAMS=""
if [ "$REALISTIC" = true ]; then
    log_info "Realistic mode enabled, loading sim/realistic.params"
    EXTRA_PARAMS="--add-param-file=$PROJECT_DIR/sim/realistic.params"
fi

# Main
log_info "AegisAV Simulation Launcher"
check_ardupilot
start_sitl $EXTRA_PARAMS "$@"
