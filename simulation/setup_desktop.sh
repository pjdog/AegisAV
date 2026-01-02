#!/bin/bash
#
# AegisAV Desktop Setup Script
#
# Comprehensive setup for high-fidelity drone simulation with:
# - ArduPilot SITL (flight controller)
# - Cosys-AirSim (Unreal Engine 5.5 rendering)
# - GPU-accelerated vision (YOLO)
# - Interactive configuration
#
# Supports:
# - Ubuntu 20.04/22.04 (native Linux)
# - Windows 11 with WSL2 (GPU passthrough)
# - AMD/NVIDIA GPUs
#
# Usage:
#   chmod +x setup_desktop.sh
#   ./setup_desktop.sh
#

set -e

echo "============================================================"
echo "  AegisAV High-Fidelity Simulation Setup"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }
print_prompt() { echo -e "${CYAN}[?]${NC} $1"; }

# Verbose install logging (set VERBOSE_INSTALL=false to disable)
VERBOSE_INSTALL=${VERBOSE_INSTALL:-true}
enable_install_trace() { if [[ "$VERBOSE_INSTALL" == "true" ]]; then set -x; fi; }
disable_install_trace() { if [[ "$VERBOSE_INSTALL" == "true" ]]; then set +x; fi; }

# Configuration defaults
CONFIG_FILE=""
AEGISAV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="$AEGISAV_DIR/configs"
STATE_DIR="$AEGISAV_DIR/.setup_state"

# Install caching controls (set FORCE_REINSTALL=true to bypass all caches)
FORCE_REINSTALL=${FORCE_REINSTALL:-false}
FORCE_SYSTEM_DEPS=${FORCE_SYSTEM_DEPS:-false}
FORCE_PYTHON_DEPS=${FORCE_PYTHON_DEPS:-false}
FORCE_ARDUPILOT=${FORCE_ARDUPILOT:-false}
FORCE_AIRSIM=${FORCE_AIRSIM:-false}

# CLI defaults (non-interactive)
RUN_INTERACTIVE="false"
CLEANUP="false"
USE_DOCKER="true"
INSTALL_DEPS="true"
UPDATE_ARDUPILOT="false"

SERVER_HOST="0.0.0.0"
SERVER_PORT=8000
REDIS_ENABLED="true"
REDIS_HOST="localhost"
REDIS_PORT=6379
VISION_ENABLED="true"
USE_REAL_DETECTOR="false"
VISION_DEVICE="auto"
YOLO_MODEL="yolov8n.pt"
CONFIDENCE_THRESHOLD=0.5
SIM_ENABLED="true"
AIRSIM_ENABLED="true"
AIRSIM_HOST="127.0.0.1"
SITL_ENABLED="true"
ARDUPILOT_PATH="~/ardupilot"
SITL_SPEEDUP=1.0
HOME_LAT=37.7749
HOME_LON="-122.4194"
HOME_ADDRESS="${HOME_ADDRESS:-}"
USE_LLM="true"
LLM_MODEL="gpt-4o-mini"
AUTH_ENABLED="false"

ensure_state_dir() { mkdir -p "$STATE_DIR"; }
mark_done() { ensure_state_dir; touch "$1"; }

geocode_address() {
    local address="$1"
    if [ -z "$address" ]; then
        return 1
    fi

    python3 - "$address" << 'PY'
import json
import sys
import urllib.parse
import urllib.request

address = sys.argv[1].strip()
if not address:
    sys.exit(1)

url = "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" + urllib.parse.quote(address)
req = urllib.request.Request(url, headers={"User-Agent": "AegisAV-Setup/1.0"})

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data:
        sys.exit(1)
    lat = data[0].get("lat")
    lon = data[0].get("lon")
    if not lat or not lon:
        sys.exit(1)
    print(f"{lat} {lon}")
except Exception:
    sys.exit(1)
PY
}

docker_available() {
    command -v docker &> /dev/null && docker compose version &> /dev/null
}

setup_docker_compose() {
    echo ""
    echo "============================================================"
    echo "  Docker Compose Setup"
    echo "============================================================"
    echo ""

    if ! docker_available; then
        print_warning "Docker Compose not available."
        print_info "Install Docker Desktop (Windows) or Docker Engine + compose plugin (Linux), then re-run."
        return 0
    fi

    print_info "Building and starting containers..."
    docker compose up -d --build
    print_status "Docker Compose services started"
}

usage() {
    cat << 'EOF'
Usage: ./simulation/setup_desktop.sh [options]

Core options:
  -h, --help                 Show this help text
  --cleanup                  Stop containers and remove generated setup artifacts
  --interactive              Run interactive prompts (overrides CLI defaults)

Setup toggles:
  --docker / --no-docker      Enable/disable Docker Compose setup (default: enabled)
  --install-deps / --no-install-deps
                             Enable/disable dependency installs (default: enabled)
  --update-ardupilot         Update existing ArduPilot checkout before build
  --no-update-ardupilot      Skip ArduPilot update (default)

Config overrides:
  --server-host HOST          Server bind host (default: 0.0.0.0)
  --server-port PORT          Server port (default: 8000)
  --redis / --no-redis         Enable/disable Redis persistence (default: enabled)
  --redis-host HOST           Redis host (default: localhost)
  --redis-port PORT           Redis port (default: 6379)
  --vision / --no-vision       Enable/disable vision pipeline (default: enabled)
  --real-detector             Use real YOLO detector
  --mock-detector             Use mock detector
  --vision-device DEVICE       auto|cpu|cuda (default: auto)
  --yolo-model MODEL           YOLO model path (default: yolov8n.pt)
  --confidence FLOAT           Detection confidence threshold (default: 0.5)
  --sim / --no-sim             Enable/disable simulation (default: enabled)
  --airsim / --no-airsim       Enable/disable Cosys-AirSim integration (default: enabled)
  --airsim-host HOST           Cosys-AirSim host (default: 127.0.0.1)
  --sitl / --no-sitl           Enable/disable ArduPilot SITL (default: enabled)
  --ardupilot-path PATH         ArduPilot path (default: ~/ardupilot)
  --sitl-speedup FLOAT          SITL speedup (default: 1.0)
  --home-address "ADDRESS"      Geocode address to home lat/lon
  --home-lat LAT                Home latitude (default: 37.7749)
  --home-lon LON                Home longitude (default: -122.4194)
  --llm / --no-llm              Enable/disable LLM decisions (default: enabled)
  --llm-model MODEL             LLM model (default: gpt-4o-mini)
  --auth / --no-auth            Enable/disable API auth (default: disabled)

Examples:
  ./simulation/setup_desktop.sh --no-install-deps --docker
  ./simulation/setup_desktop.sh --home-address "Austin, TX" --redis-host redis
EOF
}

require_arg() {
    local flag="$1"
    local value="${2:-}"
    if [ -z "$value" ] || [[ "$value" == -* ]]; then
        print_error "Missing value for $flag"
        exit 1
    fi
}

apply_home_address() {
    if [ -n "${HOME_ADDRESS:-}" ]; then
        print_info "Geocoding HOME_ADDRESS..."
        GEO_RESULT=$(geocode_address "$HOME_ADDRESS" || true)
        if [ -n "$GEO_RESULT" ]; then
            HOME_LAT=$(echo "$GEO_RESULT" | awk '{print $1}')
            HOME_LON=$(echo "$GEO_RESULT" | awk '{print $2}')
            print_status "Home set to lat=${HOME_LAT}, lon=${HOME_LON}"
        else
            print_warning "HOME_ADDRESS lookup failed; using existing lat/lon."
        fi
    fi
}

cleanup_setup() {
    echo ""
    echo "============================================================"
    echo "  Cleanup"
    echo "============================================================"
    echo ""

    if docker_available; then
        print_info "Stopping Docker Compose services..."
        docker compose down --volumes --remove-orphans
    fi

    print_info "Removing generated setup artifacts..."
    rm -f "$AEGISAV_DIR/.env" \
        "$AEGISAV_DIR/start_simulation.sh" \
        "$AEGISAV_DIR/start_server.sh" \
        "$AEGISAV_DIR/start_airsim.bat"
    rm -f "$AEGISAV_DIR/configs/aegis_config.yaml"
    rm -rf "$STATE_DIR"

    print_status "Cleanup complete"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --cleanup)
                CLEANUP="true"
                shift
                ;;
            --interactive)
                RUN_INTERACTIVE="true"
                shift
                ;;
            --docker)
                USE_DOCKER="true"
                shift
                ;;
            --no-docker)
                USE_DOCKER="false"
                shift
                ;;
            --install-deps)
                INSTALL_DEPS="true"
                shift
                ;;
            --no-install-deps)
                INSTALL_DEPS="false"
                shift
                ;;
            --update-ardupilot)
                UPDATE_ARDUPILOT="true"
                shift
                ;;
            --no-update-ardupilot)
                UPDATE_ARDUPILOT="false"
                shift
                ;;
            --server-host)
                require_arg "$1" "$2"
                SERVER_HOST="$2"
                shift 2
                ;;
            --server-port)
                require_arg "$1" "$2"
                SERVER_PORT="$2"
                shift 2
                ;;
            --redis)
                REDIS_ENABLED="true"
                shift
                ;;
            --no-redis)
                REDIS_ENABLED="false"
                shift
                ;;
            --redis-host)
                require_arg "$1" "$2"
                REDIS_HOST="$2"
                shift 2
                ;;
            --redis-port)
                require_arg "$1" "$2"
                REDIS_PORT="$2"
                shift 2
                ;;
            --vision)
                VISION_ENABLED="true"
                shift
                ;;
            --no-vision)
                VISION_ENABLED="false"
                shift
                ;;
            --real-detector)
                USE_REAL_DETECTOR="true"
                shift
                ;;
            --mock-detector)
                USE_REAL_DETECTOR="false"
                shift
                ;;
            --vision-device)
                require_arg "$1" "$2"
                VISION_DEVICE="$2"
                shift 2
                ;;
            --yolo-model)
                require_arg "$1" "$2"
                YOLO_MODEL="$2"
                shift 2
                ;;
            --confidence)
                require_arg "$1" "$2"
                CONFIDENCE_THRESHOLD="$2"
                shift 2
                ;;
            --sim)
                SIM_ENABLED="true"
                shift
                ;;
            --no-sim)
                SIM_ENABLED="false"
                shift
                ;;
            --airsim)
                AIRSIM_ENABLED="true"
                shift
                ;;
            --no-airsim)
                AIRSIM_ENABLED="false"
                shift
                ;;
            --airsim-host)
                require_arg "$1" "$2"
                AIRSIM_HOST="$2"
                shift 2
                ;;
            --sitl)
                SITL_ENABLED="true"
                shift
                ;;
            --no-sitl)
                SITL_ENABLED="false"
                shift
                ;;
            --ardupilot-path)
                require_arg "$1" "$2"
                ARDUPILOT_PATH="$2"
                shift 2
                ;;
            --sitl-speedup)
                require_arg "$1" "$2"
                SITL_SPEEDUP="$2"
                shift 2
                ;;
            --home-address)
                require_arg "$1" "$2"
                HOME_ADDRESS="$2"
                shift 2
                ;;
            --home-lat)
                require_arg "$1" "$2"
                HOME_LAT="$2"
                shift 2
                ;;
            --home-lon)
                require_arg "$1" "$2"
                HOME_LON="$2"
                shift 2
                ;;
            --llm)
                USE_LLM="true"
                shift
                ;;
            --no-llm)
                USE_LLM="false"
                shift
                ;;
            --llm-model)
                require_arg "$1" "$2"
                LLM_MODEL="$2"
                shift 2
                ;;
            --auth)
                AUTH_ENABLED="true"
                shift
                ;;
            --no-auth)
                AUTH_ENABLED="false"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Detect environment
detect_environment() {
    echo "Detecting environment..."
    echo ""

    # Check if running in WSL
    if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
        OS="wsl"
        WINDOWS_USER=$(cmd.exe /c "echo %USERNAME%" </dev/null 2>/dev/null | tr -d '\r')
        print_status "Detected Windows Subsystem for Linux (WSL2)"
        print_info "Windows user: $WINDOWS_USER"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Native Linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_status "Detected Windows (Git Bash/Cygwin)"
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi

    # Detect GPU
    echo ""
    echo "Detecting GPU..."

    GPU_TYPE="none"
    GPU_NAME=""

    if [[ "$OS" == "wsl" ]]; then
        WSL_GPU_PASSTHROUGH="false"
        if [ -e "/dev/dxg" ]; then
            WSL_GPU_PASSTHROUGH="true"
            print_status "WSL GPU passthrough device detected (/dev/dxg)"
        else
            print_warning "WSL GPU passthrough device not found (/dev/dxg)"
            print_info "GPU acceleration may be disabled in WSL"
        fi

        WINDOWS_GPU_LIST=$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" </dev/null 2>/dev/null | tr -d '\r')
        if [ -n "$WINDOWS_GPU_LIST" ]; then
            WINDOWS_GPU_NAME=$(echo "$WINDOWS_GPU_LIST" | grep -Ev -i "Virtual|Remote|Basic Display|Hyper-V|VMware|DisplayLink|Virtual Desktop Monitor" | head -1)
            if [ -z "$WINDOWS_GPU_NAME" ]; then
                WINDOWS_GPU_NAME=$(echo "$WINDOWS_GPU_LIST" | head -1)
            fi

            GPU_NAME="$WINDOWS_GPU_NAME"
            if echo "$GPU_NAME" | grep -qi "NVIDIA"; then
                GPU_TYPE="nvidia"
            elif echo "$GPU_NAME" | grep -qiE "AMD|Radeon"; then
                GPU_TYPE="amd"
            else
                GPU_TYPE="unknown"
            fi
            print_status "Windows GPU detected: $GPU_NAME"
        fi
    fi

    if [[ "$GPU_TYPE" == "none" ]] || [[ "$GPU_TYPE" == "unknown" ]]; then
        if command -v nvidia-smi &> /dev/null; then
            GPU_TYPE="nvidia"
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown NVIDIA")
            print_status "NVIDIA GPU detected: $GPU_NAME"

            # Check CUDA version
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
                print_info "CUDA version: $CUDA_VERSION"
            fi
        elif [ -d "/dev/dri" ]; then
            # Check for AMD GPU
            if lspci 2>/dev/null | grep -i "vga\|3d\|display" | grep -i "AMD\|ATI" > /dev/null; then
                GPU_TYPE="amd"
                GPU_NAME=$(lspci 2>/dev/null | grep -i "vga\|3d\|display" | grep -i "AMD\|ATI" | head -1 | sed 's/.*: //')
                print_status "AMD GPU detected: $GPU_NAME"

                # Check ROCm
                if command -v rocm-smi &> /dev/null; then
                    ROCM_VERSION=$(rocm-smi --showversion 2>/dev/null | grep "ROCm" || echo "Unknown")
                    print_info "ROCm: $ROCM_VERSION"
                else
                    print_warning "ROCm not installed. GPU acceleration may be limited."
                fi
            else
                GPU_TYPE="intel"
                GPU_NAME="Intel Integrated"
                print_warning "Intel/Integrated GPU detected. Using CPU for ML inference."
            fi
        else
            if [[ "$OS" == "wsl" ]] && [[ "$WSL_GPU_PASSTHROUGH" == "true" ]]; then
                print_warning "WSL GPU passthrough detected but GPU name not found. Using CPU for ML inference."
            else
                print_warning "No GPU detected. Using CPU for ML inference."
            fi
        fi
    fi

    if [[ "$OS" == "wsl" ]] && [[ "$GPU_TYPE" == "amd" ]]; then
        print_warning "AMD GPU detected via Windows. WSL GPU support for AMD is limited."
        print_info "If /dev/dxg is missing, update Windows GPU drivers and WSL."
    fi

    # Check memory
    echo ""
    TOTAL_MEM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
    if [ "$TOTAL_MEM_GB" -ge 32 ]; then
        print_status "RAM: ${TOTAL_MEM_GB}GB (excellent)"
    elif [ "$TOTAL_MEM_GB" -ge 16 ]; then
        print_warning "RAM: ${TOTAL_MEM_GB}GB (adequate)"
    else
        print_error "RAM: ${TOTAL_MEM_GB}GB (may be insufficient for simulation)"
    fi

    echo ""
}

# Interactive configuration
interactive_config() {
    echo "============================================================"
    echo "  Interactive Configuration"
    echo "============================================================"
    echo ""

    # Server settings
    print_prompt "Server host (default: 0.0.0.0):"
    read -r SERVER_HOST
    SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}

    print_prompt "Server port (default: 8000):"
    read -r SERVER_PORT
    SERVER_PORT=${SERVER_PORT:-8000}

    # Redis settings
    echo ""
    print_prompt "Enable Redis persistence? (Y/n):"
    read -r REDIS_ENABLED
    if [[ ! $REDIS_ENABLED =~ ^[Nn]$ ]]; then
        REDIS_ENABLED="true"
        print_prompt "Redis host (default: localhost):"
        read -r REDIS_HOST
        REDIS_HOST=${REDIS_HOST:-"localhost"}

        print_prompt "Redis port (default: 6379):"
        read -r REDIS_PORT
        REDIS_PORT=${REDIS_PORT:-6379}
    else
        REDIS_ENABLED="false"
        REDIS_HOST="localhost"
        REDIS_PORT=6379
    fi

    # Vision settings
    echo ""
    print_prompt "Enable Vision System? (Y/n):"
    read -r VISION_ENABLED
    if [[ ! $VISION_ENABLED =~ ^[Nn]$ ]]; then
        VISION_ENABLED="true"

        # Determine default device based on GPU
        if [ "$GPU_TYPE" == "nvidia" ]; then
            DEFAULT_DEVICE="cuda"
        elif [ "$GPU_TYPE" == "amd" ]; then
            # ROCm uses hip, but PyTorch/ultralytics may use 'cuda' or 'rocm'
            DEFAULT_DEVICE="cpu"
            print_info "AMD GPU detected. ROCm support varies by library."
            print_info "For best results, install PyTorch with ROCm support."
        else
            DEFAULT_DEVICE="cpu"
        fi

        print_prompt "Use real YOLO detector (requires GPU)? (y/N):"
        read -r USE_REAL_DETECTOR
        if [[ $USE_REAL_DETECTOR =~ ^[Yy]$ ]]; then
            USE_REAL_DETECTOR="true"
        else
            USE_REAL_DETECTOR="false"
        fi

        print_prompt "Vision device (auto/cpu/cuda) (default: $DEFAULT_DEVICE):"
        read -r VISION_DEVICE
        VISION_DEVICE=${VISION_DEVICE:-$DEFAULT_DEVICE}

        print_prompt "YOLO model (yolov8n.pt/yolov8s.pt/yolov8m.pt) (default: yolov8n.pt):"
        read -r YOLO_MODEL
        YOLO_MODEL=${YOLO_MODEL:-"yolov8n.pt"}

        print_prompt "Detection confidence threshold (0.0-1.0) (default: 0.5):"
        read -r CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.5}
    else
        VISION_ENABLED="false"
        USE_REAL_DETECTOR="false"
        VISION_DEVICE="cpu"
        YOLO_MODEL="yolov8n.pt"
        CONFIDENCE_THRESHOLD=0.5
    fi

    # Simulation settings
    echo ""
    print_prompt "Enable Simulation Mode? (Y/n):"
    read -r SIM_ENABLED
    if [[ ! $SIM_ENABLED =~ ^[Nn]$ ]]; then
        SIM_ENABLED="true"

        print_prompt "Enable Cosys-AirSim integration? (Y/n):"
        read -r AIRSIM_ENABLED
        if [[ ! $AIRSIM_ENABLED =~ ^[Nn]$ ]]; then
            AIRSIM_ENABLED="true"

            print_prompt "Cosys-AirSim host (default: 127.0.0.1):"
            read -r AIRSIM_HOST
            AIRSIM_HOST=${AIRSIM_HOST:-"127.0.0.1"}
        else
            AIRSIM_ENABLED="false"
            AIRSIM_HOST="127.0.0.1"
        fi

        print_prompt "Enable ArduPilot SITL? (Y/n):"
        read -r SITL_ENABLED
        if [[ ! $SITL_ENABLED =~ ^[Nn]$ ]]; then
            SITL_ENABLED="true"

            print_prompt "ArduPilot path (default: ~/ardupilot):"
            read -r ARDUPILOT_PATH
            ARDUPILOT_PATH=${ARDUPILOT_PATH:-"~/ardupilot"}

            print_prompt "SITL speed multiplier (default: 1.0):"
            read -r SITL_SPEEDUP
            SITL_SPEEDUP=${SITL_SPEEDUP:-1.0}
        else
            SITL_ENABLED="false"
            ARDUPILOT_PATH="~/ardupilot"
            SITL_SPEEDUP=1.0
        fi

        print_prompt "Home address (optional, press Enter to skip):"
        read -r HOME_ADDRESS
        if [ -n "$HOME_ADDRESS" ]; then
            print_info "Geocoding address..."
            GEO_RESULT=$(geocode_address "$HOME_ADDRESS" || true)
            if [ -n "$GEO_RESULT" ]; then
                HOME_LAT=$(echo "$GEO_RESULT" | awk '{print $1}')
                HOME_LON=$(echo "$GEO_RESULT" | awk '{print $2}')
                print_status "Home set to lat=${HOME_LAT}, lon=${HOME_LON}"
            else
                print_warning "Address lookup failed; falling back to lat/lon."
            fi
        fi

        if [ -z "$HOME_LAT" ] || [ -z "$HOME_LON" ]; then
            print_prompt "Home latitude (default: 37.7749):"
            read -r HOME_LAT
            HOME_LAT=${HOME_LAT:-37.7749}

            print_prompt "Home longitude (default: -122.4194):"
            read -r HOME_LON
            HOME_LON=${HOME_LON:-"-122.4194"}
        fi
    else
        SIM_ENABLED="false"
        AIRSIM_ENABLED="false"
        AIRSIM_HOST="127.0.0.1"
        SITL_ENABLED="false"
        ARDUPILOT_PATH="~/ardupilot"
        SITL_SPEEDUP=1.0
        HOME_LAT=37.7749
        HOME_LON="-122.4194"
    fi

    # Agent settings
    echo ""
    print_prompt "Enable LLM-based decision making? (Y/n):"
    read -r USE_LLM
    if [[ ! $USE_LLM =~ ^[Nn]$ ]]; then
        USE_LLM="true"

        print_prompt "LLM model (default: gpt-4o-mini):"
        read -r LLM_MODEL
        LLM_MODEL=${LLM_MODEL:-"gpt-4o-mini"}

        echo ""
        print_info "Set OPENAI_API_KEY environment variable for LLM access"
    else
        USE_LLM="false"
        LLM_MODEL="gpt-4o-mini"
    fi

    # Auth settings
    echo ""
    print_prompt "Enable API authentication? (y/N):"
    read -r AUTH_ENABLED
    if [[ $AUTH_ENABLED =~ ^[Yy]$ ]]; then
        AUTH_ENABLED="true"
        print_info "API key will be auto-generated on first run"
    else
        AUTH_ENABLED="false"
    fi

    echo ""
}

# Generate configuration file
generate_config() {
    mkdir -p "$CONFIG_DIR"

    cat > "$CONFIG_DIR/aegis_config.yaml" << EOF
# AegisAV Configuration
# Generated by setup_desktop.sh on $(date)
# Modify via webapp at http://localhost:${SERVER_PORT}/dashboard -> Settings

config_version: "1.0.0"

server:
  host: "${SERVER_HOST}"
  port: ${SERVER_PORT}
  log_level: "INFO"
  cors_origins:
    - "*"

redis:
  enabled: ${REDIS_ENABLED}
  host: "${REDIS_HOST}"
  port: ${REDIS_PORT}
  db: 0
  telemetry_ttl_hours: 1
  detection_ttl_days: 7
  anomaly_ttl_days: 30
  mission_ttl_days: 90

auth:
  enabled: ${AUTH_ENABLED}
  rate_limit_per_minute: 100
  public_endpoints:
    - "/health"
    - "/docs"
    - "/openapi.json"
    - "/redoc"
    - "/"
    - "/dashboard"
    - "/static"

vision:
  enabled: ${VISION_ENABLED}
  use_real_detector: ${USE_REAL_DETECTOR}
  model_path: "${YOLO_MODEL}"
  device: "${VISION_DEVICE}"
  confidence_threshold: ${CONFIDENCE_THRESHOLD}
  iou_threshold: 0.45
  image_size: 640
  camera_resolution:
    - 1920
    - 1080
  save_images: true
  image_output_dir: "data/vision/captures"

simulation:
  enabled: ${SIM_ENABLED}
  airsim_enabled: ${AIRSIM_ENABLED}
  airsim_host: "${AIRSIM_HOST}"
  airsim_vehicle_name: "Drone1"
  sitl_enabled: ${SITL_ENABLED}
  ardupilot_path: "${ARDUPILOT_PATH}"
  sitl_speedup: ${SITL_SPEEDUP}
  home_latitude: ${HOME_LAT}
  home_longitude: ${HOME_LON}
  home_altitude: 0.0

agent:
  use_llm: ${USE_LLM}
  llm_model: "${LLM_MODEL}"
  battery_warning_percent: 30.0
  battery_critical_percent: 15.0
  wind_warning_ms: 8.0
  wind_abort_ms: 12.0
  decision_interval_seconds: 1.0
  max_decisions_per_mission: 1000

dashboard:
  refresh_rate_ms: 1000
  map_provider: "openstreetmap"
  show_telemetry: true
  show_vision: true
  show_reasoning: true
  theme: "dark"
EOF

    print_status "Configuration saved to $CONFIG_DIR/aegis_config.yaml"
}

# Generate environment file
generate_env_file() {
    cat > "$AEGISAV_DIR/.env" << EOF
# AegisAV Environment Variables
# Generated by setup_desktop.sh on $(date)

# Server
AEGIS_HOST=${SERVER_HOST}
AEGIS_PORT=${SERVER_PORT}
AEGIS_LOG_LEVEL=INFO

# Authentication
AEGIS_AUTH_ENABLED=${AUTH_ENABLED}
# AEGIS_API_KEY=your-api-key-here

# Redis Persistence
AEGIS_REDIS_ENABLED=${REDIS_ENABLED}
AEGIS_REDIS_HOST=${REDIS_HOST}
AEGIS_REDIS_PORT=${REDIS_PORT}

# Vision System
AEGIS_VISION_ENABLED=${VISION_ENABLED}
AEGIS_VISION_REAL_DETECTOR=${USE_REAL_DETECTOR}
AEGIS_VISION_MODEL=${YOLO_MODEL}
AEGIS_VISION_DEVICE=${VISION_DEVICE}

# Simulation
AEGIS_SIM_ENABLED=${SIM_ENABLED}
AEGIS_AIRSIM_ENABLED=${AIRSIM_ENABLED}
AEGIS_SITL_ENABLED=${SITL_ENABLED}
AEGIS_ARDUPILOT_PATH=${ARDUPILOT_PATH}

# LLM
AEGIS_USE_LLM=${USE_LLM}
AEGIS_LLM_MODEL=${LLM_MODEL}
# OPENAI_API_KEY=your-openai-key
EOF

    print_status "Environment file saved to $AEGISAV_DIR/.env"
}

# Install system dependencies
install_dependencies() {
    echo ""
    echo "============================================================"
    echo "  Installing Dependencies"
    echo "============================================================"
    echo ""

    if [[ "$OS" == "linux" ]] || [[ "$OS" == "wsl" ]]; then
        local system_marker="$STATE_DIR/system_deps.done"
        local system_packages=(
            git
            gitk
            git-gui
            python3-dev
            python3-pip
            python3-venv
            python3-matplotlib
            python3-lxml
            build-essential
            ccache
            gawk
            libffi-dev
            libxml2-dev
            libxslt1-dev
            screen
            socat
            redis-server
            ffmpeg
            libsm6
            libxext6
        )

        local missing_packages=()
        if command -v dpkg &> /dev/null; then
            for pkg in "${system_packages[@]}"; do
                if ! dpkg -s "$pkg" &> /dev/null; then
                    missing_packages+=("$pkg")
                fi
            done
        fi

        if [[ "$FORCE_REINSTALL" != "true" ]] && [[ "$FORCE_SYSTEM_DEPS" != "true" ]] && [ -f "$system_marker" ] && [ "${#missing_packages[@]}" -eq 0 ]; then
            print_status "System dependencies already installed; skipping (set FORCE_SYSTEM_DEPS=true to reinstall)."
        else
            enable_install_trace
            print_info "Installing system packages..."
            print_info "apt-get update can take a few minutes..."
            sudo apt-get update
            print_info "apt-get install can take several minutes..."
            if [[ "$FORCE_REINSTALL" == "true" ]] || [[ "$FORCE_SYSTEM_DEPS" == "true" ]] || [ "${#missing_packages[@]}" -eq 0 ]; then
                sudo apt-get install -y "${system_packages[@]}"
            else
                sudo apt-get install -y "${missing_packages[@]}"
            fi

            print_status "System dependencies installed"
            mark_done "$system_marker"
            disable_install_trace
        fi

        # WSL-specific: Install GPU support
        if [[ "$OS" == "wsl" ]]; then
            echo ""
            print_info "WSL2 GPU passthrough setup..."

            if [ "$GPU_TYPE" == "nvidia" ]; then
                print_info "For NVIDIA GPU in WSL2:"
                echo "  1. Install NVIDIA drivers on Windows (not in WSL)"
                echo "  2. CUDA is automatically available via WSL2 GPU paravirtualization"
                echo ""

                # Check if CUDA is available in WSL
                if [ -d "/usr/lib/wsl/lib" ]; then
                    print_status "WSL2 GPU libraries detected"
                    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
                fi
            elif [ "$GPU_TYPE" == "amd" ]; then
                print_warning "AMD GPU support in WSL2 is experimental"
                print_info "Consider running natively on Windows or Linux for best AMD GPU support"
            fi
        fi

        # Install PyTorch with GPU support
        echo ""
        local torch_marker="$STATE_DIR/torch.done"
        local torch_installed="false"
        if python3 -c "import torch" &> /dev/null; then
            torch_installed="true"
        fi

        if [[ "$FORCE_REINSTALL" != "true" ]] && [ "$torch_installed" == "true" ] && [ -f "$torch_marker" ]; then
            print_status "PyTorch already installed; skipping (set FORCE_REINSTALL=true to reinstall)."
        else
            enable_install_trace
            print_info "Installing PyTorch..."

            if [ "$GPU_TYPE" == "nvidia" ]; then
                print_info "Installing NVIDIA CUDA PyTorch wheels..."
                pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
                print_status "PyTorch with CUDA support installed"
            elif [ "$GPU_TYPE" == "amd" ]; then
                # ROCm support
                print_info "Installing AMD ROCm PyTorch wheels..."
                pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
                print_status "PyTorch with ROCm support installed"
            else
                print_info "Installing CPU PyTorch wheels..."
                pip3 install torch torchvision
                print_status "PyTorch (CPU) installed"
            fi
            mark_done "$torch_marker"
            disable_install_trace
        fi
    fi
}

# Install ArduPilot
install_ardupilot() {
    if [[ "$SITL_ENABLED" != "true" ]]; then
        print_info "Skipping ArduPilot installation (SITL disabled)"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  Setting up ArduPilot SITL"
    echo "============================================================"
    echo ""

    ARDUPILOT_DIR=$(eval echo "$ARDUPILOT_PATH")
    enable_install_trace
    local ardupilot_marker="$STATE_DIR/ardupilot.done"
    local ardupilot_build_marker="$STATE_DIR/ardupilot_build.done"
    local copter_binary="$ARDUPILOT_DIR/build/sitl/bin/arducopter"

    if [ -d "$ARDUPILOT_DIR" ] && [ -f "$copter_binary" ] && [[ "$FORCE_REINSTALL" != "true" ]] && [[ "$FORCE_ARDUPILOT" != "true" ]]; then
        print_status "ArduPilot SITL already built; skipping (set FORCE_ARDUPILOT=true to rebuild)."
        mark_done "$ardupilot_marker"
        mark_done "$ardupilot_build_marker"
        disable_install_trace
        return
    fi

    if [ -d "$ARDUPILOT_DIR" ]; then
        print_warning "ArduPilot already exists at $ARDUPILOT_DIR"
        if [[ "$UPDATE_ARDUPILOT" == "true" ]]; then
            print_info "Updating ArduPilot repo and submodules..."
            cd "$ARDUPILOT_DIR"
            git pull
            git submodule update --init --recursive
        else
            print_info "Skipping ArduPilot update (use --update-ardupilot to update)."
        fi
    else
        print_info "Cloning ArduPilot (large repo, may take a while)..."
        git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "$ARDUPILOT_DIR"
        print_status "ArduPilot cloned"
    fi

    cd "$ARDUPILOT_DIR"

    if [[ "$OS" == "linux" ]] || [[ "$OS" == "wsl" ]]; then
        print_info "Installing ArduPilot build prerequisites..."
        Tools/environment_install/install-prereqs-ubuntu.sh -y
        . ~/.profile 2>/dev/null || true
    fi

    print_info "Building ArduPilot SITL (this may take a while)..."
    ./waf configure --board sitl
    ./waf copter

    print_status "ArduPilot SITL built successfully"
    mark_done "$ardupilot_marker"
    mark_done "$ardupilot_build_marker"
    disable_install_trace
}

# Install Cosys-AirSim
install_airsim() {
    if [[ "$AIRSIM_ENABLED" != "true" ]]; then
        print_info "Skipping Cosys-AirSim installation (disabled)"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  Setting up Cosys-AirSim"
    echo "============================================================"
    echo ""

    AIRSIM_DIR="$HOME/Cosys-AirSim"
    local airsim_marker="$STATE_DIR/airsim.done"
    local settings_marker="$STATE_DIR/airsim_settings.done"

    if [[ "$OS" == "wsl" ]]; then
        AIRSIM_SETTINGS_DIR="/mnt/c/Users/$WINDOWS_USER/Documents/AirSim"
    else
        AIRSIM_SETTINGS_DIR="$HOME/Documents/AirSim"
    fi
    mkdir -p "$AIRSIM_SETTINGS_DIR"

    if [[ "$FORCE_REINSTALL" != "true" ]] && [[ "$FORCE_AIRSIM" != "true" ]] && [ -f "$airsim_marker" ]; then
        print_status "Cosys-AirSim already configured; skipping install (set FORCE_AIRSIM=true to reinstall)."
    else
        enable_install_trace
        if [[ "$OS" == "wsl" ]]; then
            print_info "Cosys-AirSim for WSL2 setup..."
            print_warning "Cosys-AirSim must run on Windows side for GPU rendering"
            echo ""
            echo "To set up Cosys-AirSim on Windows:"
            echo "  1. Download from: https://github.com/Cosys-Lab/Cosys-AirSim/releases"
            echo "  2. Extract to C:\\Users\\$WINDOWS_USER\\Cosys-AirSim"
            echo "  3. Use Windows path in WSL: /mnt/c/Users/$WINDOWS_USER/Cosys-AirSim"
            echo ""
            echo "Cosys-AirSim supports UE 5.5 (recommended) or UE 5.2 (LTS)"
            echo ""
        else
            if [ -d "$AIRSIM_DIR" ]; then
                print_warning "Cosys-AirSim already exists at $AIRSIM_DIR"
            else
                print_info "Cloning Cosys-AirSim (large repo, may take a while)..."
                git clone https://github.com/Cosys-Lab/Cosys-AirSim.git "$AIRSIM_DIR"
                print_status "Cosys-AirSim cloned"
            fi

            cd "$AIRSIM_DIR"

            if [[ "$OS" == "linux" ]]; then
                print_info "Building Cosys-AirSim (this can take 10+ minutes)..."
                ./setup.sh
                ./build.sh
                print_status "Cosys-AirSim built"
            fi
        fi
        mark_done "$airsim_marker"
        disable_install_trace
    fi

    # Generate AirSim settings
    if [[ "$FORCE_REINSTALL" == "true" ]] || [[ "$FORCE_AIRSIM" == "true" ]] || [ ! -f "$AIRSIM_SETTINGS_DIR/settings.json" ]; then
        cat > "$AIRSIM_SETTINGS_DIR/settings.json" << 'EOF'
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockType": "SteppableClock",
  "LocalHostIp": "127.0.0.1",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "ArduCopter",
      "UseSerial": false,
      "LocalHostIp": "127.0.0.1",
      "UdpIp": "127.0.0.1",
      "UdpPort": 9003,
      "SitlPort": 5760,
      "ControlPort": 14550,
      "AutoCreate": true,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 1920,
              "Height": 1080,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.5,
          "Y": 0,
          "Z": -0.3,
          "Pitch": -15,
          "Roll": 0,
          "Yaw": 0
        },
        "bottom": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 1920,
              "Height": 1080,
              "FOV_Degrees": 90
            }
          ],
          "X": 0,
          "Y": 0,
          "Z": 0.5,
          "Pitch": -90,
          "Roll": 0,
          "Yaw": 0
        }
      },
      "X": 0,
      "Y": 0,
      "Z": 0,
      "Pitch": 0,
      "Roll": 0,
      "Yaw": 0
    }
  },
  "SubWindows": [
    {"WindowID": 0, "CameraName": "front_center", "ImageType": 0, "VehicleName": "Drone1"}
  ]
}
EOF

        print_status "AirSim settings configured at $AIRSIM_SETTINGS_DIR/settings.json"
        mark_done "$settings_marker"
    else
        print_status "AirSim settings already present; skipping (set FORCE_AIRSIM=true to regenerate)."
    fi
}

# Install Python dependencies
install_python_deps() {
    echo ""
    echo "============================================================"
    echo "  Installing Python Dependencies"
    echo "============================================================"
    echo ""

    cd "$AEGISAV_DIR"
    enable_install_trace
    local python_marker="$STATE_DIR/python_deps.done"
    local deps_changed="false"

    if [ -f "$python_marker" ]; then
        if [ -f "pyproject.toml" ] && [ "pyproject.toml" -nt "$python_marker" ]; then
            deps_changed="true"
        fi
        if [ -f "uv.lock" ] && [ "uv.lock" -nt "$python_marker" ]; then
            deps_changed="true"
        fi
    fi

    if [[ "$FORCE_REINSTALL" != "true" ]] && [[ "$FORCE_PYTHON_DEPS" != "true" ]] && [ -f "$python_marker" ] && [ "$deps_changed" == "false" ]; then
        print_status "Python dependencies up to date; skipping (set FORCE_PYTHON_DEPS=true to reinstall)."
        disable_install_trace
        return
    fi

    # Install main dependencies
    if [ -f "pyproject.toml" ]; then
        print_info "Installing AegisAV dependencies with uv..."
        if ! command -v uv &> /dev/null; then
            pip3 install uv
        fi
        print_info "Running uv sync (can take a few minutes)..."
        uv sync
        print_status "AegisAV dependencies installed"
    else
        print_info "Installing dependencies with pip..."
        pip3 install -e .
    fi

    # Install additional simulation dependencies
    print_info "Installing simulation Python packages..."
    pip3 install \
        cosysairsim \
        pymavlink \
        MAVProxy \
        opencv-python \
        Pillow \
        pexpect

    print_status "Python dependencies installed"
    mark_done "$python_marker"
    disable_install_trace
}

# Create convenience scripts
create_scripts() {
    echo ""
    echo "============================================================"
    echo "  Creating Convenience Scripts"
    echo "============================================================"
    echo ""

    # Start simulation script
    cat > "$AEGISAV_DIR/start_simulation.sh" << EOF
#!/bin/bash
# Start complete AegisAV simulation

echo "Starting AegisAV High-Fidelity Simulation..."
echo ""

# Load environment
if [ -f "\$(dirname "\$0")/.env" ]; then
    set -a
    source "\$(dirname "\$0")/.env"
    set +a
fi

cd "\$(dirname "\$0")"
python3 simulation/run_simulation.py --airsim --sitl "\$@"
EOF
    chmod +x "$AEGISAV_DIR/start_simulation.sh"
    print_status "Created start_simulation.sh"

    # Start server script
    cat > "$AEGISAV_DIR/start_server.sh" << EOF
#!/bin/bash
# Start AegisAV server only

echo "Starting AegisAV Server..."
echo ""

# Load environment
if [ -f "\$(dirname "\$0")/.env" ]; then
    set -a
    source "\$(dirname "\$0")/.env"
    set +a
fi

cd "\$(dirname "\$0")"
python3 -m uvicorn agent.server.main:app --host \${AEGIS_HOST:-0.0.0.0} --port \${AEGIS_PORT:-8000} --reload
EOF
    chmod +x "$AEGISAV_DIR/start_server.sh"
    print_status "Created start_server.sh"

    # WSL-specific: Create Windows batch file
    if [[ "$OS" == "wsl" ]]; then
        cat > "$AEGISAV_DIR/start_airsim.bat" << EOF
@echo off
REM Start Cosys-AirSim on Windows
REM Run this from Windows Command Prompt, not WSL

echo Starting Cosys-AirSim...
cd %USERPROFILE%\\Cosys-AirSim
start Blocks.exe -ResX=1920 -ResY=1080 -windowed
EOF
        print_status "Created start_airsim.bat (run from Windows)"
    fi
}

# Verify Cosys-AirSim is running and reachable
verify_airsim_running() {
    if [[ "$AIRSIM_ENABLED" != "true" ]]; then
        return
    fi

    echo ""
    echo "============================================================"
    echo "  Verifying Cosys-AirSim Runtime"
    echo "============================================================"
    echo ""
    print_info "Checking Cosys-AirSim RPC on 127.0.0.1:41451..."

    if python3 - << 'PY'
import socket
import sys

host = "127.0.0.1"
port = 41451
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect((host, port))
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
    then
        print_status "Cosys-AirSim RPC reachable"
    else
        print_warning "Cosys-AirSim RPC not reachable"
        if [[ "$OS" == "wsl" ]]; then
            print_info "Start Cosys-AirSim on Windows: run start_airsim.bat, then re-run this check."
        else
            print_info "Start Cosys-AirSim (Unreal) and try again."
        fi
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "============================================================"
    echo "  Setup Complete!"
    echo "============================================================"
    echo ""
    echo "Configuration:"
    echo "  Config file: $CONFIG_DIR/aegis_config.yaml"
    echo "  Environment: $AEGISAV_DIR/.env"
    echo ""
    echo "GPU: $GPU_NAME ($GPU_TYPE)"
    echo "Vision device: $VISION_DEVICE"
    echo ""
    if [[ "$USE_DOCKER" == "true" ]]; then
        echo "Docker Compose:"
        echo "  docker compose ps"
        echo "  docker compose logs -f"
        echo ""
    fi

    if [[ "$OS" == "wsl" ]]; then
        echo "WSL2 Setup Instructions:"
        echo ""
        echo "  1. On Windows: Download Cosys-AirSim from GitHub releases"
        echo "     https://github.com/Cosys-Lab/Cosys-AirSim/releases"
        echo "  2. On Windows: Run start_airsim.bat"
        echo "  3. In WSL: ./start_simulation.sh"
        echo "  4. Open: http://localhost:${SERVER_PORT}/dashboard"
        echo ""
    else
        echo "To run the simulation:"
        echo ""
        echo "  Terminal 1 - Start Cosys-AirSim:"
        echo "    ~/Cosys-AirSim/Blocks.sh -ResX=1920 -ResY=1080 -windowed"
        echo ""
        echo "  Terminal 2 - Start ArduPilot SITL:"
        echo "    cd $(eval echo $ARDUPILOT_PATH)/ArduCopter"
        echo "    ../Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter --console --map"
        echo ""
        echo "  Terminal 3 - Start AegisAV:"
        echo "    ./start_simulation.sh"
        echo ""
        echo "  Dashboard:"
        echo "    http://localhost:${SERVER_PORT}/dashboard"
        echo ""
    fi

    echo "Configuration can be modified:"
    echo "  - Via dashboard Settings panel"
    echo "  - By editing $CONFIG_DIR/aegis_config.yaml"
    echo "  - Via environment variables (see .env file)"
    echo ""
    echo "============================================================"
}

# Main execution
main() {
    parse_args "$@"

    if [[ "$CLEANUP" == "true" ]]; then
        cleanup_setup
        exit 0
    fi

    detect_environment

    if [[ "$RUN_INTERACTIVE" == "true" ]]; then
        interactive_config
    else
        apply_home_address
    fi

    generate_config
    generate_env_file

    if [[ "$INSTALL_DEPS" == "true" ]]; then
        install_dependencies
        install_python_deps
        install_ardupilot
        install_airsim
    fi

    create_scripts
    if [[ "$USE_DOCKER" == "true" ]]; then
        setup_docker_compose
    fi
    verify_airsim_running
    print_summary
}

# Run main
main "$@"
