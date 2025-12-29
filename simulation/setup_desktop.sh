#!/bin/bash
#
# AegisAV Desktop Setup Script
#
# Comprehensive setup for high-fidelity drone simulation with:
# - ArduPilot SITL (flight controller)
# - AirSim (Unreal Engine rendering)
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

# Configuration defaults
CONFIG_FILE=""
AEGISAV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="$AEGISAV_DIR/configs"

# Detect environment
detect_environment() {
    echo "Detecting environment..."
    echo ""

    # Check if running in WSL
    if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
        OS="wsl"
        WINDOWS_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
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
        print_warning "No GPU detected. Using CPU for ML inference."
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
    print_prompt "Enable Redis persistence? (y/N):"
    read -r REDIS_ENABLED
    if [[ $REDIS_ENABLED =~ ^[Yy]$ ]]; then
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

        print_prompt "Enable AirSim integration? (Y/n):"
        read -r AIRSIM_ENABLED
        if [[ ! $AIRSIM_ENABLED =~ ^[Nn]$ ]]; then
            AIRSIM_ENABLED="true"

            print_prompt "AirSim host (default: 127.0.0.1):"
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

        print_prompt "Home latitude (default: 37.7749):"
        read -r HOME_LAT
        HOME_LAT=${HOME_LAT:-37.7749}

        print_prompt "Home longitude (default: -122.4194):"
        read -r HOME_LON
        HOME_LON=${HOME_LON:-"-122.4194"}
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
        print_info "Installing system packages..."
        sudo apt-get update
        sudo apt-get install -y \
            git \
            gitk \
            git-gui \
            python3-dev \
            python3-pip \
            python3-venv \
            python3-matplotlib \
            python3-lxml \
            build-essential \
            ccache \
            gawk \
            libffi-dev \
            libxml2-dev \
            libxslt1-dev \
            screen \
            socat \
            redis-server \
            ffmpeg \
            libsm6 \
            libxext6

        print_status "System dependencies installed"

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
        print_info "Installing PyTorch..."

        if [ "$GPU_TYPE" == "nvidia" ]; then
            pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            print_status "PyTorch with CUDA support installed"
        elif [ "$GPU_TYPE" == "amd" ]; then
            # ROCm support
            pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
            print_status "PyTorch with ROCm support installed"
        else
            pip3 install torch torchvision
            print_status "PyTorch (CPU) installed"
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

    if [ -d "$ARDUPILOT_DIR" ]; then
        print_warning "ArduPilot already exists at $ARDUPILOT_DIR"
        print_prompt "Update existing installation? (y/N):"
        read -r UPDATE_ARDUPILOT
        if [[ $UPDATE_ARDUPILOT =~ ^[Yy]$ ]]; then
            cd "$ARDUPILOT_DIR"
            git pull
            git submodule update --init --recursive
        fi
    else
        git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "$ARDUPILOT_DIR"
        print_status "ArduPilot cloned"
    fi

    cd "$ARDUPILOT_DIR"

    if [[ "$OS" == "linux" ]] || [[ "$OS" == "wsl" ]]; then
        Tools/environment_install/install-prereqs-ubuntu.sh -y
        . ~/.profile 2>/dev/null || true
    fi

    print_info "Building ArduPilot SITL (this may take a while)..."
    ./waf configure --board sitl
    ./waf copter

    print_status "ArduPilot SITL built successfully"
}

# Install AirSim
install_airsim() {
    if [[ "$AIRSIM_ENABLED" != "true" ]]; then
        print_info "Skipping AirSim installation (AirSim disabled)"
        return
    fi

    echo ""
    echo "============================================================"
    echo "  Setting up AirSim"
    echo "============================================================"
    echo ""

    AIRSIM_DIR="$HOME/AirSim"

    if [[ "$OS" == "wsl" ]]; then
        print_info "AirSim for WSL2 setup..."
        print_warning "AirSim must run on Windows side for GPU rendering"
        echo ""
        echo "To set up AirSim on Windows:"
        echo "  1. Download AirSim from: https://github.com/Microsoft/AirSim/releases"
        echo "  2. Extract to C:\\Users\\$WINDOWS_USER\\AirSim"
        echo "  3. Use Windows path in WSL: /mnt/c/Users/$WINDOWS_USER/AirSim"
        echo ""

        # Create Windows-compatible AirSim settings
        AIRSIM_SETTINGS_DIR="/mnt/c/Users/$WINDOWS_USER/Documents/AirSim"
        mkdir -p "$AIRSIM_SETTINGS_DIR"
    else
        if [ -d "$AIRSIM_DIR" ]; then
            print_warning "AirSim already exists at $AIRSIM_DIR"
        else
            git clone https://github.com/Microsoft/AirSim.git "$AIRSIM_DIR"
            print_status "AirSim cloned"
        fi

        cd "$AIRSIM_DIR"

        if [[ "$OS" == "linux" ]]; then
            ./setup.sh
            ./build.sh
            print_status "AirSim built"
        fi

        AIRSIM_SETTINGS_DIR="$HOME/Documents/AirSim"
        mkdir -p "$AIRSIM_SETTINGS_DIR"
    fi

    # Generate AirSim settings
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
}

# Install Python dependencies
install_python_deps() {
    echo ""
    echo "============================================================"
    echo "  Installing Python Dependencies"
    echo "============================================================"
    echo ""

    cd "$AEGISAV_DIR"

    # Install main dependencies
    if [ -f "pyproject.toml" ]; then
        print_info "Installing AegisAV dependencies with uv..."
        pip3 install uv
        uv sync
        print_status "AegisAV dependencies installed"
    else
        print_info "Installing dependencies with pip..."
        pip3 install -e .
    fi

    # Install additional simulation dependencies
    pip3 install \
        airsim \
        pymavlink \
        MAVProxy \
        msgpack-rpc-python \
        opencv-python \
        Pillow

    print_status "Python dependencies installed"
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
REM Start AirSim on Windows
REM Run this from Windows Command Prompt, not WSL

echo Starting AirSim...
cd %USERPROFILE%\\AirSim\\AirSimNH
start AirSimNH.exe -ResX=1920 -ResY=1080 -windowed
EOF
        print_status "Created start_airsim.bat (run from Windows)"
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

    if [[ "$OS" == "wsl" ]]; then
        echo "WSL2 Setup Instructions:"
        echo ""
        echo "  1. On Windows: Download AirSim from GitHub releases"
        echo "  2. On Windows: Run start_airsim.bat"
        echo "  3. In WSL: ./start_simulation.sh"
        echo "  4. Open: http://localhost:${SERVER_PORT}/dashboard"
        echo ""
    else
        echo "To run the simulation:"
        echo ""
        echo "  Terminal 1 - Start AirSim:"
        echo "    ~/AirSimEnv/AirSimNH.sh -ResX=1920 -ResY=1080 -windowed"
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
    detect_environment

    echo ""
    print_prompt "Run interactive configuration? (Y/n):"
    read -r RUN_INTERACTIVE

    if [[ ! $RUN_INTERACTIVE =~ ^[Nn]$ ]]; then
        interactive_config
    else
        # Use defaults
        SERVER_HOST="0.0.0.0"
        SERVER_PORT=8000
        REDIS_ENABLED="false"
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
        USE_LLM="true"
        LLM_MODEL="gpt-4o-mini"
        AUTH_ENABLED="false"
    fi

    generate_config
    generate_env_file

    echo ""
    print_prompt "Install dependencies and simulation tools? (Y/n):"
    read -r INSTALL_DEPS

    if [[ ! $INSTALL_DEPS =~ ^[Nn]$ ]]; then
        install_dependencies
        install_python_deps
        install_ardupilot
        install_airsim
    fi

    create_scripts
    print_summary
}

# Run main
main "$@"
