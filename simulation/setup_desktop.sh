#!/bin/bash
#
# AegisAV Simulation Desktop Setup
#
# This script sets up your desktop for high-fidelity drone simulation:
# - ArduPilot SITL (flight controller)
# - AirSim (Unreal Engine rendering)
# - Required dependencies
#
# Requirements:
# - Ubuntu 20.04/22.04 or Windows 11
# - 36GB+ RAM ✓
# - AMD 6950XT or equivalent GPU ✓
# - 100GB+ free disk space
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
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    print_status "Detected Linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    print_status "Detected Windows"
else
    print_error "Unsupported OS: $OSTYPE"
    exit 1
fi

# ============================================================
# Step 1: Install system dependencies
# ============================================================
echo ""
echo "Step 1: Installing system dependencies..."

if [[ "$OS" == "linux" ]]; then
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
        python3-pygame \
        python3-wxgtk4.0 \
        build-essential \
        ccache \
        gawk \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        screen \
        socat \
        xterm

    print_status "System dependencies installed"
fi

# ============================================================
# Step 2: Install ArduPilot
# ============================================================
echo ""
echo "Step 2: Setting up ArduPilot..."

ARDUPILOT_DIR="$HOME/ardupilot"

if [ -d "$ARDUPILOT_DIR" ]; then
    print_warning "ArduPilot already exists at $ARDUPILOT_DIR"
    read -p "Update existing installation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$ARDUPILOT_DIR"
        git pull
        git submodule update --init --recursive
    fi
else
    git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "$ARDUPILOT_DIR"
    print_status "ArduPilot cloned"
fi

# Install ArduPilot prereqs
cd "$ARDUPILOT_DIR"
if [[ "$OS" == "linux" ]]; then
    Tools/environment_install/install-prereqs-ubuntu.sh -y
    . ~/.profile
fi

# Build SITL
print_status "Building ArduPilot SITL (this may take a while)..."
./waf configure --board sitl
./waf copter

print_status "ArduPilot SITL built successfully"

# ============================================================
# Step 3: Install AirSim
# ============================================================
echo ""
echo "Step 3: Setting up AirSim..."

AIRSIM_DIR="$HOME/AirSim"

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
else
    print_warning "On Windows, run build.cmd from Developer Command Prompt"
fi

# ============================================================
# Step 4: Configure AirSim for ArduPilot
# ============================================================
echo ""
echo "Step 4: Configuring AirSim..."

AIRSIM_SETTINGS_DIR="$HOME/Documents/AirSim"
mkdir -p "$AIRSIM_SETTINGS_DIR"

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

print_status "AirSim configured for ArduPilot"

# ============================================================
# Step 5: Install Python dependencies
# ============================================================
echo ""
echo "Step 5: Installing Python dependencies..."

pip3 install --user \
    airsim \
    pymavlink \
    MAVProxy \
    msgpack-rpc-python \
    numpy \
    opencv-python \
    Pillow

print_status "Python dependencies installed"

# ============================================================
# Step 6: Download pre-built AirSim environment (optional)
# ============================================================
echo ""
echo "Step 6: AirSim environment..."

AIRSIM_ENV_DIR="$HOME/AirSimEnv"
mkdir -p "$AIRSIM_ENV_DIR"

print_warning "Pre-built environments can be downloaded from:"
echo "  https://github.com/Microsoft/AirSim/releases"
echo ""
echo "  Recommended: AirSimNH (Neighborhood environment)"
echo "  Download and extract to: $AIRSIM_ENV_DIR"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "Installed components:"
echo "  [✓] ArduPilot SITL: $ARDUPILOT_DIR"
echo "  [✓] AirSim: $AIRSIM_DIR"
echo "  [✓] AirSim settings: $AIRSIM_SETTINGS_DIR/settings.json"
echo ""
echo "To run the simulation:"
echo ""
echo "  Terminal 1 - Start AirSim:"
echo "    $AIRSIM_ENV_DIR/AirSimNH.sh -ResX=1920 -ResY=1080 -windowed"
echo ""
echo "  Terminal 2 - Start ArduPilot SITL:"
echo "    cd $ARDUPILOT_DIR/ArduCopter"
echo "    ../Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter --console --map"
echo ""
echo "  Terminal 3 - Start AegisAV:"
echo "    cd $(dirname "$0")/.."
echo "    python3 simulation/run_simulation.py --airsim --sitl"
echo ""
echo "  Terminal 4 - Open Dashboard:"
echo "    http://localhost:8000/dashboard"
echo ""
echo "============================================================"
echo ""

# Create convenience scripts
AEGISAV_DIR="$(dirname "$0")/.."

cat > "$AEGISAV_DIR/start_simulation.sh" << EOF
#!/bin/bash
# Start complete AegisAV simulation

echo "Starting AegisAV High-Fidelity Simulation..."
echo ""
echo "Make sure AirSim is running first!"
echo ""

cd "\$(dirname "\$0")"
python3 simulation/run_simulation.py --airsim --sitl "\$@"
EOF
chmod +x "$AEGISAV_DIR/start_simulation.sh"

print_status "Created start_simulation.sh convenience script"
echo ""
echo "You can now run: ./start_simulation.sh"
