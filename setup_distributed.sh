#!/bin/bash
# AegisAV Distributed Setup Helper
# Helps configure AegisAV to run simulation on desktop, agents on laptop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/configs/agent_config.yaml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   AegisAV Distributed Setup Helper                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect current machine
echo -e "${YELLOW}Detecting system...${NC}"
HOSTNAME=$(hostname)
echo "Hostname: $HOSTNAME"
echo ""

# Menu
echo "What is this machine?"
echo "  1) Desktop (will run simulation)"
echo "  2) Laptop (will run agents + development)"
echo ""
read -p "Select [1 or 2]: " MACHINE_TYPE

if [ "$MACHINE_TYPE" == "1" ]; then
    echo ""
    echo -e "${GREEN}=== Desktop Setup ===${NC}"
    echo ""

    # Check ArduPilot
    echo "Checking ArduPilot SITL..."
    if command -v sim_vehicle.py &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} ArduPilot SITL found"
    else
        echo -e "  ${RED}✗${NC} ArduPilot SITL not found"
        echo ""
        echo "Would you like to install ArduPilot? (requires ~2GB download)"
        read -p "[y/N]: " INSTALL_AP
        if [ "$INSTALL_AP" == "y" ] || [ "$INSTALL_AP" == "Y" ]; then
            echo "Installing ArduPilot..."
            cd ~
            git clone https://github.com/ArduPilot/ardupilot.git
            cd ardupilot
            git submodule update --init --recursive
            Tools/environment_install/install-prereqs-ubuntu.sh -y

            echo ""
            echo -e "${YELLOW}Add this to your ~/.bashrc:${NC}"
            echo 'export PATH="$HOME/ardupilot/Tools/autotest:$PATH"'
            echo ""
            echo "Then run: source ~/.bashrc"
        fi
    fi

    # Get IP address
    echo ""
    echo "Network interfaces:"
    ip -br addr show | grep -v "lo\|DOWN" || true
    echo ""
    read -p "Enter desktop IP address (e.g., 192.168.1.100): " DESKTOP_IP

    # Firewall
    echo ""
    echo "Configuring firewall for MAVLink (UDP 14550)..."
    if command -v ufw &> /dev/null; then
        sudo ufw allow 14550/udp
        echo -e "  ${GREEN}✓${NC} UFW rule added"
    elif command -v firewall-cmd &> /dev/null; then
        sudo firewall-cmd --add-port=14550/udp --permanent
        sudo firewall-cmd --reload
        echo -e "  ${GREEN}✓${NC} firewalld rule added"
    else
        echo -e "  ${YELLOW}⚠${NC} No firewall detected, skipping"
    fi

    echo ""
    echo -e "${GREEN}=== Desktop Setup Complete ===${NC}"
    echo ""
    echo "To start simulation:"
    echo -e "  ${YELLOW}sim_vehicle.py -v ArduCopter -f quad --out udp:0.0.0.0:14550 --map --console${NC}"
    echo ""
    echo "Configure your laptop with:"
    echo -e "  ${YELLOW}Desktop IP: $DESKTOP_IP${NC}"
    echo ""

elif [ "$MACHINE_TYPE" == "2" ]; then
    echo ""
    echo -e "${GREEN}=== Laptop Setup ===${NC}"
    echo ""

    # Get desktop IP
    read -p "Enter desktop IP address (e.g., 192.168.1.100): " DESKTOP_IP

    # Configure MAVLink connection
    echo ""
    echo "Updating MAVLink connection in $CONFIG_FILE..."
    if [ -f "$CONFIG_FILE" ]; then
        # Backup original
        cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

        # Update connection string (simple sed replacement)
        sed -i "s|connection:.*|connection: \"udp:$DESKTOP_IP:14550\"|g" "$CONFIG_FILE"

        echo -e "  ${GREEN}✓${NC} Updated MAVLink connection to udp:$DESKTOP_IP:14550"
    else
        echo -e "  ${RED}✗${NC} Config file not found: $CONFIG_FILE"
    fi

    # LLM API setup
    echo ""
    echo "LLM Configuration:"
    echo "  1) OpenAI (GPT-4o)"
    echo "  2) Anthropic (Claude)"
    echo "  3) Skip (configure manually later)"
    echo ""
    read -p "Select LLM provider [1-3]: " LLM_CHOICE

    if [ "$LLM_CHOICE" == "1" ]; then
        read -p "Enter OpenAI API key (sk-...): " OPENAI_KEY
        echo "export OPENAI_API_KEY=\"$OPENAI_KEY\"" >> ~/.bashrc
        export OPENAI_API_KEY="$OPENAI_KEY"
        echo -e "  ${GREEN}✓${NC} OpenAI API key saved to ~/.bashrc"
    elif [ "$LLM_CHOICE" == "2" ]; then
        read -p "Enter Anthropic API key (sk-ant-...): " ANTHROPIC_KEY
        echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_KEY\"" >> ~/.bashrc
        export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
        echo -e "  ${GREEN}✓${NC} Anthropic API key saved to ~/.bashrc"

        echo ""
        echo -e "${YELLOW}Note: You'll need to edit agent/server/advanced_decision.py${NC}"
        echo "Change line 109 from:"
        echo "  \"openai:gpt-4o\""
        echo "to:"
        echo "  \"anthropic:claude-3-5-sonnet-20241022\""
    fi

    # Test connection
    echo ""
    echo "Testing connection to desktop..."
    if ping -c 1 -W 2 "$DESKTOP_IP" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Desktop is reachable"
    else
        echo -e "  ${RED}✗${NC} Cannot reach desktop at $DESKTOP_IP"
        echo "  Make sure desktop is on and connected to same network"
    fi

    echo ""
    echo -e "${GREEN}=== Laptop Setup Complete ===${NC}"
    echo ""
    echo "To start AegisAV:"
    echo ""
    echo "  Terminal 1 (Agent Server):"
    echo -e "    ${YELLOW}uv run aegis-server${NC}"
    echo ""
    echo "  Terminal 2 (Agent Client):"
    echo -e "    ${YELLOW}uv run aegis-demo --scenario anomaly${NC}"
    echo ""
    echo "Make sure SITL is running on desktop first!"
    echo ""

else
    echo "Invalid selection. Exiting."
    exit 1
fi

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "For full documentation, see:"
echo "  - aegisav_distributed_architecture.md"
echo "  - README.md"
