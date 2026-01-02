#!/bin/bash
# ============================================================================
# Invoke Windows Install Scripts from WSL
# ============================================================================
# This script allows running Windows batch files from within WSL.
# It converts WSL paths to Windows paths and executes via cmd.exe.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEGISAV_ROOT="$(dirname "$SCRIPT_DIR")"

# Convert WSL path to Windows path
wsl_to_windows_path() {
    local wsl_path="$1"
    wslpath -w "$wsl_path" 2>/dev/null || echo "$wsl_path"
}

# Get Windows path to the scripts
WINDOWS_SCRIPTS_DIR=$(wsl_to_windows_path "$SCRIPT_DIR")
WINDOWS_ROOT=$(wsl_to_windows_path "$AEGISAV_ROOT")

echo "=============================================="
echo "  AegisAV Windows Install Script Invoker"
echo "=============================================="
echo ""
echo "AegisAV Root (WSL):     $AEGISAV_ROOT"
echo "AegisAV Root (Windows): $WINDOWS_ROOT"
echo ""

# Function to run a Windows batch script
run_windows_script() {
    local script_name="$1"
    local script_path="${SCRIPT_DIR}/${script_name}"
    local windows_path=$(wsl_to_windows_path "$script_path")

    if [ ! -f "$script_path" ]; then
        echo "ERROR: Script not found: $script_path"
        return 1
    fi

    echo "Running: $script_name"
    echo "Windows Path: $windows_path"
    echo ""

    # Run the batch file via cmd.exe
    # The /C flag runs the command and then terminates
    cmd.exe /C "cd /d \"$WINDOWS_ROOT\" && \"$windows_path\""

    return $?
}

# Function to check if running in WSL
check_wsl() {
    if ! grep -qi microsoft /proc/version 2>/dev/null; then
        echo "WARNING: This script is designed to run in WSL."
        echo "It may not work correctly in other environments."
    fi
}

# Main menu
show_menu() {
    echo "Available install scripts:"
    echo ""
    echo "  1) install_airsim.bat - Install AirSim + AegisAV Overlay"
    echo "  2) Custom script path"
    echo "  3) Exit"
    echo ""
    read -p "Select option [1-3]: " choice

    case $choice in
        1)
            run_windows_script "install_airsim.bat"
            ;;
        2)
            read -p "Enter script name (in scripts/): " script_name
            run_windows_script "$script_name"
            ;;
        3)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "Invalid option"
            show_menu
            ;;
    esac
}

# Direct invocation with argument
if [ -n "$1" ]; then
    check_wsl
    run_windows_script "$1"
else
    check_wsl
    show_menu
fi
