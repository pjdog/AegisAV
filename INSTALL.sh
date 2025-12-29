#!/bin/bash
# AegisAV Installer Launcher for Linux/macOS
# Run: chmod +x INSTALL.sh && ./INSTALL.sh

echo "================================================"
echo "         AegisAV Setup Wizard Launcher"
echo "================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3 is not installed"
    echo ""
    echo "Please install Python 3.10+:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-tk"
    echo "  Fedora: sudo dnf install python3 python3-tkinter"
    echo "  macOS: brew install python python-tk"
    echo ""
    exit 1
fi

echo "Found Python:"
$PYTHON_CMD --version
echo ""

# Check for tkinter
$PYTHON_CMD -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Python tkinter module is not installed"
    echo ""
    echo "Please install tkinter:"
    echo "  Ubuntu/Debian: sudo apt-get install python3-tk"
    echo "  Fedora: sudo dnf install python3-tkinter"
    echo "  macOS: brew install python-tk"
    echo ""
    exit 1
fi

echo "Starting setup wizard..."
echo ""

$PYTHON_CMD scripts/installer/setup_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Setup wizard exited with an error."
    echo ""
fi
