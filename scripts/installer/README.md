# AegisAV Setup Wizard

A cross-platform GUI installer for setting up AegisAV with Unreal Engine and AirSim.

## Quick Start

### Windows
Double-click `INSTALL.bat` in the project root, or run:
```batch
python scripts\installer\setup_gui.py
```

### Linux / macOS
```bash
./INSTALL.sh
# or directly:
python3 scripts/installer/setup_gui.py
```

## Requirements

- **Python 3.10+** with tkinter
- **Git** (for dependency installation)
- **Node.js 18+** (optional, for dashboard frontend)

### Installing tkinter

If you see "tkinter is required but not installed":

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**Fedora:**
```bash
sudo dnf install python3-tkinter
```

**Arch Linux:**
```bash
sudo pacman -S tk
```

**macOS:**
```bash
brew install python-tk
```

**Windows:**
Reinstall Python from python.org and check "tcl/tk and IDLE" option.

## What the Wizard Does

1. **System Requirements Check**
   - Verifies Python, pip, Git, Node.js, npm
   - Shows install buttons for missing dependencies

2. **Python Environment Setup**
   - Installs AegisAV Python packages using uv or pip
   - Optional dev dependencies for testing

3. **Unreal Engine Setup**
   - Auto-detects existing UE5 installation
   - Provides download links if not installed
   - Option to skip for OBS-only setup

4. **AirSim Configuration**
   - Creates optimized settings.json
   - Tests connection to AirSim
   - Guides through installation

5. **AegisAV Configuration**
   - Server options (Redis, GPU)
   - Frontend build settings
   - Port configuration

6. **Final Installation**
   - Generates config files
   - Builds frontend (if enabled)
   - Creates launch scripts
   - Verifies installation

## Command-Line Installation (Alternative)

If you prefer not to use the GUI:

```bash
# 1. Install Python dependencies
pip install -e .
# or with uv:
uv sync

# 2. Build frontend (optional)
cd frontend && npm install && npm run build

# 3. Start server
python -m agent.server.main
```

## Generated Files

After installation, the wizard creates:

- `.aegis_setup.json` - Setup configuration
- `scripts/start_server.sh` (or `.bat`) - Server launcher
- `scripts/run_demo.sh` (or `.bat`) - Demo launcher

## Troubleshooting

### "No module named tkinter"
Install tkinter for your platform (see above).

### "Permission denied" on Linux
```bash
chmod +x INSTALL.sh
chmod +x scripts/installer/setup_gui.py
```

### GUI doesn't appear
- Check if you're in a graphical environment
- On WSL2, install an X server like VcXsrv
- Try running with `python3 -u scripts/installer/setup_gui.py`

### AirSim connection fails
1. Make sure AirSim is running first
2. Check that `airsim` Python package is installed: `pip install airsim`
3. Verify settings.json exists in `~/Documents/AirSim/`

## Support

For issues with the installer, please open an issue on the repository.
