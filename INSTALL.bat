@echo off
REM AegisAV Installer Launcher for Windows
REM Double-click this file to start the setup wizard

echo ================================================
echo          AegisAV Setup Wizard Launcher
echo ================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Found Python:
python --version
echo.

echo Starting setup wizard...
echo.

cd /d "%~dp0"
python scripts\installer\setup_gui.py

if errorlevel 1 (
    echo.
    echo Setup wizard exited with an error.
    echo.
    pause
)
