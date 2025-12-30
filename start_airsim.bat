@echo off
REM Start AirSim on Windows
REM Run this from Windows Command Prompt, not WSL

echo Starting AirSim...
cd /d "C:\Users\games\Desktop\AirSimNH\WindowsNoEditor"
start "" "AirSimNH.exe" -ResX=1920 -ResY=1080 -windowed
