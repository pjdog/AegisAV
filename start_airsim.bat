@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM AegisAV Cosys-AirSim Launcher
REM ============================================================================
REM Searches for Cosys-AirSim executable in multiple locations and launches it.
REM ============================================================================

echo.
echo ================================================================
echo            AegisAV Cosys-AirSim Launcher
echo ================================================================
echo.

set "AIRSIM_EXE="
set "AEGIS_LOCAL=%LOCALAPPDATA%\AegisAV"

REM Check packaged build output first (from installer packaging step)
if exist "%AEGIS_LOCAL%\AirSim\Blocks\Blocks.exe" (
    set "AIRSIM_EXE=%AEGIS_LOCAL%\AirSim\Blocks\Blocks.exe"
    goto :found
)
if exist "%AEGIS_LOCAL%\Cosys-AirSim\Blocks\Blocks.exe" (
    set "AIRSIM_EXE=%AEGIS_LOCAL%\Cosys-AirSim\Blocks\Blocks.exe"
    goto :found
)

REM Check UAT Build output (Build\WindowsNoEditor)
if exist "%AEGIS_LOCAL%\AirSim\Build\WindowsNoEditor\Blocks.exe" (
    set "AIRSIM_EXE=%AEGIS_LOCAL%\AirSim\Build\WindowsNoEditor\Blocks.exe"
    goto :found
)
if exist "%AEGIS_LOCAL%\Cosys-AirSim\Build\WindowsNoEditor\Blocks.exe" (
    set "AIRSIM_EXE=%AEGIS_LOCAL%\Cosys-AirSim\Build\WindowsNoEditor\Blocks.exe"
    goto :found
)

REM Check Editor build output (Development Editor build from Visual Studio)
if exist "%AEGIS_LOCAL%\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe" (
    set "AIRSIM_EXE=%AEGIS_LOCAL%\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    goto :found
)
if exist "%AEGIS_LOCAL%\Cosys-AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe" (
    set "AIRSIM_EXE=%AEGIS_LOCAL%\Cosys-AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    goto :found
)

REM Check user's Desktop for pre-built binaries
if exist "%USERPROFILE%\Desktop\Cosys-AirSim\WindowsNoEditor\Blocks.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Desktop\Cosys-AirSim\WindowsNoEditor\Blocks.exe"
    goto :found
)
if exist "%USERPROFILE%\Desktop\Cosys-AirSim\Blocks.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Desktop\Cosys-AirSim\Blocks.exe"
    goto :found
)

REM Check Downloads folder
if exist "%USERPROFILE%\Downloads\Cosys-AirSim\WindowsNoEditor\Blocks.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Downloads\Cosys-AirSim\WindowsNoEditor\Blocks.exe"
    goto :found
)
if exist "%USERPROFILE%\Downloads\Cosys-AirSim\Blocks.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Downloads\Cosys-AirSim\Blocks.exe"
    goto :found
)

REM Check common install locations
if exist "C:\Cosys-AirSim\Blocks.exe" (
    set "AIRSIM_EXE=C:\Cosys-AirSim\Blocks.exe"
    goto :found
)
if exist "C:\Cosys-AirSim\WindowsNoEditor\Blocks.exe" (
    set "AIRSIM_EXE=C:\Cosys-AirSim\WindowsNoEditor\Blocks.exe"
    goto :found
)

REM Check legacy AirSimNH locations
if exist "%USERPROFILE%\Desktop\AirSimNH\WindowsNoEditor\AirSimNH.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Desktop\AirSimNH\WindowsNoEditor\AirSimNH.exe"
    goto :found
)

REM Not found - show helpful message
echo ERROR: Cosys-AirSim executable (Blocks.exe) not found!
echo.
echo Searched these locations:
echo   - %AEGIS_LOCAL%\AirSim\Blocks\
echo   - %AEGIS_LOCAL%\AirSim\Build\WindowsNoEditor\
echo   - %AEGIS_LOCAL%\AirSim\Unreal\Environments\Blocks\Binaries\Win64\
echo   - %%USERPROFILE%%\Desktop\Cosys-AirSim\
echo   - %%USERPROFILE%%\Downloads\Cosys-AirSim\
echo   - C:\Cosys-AirSim\
echo.
echo ================================================================
echo                      HOW TO FIX
echo ================================================================
echo.
echo The AirSim C++ libraries were built, but the Unreal Blocks
echo environment needs to be packaged separately.
echo.
echo Option 1: Package from Unreal Editor
echo   1. Open: %AEGIS_LOCAL%\AirSim\Unreal\Environments\Blocks\Blocks.uproject
echo   2. Wait for shaders to compile (first time only)
echo   3. File -^> Package Project -^> Windows (64-bit)
echo   4. Choose output: %AEGIS_LOCAL%\AirSim\Blocks
echo.
echo Option 2: Run the build script
echo   Run: scripts\build_blocks_ue5.bat
echo   Choose Option 2 to package a standalone executable.
echo.
echo Option 3: Download pre-built binaries
echo   https://github.com/Cosys-Lab/Cosys-AirSim/releases
echo   Extract to Desktop\Cosys-AirSim
echo.
pause
exit /b 1

:found
echo Found Cosys-AirSim at: %AIRSIM_EXE%
echo.

REM Check if already running
tasklist /FI "IMAGENAME eq Blocks.exe" 2>NUL | find /I /N "Blocks.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Cosys-AirSim is already running!
    echo.
    pause
    exit /b 0
)

REM Launch Cosys-AirSim
echo Starting Cosys-AirSim...
for %%F in ("%AIRSIM_EXE%") do set "EXE_DIR=%%~dpF"
for %%F in ("%AIRSIM_EXE%") do set "EXE_NAME=%%~nxF"

cd /d "%EXE_DIR%"
start "" "%EXE_NAME%" -windowed -ResX=1280 -ResY=720

echo.
echo Cosys-AirSim started. Window should appear shortly.
echo (It may take 30-60 seconds to fully initialize)
echo.
