@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: AegisAV Cosys-AirSim Launcher
:: ============================================================================
:: Launches Cosys-AirSim and waits for it to be ready for connections.
:: ============================================================================

echo.
echo ================================================================
echo            AegisAV Cosys-AirSim Launcher
echo ================================================================
echo.

:: Check common Cosys-AirSim and AirSim locations
set "AIRSIM_EXE="

:: Get script location
set "SCRIPT_DIR=%~dp0"
set "AEGISAV_ROOT=%SCRIPT_DIR%.."

:: Check for Cosys-AirSim Blocks in AegisAV folder (preferred)
if exist "%AEGISAV_ROOT%\Cosys-AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe" (
    set "AIRSIM_EXE=%AEGISAV_ROOT%\Cosys-AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    goto :found
)

:: Check for legacy AirSim folder name
if exist "%AEGISAV_ROOT%\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe" (
    set "AIRSIM_EXE=%AEGISAV_ROOT%\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    goto :found
)

:: Check user's desktop for pre-built binaries
if exist "%USERPROFILE%\Desktop\Cosys-AirSim\WindowsNoEditor\Blocks.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Desktop\Cosys-AirSim\WindowsNoEditor\Blocks.exe"
    goto :found
)

:: Check for AirSimNH (legacy, for backwards compatibility)
if exist "%USERPROFILE%\Desktop\AirSimNH\WindowsNoEditor\AirSimNH.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Desktop\AirSimNH\WindowsNoEditor\AirSimNH.exe"
    goto :found
)

:: Check common download locations
if exist "%USERPROFILE%\Downloads\Cosys-AirSim\WindowsNoEditor\Blocks.exe" (
    set "AIRSIM_EXE=%USERPROFILE%\Downloads\Cosys-AirSim\WindowsNoEditor\Blocks.exe"
    goto :found
)

if exist "C:\Cosys-AirSim\Blocks.exe" (
    set "AIRSIM_EXE=C:\Cosys-AirSim\Blocks.exe"
    goto :found
)

echo ERROR: Could not find Cosys-AirSim executable.
echo.
echo Please either:
echo   1. Download pre-built binaries from:
echo      https://github.com/Cosys-Lab/Cosys-AirSim/releases
echo      and extract to Desktop\Cosys-AirSim
echo.
echo   2. Or run install_airsim.bat to build from source
echo.
pause
exit /b 1

:found
echo Found Cosys-AirSim at: %AIRSIM_EXE%
echo.

:: Check if already running
tasklist /FI "IMAGENAME eq Blocks.exe" 2>NUL | find /I /N "Blocks.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Cosys-AirSim is already running!
    goto :wait_ready
)

:: Legacy check
tasklist /FI "IMAGENAME eq AirSimNH.exe" 2>NUL | find /I /N "AirSimNH.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo AirSim is already running!
    goto :wait_ready
)

:: Launch Cosys-AirSim
echo Starting Cosys-AirSim...
start "" "%AIRSIM_EXE%" -windowed -ResX=1280 -ResY=720

:: Wait for startup
echo Waiting for Cosys-AirSim to initialize (this may take 30-60 seconds)...
timeout /t 10 /nobreak >nul

:wait_ready
:: Check if Cosys-AirSim API is responding
echo Checking if Cosys-AirSim API is ready...

:: Try to connect via Python using cosysairsim
python -c "import cosysairsim as airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('Connected!')" 2>nul
if %ERRORLEVEL%==0 (
    echo.
    echo ================================================================
    echo            Cosys-AirSim is ready!
    echo ================================================================
    echo.
    echo You can now start the AegisAV server:
    echo   python -m uvicorn agent.server.main:app --port 8090
    echo.
    goto :end
)

:: Wait more and retry
echo Still waiting for Cosys-AirSim to be ready...
timeout /t 10 /nobreak >nul

python -c "import cosysairsim as airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('Connected!')" 2>nul
if %ERRORLEVEL%==0 (
    echo.
    echo Cosys-AirSim is ready!
    goto :end
)

echo.
echo WARNING: Could not confirm Cosys-AirSim connection.
echo Please wait for the Cosys-AirSim window to fully load, then start the AegisAV server.
echo.

:end
pause
