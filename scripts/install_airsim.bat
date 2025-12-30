@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: AegisAV AirSim + Overlay Plugin Setup Script
:: ============================================================================
::
:: This script:
:: 1. Checks for Unreal Engine 5 installation
:: 2. Clones AirSim repository if not present
:: 3. Builds AirSim
:: 4. Copies the AegisAVOverlay plugin to the AirSim project
:: 5. Generates project files and builds
::
:: Prerequisites:
:: - Unreal Engine 5.x installed via Epic Games Launcher
:: - Visual Studio 2022 with C++ game development workload
:: - Git installed and in PATH
:: - CMake (usually installed with VS)
::
:: ============================================================================

echo.
echo ================================================================
echo          AegisAV AirSim + Overlay Plugin Setup
echo ================================================================
echo.

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "AEGISAV_ROOT=%SCRIPT_DIR%.."

:: Default UE5 paths to check
set "UE5_PATHS=C:\Program Files\Epic Games\UE_5.5;C:\Program Files\Epic Games\UE_5.4;C:\Program Files\Epic Games\UE_5.3;C:\Program Files\Epic Games\UE_5.2;C:\Program Files\Epic Games\UE_5.1;C:\Program Files\Epic Games\UE_5.0"

:: ============================================================================
:: Step 1: Find Unreal Engine 5
:: ============================================================================
echo [1/6] Searching for Unreal Engine 5...

set "UE5_ROOT="
for %%p in (%UE5_PATHS%) do (
    if exist "%%p\Engine\Build\BatchFiles\Build.bat" (
        set "UE5_ROOT=%%p"
        goto :found_ue5
    )
)

:: Allow user to specify custom path
if "%UE5_ROOT%"=="" (
    echo.
    echo ERROR: Unreal Engine 5 not found in common locations.
    echo.
    echo Please install UE5 via Epic Games Launcher, or
    echo set the UE5_ROOT environment variable to your UE5 installation.
    echo.
    echo Example: set UE5_ROOT=D:\Epic Games\UE_5.4
    echo.
    pause
    exit /b 1
)

:found_ue5
echo Found UE5 at: %UE5_ROOT%

:: ============================================================================
:: Step 2: Check Prerequisites
:: ============================================================================
echo.
echo [2/6] Checking prerequisites...

:: Check Git
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Git not found. Please install Git and add it to PATH.
    pause
    exit /b 1
)
echo - Git: OK

:: Check Visual Studio / MSBuild
where msbuild >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: MSBuild not in PATH. Trying to find Visual Studio...

    :: Try common VS paths
    set "VS_PATHS=%ProgramFiles%\Microsoft Visual Studio\2022\Community;%ProgramFiles%\Microsoft Visual Studio\2022\Professional;%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise"

    for %%p in (%VS_PATHS%) do (
        if exist "%%p\MSBuild\Current\Bin\MSBuild.exe" (
            set "PATH=%%p\MSBuild\Current\Bin;%PATH%"
            goto :found_msbuild
        )
    )

    echo ERROR: Visual Studio 2022 not found. Please install Visual Studio 2022 with C++ workload.
    pause
    exit /b 1
)
:found_msbuild
echo - MSBuild: OK

:: ============================================================================
:: Step 3: Clone or Update AirSim
:: ============================================================================
echo.
echo [3/6] Setting up AirSim...

cd /d "%AEGISAV_ROOT%"

if exist "AirSim\.git" (
    echo AirSim repository already exists. Updating...
    cd AirSim
    git fetch origin
    git pull origin main 2>nul || git pull origin master 2>nul
) else (
    if exist "AirSim" (
        echo WARNING: AirSim folder exists but is not a git repository.
        echo Please remove or rename the existing AirSim folder.
        pause
        exit /b 1
    )

    echo Cloning AirSim repository...
    git clone https://github.com/microsoft/AirSim.git
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to clone AirSim repository.
        pause
        exit /b 1
    )
    cd AirSim
)

echo AirSim setup complete.

:: ============================================================================
:: Step 4: Build AirSim
:: ============================================================================
echo.
echo [4/6] Building AirSim (this may take several minutes)...

:: Run AirSim's build script
if exist "build.cmd" (
    call build.cmd
    if %ERRORLEVEL% neq 0 (
        echo WARNING: AirSim build had errors. Continuing anyway...
    )
) else (
    echo WARNING: build.cmd not found. Skipping AirSim native build.
)

:: ============================================================================
:: Step 5: Copy AegisAVOverlay Plugin
:: ============================================================================
echo.
echo [5/6] Installing AegisAVOverlay plugin...

set "PLUGIN_SRC=%AEGISAV_ROOT%\unreal\AegisAVOverlay"
set "PLUGIN_DST=%AEGISAV_ROOT%\AirSim\Unreal\Plugins\AegisAVOverlay"

if not exist "%PLUGIN_SRC%" (
    echo ERROR: AegisAVOverlay plugin not found at: %PLUGIN_SRC%
    echo Please ensure the plugin source exists in the unreal folder.
    pause
    exit /b 1
)

:: Remove existing plugin if present
if exist "%PLUGIN_DST%" (
    echo Removing existing plugin installation...
    rmdir /s /q "%PLUGIN_DST%"
)

:: Copy plugin
echo Copying plugin to AirSim...
xcopy /E /I /Y "%PLUGIN_SRC%" "%PLUGIN_DST%"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy plugin.
    pause
    exit /b 1
)

echo Plugin installed successfully.

:: ============================================================================
:: Step 6: Generate Project Files
:: ============================================================================
echo.
echo [6/6] Generating project files...

:: Find the Blocks environment project
set "PROJECT_PATH=%AEGISAV_ROOT%\AirSim\Unreal\Environments\Blocks\Blocks.uproject"

if not exist "%PROJECT_PATH%" (
    echo WARNING: Blocks.uproject not found. You may need to generate project files manually.
    goto :done
)

:: Generate project files using UE5's UnrealBuildTool
echo Generating Visual Studio project files...

"%UE5_ROOT%\Engine\Binaries\Win64\UnrealBuildTool.exe" -projectfiles -project="%PROJECT_PATH%" -game -engine -rocket

if %ERRORLEVEL% neq 0 (
    echo WARNING: Project file generation may have had issues.
    echo You can regenerate manually by right-clicking Blocks.uproject
    echo and selecting "Generate Visual Studio project files"
)

:: ============================================================================
:: Done
:: ============================================================================
:done
echo.
echo ================================================================
echo                     Setup Complete!
echo ================================================================
echo.
echo Next Steps:
echo.
echo 1. Open the project in Visual Studio:
echo    %AEGISAV_ROOT%\AirSim\Unreal\Environments\Blocks\Blocks.sln
echo.
echo 2. Build the project (Development Editor, Win64)
echo.
echo 3. Or open directly in Unreal Editor:
echo    %PROJECT_PATH%
echo.
echo 4. Start the AegisAV server:
echo    python -m agent.server.main
echo.
echo 5. Press Play in Unreal Editor - the overlay will auto-connect
echo.
echo To toggle the overlay: Press F1 (can be configured)
echo.
echo ================================================================
echo.

pause
