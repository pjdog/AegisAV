@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: AegisAV Cosys-AirSim + Overlay Plugin Setup Script
:: ============================================================================
::
:: This script:
:: 1. Checks for Unreal Engine 5.5 installation
:: 2. Clones Cosys-AirSim repository if not present
:: 3. Builds Cosys-AirSim
:: 4. Copies the AegisAVOverlay plugin to the Cosys-AirSim project
:: 5. Generates project files and builds
::
:: Prerequisites:
:: - Unreal Engine 5.5 installed via Epic Games Launcher
:: - Visual Studio 2022/2026 with C++ game development workload
:: - Git installed and in PATH
:: - CMake (usually installed with VS)
::
:: Why Cosys-AirSim?
:: - Actively maintained fork of Microsoft AirSim (deprecated 2022)
:: - Supports UE 5.5 (latest stable)
:: - Enhanced sensors: GPU LiDAR, Echo, Infrared
:: - https://github.com/Cosys-Lab/Cosys-AirSim
::
:: ============================================================================

echo.
echo ================================================================
echo       AegisAV Cosys-AirSim + Overlay Plugin Setup
echo ================================================================
echo.

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "AEGISAV_ROOT=%SCRIPT_DIR%.."

:: ============================================================================
:: Step 1: Find Unreal Engine 5
:: ============================================================================
echo [1/6] Searching for Unreal Engine 5.5...

set "UE5_ROOT="

:: Check common UE5 installation paths (Cosys-AirSim v3.3 supports UE 5.5)
:: Check custom/game pass locations first
if exist "D:\game_pass_games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\game_pass_games\UE_5.5"
    goto :found_ue5
)
if exist "C:\Program Files\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=C:\Program Files\Epic Games\UE_5.5"
    goto :found_ue5
)
if exist "C:\Program Files\Epic Games\UE_5.4\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=C:\Program Files\Epic Games\UE_5.4"
    goto :found_ue5
)
if exist "C:\Program Files\Epic Games\UE_5.3\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=C:\Program Files\Epic Games\UE_5.3"
    goto :found_ue5
)
if exist "C:\Program Files\Epic Games\UE_5.2\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=C:\Program Files\Epic Games\UE_5.2"
    goto :found_ue5
)
if exist "D:\Program Files\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Program Files\Epic Games\UE_5.5"
    goto :found_ue5
)
if exist "D:\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Epic Games\UE_5.5"
    goto :found_ue5
)
if exist "E:\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=E:\Epic Games\UE_5.5"
    goto :found_ue5
)

:: Allow user to specify custom path
if "%UE5_ROOT%"=="" (
    echo.
    echo Unreal Engine 5.5 not found in common locations.
    echo.
    echo Please enter the full path to your UE5.5 installation:
    echo (Example: D:\game_pass_games\UE_5.5)
    echo.
    set /p UE5_ROOT="UE5 Path: "

    if not exist "!UE5_ROOT!\Engine\Build\BatchFiles\Build.bat" (
        echo ERROR: Invalid UE5 path. Build.bat not found.
        pause
        exit /b 1
    )
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

    :: Try common VS paths (including VS 2026 folder "18")
    if exist "%ProgramFiles%\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" (
        set "PATH=%ProgramFiles%\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin;%PATH%"
        goto :found_msbuild
    )
    if exist "%ProgramFiles%\Microsoft Visual Studio\18\Professional\MSBuild\Current\Bin\MSBuild.exe" (
        set "PATH=%ProgramFiles%\Microsoft Visual Studio\18\Professional\MSBuild\Current\Bin;%PATH%"
        goto :found_msbuild
    )
    if exist "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\MSBuild\Current\Bin\MSBuild.exe" (
        set "PATH=%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\MSBuild\Current\Bin;%PATH%"
        goto :found_msbuild
    )
    if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" (
        set "PATH=%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin;%PATH%"
        goto :found_msbuild
    )
    if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe" (
        set "PATH=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin;%PATH%"
        goto :found_msbuild
    )
    if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe" (
        set "PATH=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin;%PATH%"
        goto :found_msbuild
    )

    echo ERROR: Visual Studio 2022/2026 not found.
    echo Please install Visual Studio with C++ game development workload.
    pause
    exit /b 1
)
:found_msbuild
echo - MSBuild: OK

:: ============================================================================
:: Step 3: Clone or Update Cosys-AirSim
:: ============================================================================
echo.
echo [3/6] Setting up Cosys-AirSim...

cd /d "%AEGISAV_ROOT%"

:: Check if existing "AirSim" folder is actually Cosys-AirSim
set "AIRSIM_DIR="

if exist "AirSim\.git" (
    echo Checking existing AirSim folder...
    cd AirSim
    for /f "tokens=*" %%i in ('git remote get-url origin 2^>nul') do set "GIT_REMOTE=%%i"
    cd ..

    echo !GIT_REMOTE! | findstr /i "Cosys-AirSim" >nul
    if !ERRORLEVEL! equ 0 (
        echo Found existing Cosys-AirSim installation in AirSim folder.
        set "AIRSIM_DIR=AirSim"
        cd AirSim
        echo Updating repository...
        git fetch origin
        git pull origin main 2>nul
        goto :airsim_ready
    )
)

:: Check for Cosys-AirSim folder
if exist "Cosys-AirSim\.git" (
    echo Found existing Cosys-AirSim folder. Updating...
    set "AIRSIM_DIR=Cosys-AirSim"
    cd Cosys-AirSim
    git fetch origin
    git pull origin main 2>nul
    goto :airsim_ready
)

:: No existing installation found, clone fresh
if exist "Cosys-AirSim" (
    echo WARNING: Cosys-AirSim folder exists but is not a git repository.
    echo Please remove or rename the existing folder.
    pause
    exit /b 1
)

echo Cloning Cosys-AirSim repository...
echo Repository: https://github.com/Cosys-Lab/Cosys-AirSim
set "AIRSIM_DIR=Cosys-AirSim"
git clone --branch main https://github.com/Cosys-Lab/Cosys-AirSim.git "%AIRSIM_DIR%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to clone Cosys-AirSim repository.
    pause
    exit /b 1
)
cd "%AIRSIM_DIR%"

:airsim_ready
echo Cosys-AirSim setup complete (using %AIRSIM_DIR% folder).

:: ============================================================================
:: Step 4: Build Cosys-AirSim
:: ============================================================================
echo.
echo [4/6] Building Cosys-AirSim (this may take several minutes)...

:: Run Cosys-AirSim's build script
if exist "build.cmd" (
    call build.cmd
    if %ERRORLEVEL% neq 0 (
        echo WARNING: Cosys-AirSim build had errors. Continuing anyway...
    )
) else (
    echo WARNING: build.cmd not found. Skipping native build.
)

:: ============================================================================
:: Step 5: Copy AegisAVOverlay Plugin
:: ============================================================================
echo.
echo [5/6] Installing AegisAVOverlay plugin...

set "PLUGIN_SRC=%AEGISAV_ROOT%\unreal\AegisAVOverlay"
set "PLUGIN_DST=%AEGISAV_ROOT%\%AIRSIM_DIR%\Unreal\Plugins\AegisAVOverlay"

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
echo Copying plugin to Cosys-AirSim...
xcopy /E /I /Y "%PLUGIN_SRC%" "%PLUGIN_DST%"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy plugin.
    pause
    exit /b 1
)

echo Plugin installed successfully.

:: ============================================================================
:: Step 6: Update Blocks.uproject for UE5
:: ============================================================================
echo.
echo [6/7] Configuring Blocks project for UE5...

:: Find the Blocks environment project
set "PROJECT_PATH=%AEGISAV_ROOT%\%AIRSIM_DIR%\Unreal\Environments\Blocks\Blocks.uproject"
set "BLOCKS_DIR=%AEGISAV_ROOT%\%AIRSIM_DIR%\Unreal\Environments\Blocks"

if not exist "%PROJECT_PATH%" (
    echo WARNING: Blocks.uproject not found. You may need to configure manually.
    goto :skip_uproject_update
)

:: Update EngineAssociation in Blocks.uproject to 5.5
echo Updating Blocks.uproject for UE 5.5...
powershell -Command "(Get-Content '%PROJECT_PATH%') -replace '\"EngineAssociation\": \"[^\"]*\"', '\"EngineAssociation\": \"5.5\"' | Set-Content '%PROJECT_PATH%'"

:: Add AegisAVOverlay plugin if not present
powershell -Command "$content = Get-Content '%PROJECT_PATH%' -Raw; if ($content -notmatch 'AegisAVOverlay') { $content = $content -replace '(\"Name\": \"AirSim\",\s*\"Enabled\": true\s*})', '$1,`n`t`t{`n`t`t`t\"Name\": \"AegisAVOverlay\",`n`t`t`t\"Enabled\": true`n`t`t}'; Set-Content '%PROJECT_PATH%' $content }"

echo Blocks.uproject updated for UE 5.5 with AegisAVOverlay plugin.

:skip_uproject_update

:: ============================================================================
:: Step 7: Copy Plugins to Blocks Project
:: ============================================================================
echo.
echo [7/7] Copying plugins to Blocks project...

set "PLUGINS_SRC=%AEGISAV_ROOT%\%AIRSIM_DIR%\Unreal\Plugins"
set "PLUGINS_DST=%BLOCKS_DIR%\Plugins"

:: Create Plugins directory if it doesn't exist
if not exist "%PLUGINS_DST%" mkdir "%PLUGINS_DST%"

:: Copy AirSim plugin to Blocks project
if exist "%PLUGINS_SRC%\AirSim" (
    echo Copying AirSim plugin to Blocks project...
    if exist "%PLUGINS_DST%\AirSim" rmdir /s /q "%PLUGINS_DST%\AirSim"
    xcopy /E /I /Y /Q "%PLUGINS_SRC%\AirSim" "%PLUGINS_DST%\AirSim" >nul
    echo   - AirSim plugin copied
)

:: Copy AegisAVOverlay plugin to Blocks project
if exist "%PLUGINS_SRC%\AegisAVOverlay" (
    echo Copying AegisAVOverlay plugin to Blocks project...
    if exist "%PLUGINS_DST%\AegisAVOverlay" rmdir /s /q "%PLUGINS_DST%\AegisAVOverlay"
    xcopy /E /I /Y /Q "%PLUGINS_SRC%\AegisAVOverlay" "%PLUGINS_DST%\AegisAVOverlay" >nul
    echo   - AegisAVOverlay plugin copied
)

:: Generate project files using UE5's UnrealBuildTool
echo.
echo Generating Visual Studio project files...

"%UE5_ROOT%\Engine\Binaries\Win64\UnrealBuildTool.exe" -projectfiles -project="%PROJECT_PATH%" -game -engine -rocket 2>nul

if %ERRORLEVEL% neq 0 (
    echo WARNING: Project file generation may have had issues.
    echo You can regenerate manually by right-clicking Blocks.uproject
    echo and selecting "Generate Visual Studio project files"
)

:: ============================================================================
:: Done - Ask what to do next
:: ============================================================================
:done
echo.
echo ================================================================
echo                     Setup Complete!
echo ================================================================
echo.
echo Cosys-AirSim installed at: %AEGISAV_ROOT%\%AIRSIM_DIR%
echo UE5 Path: %UE5_ROOT%
echo.
echo ================================================================
echo                     What would you like to do?
echo ================================================================
echo.
echo   1. Package standalone executable (Blocks.exe)
echo      - Creates a standalone .exe for the "Start AirSim" button
echo      - Takes 10-30 minutes on first build
echo.
echo   2. Open in Unreal Editor
echo      - For editing the scene, adding assets, etc.
echo      - First launch compiles shaders (may take several minutes)
echo.
echo   3. Exit (build later with scripts\build_blocks_ue5.bat)
echo.
set /p NEXT_ACTION="Enter choice (1/2/3): "

if "%NEXT_ACTION%"=="1" goto :package_build
if "%NEXT_ACTION%"=="2" goto :open_editor
goto :exit_script

:package_build
echo.
echo ================================================================
echo         Packaging Blocks.exe (this will take a while)
echo ================================================================
echo.

:: Setup VS environment
call :setup_vs_env
if %ERRORLEVEL% neq 0 goto :exit_script

set "UE_BUILD=%UE5_ROOT%\Engine\Build\BatchFiles"
set "OUTPUT_DIR=%LOCALAPPDATA%\AegisAV\AirSim"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Starting package build...
echo Output will be at: %OUTPUT_DIR%\Blocks\Blocks.exe
echo.

"%UE_BUILD%\RunUAT.bat" BuildCookRun ^
    -project="%PROJECT_PATH%" ^
    -noP4 ^
    -platform=Win64 ^
    -clientconfig=Development ^
    -cook ^
    -build ^
    -stage ^
    -pak ^
    -archive ^
    -archivedirectory="%OUTPUT_DIR%" ^
    -utf8output ^
    -compressed ^
    -prereqs

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Packaging failed. Check the output above for errors.
    pause
    goto :exit_script
)

:: Rename output folder (UE5 uses 'Windows', UE4 uses 'WindowsNoEditor')
echo.
echo Checking build output...
if exist "%OUTPUT_DIR%\Windows\Blocks.exe" (
    echo Found UE5 output at Windows folder
    if exist "%OUTPUT_DIR%\Blocks" rmdir /s /q "%OUTPUT_DIR%\Blocks"
    move "%OUTPUT_DIR%\Windows" "%OUTPUT_DIR%\Blocks" >nul
) else if exist "%OUTPUT_DIR%\WindowsNoEditor\Blocks.exe" (
    echo Found UE4-style output at WindowsNoEditor folder
    if exist "%OUTPUT_DIR%\Blocks" rmdir /s /q "%OUTPUT_DIR%\Blocks"
    move "%OUTPUT_DIR%\WindowsNoEditor" "%OUTPUT_DIR%\Blocks" >nul
) else (
    echo WARNING: Could not find Blocks.exe in expected output locations
    echo Checking archive directory contents:
    dir "%OUTPUT_DIR%" /b
)

echo.
echo ================================================================
echo                  Packaging Successful!
echo ================================================================
echo.
echo Executable created at:
echo   %OUTPUT_DIR%\Blocks\Blocks.exe
echo.
echo The AegisAV "Start AirSim" button should now work!
echo.
pause
goto :exit_script

:open_editor
echo.
echo Opening Blocks project in Unreal Editor...
echo NOTE: First launch may take several minutes while shaders compile.
echo.
start "" "%UE5_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe" "%PROJECT_PATH%"
goto :exit_script

:setup_vs_env
:: Find and setup Visual Studio environment
set "VCVARS="

if exist "%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vcvars
)
if exist "%ProgramFiles%\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vcvars
)
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vcvars
)
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vcvars
)
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vcvars
)

echo ERROR: Visual Studio not found for packaging.
exit /b 1

:found_vcvars
call "%VCVARS%" x64
exit /b 0

:exit_script
echo.
echo Resources:
echo - Cosys-AirSim Docs: https://cosys-lab.github.io/Cosys-AirSim/
echo - Build script: scripts\build_blocks_ue5.bat
echo.
pause
