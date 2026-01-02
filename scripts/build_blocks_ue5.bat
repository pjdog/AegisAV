@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: AegisAV - Build Blocks Environment for UE5
:: ============================================================================
::
:: This script builds the Cosys-AirSim Blocks environment from source using UE5.
:: It handles:
:: 1. Finding your UE5 installation
:: 2. Copying AirSim and AegisAVOverlay plugins to the project
:: 3. Building the project for editor use or packaging standalone exe
::
:: ============================================================================

echo.
echo ================================================================
echo       AegisAV - Build Blocks Environment (UE5)
echo ================================================================
echo.

:: Get paths
set "SCRIPT_DIR=%~dp0"
set "AEGISAV_ROOT=%SCRIPT_DIR%.."
set "AEGIS_LOCAL=%LOCALAPPDATA%\AegisAV"

:: Determine AirSim location - check local install first, then repo locations
set "AIRSIM_DIR="
if exist "%AEGIS_LOCAL%\Cosys-AirSim" (
    set "AIRSIM_DIR=%AEGIS_LOCAL%\Cosys-AirSim"
) else if exist "%AEGIS_LOCAL%\AirSim" (
    set "AIRSIM_DIR=%AEGIS_LOCAL%\AirSim"
) else if exist "%AEGISAV_ROOT%\AirSim" (
    set "AIRSIM_DIR=%AEGISAV_ROOT%\AirSim"
) else if exist "%AEGISAV_ROOT%\Cosys-AirSim" (
    set "AIRSIM_DIR=%AEGISAV_ROOT%\Cosys-AirSim"
)

if "%AIRSIM_DIR%"=="" (
    echo ERROR: Could not find Cosys-AirSim installation.
    echo.
    echo Expected at: %AEGIS_LOCAL%\Cosys-AirSim
    echo          or: %AEGISAV_ROOT%\AirSim
    echo.
    echo Run INSTALL.bat first to clone and build Cosys-AirSim.
    pause
    exit /b 1
)

echo Found AirSim at: %AIRSIM_DIR%

set "BLOCKS_DIR=%AIRSIM_DIR%\Unreal\Environments\Blocks"
set "PROJECT_FILE=%BLOCKS_DIR%\Blocks.uproject"

if not exist "%PROJECT_FILE%" (
    echo ERROR: Blocks.uproject not found at: %PROJECT_FILE%
    pause
    exit /b 1
)

echo Found Blocks project at: %BLOCKS_DIR%

:: ============================================================================
:: Step 1: Find Unreal Engine 5.5
:: ============================================================================
echo.
echo [1/5] Searching for Unreal Engine 5.x...

set "UE5_ROOT="

:: Check custom/game pass paths first (newer versions first)
if exist "D:\game_pass_games\UE_5.7\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\game_pass_games\UE_5.7"
    goto :found_ue5
)
if exist "D:\game_pass_games\UE_5.6\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\game_pass_games\UE_5.6"
    goto :found_ue5
)
if exist "D:\game_pass_games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\game_pass_games\UE_5.5"
    goto :found_ue5
)

:: Check common paths (newer versions first)
if exist "C:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=C:\Program Files\Epic Games\UE_5.7"
    goto :found_ue5
)
if exist "C:\Program Files\Epic Games\UE_5.6\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=C:\Program Files\Epic Games\UE_5.6"
    goto :found_ue5
)
if exist "C:\Program Files\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=C:\Program Files\Epic Games\UE_5.5"
    goto :found_ue5
)
if exist "D:\Program Files\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Program Files\Epic Games\UE_5.7"
    goto :found_ue5
)
if exist "D:\Program Files\Epic Games\UE_5.6\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Program Files\Epic Games\UE_5.6"
    goto :found_ue5
)
if exist "D:\Program Files\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Program Files\Epic Games\UE_5.5"
    goto :found_ue5
)
if exist "D:\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Epic Games\UE_5.7"
    goto :found_ue5
)
if exist "D:\Epic Games\UE_5.6\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Epic Games\UE_5.6"
    goto :found_ue5
)
if exist "D:\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=D:\Epic Games\UE_5.5"
    goto :found_ue5
)
if exist "E:\Epic Games\UE_5.7\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=E:\Epic Games\UE_5.7"
    goto :found_ue5
)
if exist "E:\Epic Games\UE_5.6\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=E:\Epic Games\UE_5.6"
    goto :found_ue5
)
if exist "E:\Epic Games\UE_5.5\Engine\Build\BatchFiles\Build.bat" (
    set "UE5_ROOT=E:\Epic Games\UE_5.5"
    goto :found_ue5
)

:: Not found - prompt user
echo.
echo ERROR: Unreal Engine 5.x not found in common locations.
echo.
echo Please enter the full path to your UE5.5 installation:
echo (Example: D:\game_pass_games\UE_5.5)
echo.
set /p UE5_ROOT="UE5 Path: "

if not exist "%UE5_ROOT%\Engine\Build\BatchFiles\Build.bat" (
    echo ERROR: Invalid UE5 path. Build.bat not found.
    pause
    exit /b 1
)

:found_ue5
echo Found UE5 at: %UE5_ROOT%

:: ============================================================================
:: Step 2: Copy Plugins to Blocks Project
:: ============================================================================
echo.
echo [2/5] Copying plugins to Blocks project...

set "PLUGINS_SRC=%AIRSIM_DIR%\Unreal\Plugins"
set "PLUGINS_DST=%BLOCKS_DIR%\Plugins"

:: Create Plugins directory if it doesn't exist
if not exist "%PLUGINS_DST%" mkdir "%PLUGINS_DST%"

:: Copy AirSim plugin
echo Copying AirSim plugin...
if exist "%PLUGINS_SRC%\AirSim" (
    if exist "%PLUGINS_DST%\AirSim" rmdir /s /q "%PLUGINS_DST%\AirSim"
    xcopy /E /I /Y /Q "%PLUGINS_SRC%\AirSim" "%PLUGINS_DST%\AirSim" >nul
    echo   - AirSim plugin copied
) else (
    echo   WARNING: AirSim plugin not found at %PLUGINS_SRC%\AirSim
)

:: Copy AegisAVOverlay plugin
echo Copying AegisAVOverlay plugin...
if exist "%PLUGINS_SRC%\AegisAVOverlay" (
    if exist "%PLUGINS_DST%\AegisAVOverlay" rmdir /s /q "%PLUGINS_DST%\AegisAVOverlay"
    xcopy /E /I /Y /Q "%PLUGINS_SRC%\AegisAVOverlay" "%PLUGINS_DST%\AegisAVOverlay" >nul
    echo   - AegisAVOverlay plugin copied
) else (
    :: Try alternate location
    if exist "%AEGISAV_ROOT%\unreal\AegisAVOverlay" (
        if exist "%PLUGINS_DST%\AegisAVOverlay" rmdir /s /q "%PLUGINS_DST%\AegisAVOverlay"
        xcopy /E /I /Y /Q "%AEGISAV_ROOT%\unreal\AegisAVOverlay" "%PLUGINS_DST%\AegisAVOverlay" >nul
        echo   - AegisAVOverlay plugin copied from alternate location
    ) else (
        echo   WARNING: AegisAVOverlay plugin not found
    )
)

:: ============================================================================
:: Step 3: Ask Build Type
:: ============================================================================
echo.
echo [3/5] Select build type:
echo.
echo   1. Editor Build (Development) - For editing in Unreal Editor
echo   2. Package Standalone (Shipping) - Creates Blocks.exe for distribution
echo   3. Open in Unreal Editor - Just open, no build
echo.
set /p BUILD_TYPE="Enter choice (1/2/3): "

if "%BUILD_TYPE%"=="3" goto :open_editor

:: ============================================================================
:: Step 4: Setup Visual Studio Environment
:: ============================================================================
echo.
echo [4/5] Setting up build environment...

:: Find vcvarsall.bat for VS environment
set "VCVARS="

:: VS 2026 (version 18)
if exist "%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vs
)
if exist "%ProgramFiles%\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vs
)
if exist "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vs
)

:: VS 2022
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vs
)
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vs
)
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vs
)

:: VS Build Tools
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    goto :found_vs
)

echo ERROR: Visual Studio not found. Please install VS 2022 or 2026.
pause
exit /b 1

:found_vs
echo Found Visual Studio: %VCVARS%
call "%VCVARS%" x64

:: ============================================================================
:: Step 5: Build
:: ============================================================================
echo.
echo [5/5] Building Blocks project...

set "UE_BUILD=%UE5_ROOT%\Engine\Build\BatchFiles"
set "UE_BINARIES=%UE5_ROOT%\Engine\Binaries\Win64"

if "%BUILD_TYPE%"=="1" goto :editor_build
if "%BUILD_TYPE%"=="2" goto :package_build

:editor_build
echo.
echo Building for Editor (Development)...
echo This will take several minutes on first build...
echo.

:: Generate project files first
echo Generating project files...
"%UE_BINARIES%\UnrealBuildTool.exe" -projectfiles -project="%PROJECT_FILE%" -game -engine -rocket

:: Build the editor target
echo.
echo Building BlocksEditor...
"%UE_BUILD%\Build.bat" BlocksEditor Win64 Development -project="%PROJECT_FILE%" -WaitMutex

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Build failed with error code %ERRORLEVEL%
    echo.
    echo Common fixes:
    echo   1. Make sure Visual Studio C++ workload is installed
    echo   2. Delete Intermediate and Saved folders and rebuild
    echo   3. Check the build log for specific errors
    pause
    exit /b 1
)

echo.
echo ================================================================
echo                    Build Successful!
echo ================================================================
echo.
echo You can now open the project in Unreal Editor:
echo   %PROJECT_FILE%
echo.
echo Or open the Visual Studio solution:
echo   %BLOCKS_DIR%\Blocks.sln
echo.
goto :done

:package_build
echo.
echo Packaging standalone executable...
echo This will take 10-30 minutes depending on your system...
echo.

set "OUTPUT_DIR=%AEGIS_LOCAL%\Cosys-AirSim"

:: Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Run UAT to package
"%UE_BUILD%\RunUAT.bat" BuildCookRun ^
    -project="%PROJECT_FILE%" ^
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
    echo ERROR: Packaging failed with error code %ERRORLEVEL%
    pause
    exit /b 1
)

:: Rename output folder (UE5 uses 'Windows', UE4 uses 'WindowsNoEditor')
echo.
echo Checking build output...
if exist "%OUTPUT_DIR%\Windows\Blocks.exe" (
    echo Found UE5 output at Windows folder
    if exist "%OUTPUT_DIR%\Blocks" rmdir /s /q "%OUTPUT_DIR%\Blocks"
    move "%OUTPUT_DIR%\Windows" "%OUTPUT_DIR%\Blocks"
) else if exist "%OUTPUT_DIR%\WindowsNoEditor\Blocks.exe" (
    echo Found UE4-style output at WindowsNoEditor folder
    if exist "%OUTPUT_DIR%\Blocks" rmdir /s /q "%OUTPUT_DIR%\Blocks"
    move "%OUTPUT_DIR%\WindowsNoEditor" "%OUTPUT_DIR%\Blocks"
) else (
    echo WARNING: Could not find Blocks.exe in expected output locations
    echo Checking archive directory contents:
    dir "%OUTPUT_DIR%" /b
)

:: Verify final executable location
set "FINAL_EXE=%OUTPUT_DIR%\Blocks\Blocks.exe"
if exist "%FINAL_EXE%" (
    echo.
    echo ================================================================
    echo                  Packaging Successful!
    echo ================================================================
    echo.
    echo Executable created at:
    echo   %FINAL_EXE%
    echo.
    echo The AegisAV "Start AirSim" button should now work!
    echo.
) else (
    echo.
    echo WARNING: Expected executable not found at %FINAL_EXE%
    echo Check the output directory: %OUTPUT_DIR%
    echo.
    echo Contents of output directory:
    dir "%OUTPUT_DIR%" /b /s 2>nul | findstr /i "Blocks.exe"
    echo.
)

goto :done

:open_editor
echo.
echo Opening Blocks project in Unreal Editor...
echo.
echo NOTE: First launch may take several minutes while shaders compile.
echo.
start "" "%UE_BINARIES%\UnrealEditor.exe" "%PROJECT_FILE%"
goto :done

:done
echo.
pause
