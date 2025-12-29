@echo off
setlocal enableextensions
REM AegisAV Installer Launcher for Windows
REM Double-click this file to start the setup wizard

echo ================================================
echo          AegisAV Setup Wizard Launcher
echo ================================================
echo.

REM Map UNC paths (like \\wsl.localhost\...) to a drive letter
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Unable to access %SCRIPT_DIR%
    echo.
    echo Please copy this repo to a local drive or map a drive letter.
    echo.
    pause
    exit /b 1
)

REM Check for Python 3.10+
set "PYTHON_EXE="
set "PYTHON_ARGS="

python -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
) else (
    py -3.12 -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3.12"
    ) else (
        py -3.11 -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=py"
            set "PYTHON_ARGS=-3.11"
        ) else (
            py -3.10 -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_EXE=py"
                set "PYTHON_ARGS=-3.10"
            )
        )
    )
)

if not defined PYTHON_EXE (
    call :install_python
)

if not defined PYTHON_EXE (
    echo.
    echo ERROR: Python 3.10+ is required to continue.
    echo.
    popd
    pause
    exit /b 1
)

echo Found Python:
"%PYTHON_EXE%" %PYTHON_ARGS% --version
echo.

echo Starting setup wizard...
echo.

"%PYTHON_EXE%" %PYTHON_ARGS% scripts\installer\setup_gui.py

if errorlevel 1 (
    echo.
    echo Setup wizard exited with an error.
    echo.
    popd
    pause
    exit /b 1
)

popd
exit /b 0

:install_python
echo Python 3.10+ not found. Attempting to install Python 3.12...
where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo Winget is not available. Opening the Python download page.
    start "" "https://www.python.org/downloads/windows/"
    goto :eof
)

winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo Winget install failed. Opening the Python download page.
    start "" "https://www.python.org/downloads/windows/"
    goto :eof
)

REM Try to locate Python in common install locations for this session
set "PY_HOME="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PY_HOME=%LocalAppData%\Programs\Python\Python312"
if exist "%ProgramFiles%\Python312\python.exe" set "PY_HOME=%ProgramFiles%\Python312"
if defined PY_HOME (
    set "PATH=%PY_HOME%;%PY_HOME%\Scripts;%PATH%"
    set "PYTHON_EXE=%PY_HOME%\python.exe"
    set "PYTHON_ARGS="
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if defined PYTHON_EXE (
    "%PYTHON_EXE%" -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if errorlevel 1 (
        set "PYTHON_EXE="
        set "PYTHON_ARGS="
    )
)
goto :eof
