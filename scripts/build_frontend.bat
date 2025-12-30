@echo off
setlocal
if "%AEGIS_SKIP_FRONTEND_BUILD%"=="1" exit /b 0

set "FRONTEND_DIR=%~dp0..\frontend"
if not exist "%FRONTEND_DIR%" (
  echo Frontend folder not found, skipping build.
  exit /b 0
)

for %%N in (npm.cmd npm) do (
  where %%N >nul 2>&1 && set "NPM_BIN=%%N" && goto :found_npm
)
echo npm not found in PATH, skipping frontend build.
exit /b 0

:found_npm
pushd "%FRONTEND_DIR%" >nul 2>&1
if errorlevel 1 (
  echo Failed to enter frontend directory.
  exit /b 1
)

echo Building frontend...
if not exist "node_modules" (
  %NPM_BIN% install
  if errorlevel 1 (
    echo npm install failed.
    popd
    exit /b 1
  )
)

%NPM_BIN% run build
if errorlevel 1 (
  echo npm build failed.
  popd
  exit /b 1
)

popd
exit /b 0
