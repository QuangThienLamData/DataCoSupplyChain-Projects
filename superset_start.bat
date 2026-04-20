@echo off
REM Starts the Superset development server on http://localhost:8088
REM Usage: superset_start.bat [venv|conda]

set MODE=%1
if "%MODE%"=="" set MODE=venv

if "%MODE%"=="venv" (
    set SUPERSET=superset-venv\Scripts\superset.exe
) else (
    set SUPERSET=superset
)

set SUPERSET_SECRET_KEY=dataco_supply_chain_local_secret
set FLASK_APP=superset

echo Starting Superset on http://localhost:8088  (Ctrl+C to stop)
%SUPERSET% run -p 8088 --with-threads
