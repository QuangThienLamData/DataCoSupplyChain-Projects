@echo off
REM One-time Superset initialization.
REM Run AFTER installing apache-superset into superset-venv OR conda env "superset".
REM Usage: superset_init.bat [venv|conda]

set MODE=%1
if "%MODE%"=="" set MODE=venv

if "%MODE%"=="venv" (
    set PYTHON=superset-venv\Scripts\python.exe
    set SUPERSET=superset-venv\Scripts\superset.exe
) else (
    set PYTHON=python
    set SUPERSET=superset
)

set SUPERSET_SECRET_KEY=dataco_supply_chain_local_secret
set FLASK_APP=superset

echo [1/3] Upgrading database schema...
%SUPERSET% db upgrade

echo [2/3] Creating admin user (admin / admin)...
%SUPERSET% fab create-admin ^
    --username admin ^
    --firstname Admin ^
    --lastname User ^
    --email admin@example.com ^
    --password admin

echo [3/3] Loading default roles and permissions...
%SUPERSET% init

echo.
echo Done. Run superset_start.bat to start the server.
