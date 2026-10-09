@echo off
setlocal
cd /d "%~dp0"
title Gemma Live
if "%~1"=="" (
    start "" pythonw "%~dp0launch_gemma.py"
    exit /b
)
python launch_gemma.py %*
if errorlevel 1 pause
