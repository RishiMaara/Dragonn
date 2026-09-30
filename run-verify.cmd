@echo off
REM Dragonn - double-click, or run from cmd: run-verify.cmd
REM Installs anything missing (Python included), measures, writes a report to your Desktop.
powershell -ExecutionPolicy Bypass -File "%~dp0verify-on-snapdragon.ps1" %*
echo.
pause
