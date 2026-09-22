@echo off
chcp 65001 >nul
REM ============================================================
REM  一键停止所有服务
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-all.ps1"
echo.
pause
