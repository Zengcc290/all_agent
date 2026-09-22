@echo off
chcp 65001 >nul
REM ============================================================
REM  查看服务状态 + 连通性自检
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\status.ps1"
echo.
pause
