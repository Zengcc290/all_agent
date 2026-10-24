@echo off
chcp 65001 >nul
REM ============================================================
REM  知识图谱入库系统 — 一键启动（双击这个文件即可）
REM ============================================================
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-all.ps1" %*
echo.
pause
