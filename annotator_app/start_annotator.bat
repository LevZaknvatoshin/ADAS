@echo off
chcp 65001 >nul
title ADAS Road Annotator Pro Launcher
echo ========================================================
echo   ADAS Road ^& Lane Annotator Pro
echo   Интерактивная разметка полос и геометрии дороги
echo ========================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Python найден, запускаем локальный веб-сервер...
    python "%~dp0run_app.py"
) else (
    echo [!] Python не найден в PATH, открываем index.html в браузере напрямую...
    start "" "%~dp0index.html"
)
pause
