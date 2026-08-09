@echo off
title TaijiMotion Lab
chcp 65001 >nul 2>&1

echo.
echo   ============================================
echo     Tai Ji Motion Capture / TaijiMotion Lab
echo     MediaPipe Tasks Vision  ^|  Prototype v1
echo   ============================================
echo.

set "PORT=8080"

py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Found Python ^(py^)
    echo   [*] http://localhost:%PORT%
    echo.
    start "" http://localhost:%PORT%
    py -m http.server %PORT%
    goto :end
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Found Python
    echo   [*] http://localhost:%PORT%
    echo.
    start "" http://localhost:%PORT%
    python -m http.server %PORT%
    goto :end
)

npx --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Found Node.js / npx
    echo   [*] http://localhost:%PORT%
    echo.
    start "" http://localhost:%PORT%
    npx serve . -l %PORT%
    goto :end
)

echo   [FAIL] No suitable server runtime found.
echo   Please install Python 3 or Node.js, then retry.
echo.

:end
pause
