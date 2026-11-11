@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "APP_PY=.venv\Scripts\python.exe"

if not exist "%APP_PY%" (
    echo Dang tao moi truong Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :failed
)

"%APP_PY%" -c "import streamlit, lightgbm, pandas, numpy, altair, openpyxl, sklearn; assert sklearn.__version__ == '1.8.0'" >nul 2>&1
if errorlevel 1 (
    echo Dang cai thu vien. Lan dau can ket noi Internet.
    "%APP_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

echo Mo http://localhost:8502 trong Chrome hoac Edge.
"%APP_PY%" -m streamlit run app.py --server.address localhost --server.port 8502 --browser.gatherUsageStats false
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo Khong the khoi dong. Kiem tra Python 3.12 va thong bao loi phia tren.
pause
exit /b 1
