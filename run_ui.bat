@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  UI WEB STREAMLIT - Clasificador de Hortalizas (Act. 4.3)
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo [INFO] Iniciando interfaz web en http://localhost:8501 ...
streamlit run src/app_ui.py --server.port 8501

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] No se pudo iniciar Streamlit. Verifique haber instalado: pip install -r requirements_ui.txt
    exit /b %ERRORLEVEL%
)
exit /b 0
