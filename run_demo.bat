@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  DEMO REPRODUCIBLE - Clasificador de Hortalizas (Act. 4.2)
echo ============================================================
echo.

set "PYTHON_EXE=python"
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

echo [INFO] Python ejecutable: %PYTHON_EXE%
"%PYTHON_EXE%" src\pipeline_inferencia.py --modelo models\modelo_hortalizas_act3_3.keras --clases data\clases.txt --modo batch --directorio data\ejemplos --output outputs\demo_resultado.json

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] La ejecucion fallo con codigo %ERRORLEVEL%. Verifique que las dependencias de requirements.txt esten instaladas en el entorno activo.
    exit /b %ERRORLEVEL%
)

echo.
echo [OK] Demo completada exitosamente. Resultados en: outputs\demo_resultado.json
exit /b 0
