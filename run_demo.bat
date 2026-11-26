@echo off
echo ======================================
echo  DEMO - Clasificador de Hortalizas
echo ======================================
echo.
python src/pipeline_inferencia.py --modelo models\mi_modelo_final.h5 --imagen data\ejemplos\ejemplo_1.jpg --clases data\clases.txt
pause
