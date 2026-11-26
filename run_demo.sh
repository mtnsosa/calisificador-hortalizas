#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo " DEMO REPRODUCIBLE - Clasificador de Hortalizas (Act. 4.2)"
echo "============================================================"
echo

if [ -x "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "[INFO] Intérprete Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" src/pipeline_inferencia.py \
    --modelo models/modelo_hortalizas_act3_3.keras \
    --clases data/clases.txt \
    --modo batch \
    --directorio data/ejemplos \
    --output outputs/demo_resultado.json

echo
echo "[OK] Demo completada exitosamente. Resultados en: outputs/demo_resultado.json"
