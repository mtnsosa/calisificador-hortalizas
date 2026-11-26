#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo " UI WEB STREAMLIT - Clasificador de Hortalizas (Act. 4.3)"
echo "============================================================"
echo

if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

echo "[INFO] Iniciando interfaz web en http://localhost:8501 ..."
exec streamlit run src/app_ui.py --server.port 8501
