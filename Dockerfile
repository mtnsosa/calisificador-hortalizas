# ==============================================================================
# Dockerfile — Clasificador de Hortalizas (MobileNetV2 | Actividad 4.2)
# Imagen ligera de producción optimizada por capas para despliegue Contenedor/Edge
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Capa 1: Instalación de dependencias mínimas de inferencia (aprovecha caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Capa 2: Copia estricta de código productivo, modelo oficial, etiquetas y ejemplos
COPY src/ ./src/
COPY models/modelo_hortalizas_act3_3.keras ./models/modelo_hortalizas_act3_3.keras
COPY models/model_metadata.json ./models/model_metadata.json
COPY data/clases.txt ./data/clases.txt
COPY data/ejemplos/ ./data/ejemplos/

# Directorio de salida para persistencia o montaje de volúmenes
RUN mkdir -p /app/outputs

# ------------------------------------------------------------------------------
# OPCIONAL (Actividad 4.3 — Interfaz Web Streamlit dentro del contenedor):
# Si desea ejecutar la interfaz gráfica web en Docker sin alterar la imagen base,
# descomente las siguientes líneas o instale requirements_ui.txt al construir:
#   COPY requirements_ui.txt .
#   RUN pip install --no-cache-dir -r requirements_ui.txt
#   EXPOSE 8501
#   CMD ["streamlit", "run", "src/app_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ------------------------------------------------------------------------------

CMD ["python", "src/pipeline_inferencia.py", "--modelo", "models/modelo_hortalizas_act3_3.keras", "--clases", "data/clases.txt", "--modo", "batch", "--directorio", "data/ejemplos", "--output", "outputs/docker_resultado.json"]
