# Clasificador de Hortalizas — Empaquetado, Despliegue e Interfaz Web (Actividades 4.2 y 4.3)

Sistema autónomo de visión por computadora para la clasificación automática de tres hortalizas (**lechuga**, **brócoli** y **col**), listo para ejecución reproducible en entorno local por CLI, interfaz gráfica interactiva en **Streamlit** con telemetría `stdout` en vivo, contenedor **Docker** y dispositivos de borde (**Edge Computing**).

---

## 1. Descripción del Sistema

El modelo productivo oficial (`models/modelo_hortalizas_act3_3.keras`) está basado en **MobileNetV2** preentrenado sobre ImageNet con **Fine-Tuning parcial (54 capas descongeladas)** y una capa interna de reescalado (`MobileNetV2_Preprocess`), entrenado en **Keras 3 / TensorFlow**:

| Atributo | Valor |
| :--- | :--- |
| **Arquitectura** | MobileNetV2 (ImageNet Backbone + Fine-Tuning Parcial 54 capas) |
| **Resolución de entrada** | `224 × 224 × 3` (RGB) |
| **Clases soportadas (`data/clases.txt`)** | `lechuga`, `brocoli`, `col` |
| **Exactitud en Test (`test_accuracy`)** | `85.00%` |
| **F1-Score Macro (`test_f1_macro`)** | `83.66%` |
| **Tamaño del artefacto `.keras`** | `25.13 MB` |
| **Regla de decisión operativa** | `AUTOMATICO` si confianza $\ge 60\%$; `REVISION_MANUAL` en caso contrario |

---

## 2. Estructura del Paquete Mínimo de Entrega

```text
clasificador_hortalizas/
├── README.md                              # Guía ejecutiva del sistema, CLI, Docker y UI Web
├── requirements.txt                       # Dependencias estrictas de inferencia CPU (Docker/CLI)
├── requirements_ui.txt                    # Dependencias extendidas para interfaz web Streamlit
├── Dockerfile                             # Imagen ligera python:3.11-slim optimizada por capas
├── .dockerignore                          # Exclusión de datos crudos, venv y artefactos de desarrollo
├── run_demo.bat                           # Arranque 1-clic CLI Batch en Windows
├── run_demo.sh                            # Arranque 1-clic CLI Batch en Linux / macOS
├── run_ui.bat                             # Arranque 1-clic Interfaz Web Streamlit en Windows
├── run_ui.sh                              # Arranque 1-clic Interfaz Web Streamlit en Linux / macOS
├── src/
│   ├── pipeline_inferencia.py             # Motor CLI de inferencia (individual, batch, stream)
│   ├── benchmark_edge.py                  # Benchmark de Cold Start, RAM RSS, latencia P95 y FPS
│   └── app_ui.py                          # Interfaz gráfica web Streamlit con stdout en vivo
├── models/
│   ├── modelo_hortalizas_act3_3.keras     # Modelo productivo oficial (Keras 3)
│   └── model_metadata.json                # Metadatos técnicos y métricas de validación/prueba
├── data/
│   ├── clases.txt                         # Etiquetas ordenadas en UTF-8 sin BOM
│   └── ejemplos/                          # 6 imágenes reales de prueba (2 por clase)
├── docs/
│   └── manual_instalacion.md              # Manual detallado de instalación, operación y soporte
└── outputs/
    └── .gitkeep                           # Directorio de persistencia de reportes JSON
```

---

## 3. Instalación Local (`venv`)

Se requiere **Python 3.10 – 3.12** (64 bits). Desde la raíz del proyecto:

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Solo motor CLI:
pip install -r requirements.txt
# Motor CLI + Interfaz Gráfica Streamlit (Actividad 4.3):
pip install -r requirements_ui.txt
```

### Linux / macOS (Bash)
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_ui.txt
chmod +x run_demo.sh run_ui.sh
```

---

## 4. Interfaz Gráfica Web en Streamlit (Actividad 4.3 — Video-Demo)

La aplicación `src/app_ui.py` ofrece una consola visual institucional (`#0F4761`) con transmisión de `stdout` en tiempo real sin alterar el pipeline CLI:

1. **Pestaña 1 — Clasificación Individual (Interactiva):** Selección desde `data/ejemplos/` o subida de imágenes propias, control deslizante de umbral en barra lateral, carga cacheada (`@st.cache_resource`), badge operativo (`AUTOMATICO` / `REVISION_MANUAL`), gráfica Softmax y panel de **Consola `stdout` en vivo** (`Ingesta → Preprocesamiento → Inferencia → Posprocesamiento → JSON`).
2. **Pestaña 2 — Procesamiento por Lote (`stdout` en Vivo):** Lanza `src/pipeline_inferencia.py` en un subproceso sin búfer (`PYTHONUNBUFFERED=1`), transmite la salida de terminal línea por línea, muestra la tabla consolidada y habilita la descarga de `ui_batch_resultado.json`.
3. **Pestaña 3 — Benchmark Edge / Rendimiento (`stdout` en Vivo):** Ejecuta `src/benchmark_edge.py` para `N` iteraciones configurables (`10–50`), mostrando el `stdout` en vivo y graficando Cold Start, RAM RSS, Latencia Media, P95 y FPS.

### Arranque 1-Clic de la Interfaz Web (`http://localhost:8501`)
- **Windows:**
  ```powershell
  .\run_ui.bat
  ```
- **Linux / macOS:**
  ```bash
  ./run_ui.sh
  ```
- **Comando directo:**
  ```bash
  streamlit run src/app_ui.py --server.port 8501
  ```

---

## 5. Ejecución 1-Clic CLI y Uso Avanzado por Terminal

### 5.1 Arranque Rápido CLI (1-Clic)
Procesa automáticamente las 6 imágenes de `data/ejemplos/` y guarda el reporte estructurado en `outputs/demo_resultado.json`:

- **Windows:** `.\run_demo.bat`
- **Linux / macOS:** `./run_demo.sh`

### 5.2 Uso Avanzado (`src/pipeline_inferencia.py`)

- **Modo individual (una sola imagen):**
  ```bash
  python src/pipeline_inferencia.py \
    --modelo models/modelo_hortalizas_act3_3.keras \
    --clases data/clases.txt \
    --modo individual \
    --imagen data/ejemplos/ejemplo_lechuga_1.jpg \
    --umbral 0.60 \
    --output outputs/individual_resultado.json
  ```

- **Modo por lotes (`batch`):**
  ```bash
  python src/pipeline_inferencia.py \
    --modelo models/modelo_hortalizas_act3_3.keras \
    --clases data/clases.txt \
    --modo batch \
    --directorio data/ejemplos \
    --output outputs/batch_resultado.json
  ```

- **Modo flujo continuo simulado (`stream`):**
  ```bash
  python src/pipeline_inferencia.py \
    --modelo models/modelo_hortalizas_act3_3.keras \
    --clases data/clases.txt \
    --modo stream \
    --directorio data/ejemplos \
    --output outputs/stream_resultado.json
  ```

### 5.3 Benchmark de Rendimiento Local (`src/benchmark_edge.py`)
```bash
python src/benchmark_edge.py \
  --modelo models/modelo_hortalizas_act3_3.keras \
  --clases data/clases.txt \
  --directorio data/ejemplos \
  --iteraciones 30 \
  --perfil "Host Local CPU" \
  --output outputs/benchmark_local.json
```

---

## 6. Construcción y Despliegue con Docker (Contenedor y Simulación Edge)

### 6.1 Construcción de la Imagen
```bash
docker build -t clasificador-hortalizas:act4.2 .
```

### 6.2 Ejecución Estándar del Contenedor
Montando la carpeta `outputs/` para persistir el archivo JSON en el host:

- **PowerShell (Windows):**
  ```powershell
  docker run --rm -v "${PWD}/outputs:/app/outputs" clasificador-hortalizas:act4.2
  ```
- **Bash (Linux / macOS):**
  ```bash
  docker run --rm -v "$(pwd)/outputs:/app/outputs" clasificador-hortalizas:act4.2
  ```

### 6.3 Simulación de Nodos Edge (`--cpus` y `--memory`)

- **Perfil Edge Estándar (1 vCPU, 1 GB RAM — ej. Raspberry Pi 4 / Gateway Industrial):**
  ```powershell
  docker run --rm --cpus="1.0" --memory="1g" -v "${PWD}/outputs:/app/outputs" clasificador-hortalizas:act4.2 python src/benchmark_edge.py --iteraciones 30 --perfil "Edge 1vCPU 1GB" --output outputs/bench_edge_1cpu.json
  ```

- **Perfil Edge Restringido (0.5 vCPU, 512 MB RAM — nodo IoT ligero):**
  ```powershell
  docker run --rm --cpus="0.5" --memory="512m" -v "${PWD}/outputs:/app/outputs" clasificador-hortalizas:act4.2 python src/benchmark_edge.py --iteraciones 30 --perfil "Edge 0.5vCPU 512MB" --output outputs/bench_edge_05cpu.json
  ```

---

## 7. Empaquetado Limpio de Entrega

Para generar `clasificador_hortalizas_entrega_act4_2.zip` únicamente con los artefactos productivos (incluyendo CLI, Docker y la interfaz Streamlit, sin `venv/`, `data/raw/`, `data/splits/` ni `modelos/`):

```powershell
powershell -ExecutionPolicy Bypass -File .\empaquetar_entrega.ps1
```

---

## 8. Solución de Problemas Comunes

| Síntoma / Error | Causa Probable | Solución |
| :--- | :--- | :--- |
| `ValueError` al cargar `.keras` | Versión antigua de TensorFlow/Keras (`< 2.16` / Keras 2) | Instalar dependencias desde `requirements.txt` o `requirements_ui.txt` (`tensorflow-cpu>=2.16.0`). |
| `streamlit: command not found` | No se instaló `requirements_ui.txt` o `venv` inactivo | Activar `venv` y ejecutar `pip install -r requirements_ui.txt`. |
| `VersionError: Detected incompatible Protobuf Gencode/Runtime` | Conflicto de `protobuf` en el Python global del sistema | Usar siempre un entorno virtual aislado (`venv`) o el contenedor Docker. |
| Nombre de primera clase con carácter extraño (`\ufefflechuga`) | Archivo `clases.txt` guardado con UTF-8 BOM | `data/clases.txt` está guardado en UTF-8 sin BOM y el pipeline lee con `encoding="utf-8-sig"`. |
| `bad interpreter: /bin/bash^M` o `Permission denied` en Linux | Saltos de línea CRLF o falta de permiso de ejecución | Ejecutar `sed -i 's/\r$//' run_demo.sh run_ui.sh && chmod +x run_demo.sh run_ui.sh`. |
| Contenedor finaliza con código `137` (`OOMKilled`) | Límite `--memory` demasiado bajo (`< 450m`) para el runtime de TensorFlow | Asignar al menos `--memory="512m"` (recomendado `1g`) al ejecutar el contenedor. |

Para más detalles operativos, consulte [docs/manual_instalacion.md](docs/manual_instalacion.md).
