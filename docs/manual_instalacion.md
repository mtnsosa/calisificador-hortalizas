# Manual de Instalación, Operación y Despliegue — Clasificador de Hortalizas

**Proyecto Integrador:** Clasificación Automática de Hortalizas (Lechuga, Brócoli y Col)  
**Asignatura:** Visión Inteligente — Unidad 4 (Actividad 4.2: Empaquetado y Despliegue)  
**Programa:** Maestría en Inteligencia Artificial (TecNM Virtual)

---

## 1. Descripción General de la Solución

El sistema provee un motor autónomo de inferencia en visión artificial diseñado para clasificar imágenes RGB de hortalizas en tres categorías comerciales:
1. `lechuga`
2. `brocoli`
3. `col`

### Especificaciones del Modelo Productivo
- **Archivo oficial:** `models/modelo_hortalizas_act3_3.keras` (`25.13 MB`)
- **Arquitectura:** `MobileNetV2` (backbone ImageNet) + Fine-Tuning parcial sobre las últimas 54 capas + cabezal denso regularizado (`Dense 128` + `BatchNormalization` + `ReLU` + `Dropout` + `Softmax`).
- **Preprocesamiento integrado:** El grafo incorpora la capa `Rescaling` (`MobileNetV2_Preprocess`), por lo que recibe tensores RGB `[0, 255]` de tamaño `224 × 224 × 3`. El pipeline detecta automáticamente esta capa para evitar errores de doble normalización.
- **Desempeño validado en conjunto de prueba (`models/model_metadata.json`):**
  - Exactitud (`test_accuracy`): **85.00%**
  - Precisión Macro (`test_precision_macro`): **84.03%**
  - Recall Macro (`test_recall_macro`): **84.44%**
  - F1-Score Macro (`test_f1_macro`): **83.66%**
- **Criterio operativo:** Si la probabilidad máxima es mayor o igual al umbral configurable (`--umbral`, por defecto `0.60`), el dictamen es `AUTOMATICO`; en caso contrario se marca como `REVISION_MANUAL`.

---

## 2. Requisitos Previos del Entorno

### Despliegue Local (Bare-Metal / Workstation)
- **Sistema Operativo:** Windows 10/11 (64 bits), Ubuntu 20.04+/Debian 12+, o macOS 13+.
- **Python:** Versión **3.10, 3.11 o 3.12** (64 bits). Se recomienda Python 3.11 para paridad exacta con la imagen Docker.
- **Memoria RAM mínima:** 1 GB libre (2 GB recomendados).
- **Espacio en disco:** ~650 MB libres para el entorno virtual con `tensorflow-cpu` y el modelo.

### Despliegue en Contenedor / Simulación Edge
- **Docker Engine** 24.0+ o **Docker Desktop** 4.25+ con soporte para contenedores Linux (`linux/amd64` o `linux/arm64`).

---

## 3. Estructura del Paquete Mínimo de Entrega

El paquete limpio de producción (`clasificador_hortalizas_entrega_act4_2.zip`) contiene exclusivamente los componentes indispensables para inferencia y auditoría de rendimiento:

```text
clasificador_hortalizas/
├── README.md                              # Resumen general y comandos rápidos
├── requirements.txt                       # Dependencias mínimas de inferencia (CPU)
├── Dockerfile                             # Definición de imagen ligera python:3.11-slim
├── .dockerignore                          # Reglas de exclusión de contexto Docker
├── run_demo.bat                           # Ejecución reproducible 1-clic (Windows)
├── run_demo.sh                            # Ejecución reproducible 1-clic (Linux/macOS)
├── src/
│   ├── pipeline_inferencia.py             # Pipeline CLI (modos individual, batch y stream)
│   └── benchmark_edge.py                  # Medición de Cold Start, RAM RSS, latencia y FPS
├── models/
│   ├── modelo_hortalizas_act3_3.keras     # Modelo entrenado en formato nativo Keras 3
│   └── model_metadata.json                # Trazabilidad de arquitectura y métricas
├── data/
│   ├── clases.txt                         # Catálogo de clases en UTF-8 sin BOM
│   └── ejemplos/                          # 6 imágenes reales de prueba (2 por clase)
│       ├── ejemplo_lechuga_1.jpg
│       ├── ejemplo_lechuga_2.jpg
│       ├── ejemplo_brocoli_1.jpg
│       ├── ejemplo_brocoli_2.jpg
│       ├── ejemplo_col_1.jpg
│       └── ejemplo_col_2.jpg
├── docs/
│   └── manual_instalacion.md              # Este manual técnico
└── outputs/
    └── .gitkeep                           # Carpeta destino para reportes JSON
```

---

## 4. Instalación Paso a Paso en Entorno Local

### 4.1 En Windows (PowerShell)
1. Abrir PowerShell y ubicarse en la carpeta raíz del proyecto:
   ```powershell
   cd ruta\a\clasificador_hortalizas
   ```
2. Crear y activar un entorno virtual limpio:
   ```powershell
   python -m venv venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
   ```
3. Actualizar `pip` e instalar únicamente las dependencias de producción:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 4.2 En Linux / macOS (Bash / Zsh)
1. Ubicarse en la carpeta raíz del paquete descomprimido:
   ```bash
   cd /ruta/a/clasificador_hortalizas
   ```
2. Crear y activar el entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Instalar dependencias y otorgar permisos al script de arranque:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   chmod +x run_demo.sh
   ```

---

## 5. Ejecución 1-Clic y Uso Avanzado por Línea de Comandos

### 5.1 Validación Inmediata (1-Clic)
Ejecuta el clasificador sobre las 6 imágenes de `data/ejemplos/` y genera `outputs/demo_resultado.json`:

- **Windows:** `.\run_demo.bat`
- **Linux / macOS:** `./run_demo.sh`

### 5.2 Parámetros de `src/pipeline_inferencia.py`

| Argumento | Default | Descripción |
| :--- | :--- | :--- |
| `--modelo` | `models/modelo_hortalizas_act3_3.keras` | Ruta al archivo `.keras` del modelo entrenado |
| `--clases` | `data/clases.txt` | Archivo de etiquetas (una por línea) |
| `--modo` | `individual` | Modalidad: `individual`, `batch` o `stream` |
| `--imagen` | `None` | Ruta a una imagen puntual (obligatorio en `individual`) |
| `--directorio` | `None` | Carpeta con imágenes `.jpg/.jpeg/.png` (`batch` o `stream`) |
| `--umbral` | `0.60` | Umbral mínimo `[0-1]` para decisión `AUTOMATICO` |
| `--output` | `outputs/cli_resultado.json` | Ruta del archivo JSON de salida |

#### Ejemplos de invocación
```bash
# Inferencia sobre una sola fotografía
python src/pipeline_inferencia.py --modo individual --imagen data/ejemplos/ejemplo_brocoli_1.jpg --output outputs/unica.json

# Inferencia por lote sobre una carpeta completa
python src/pipeline_inferencia.py --modo batch --directorio data/ejemplos --output outputs/lote.json

# Inferencia simulando telemetría cuadro por cuadro en tiempo real
python src/pipeline_inferencia.py --modo stream --directorio data/ejemplos --output outputs/stream.json
```

### 5.3 Benchmark Operativo Edge (`src/benchmark_edge.py`)
Permite cuantificar tiempos de arranque en frío (**Cold Start**), pico de **RAM RSS**, **latencia media**, **jitter (desviación estándar)**, **P95** y **FPS sostenidos**:

```bash
python src/benchmark_edge.py \
  --modelo models/modelo_hortalizas_act3_3.keras \
  --clases data/clases.txt \
  --directorio data/ejemplos \
  --iteraciones 30 \
  --perfil "Workstation Local" \
  --output outputs/benchmark_local.json
```

---

## 6. Contenedorización y Despliegue con Docker

### 6.1 Construcción de la Imagen Optimizada
El `Dockerfile` utiliza `python:3.11-slim` y separa la instalación de `requirements.txt` de la copia de código y pesos para maximizar el aprovechamiento de caché por capas. Además, `.dockerignore` impide transferir datasets de entrenamiento o entornos locales al demonio de Docker.

```bash
docker build -t clasificador-hortalizas:act4.2 .
```

### 6.2 Inferencia por Lotes en Contenedor (Comando por Defecto)
- **Windows PowerShell:**
  ```powershell
  docker run --rm -v "${PWD}/outputs:/app/outputs" clasificador-hortalizas:act4.2
  ```
- **Linux / macOS Bash:**
  ```bash
  docker run --rm -v "$(pwd)/outputs:/app/outputs" clasificador-hortalizas:act4.2
  ```
Al finalizar, el reporte quedará disponible en el host en `outputs/docker_resultado.json`.

### 6.3 Pruebas de Estrés y Simulación de Dispositivos Edge
Utilizando las banderas `--cpus` y `--memory` de Docker se emulan perfiles reales de hardware embebido:

1. **Gateway Edge Medio (1.0 vCPU / 1 GB RAM):**
   ```powershell
   docker run --rm --cpus="1.0" --memory="1g" `
     -v "${PWD}/outputs:/app/outputs" `
     clasificador-hortalizas:act4.2 `
     python src/benchmark_edge.py --iteraciones 30 --perfil "Edge 1vCPU 1GB" --output outputs/benchmark_1vcpu.json
   ```

2. **Nodo Edge Restringido (0.5 vCPU / 512 MB RAM):**
   ```powershell
   docker run --rm --cpus="0.5" --memory="512m" `
     -v "${PWD}/outputs:/app/outputs" `
     clasificador-hortalizas:act4.2 `
     python src/benchmark_edge.py --iteraciones 30 --perfil "Edge 0.5vCPU 512MB" --output outputs/benchmark_05vcpu.json
   ```

3. **Servidor / Contenedor sin restricción de CPU (2.0 vCPU / 2 GB RAM):**
   ```powershell
   docker run --rm --cpus="2.0" --memory="2g" `
     -v "${PWD}/outputs:/app/outputs" `
     clasificador-hortalizas:act4.2 `
     python src/benchmark_edge.py --iteraciones 30 --perfil "Container 2vCPU 2GB" --output outputs/benchmark_2vcpu.json
   ```

---

## 7. Solución de Problemas Comunes (Troubleshooting)

1. **Error al deserializar el archivo `.keras` (`ValueError: Unknown layer` o `File format not supported`)**
   - *Causa:* Se está ejecutando una versión antigua de TensorFlow (`<= 2.15`) con Keras 2.
   - *Solución:* Verificar la versión activa con `python -c "import keras; print(keras.__version__)"` y actualizar usando `pip install --upgrade -r requirements.txt` (`tensorflow-cpu>=2.16.0` con Keras 3).

2. **Error `google.protobuf.runtime_version.VersionError` al importar TensorFlow**
   - *Causa:* La instalación global de Python tiene una versión de `protobuf` incompatible con la compilación de TensorFlow.
   - *Solución:* No utilizar el Python global del sistema; activar el entorno virtual dedicado (`venv`) o ejecutar mediante Docker.

3. **El contenedor se detiene abruptamente con `Exit Code 137` (`Killed`)**
   - *Causa:* El kernel de Linux terminó el proceso por exceder el límite de RAM (`--memory`) durante la carga de `tensorflow-cpu` y el grafo MobileNetV2 (~350–450 MB RSS).
   - *Solución:* Incrementar el límite de memoria del contenedor a un mínimo de `--memory="512m"` o `--memory="1g"`.

4. **No aparecen archivos nuevos en `outputs/` tras correr `docker run`**
   - *Causa:* No se montó el volumen `-v` o la ruta relativa tiene espacios sin comillas.
   - *Solución:* Envolver siempre el mapeo de volumen entre comillas: `-v "${PWD}/outputs:/app/outputs"` en PowerShell o `-v "$(pwd)/outputs:/app/outputs"` en Bash.

5. **Error `Permission denied` o `^M` al ejecutar `./run_demo.sh` en Linux/WSL**
   - *Causa:* El archivo perdió el bit de ejecución o fue editado en Windows con saltos CRLF.
   - *Solución:* Ejecutar `sed -i 's/\r$//' run_demo.sh && chmod +x run_demo.sh`.
