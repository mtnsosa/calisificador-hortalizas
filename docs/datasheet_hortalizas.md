# Ficha Técnica del Dataset (Datasheet)
## Clasificación de Hortalizas: Lechuga, Brócoli y Col

### 1. Identificación y Propósito
- **Asignatura:** Visión Inteligente — Maestría en Inteligencia Artificial (TecNM Virtual)
- **Actividad:** Actividad 1.4 — Avance de Proyecto 1: Anotación y Creación de Dataset
- **Autor:** Ing. Martin Ernesto Sosa Cervantes
- **Tarea de Visión Artificial:** Clasificación supervisada de imágenes (asignación de una etiqueta global por imagen completa).
- **Propósito:** Construir un conjunto de datos limpio, balanceado y estandarizado que servirá como insumo para entrenar y validar un modelo de aprendizaje profundo (Unidad 3) y desplegarlo en un pipeline operativo reproducible (Unidad 4).

### 2. Composición, Balance de Clases y Partición
El conjunto de datos consta de **300 imágenes** a color (espacio RGB, formato JPEG) distribuidas de manera perfectamente equitativa entre **3 categorías** (33.33 % por clase), evitando el problema de desbalance donde un clasificador se sesga hacia una clase mayoritaria:

| ID Categoría | Clase (`clases.txt`) | Total (`raw`) | Entrenamiento (`train` 70%) | Validación (`val` 15%) | Prueba (`test` 15%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | `lechuga` | 100 | 70 | 15 | 15 |
| `1` | `brocoli` | 100 | 70 | 15 | 15 |
| `2` | `col` | 100 | 70 | 15 | 15 |
| **Total** | **3 clases** | **300** | **210** | **45** | **45** |

- **Protocolo de partición previa:** Siguiendo las buenas prácticas de ingeniería de datos, la división en `train`, `val` y `test` se ejecutó mediante muestreo aleatorio reproducible (`seed = 42`) antes de realizar cualquier exploración visual profunda o modelado. El subconjunto `test` (45 imágenes) queda estrictamente reservado para la evaluación final del proyecto integrador con el fin de evitar fuga de información (*data leakage*).

### 3. Recopilación y Preprocesamiento Técnico
- **Fuentes de adquisición:**
  - **`brocoli` (100 imágenes) y `col` (100 imágenes):** Recopiladas desde *Open Images Dataset v7* mediante el gestor de datasets `fiftyone`, filtrando exclusivamente muestras con etiqueta positiva inequívoca para cada clase.
  - **`lechuga` (100 imágenes):** Recopiladas mediante consulta programática a la API de *Wikimedia Commons* empleando múltiples términos de búsqueda (*Lactuca sativa*, *romaine lettuce*, *iceberg lettuce*, *butterhead lettuce*, *lettuce head*) para capturar distintas variedades morfológicas.
- **Estandarización:** Todas las imágenes fueron verificadas en integridad y estandarizadas al espacio de color **RGB** de 3 canales en formato `.jpg` para garantizar compatibilidad directa con arquitecturas convolucionales.

### 4. Anotación y Formato de Exportación Estándar
- **Criterio de etiquetado:** Clasificación global de escena según la hortaliza predominante en la imagen (`lechuga`, `brocoli` o `col`).
- **Formato de exportación:**
  1. Estructura jerárquica de directorios por partición y clase (`data/splits/<split>/<clase>/`).
  2. Exportación en formato estándar **COCO** (`data/annotations/dataset_coco.json`), que centraliza en un único archivo JSON los metadatos (`info`), el catálogo de categorías (`categories`), el inventario de imágenes con dimensiones y partición asignada (`images`), y las anotaciones vinculadas (`annotations`).

### 5. Análisis Crítico: Sesgos, Limitaciones y Uso Responsable
- **Sesgo de fuente y muestreo:** Al combinar dos repositorios públicos (*Open Images v7* para brócoli/col y *Wikimedia Commons* para lechuga), existe la posibilidad de ligeras diferencias en iluminación, encuadre o contexto (cultivo frente a cocina/mercado) entre clases. Para mitigar que el modelo aprenda correlaciones espurias del fondo, durante el entrenamiento se aplicará aumento de datos (*data augmentation*: rotaciones, volteos horizontales y variaciones de brillo) junto con normalización uniforme.
- **Similitud cromática:** Las tres clases comparten tonalidades verdes predominantes; sin embargo, presentan rasgos estructurales distintivos (hojas amplias y onduladas en la lechuga, inflorescencias granulares en el brócoli y cabezas compactas de hojas superpuestas en la col) que permiten su discriminación visual.
- **Limitaciones y deriva de datos (*Data Drift*):** Con 100 muestras por clase, el dataset es adecuado para aplicar *Transfer Learning* y *Fine-Tuning*, pero no para entrenar redes profundas desde cero sin regularización. Asimismo, cambios extremos de iluminación o estados avanzados de descomposición podrían degradar la confianza en producción.
- **Consideraciones éticas y de privacidad:** El conjunto contiene exclusivamente imágenes de productos agrícolas/alimentarios de acceso público; no incluye rostros, datos personales ni información sensible.