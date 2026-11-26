#!/usr/bin/env python3
"""
Interfaz Gráfica Web Interactiva (Streamlit) — Clasificador de Hortalizas
Asignatura: Visión Inteligente | Unidad 4 (Actividad 4.3: Proyecto Integrador y Video-Demo)
Programa: Maestría en Inteligencia Artificial — TecNM Virtual

Características principales:
- Paleta institucional Navy (#0F4761) limpia y responsiva.
- Pestaña 1: Clasificación individual interactiva con @st.cache_resource y consola stdout en vivo.
- Pestaña 2: Ejecución CLI por lotes en subproceso real con streaming stdout línea por línea.
- Pestaña 3: Benchmark Edge/Contenedor en vivo con gráficas comparativas de latencia y recursos.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Rutas base del proyecto (independientes del directorio desde donde se invoque streamlit)
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_MODELO = ROOT_DIR / "models" / "modelo_hortalizas_act3_3.keras"
DEFAULT_METADATA = ROOT_DIR / "models" / "model_metadata.json"
DEFAULT_CLASES = ROOT_DIR / "data" / "clases.txt"
DEFAULT_EJEMPLOS = ROOT_DIR / "data" / "ejemplos"
OUTPUTS_DIR = ROOT_DIR / "outputs"

NAVY_PRIMARY = "#0F4761"
NAVY_SECONDARY = "#1A5F7A"
NAVY_LIGHT = "#E8F1F5"


# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS INSTITUCIONALES (#0F4761)
# ==============================================================================
st.set_page_config(
    page_title="Clasificador de Hortalizas | Visión Inteligente",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
        /* Banner principal institucional */
        .hero-banner {{
            background: linear-gradient(135deg, {NAVY_PRIMARY} 0%, {NAVY_SECONDARY} 100%);
            color: #FFFFFF;
            padding: 1.35rem 1.75rem;
            border-radius: 12px;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 14px rgba(15, 71, 97, 0.18);
        }}
        .hero-banner h1 {{
            color: #FFFFFF !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            margin: 0 0 0.35rem 0 !important;
            padding: 0 !important;
        }}
        .hero-banner p {{
            color: #DCEAF2 !important;
            font-size: 0.96rem !important;
            margin: 0 !important;
        }}

        /* Tarjetas de resultado y KPI */
        .pred-card {{
            background-color: #FFFFFF;
            border: 1px solid #D5E2EA;
            border-left: 6px solid {NAVY_PRIMARY};
            border-radius: 10px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(15, 71, 97, 0.06);
        }}
        .pred-label {{
            color: #526675;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        .pred-value {{
            color: {NAVY_PRIMARY};
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0.1rem 0 0.6rem 0;
            text-transform: capitalize;
        }}

        /* Badges de decisión operativa */
        .badge-auto {{
            display: inline-block;
            background-color: #DCFCE7;
            color: #166534;
            border: 1px solid #22C55E;
            font-weight: 700;
            font-size: 0.84rem;
            padding: 0.28rem 0.78rem;
            border-radius: 999px;
            letter-spacing: 0.03em;
        }}
        .badge-manual {{
            display: inline-block;
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #F59E0B;
            font-weight: 700;
            font-size: 0.84rem;
            padding: 0.28rem 0.78rem;
            border-radius: 999px;
            letter-spacing: 0.03em;
        }}

        /* Encabezado de consola stdout */
        .stdout-header {{
            background-color: {NAVY_PRIMARY};
            color: #FFFFFF;
            font-family: monospace;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.45rem 0.9rem;
            border-radius: 8px 8px 0 0;
            margin-top: 0.5rem;
        }}

        /* Pestañas estilizadas con Navy #0F4761 */
        button[data-baseweb="tab"] {{
            font-weight: 600 !important;
            font-size: 0.96rem !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {NAVY_PRIMARY} !important;
            border-bottom-color: {NAVY_PRIMARY} !important;
        }}

        /* Botones primarios en Navy #0F4761 */
        div.stButton > button[kind="primary"] {{
            background-color: {NAVY_PRIMARY} !important;
            border-color: {NAVY_PRIMARY} !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: {NAVY_SECONDARY} !important;
            border-color: {NAVY_SECONDARY} !important;
        }}

        /* Tarjetas informativas en barra lateral */
        .sidebar-box {{
            background-color: {NAVY_LIGHT};
            border-left: 4px solid {NAVY_PRIMARY};
            padding: 0.75rem 0.9rem;
            border-radius: 6px;
            font-size: 0.86rem;
            color: #1E293B;
            margin-bottom: 0.8rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# FUNCIONES CACHEADAS Y AUXILIARES DE EJECUCIÓN / STDOUT
# ==============================================================================
@st.cache_data
def cargar_metadatos_modelo(ruta_metadata_str: str) -> dict:
    ruta = Path(ruta_metadata_str)
    if ruta.exists():
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@st.cache_resource(show_spinner="Cargando modelo MobileNetV2 y ejecutando warm-up inicial...")
def obtener_pipeline_cacheado(ruta_modelo_str: str, ruta_clases_str: str):
    from pipeline_inferencia import PipelineInferenciaHortalizas

    pipe = PipelineInferenciaHortalizas(
        ruta_modelo=ruta_modelo_str,
        ruta_clases=ruta_clases_str,
        umbral_confianza=0.60,
    )
    pipe.cargar_modelo()
    return pipe


def ts_actual() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def emitir_traza_stdout(mensaje: str, buffer_sesion: list) -> None:
    """Registra una traza con timestamp tanto en el stdout real del proceso como en el panel UI."""
    linea = f"[{ts_actual()}] {mensaje}"
    print(linea, flush=True)
    buffer_sesion.append(linea)


def clasificar_con_trazas(pipe, ruta_imagen: Path, nombre_visible: str, umbral: float, logs: list) -> dict:
    """
    Ejecuta las etapas reales del pipeline (Ingesta -> Preprocesamiento -> Inferencia -> Posprocesamiento -> JSON)
    emitiendo telemetría detallada al stdout del proceso y a la consola en vivo de Streamlit.
    """
    import tensorflow as tf

    pipe.umbral_confianza = float(umbral)
    emitir_traza_stdout("=" * 72, logs)
    emitir_traza_stdout(f"[INGESTA] Recibiendo archivo '{nombre_visible}' ({ruta_imagen.stat().st_size / 1024:.1f} KB)", logs)

    with Image.open(ruta_imagen) as im_info:
        ancho_orig, alto_orig = im_info.size
        modo_orig = im_info.mode
    emitir_traza_stdout(
        f"[INGESTA] Resolución original={ancho_orig}x{alto_orig}px | modo={modo_orig} -> validación exitosa.",
        logs,
    )

    t0_total = time.perf_counter()
    t0_prep = time.perf_counter()
    tensor = pipe.preprocesar(ruta_imagen)
    ms_prep = (time.perf_counter() - t0_prep) * 1000.0
    emitir_traza_stdout(
        f"[PREPROCESAMIENTO] Tensor shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"| rango=[{float(np.min(tensor)):.2f}, {float(np.max(tensor)):.2f}] "
        f"| Rescaling interno={pipe.tiene_normalizacion_interna} ({ms_prep:.2f} ms)",
        logs,
    )

    t0_inf = time.perf_counter()
    probs = pipe.modelo(tensor, training=False).numpy()[0]
    ms_inf = (time.perf_counter() - t0_inf) * 1000.0
    lat_total_ms = (time.perf_counter() - t0_total) * 1000.0
    fps = 1000.0 / max(lat_total_ms, 1e-6)
    emitir_traza_stdout(
        f"[INFERENCIA] Forward-pass MobileNetV2 completado en {ms_inf:.2f} ms "
        f"(latencia end-to-end={lat_total_ms:.2f} ms | {fps:.2f} FPS)",
        logs,
    )

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    decision = "AUTOMATICO" if conf >= umbral else "REVISION_MANUAL"
    probs_dict = {pipe.clases[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(pipe.clases))}
    emitir_traza_stdout(
        f"[POSPROCESAMIENTO] Softmax={probs_dict} | argmax='{pipe.clases[idx]}' "
        f"({conf * 100.0:.2f}%) vs umbral={umbral * 100.0:.0f}% -> {decision}",
        logs,
    )

    resultado = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "archivo": nombre_visible,
        "clase_predicha": pipe.clases[idx],
        "confianza_pct": round(conf * 100.0, 2),
        "umbral_aplicado_pct": round(umbral * 100.0, 2),
        "decision_operativa": decision,
        "probabilidades": probs_dict,
        "latencia_ms": round(lat_total_ms, 3),
        "fps": round(fps, 2),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_out = OUTPUTS_DIR / "ui_individual_resultado.json"
    ruta_out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    emitir_traza_stdout(
        f"[JSON] Payload serializado y persistido en outputs/ui_individual_resultado.json",
        logs,
    )
    return resultado


def ejecutar_cli_en_vivo(comando: list, contenedor_terminal) -> tuple[int, str]:
    """
    Ejecuta un comando CLI Python con PYTHONUNBUFFERED=1 y transmite stdout/stderr
    línea por línea en tiempo real tanto a Streamlit como al stdout del servidor.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    cmd_legible = " ".join(
        f'"{arg}"' if " " in str(arg) else str(arg)
        for arg in (["python"] + comando[1:])
    )
    encabezado = f"$ {cmd_legible}\n" + ("-" * 78) + "\n"
    lineas_acumuladas = [encabezado]
    contenedor_terminal.code("".join(lineas_acumuladas), language="bash")
    print(f"\n[UI-SUBPROCESS] Ejecutando: {cmd_legible}", flush=True)

    proc = subprocess.Popen(
        comando,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    assert proc.stdout is not None
    for linea in iter(proc.stdout.readline, ""):
        sys.stdout.write(linea)
        sys.stdout.flush()
        lineas_acumuladas.append(linea)
        contenedor_terminal.code("".join(lineas_acumuladas), language="bash")

    proc.stdout.close()
    returncode = proc.wait()
    pie = f"\n[PROCESO FINALIZADO] Código de salida: {returncode}\n"
    lineas_acumuladas.append(pie)
    contenedor_terminal.code("".join(lineas_acumuladas), language="bash")
    return returncode, "".join(lineas_acumuladas)


# ==============================================================================
# ESTADO INICIAL DE SESIÓN
# ==============================================================================
if "tab1_logs" not in st.session_state:
    st.session_state["tab1_logs"] = []
if "tab1_ultimo_resultado" not in st.session_state:
    st.session_state["tab1_ultimo_resultado"] = None
if "tab2_stdout" not in st.session_state:
    st.session_state["tab2_stdout"] = ""
if "tab2_data" not in st.session_state:
    st.session_state["tab2_data"] = None
if "tab3_stdout" not in st.session_state:
    st.session_state["tab3_stdout"] = ""
if "tab3_data" not in st.session_state:
    st.session_state["tab3_data"] = None


# ==============================================================================
# BARRA LATERAL (CONFIGURACIÓN Y FICHA TÉCNICA DEL MODELO)
# ==============================================================================
metadata = cargar_metadatos_modelo(str(DEFAULT_METADATA))

with st.sidebar:
    st.markdown(f"### ⚙️ Control Operativo")
    umbral_confianza = st.slider(
        "Umbral de confianza automática",
        min_value=0.05,
        max_value=0.99,
        value=0.60,
        step=0.05,
        help="Si la probabilidad máxima supera o iguala este valor, el dictamen es AUTOMATICO; de lo contrario pasa a REVISION_MANUAL.",
    )

    st.markdown(
        f"""
        <div class="sidebar-box">
            <strong>Regla activa:</strong><br>
            • Confianza &ge; <b>{umbral_confianza * 100:.0f}%</b> &rarr; <code>AUTOMATICO</code><br>
            • Confianza &lt; <b>{umbral_confianza * 100:.0f}%</b> &rarr; <code>REVISION_MANUAL</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 🧠 Ficha del Modelo Productivo")
    st.caption(f"**Artefacto:** `{DEFAULT_MODELO.relative_to(ROOT_DIR)}`")
    if metadata:
        st.markdown(
            f"""
            <div class="sidebar-box">
                <b>Arquitectura:</b> {metadata.get("arquitectura", "MobileNetV2")}<br>
                <b>Entrada:</b> <code>{tuple(metadata.get("input_shape", [224, 224, 3]))}</code><br>
                <b>Clases:</b> <code>{", ".join(metadata.get("clases", ["lechuga", "brocoli", "col"]))}</code><br>
                <b>Exactitud Test:</b> {metadata.get("test_accuracy", 0.85) * 100:.2f}%<br>
                <b>F1-Score Macro:</b> {metadata.get("test_f1_macro", 0.8366) * 100:.2f}%<br>
                <b>Tamaño:</b> {metadata.get("tamano_mb", 25.13)} MB
            </div>
            """,
            unsafe_allow_html=True,
        )

    estado_modelo = "✅ Disponible" if DEFAULT_MODELO.exists() else "❌ No encontrado"
    estado_clases = "✅ Disponible" if DEFAULT_CLASES.exists() else "❌ No encontrado"
    st.caption(f"Estado modelo: {estado_modelo} | Clases: {estado_clases}")


# ==============================================================================
# ENCABEZADO PRINCIPAL Y PESTAÑAS
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <h1>🥬 Sistema Inteligente de Clasificación de Hortalizas</h1>
        <p>Actividad 4.3 — Proyecto Integrador y Video-Demo | Motor MobileNetV2 (Lechuga, Brócoli y Col) con Telemetría <code>stdout</code> en Vivo</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_individual, tab_batch, tab_benchmark = st.tabs(
    [
        "🔍 1. Clasificación Individual (Interactiva)",
        "📦 2. Procesamiento por Lote (CLI stdout en Vivo)",
        "⚡ 3. Benchmark Edge / Rendimiento (stdout en Vivo)",
    ]
)


# ==============================================================================
# PESTAÑA 1: CLASIFICACIÓN INDIVIDUAL INTERACTIVA
# ==============================================================================
with tab_individual:
    st.markdown("#### Inspección unitaria con trazabilidad de etapas en tiempo real")

    fuente_modo = st.radio(
        "Seleccione el origen de la imagen:",
        options=["Elegir muestra de data/ejemplos/", "Subir imagen desde mi equipo"],
        horizontal=True,
    )

    ruta_img_seleccionada = None
    nombre_img_visible = None
    archivo_subido_bytes = None

    if fuente_modo == "Elegir muestra de data/ejemplos/":
        ejemplos_disponibles = (
            sorted(
                [
                    p
                    for p in DEFAULT_EJEMPLOS.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            if DEFAULT_EJEMPLOS.exists()
            else []
        )
        if ejemplos_disponibles:
            nombres_ejemplos = [p.name for p in ejemplos_disponibles]
            seleccion_nombre = st.selectbox(
                "Imagen de referencia (`data/ejemplos/`):",
                options=nombres_ejemplos,
                index=0,
            )
            ruta_img_seleccionada = DEFAULT_EJEMPLOS / seleccion_nombre
            nombre_img_visible = seleccion_nombre
        else:
            st.warning("No se encontraron imágenes en `data/ejemplos/`.")
    else:
        archivo_up = st.file_uploader(
            "Cargue una fotografía de hortaliza (.jpg, .jpeg, .png):",
            type=["jpg", "jpeg", "png"],
        )
        if archivo_up is not None:
            archivo_subido_bytes = archivo_up.getvalue()
            nombre_img_visible = archivo_up.name

    col_btn_run, col_btn_clear, _ = st.columns([1.4, 1.2, 4.0])
    with col_btn_run:
        ejecutar_ind = st.button(
            "🚀 Clasificar Hortaliza",
            type="primary",
            use_container_width=True,
            disabled=(ruta_img_seleccionada is None and archivo_subido_bytes is None),
        )
    with col_btn_clear:
        if st.button("🧹 Limpiar Consola", use_container_width=True):
            st.session_state["tab1_logs"] = []
            st.session_state["tab1_ultimo_resultado"] = None
            st.rerun()

    if ejecutar_ind:
        try:
            pipe = obtener_pipeline_cacheado(str(DEFAULT_MODELO), str(DEFAULT_CLASES))
            if archivo_subido_bytes is not None:
                sufijo = Path(nombre_img_visible).suffix.lower() or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as tmp:
                    tmp.write(archivo_subido_bytes)
                    tmp_path = Path(tmp.name)
                try:
                    res_ind = clasificar_con_trazas(
                        pipe,
                        tmp_path,
                        nombre_img_visible,
                        umbral_confianza,
                        st.session_state["tab1_logs"],
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                res_ind = clasificar_con_trazas(
                    pipe,
                    ruta_img_seleccionada,
                    nombre_img_visible,
                    umbral_confianza,
                    st.session_state["tab1_logs"],
                )
            st.session_state["tab1_ultimo_resultado"] = res_ind
        except ModuleNotFoundError as exc:
            st.error(
                f"Dependencia faltante en el entorno actual (`{exc}`). "
                "Instale primero `pip install -r requirements_ui.txt`."
            )
        except Exception as exc:
            st.error(f"Error durante la inferencia individual: {exc}")

    col_vista_img, col_vista_res = st.columns([1.05, 1.45], gap="large")

    with col_vista_img:
        st.markdown("##### 🖼️ Imagen de Entrada")
        if archivo_subido_bytes is not None:
            st.image(archivo_subido_bytes, caption=f"Archivo cargado: {nombre_img_visible}", use_container_width=True)
        elif ruta_img_seleccionada is not None and ruta_img_seleccionada.exists():
            st.image(str(ruta_img_seleccionada), caption=f"Muestra: {nombre_img_visible}", use_container_width=True)
        else:
            st.info("Seleccione o suba una imagen para visualizarla aquí.")

    with col_vista_res:
        st.markdown("##### 📊 Dictamen y Probabilidades Softmax")
        res_actual = st.session_state.get("tab1_ultimo_resultado")
        if res_actual:
            # Recalcular decisión en vivo si el usuario mueve el slider de la barra lateral
            conf_ratio = res_actual["confianza_pct"] / 100.0
            decision_viva = "AUTOMATICO" if conf_ratio >= umbral_confianza else "REVISION_MANUAL"
            badge_html = (
                '<span class="badge-auto">✓ AUTOMATICO</span>'
                if decision_viva == "AUTOMATICO"
                else '<span class="badge-manual">⚠ REVISION_MANUAL</span>'
            )

            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="pred-label">Clase Predicha ({res_actual["archivo"]})</div>
                    <div class="pred-value">{res_actual["clase_predicha"]}</div>
                    {badge_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Confianza Máxima", f"{res_actual['confianza_pct']:.2f}%")
            m2.metric("Latencia Inferencia", f"{res_actual['latencia_ms']:.2f} ms")
            m3.metric("Throughput Instantáneo", f"{res_actual['fps']:.2f} FPS")

            df_probs = pd.DataFrame(
                [
                    {"Hortaliza": clase.capitalize(), "Probabilidad (%)": prob}
                    for clase, prob in res_actual["probabilidades"].items()
                ]
            ).set_index("Hortaliza")
            st.bar_chart(df_probs, color=NAVY_PRIMARY, height=230)
        else:
            st.info(
                "Presione **🚀 Clasificar Hortaliza** para ejecutar el modelo y visualizar la predicción, métricas y distribución Softmax."
            )

    st.markdown('<div class="stdout-header">🖥️ Consola stdout en vivo (Ingesta → Preprocesamiento → Inferencia → Posprocesamiento → JSON)</div>', unsafe_allow_html=True)
    if st.session_state["tab1_logs"]:
        st.code("\n".join(st.session_state["tab1_logs"]), language="bash")
    else:
        st.code("[ESPERANDO EJECUCIÓN] Las trazas detalladas con timestamp aparecerán aquí...", language="bash")

    if st.session_state.get("tab1_ultimo_resultado"):
        with st.expander("📄 Inspeccionar / Descargar Payload JSON Individual"):
            json_ind_str = json.dumps(
                st.session_state["tab1_ultimo_resultado"], indent=2, ensure_ascii=False
            )
            st.json(st.session_state["tab1_ultimo_resultado"])
            st.download_button(
                "⬇️ Descargar ui_individual_resultado.json",
                data=json_ind_str.encode("utf-8"),
                file_name="ui_individual_resultado.json",
                mime="application/json",
            )


# ==============================================================================
# PESTAÑA 2: PROCESAMIENTO POR LOTE (CLI STDOUT EN VIVO)
# ==============================================================================
with tab_batch:
    st.markdown("#### Ejecución autónoma del Pipeline CLI sobre `data/ejemplos/`")
    st.caption(
        "Invoca el motor productivo mediante `subprocess.Popen` con `PYTHONUNBUFFERED=1`, capturando el flujo real `stdout` línea por línea."
    )

    ruta_json_batch = OUTPUTS_DIR / "ui_batch_resultado.json"
    cmd_batch = [
        sys.executable,
        "-u",
        "src/pipeline_inferencia.py",
        "--modelo",
        "models/modelo_hortalizas_act3_3.keras",
        "--clases",
        "data/clases.txt",
        "--modo",
        "batch",
        "--directorio",
        "data/ejemplos",
        "--umbral",
        f"{umbral_confianza:.2f}",
        "--output",
        "outputs/ui_batch_resultado.json",
    ]

    st.code(
        f"python -u src/pipeline_inferencia.py --modelo models/modelo_hortalizas_act3_3.keras "
        f"--clases data/clases.txt --modo batch --directorio data/ejemplos "
        f"--umbral {umbral_confianza:.2f} --output outputs/ui_batch_resultado.json",
        language="bash",
    )

    iniciar_batch = st.button("▶️ Ejecutar Pipeline Batch en Vivo", type="primary")

    st.markdown('<div class="stdout-header">🖥️ Terminal CLI — Salida Estándar (stdout) en Tiempo Real</div>', unsafe_allow_html=True)
    contenedor_stdout_batch = st.empty()

    if iniciar_batch:
        rc_batch, texto_batch = ejecutar_cli_en_vivo(cmd_batch, contenedor_stdout_batch)
        st.session_state["tab2_stdout"] = texto_batch
        if rc_batch == 0 and ruta_json_batch.exists():
            st.session_state["tab2_data"] = json.loads(
                ruta_json_batch.read_text(encoding="utf-8")
            )
            st.success("Procesamiento por lote finalizado correctamente.")
        else:
            st.session_state["tab2_data"] = None
            st.error(f"El proceso finalizó con código de error {rc_batch}.")
    else:
        if st.session_state["tab2_stdout"]:
            contenedor_stdout_batch.code(st.session_state["tab2_stdout"], language="bash")
        else:
            contenedor_stdout_batch.code(
                "$ [LISTO] Presione 'Ejecutar Pipeline Batch en Vivo' para iniciar el subproceso...",
                language="bash",
            )

    datos_lote = st.session_state.get("tab2_data")
    if datos_lote:
        resumen_lote = datos_lote.get("resumen", {})
        lista_res = datos_lote.get("resultados", [])

        st.divider()
        st.markdown("##### 📋 Resumen y Tabla de Resultados del Lote")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Imágenes Procesadas", resumen_lote.get("total_imagenes", len(lista_res)))
        k2.metric("Latencia Media", f"{resumen_lote.get('latencia_media_ms', 0.0):.2f} ms")
        k3.metric("Latencia P95", f"{resumen_lote.get('latencia_p95_ms', 0.0):.2f} ms")
        k4.metric("Throughput Lote", f"{resumen_lote.get('fps_promedio', 0.0):.2f} FPS")

        if lista_res:
            filas_tabla = []
            for r in lista_res:
                probs_r = r.get("probabilidades", {})
                filas_tabla.append(
                    {
                        "Archivo": r.get("archivo"),
                        "Clase Real": r.get("clase_real", "desconocida"),
                        "Clase Predicha": r.get("clase_predicha"),
                        "Acierto": (
                            "✅"
                            if r.get("clase_real") == r.get("clase_predicha")
                            else ("➖" if r.get("clase_real") == "desconocida" else "❌")
                        ),
                        "Confianza (%)": r.get("confianza_pct"),
                        "Decisión Operativa": r.get("decision_operativa"),
                        "P(lechuga) %": probs_r.get("lechuga", 0.0),
                        "P(brocoli) %": probs_r.get("brocoli", 0.0),
                        "P(col) %": probs_r.get("col", 0.0),
                        "Latencia (ms)": r.get("latencia_ms"),
                        "FPS": r.get("fps"),
                    }
                )
            st.dataframe(pd.DataFrame(filas_tabla), use_container_width=True, hide_index=True)

        json_lote_bytes = json.dumps(datos_lote, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar Reporte JSON (ui_batch_resultado.json)",
            data=json_lote_bytes,
            file_name="ui_batch_resultado.json",
            mime="application/json",
            type="primary",
        )


# ==============================================================================
# PESTAÑA 3: BENCHMARK EDGE / RENDIMIENTO (STDOUT EN VIVO)
# ==============================================================================
with tab_benchmark:
    st.markdown("#### Auditoría de Rendimiento Operativo / Simulación Edge (`src/benchmark_edge.py`)")
    st.caption(
        "Mide arranque en frío (Cold Start), memoria residente pico (RAM RSS), latencia media, jitter, percentil 95 y FPS sostenidos."
    )

    col_cfg1, col_cfg2 = st.columns([1.2, 1.8])
    with col_cfg1:
        n_iteraciones = st.slider(
            "Número de iteraciones continuas (N):",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
        )
    with col_cfg2:
        etiqueta_perfil = st.text_input(
            "Etiqueta descriptiva del perfil de despliegue:",
            value="Host Local / Demo UI",
            help='Ejemplo: "Edge 1vCPU 1GB", "Docker Contenedor", "Workstation Local"',
        )

    ruta_json_bench = OUTPUTS_DIR / "ui_benchmark_resultado.json"
    cmd_bench = [
        sys.executable,
        "-u",
        "src/benchmark_edge.py",
        "--modelo",
        "models/modelo_hortalizas_act3_3.keras",
        "--clases",
        "data/clases.txt",
        "--directorio",
        "data/ejemplos",
        "--iteraciones",
        str(n_iteraciones),
        "--perfil",
        etiqueta_perfil,
        "--output",
        "outputs/ui_benchmark_resultado.json",
    ]

    st.code(
        f'python -u src/benchmark_edge.py --modelo models/modelo_hortalizas_act3_3.keras '
        f'--clases data/clases.txt --directorio data/ejemplos --iteraciones {n_iteraciones} '
        f'--perfil "{etiqueta_perfil}" --output outputs/ui_benchmark_resultado.json',
        language="bash",
    )

    iniciar_bench = st.button("⚡ Lanzar Benchmark en Vivo", type="primary")

    st.markdown('<div class="stdout-header">🖥️ Terminal Benchmark — Salida Estándar (stdout) en Tiempo Real</div>', unsafe_allow_html=True)
    contenedor_stdout_bench = st.empty()

    if iniciar_bench:
        rc_bench, texto_bench = ejecutar_cli_en_vivo(cmd_bench, contenedor_stdout_bench)
        st.session_state["tab3_stdout"] = texto_bench
        if rc_bench == 0 and ruta_json_bench.exists():
            st.session_state["tab3_data"] = json.loads(
                ruta_json_bench.read_text(encoding="utf-8")
            )
            st.success("Benchmark finalizado correctamente.")
        else:
            st.session_state["tab3_data"] = None
            st.error(f"El benchmark finalizó con código de error {rc_bench}.")
    else:
        if st.session_state["tab3_stdout"]:
            contenedor_stdout_bench.code(st.session_state["tab3_stdout"], language="bash")
        else:
            contenedor_stdout_bench.code(
                "$ [LISTO] Configure las iteraciones y presione 'Lanzar Benchmark en Vivo'...",
                language="bash",
            )

    datos_bench = st.session_state.get("tab3_data")
    if datos_bench:
        st.divider()
        st.markdown(f"##### 📈 Telemetría Consolidada — Perfil: `{datos_bench.get('perfil')}`")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Cold Start", f"{datos_bench.get('cold_start_s', 0.0):.3f} s")
        b2.metric("RAM Pico RSS", f"{datos_bench.get('ram_pico_rss_mb', 0.0):.1f} MB")
        b3.metric("Latencia Media", f"{datos_bench.get('latencia_media_ms', 0.0):.2f} ms")
        b4.metric("Latencia P95", f"{datos_bench.get('latencia_p95_ms', 0.0):.2f} ms")
        b5.metric("Throughput", f"{datos_bench.get('throughput_fps', 0.0):.2f} FPS")

        col_graf_lat, col_graf_res = st.columns(2, gap="large")
        with col_graf_lat:
            st.markdown("**Distribución de Tiempos de Respuesta por Imagen (ms)**")
            df_latencias = pd.DataFrame(
                [
                    {"Métrica": "1. Mínima", "Milisegundos (ms)": datos_bench.get("latencia_min_ms", 0.0)},
                    {"Métrica": "2. Media", "Milisegundos (ms)": datos_bench.get("latencia_media_ms", 0.0)},
                    {"Métrica": "3. Jitter (Std)", "Milisegundos (ms)": datos_bench.get("jitter_std_ms", 0.0)},
                    {"Métrica": "4. Percentil 95", "Milisegundos (ms)": datos_bench.get("latencia_p95_ms", 0.0)},
                    {"Métrica": "5. Máxima", "Milisegundos (ms)": datos_bench.get("latencia_max_ms", 0.0)},
                ]
            ).set_index("Métrica")
            st.bar_chart(df_latencias, color=NAVY_PRIMARY, height=260)

        with col_graf_res:
            st.markdown("**Capacidad Operativa y Huella en Nodo**")
            df_operativa = pd.DataFrame(
                [
                    {"Indicador": "Cold Start (s)", "Valor": datos_bench.get("cold_start_s", 0.0)},
                    {"Indicador": "Throughput Sostenido (FPS)", "Valor": datos_bench.get("throughput_fps", 0.0)},
                    {"Indicador": "Tiempo Total Bucle (s)", "Valor": datos_bench.get("tiempo_inferencia_total_s", 0.0)},
                    {"Indicador": "RAM Pico RSS (×10 MB)", "Valor": round(datos_bench.get("ram_pico_rss_mb", 0.0) / 10.0, 2)},
                ]
            ).set_index("Indicador")
            st.bar_chart(df_operativa, color=NAVY_SECONDARY, height=260)

        json_bench_bytes = json.dumps(datos_bench, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar Métricas de Benchmark (ui_benchmark_resultado.json)",
            data=json_bench_bytes,
            file_name="ui_benchmark_resultado.json",
            mime="application/json",
            type="primary",
        )
