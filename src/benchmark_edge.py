#!/usr/bin/env python3
"""
Benchmark de Rendimiento en Despliegue Edge / Contenedor — Clasificador de Hortalizas
Asignatura: Visión Inteligente | Unidad 4 (Actividad 4.2) — TecNM Virtual

Mide con precisión:
- Cold Start (s): importación de TensorFlow + carga del modelo .keras + warm-up inicial.
- Consumo pico de memoria RAM RSS (MB): resource.getrusage en Linux/Unix o psutil/Win32 como respaldo.
- Latencia media (ms), desviación estándar / jitter (ms), Percentil 95 P95 (ms) y Throughput sostenido (FPS).
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

# Silenciar logs informativos C++ de TensorFlow antes de la importación diferida
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Asegurar que el directorio src/ esté en sys.path al invocar desde la raíz o dentro de Docker
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def obtener_ram_pico_mb() -> float:
    """
    Obtiene el consumo pico de memoria RAM RSS del proceso actual en MB.
    Prioriza `resource.getrusage` en Linux/macOS y recurre a `psutil` o API Win32 en Windows.
    """
    # 1. Estándar POSIX / Contenedores Linux (Docker)
    try:
        import resource

        uso = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            # En macOS ru_maxrss se reporta en bytes
            return float(uso) / (1024.0 * 1024.0)
        # En Linux ru_maxrss se reporta en kilobytes (KiB)
        return float(uso) / 1024.0
    except (ImportError, AttributeError):
        pass

    # 2. Respaldo multiplataforma con psutil
    try:
        import psutil

        mem_info = psutil.Process(os.getpid()).memory_info()
        pico_bytes = getattr(mem_info, "peak_wset", None) or mem_info.rss
        return float(pico_bytes) / (1024.0 * 1024.0)
    except Exception:
        pass

    # 3. Respaldo nativo Windows sin dependencias externas (ctypes Win32 API)
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(pmc), pmc.cb
            ):
                return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass

    return 0.0


def listar_imagenes(directorio: Path):
    extensiones = {".jpg", ".jpeg", ".png"}
    if not directorio.exists():
        raise FileNotFoundError(f"Directorio de pruebas no encontrado: {directorio}")
    imagenes = sorted(
        [
            p
            for p in directorio.rglob("*")
            if p.is_file() and p.suffix.lower() in extensiones
        ]
    )
    if not imagenes:
        raise ValueError(f"No se encontraron imágenes válidas (.jpg/.jpeg/.png) en: {directorio}")
    return imagenes


def imprimir_tabla_resumen(metricas: dict) -> None:
    ancho = 68
    linea = "=" * ancho
    separador = "-" * ancho
    filas = [
        ("Perfil de Entorno", str(metricas["perfil"])),
        ("Plataforma / Python", f"{metricas['plataforma']} | Python {metricas['python_version']}"),
        ("Versión TensorFlow", str(metricas["tensorflow_version"])),
        ("Modelo Evaluado", str(metricas["modelo"])),
        ("Iteraciones Continuas (N)", f"{metricas['iteraciones']} ({metricas['imagenes_base']} imágenes base)"),
        ("Cold Start (TF + Carga + Warm-up)", f"{metricas['cold_start_s']:.4f} s"),
        ("Memoria RAM Pico RSS", f"{metricas['ram_pico_rss_mb']:.2f} MB"),
        ("Latencia Media", f"{metricas['latencia_media_ms']:.3f} ms"),
        ("Desviación Estándar (Jitter)", f"{metricas['jitter_std_ms']:.3f} ms"),
        ("Latencia Percentil 95 (P95)", f"{metricas['latencia_p95_ms']:.3f} ms"),
        ("Latencia Mín / Máx", f"{metricas['latencia_min_ms']:.3f} / {metricas['latencia_max_ms']:.3f} ms"),
        ("Throughput Sostenido", f"{metricas['throughput_fps']:.2f} FPS"),
    ]

    print(f"\n{linea}")
    print(" BENCHMARK DE DESPLIEGUE EDGE / CONTENEDOR — CLASIFICADOR HORTALIZAS")
    print(linea)
    print(f" {'Métrica':<36} | {'Valor':<27}")
    print(separador)
    for nombre, valor in filas:
        print(f" {nombre:<36} | {valor:<27}")
    print(f"{linea}\n")


def ejecutar_benchmark(args) -> dict:
    ruta_dir = Path(args.directorio)
    imagenes = listar_imagenes(ruta_dir)

    if args.iteraciones < 1:
        raise ValueError("El número de --iteraciones debe ser mayor o igual a 1.")

    # 1. Medición de Cold Start: importación de TensorFlow + carga de modelo + warm-up
    t0_cold = time.perf_counter()
    import numpy as np
    import tensorflow as tf
    from pipeline_inferencia import PipelineInferenciaHortalizas

    pipe = PipelineInferenciaHortalizas(
        ruta_modelo=args.modelo,
        ruta_clases=args.clases,
    )
    pipe.cargar_modelo()
    cold_start_s = time.perf_counter() - t0_cold

    # 2. Bucle continuo de N iteraciones de inferencia end-to-end (lectura + preprocesado + inferencia)
    latencias_ms = []
    t0_loop = time.perf_counter()
    for i in range(args.iteraciones):
        img_actual = imagenes[i % len(imagenes)]
        t_iter = time.perf_counter()
        tensor = pipe.preprocesar(img_actual)
        _ = pipe.modelo(tensor, training=False).numpy()
        latencias_ms.append((time.perf_counter() - t_iter) * 1000.0)
    tiempo_sostenido_s = time.perf_counter() - t0_loop

    # 3. Captura del pico de memoria RSS tras completar la carga y las inferencias
    ram_pico_mb = obtener_ram_pico_mb()

    arr_lat = np.asarray(latencias_ms, dtype=np.float64)
    ddof = 1 if len(arr_lat) > 1 else 0

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "perfil": args.perfil,
        "plataforma": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cpus_logicos_detectados": os.cpu_count(),
        "modelo": str(Path(args.modelo)),
        "normalizacion_interna_modelo": pipe.tiene_normalizacion_interna,
        "directorio_pruebas": str(ruta_dir),
        "imagenes_base": len(imagenes),
        "iteraciones": int(args.iteraciones),
        "cold_start_s": round(float(cold_start_s), 4),
        "ram_pico_rss_mb": round(float(ram_pico_mb), 2),
        "latencia_media_ms": round(float(np.mean(arr_lat)), 3),
        "jitter_std_ms": round(float(np.std(arr_lat, ddof=ddof)), 3),
        "latencia_p95_ms": round(float(np.percentile(arr_lat, 95)), 3),
        "latencia_min_ms": round(float(np.min(arr_lat)), 3),
        "latencia_max_ms": round(float(np.max(arr_lat)), 3),
        "tiempo_inferencia_total_s": round(float(tiempo_sostenido_s), 4),
        "throughput_fps": round(float(args.iteraciones) / max(tiempo_sostenido_s, 1e-9), 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Edge / Docker — Clasificador de Hortalizas (MobileNetV2)"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="models/modelo_hortalizas_act3_3.keras",
        help="Ruta al archivo del modelo (.keras)",
    )
    parser.add_argument(
        "--clases",
        type=str,
        default="data/clases.txt",
        help="Ruta al archivo clases.txt",
    )
    parser.add_argument(
        "--directorio",
        type=str,
        default="data/ejemplos",
        help="Directorio con imágenes de prueba",
    )
    parser.add_argument(
        "--iteraciones",
        type=int,
        default=30,
        help="Número de iteraciones continuas de inferencia (default: 30)",
    )
    parser.add_argument(
        "--perfil",
        type=str,
        default="Local / Host",
        help='Etiqueta descriptiva del entorno (ej. "Edge 1vCPU", "Edge 0.5vCPU 512MB")',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta opcional para exportar el resumen en formato JSON",
    )
    args = parser.parse_args()

    metricas = ejecutar_benchmark(args)
    imprimir_tabla_resumen(metricas)

    resumen_json = json.dumps(metricas, indent=2, ensure_ascii=False)
    print("=== RESUMEN JSON ===")
    print(resumen_json)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(resumen_json, encoding="utf-8")
        print(f"\n[OK] Métricas de benchmark guardadas en: {out_path}")


if __name__ == "__main__":
    main()
