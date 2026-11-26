#!/usr/bin/env python3
"""
Pipeline Operativo Autónomo de Inferencia — Clasificación de Hortalizas
Asignatura: Visión Inteligente | Unidad 4 (Actividades 4.1 y 4.2) — TecNM Virtual
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf


class PipelineInferenciaHortalizas:
    EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png"}
    IMG_SIZE = (224, 224)

    def __init__(
        self,
        ruta_modelo="models/modelo_hortalizas_act3_3.keras",
        ruta_clases="data/clases.txt",
        umbral_confianza=0.60,
    ):
        self.ruta_modelo = Path(ruta_modelo)
        self.ruta_clases = Path(ruta_clases)
        self.umbral_confianza = float(umbral_confianza)
        self.clases = self._cargar_clases()
        self.modelo = None
        self.tiene_normalizacion_interna = False

    def _cargar_clases(self):
        if not self.ruta_clases.exists():
            raise FileNotFoundError(f"Archivo de clases no encontrado: {self.ruta_clases}")
        clases = [
            c.strip()
            for c in self.ruta_clases.read_text(encoding="utf-8-sig").splitlines()
            if c.strip()
        ]
        if not clases:
            raise ValueError("El archivo clases.txt no contiene etiquetas válidas.")
        return clases

    @staticmethod
    def _detectar_normalizacion_interna(modelo):
        """
        Verifica si la arquitectura ya integra una capa Rescaling, preprocess_input
        o división de escala inicial para evitar doble normalización.
        """
        def _revisar_capas(capas):
            for layer in capas:
                nombre = layer.name.lower()
                if (
                    isinstance(layer, tf.keras.layers.Rescaling)
                    or "preprocess" in nombre
                    or "rescaling" in nombre
                    or nombre.startswith("true_divide")
                ):
                    return True
                # Revisar submodelos de preprocesamiento/augmentación (evitando recorrer el backbone MobileNetV2 completo)
                subcapas = getattr(layer, "layers", None)
                if subcapas and "mobilenet" not in nombre:
                    if _revisar_capas(subcapas):
                        return True
            return False

        return _revisar_capas(modelo.layers)

    def cargar_modelo(self):
        if not self.ruta_modelo.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {self.ruta_modelo}")
        self.modelo = tf.keras.models.load_model(self.ruta_modelo, compile=False)
        self.tiene_normalizacion_interna = self._detectar_normalizacion_interna(self.modelo)
        dummy = np.zeros((1, self.IMG_SIZE[0], self.IMG_SIZE[1], 3), dtype=np.float32)
        _ = self.modelo(dummy, training=False)  # Warm-up

    def preprocesar(self, ruta_imagen):
        if self.modelo is None:
            self.cargar_modelo()
        ruta = Path(ruta_imagen)
        if not ruta.exists() or not ruta.is_file():
            raise FileNotFoundError(f"Imagen no encontrada: {ruta}")
        if ruta.suffix.lower() not in self.EXTENSIONES_VALIDAS:
            raise ValueError(f"Extensión no soportada: {ruta.suffix}")
        try:
            with Image.open(ruta) as img:
                img_rgb = img.convert("RGB").resize(self.IMG_SIZE, Image.Resampling.BILINEAR)
                arr = np.asarray(img_rgb, dtype=np.float32)
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Imagen corrupta ({ruta.name}): {e}") from e

        tensor = (
            arr
            if self.tiene_normalizacion_interna
            else tf.keras.applications.mobilenet_v2.preprocess_input(arr.copy())
        )
        return np.expand_dims(tensor, axis=0)

    def _inferir_clase_real(self, ruta_imagen):
        p = Path(ruta_imagen)
        if p.parent.name in self.clases:
            return p.parent.name
        tokens = p.stem.lower().replace("-", "_").split("_")
        for clase in self.clases:
            if clase.lower() in tokens:
                return clase
        return "desconocida"

    def predecir_imagen(self, ruta_imagen):
        if self.modelo is None:
            self.cargar_modelo()
        t0 = time.perf_counter()
        tensor = self.preprocesar(ruta_imagen)
        probs = self.modelo(tensor, training=False).numpy()[0]
        lat_ms = (time.perf_counter() - t0) * 1000.0

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "archivo": Path(ruta_imagen).name,
            "clase_predicha": self.clases[idx],
            "confianza_pct": round(conf * 100.0, 2),
            "decision_operativa": "AUTOMATICO" if conf >= self.umbral_confianza else "REVISION_MANUAL",
            "probabilidades": {
                self.clases[i]: round(float(probs[i]) * 100.0, 2)
                for i in range(len(self.clases))
            },
            "latencia_ms": round(lat_ms, 3),
            "fps": round(1000.0 / max(lat_ms, 1e-6), 2),
        }

    def predecir_lote(self, directorio, modo_stream=False):
        if self.modelo is None:
            self.cargar_modelo()
        dir_path = Path(directorio)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directorio inexistente: {dir_path}")

        archivos = sorted(
            [
                p
                for p in dir_path.rglob("*")
                if p.is_file() and p.suffix.lower() in self.EXTENSIONES_VALIDAS
            ]
        )
        if not archivos:
            raise ValueError(f"Sin imágenes válidas en {dir_path}")

        resultados = []
        t0_global = time.perf_counter()
        for idx, p in enumerate(archivos, start=1):
            try:
                res = self.predecir_imagen(p)
                res["clase_real"] = self._inferir_clase_real(p)
                resultados.append(res)
                if modo_stream:
                    print(
                        f"[STREAM {idx:02d}/{len(archivos):02d}] {res['archivo']:<24} -> "
                        f"{res['clase_predicha']:<10} ({res['confianza_pct']:6.2f}%) | "
                        f"{res['latencia_ms']:6.2f} ms | {res['decision_operativa']}",
                        file=sys.stderr,
                    )
            except Exception as err:
                print(f"[WARN] Omitiendo {p.name}: {err}", file=sys.stderr)

        if not resultados:
            raise RuntimeError("No se pudo procesar ninguna imagen válida del lote.")

        t_total_ms = (time.perf_counter() - t0_global) * 1000.0
        latencias = [r["latencia_ms"] for r in resultados]
        resumen = {
            "modo": "tiempo_real_simulado" if modo_stream else "batch",
            "normalizacion_interna_modelo": self.tiene_normalizacion_interna,
            "total_imagenes": len(resultados),
            "latencia_media_ms": round(float(np.mean(latencias)), 3),
            "latencia_p95_ms": round(float(np.percentile(latencias, 95)), 3),
            "fps_promedio": round((len(resultados) * 1000.0) / max(t_total_ms, 1e-6), 2),
        }
        return {"resumen": resumen, "resultados": resultados}


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Operativo de Inferencia — Clasificador de Hortalizas"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="models/modelo_hortalizas_act3_3.keras",
        help="Ruta al modelo entrenado (.keras o .h5)",
    )
    parser.add_argument(
        "--clases",
        type=str,
        default="data/clases.txt",
        help="Ruta al archivo clases.txt",
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["individual", "batch", "stream"],
        default="individual",
        help="Modalidad de ejecución (individual, batch o stream)",
    )
    parser.add_argument(
        "--imagen",
        type=str,
        default=None,
        help="Ruta a imagen única (modo individual)",
    )
    parser.add_argument(
        "--directorio",
        type=str,
        default=None,
        help="Ruta a carpeta de imágenes (modo batch o stream)",
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=0.60,
        help="Umbral mínimo de confianza automática [0-1]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/cli_resultado.json",
        help="Ruta de salida JSON",
    )
    args = parser.parse_args()

    pipe = PipelineInferenciaHortalizas(
        args.modelo, args.clases, umbral_confianza=args.umbral
    )
    pipe.cargar_modelo()

    if args.modo == "individual":
        if not args.imagen:
            parser.error("El modo 'individual' requiere el argumento --imagen.")
        salida = pipe.predecir_imagen(args.imagen)
    else:
        if not args.directorio:
            parser.error(f"El modo '{args.modo}' requiere el argumento --directorio.")
        salida = pipe.predecir_lote(args.directorio, modo_stream=(args.modo == "stream"))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"\n[OK] Resultado persistido en: {out_path}")


if __name__ == "__main__":
    main()
