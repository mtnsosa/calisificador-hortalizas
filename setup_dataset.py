import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image
import fiftyone as fo
import fiftyone.zoo as foz

# ==========================================================
# CONFIGURACIÓN DEL DATASET
# ==========================================================
ROOT_DIR = Path("data")
RAW_DIR = ROOT_DIR / "raw"
SPLITS_DIR = ROOT_DIR / "splits"
EJEMPLOS_DIR = ROOT_DIR / "ejemplos"
ANNOTATIONS_DIR = ROOT_DIR / "annotations"

# Mapeo: Clase en Open Images -> Carpeta local en español
CLASES_MAP = {
    "Lettuce": "lechuga",
    "Broccoli": "brocoli",
    "Cabbage": "col",
}

IMAGENES_POR_CLASE = 100
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEMILLA = 42  # Garantiza reproducibilidad en la partición


def descargar_imagenes():
    print("[1/4] Descargando imágenes desde Open Images v7...")
    # Solicitamos un pequeño margen extra (115) por si alguna imagen viene corrupta
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["classifications"],
        classes=list(CLASES_MAP.keys()),
        max_samples=IMAGENES_POR_CLASE * len(CLASES_MAP) + 60,
        shuffle=True,
        seed=SEMILLA,
    )

    conteo = {v: 0 for v in CLASES_MAP.values()}

    for sample in dataset:
        if all(c >= IMAGENES_POR_CLASE for c in conteo.values()):
            break

        # Obtener etiquetas positivas de clasificación
        etiquetas = []
        if sample.positive_labels:
            etiquetas = [lbl.label for lbl in sample.positive_labels.classifications]

        # Filtrar imágenes que pertenezcan inequívocamente a una sola de las 3 clases
        coincidencias = [c for c in CLASES_MAP.keys() if c in etiquetas]
        if len(coincidencias) != 1:
            continue

        clase_en = coincidencias[0]
        clase_es = CLASES_MAP[clase_en]

        if conteo[clase_es] >= IMAGENES_POR_CLASE:
            continue

        origen = Path(sample.filepath)
        destino = RAW_DIR / clase_es / f"{clase_es}_{conteo[clase_es] + 1:03d}.jpg"

        # Validar integridad y estandarizar a RGB JPG
        try:
            with Image.open(origen) as img:
                img_rgb = img.convert("RGB")
                img_rgb.save(destino, format="JPEG", quality=95)
            conteo[clase_es] += 1
        except Exception:
            continue

    fo.delete_dataset(dataset.name)
    print(f"      Descarga completada: {conteo}")


def repartir_splits():
    print("[2/4] Particionando dataset en train / val / test (regla previa)...")
    random.seed(SEMILLA)

    for clase_es in CLASES_MAP.values():
        imagenes = sorted((RAW_DIR / clase_es).glob("*.jpg"))
        random.shuffle(imagenes)

        total = len(imagenes)
        n_train = int(total * RATIOS["train"])
        n_val = int(total * RATIOS["val"])

        asignaciones = {
            "train": imagenes[:n_train],
            "val": imagenes[n_train : n_train + n_val],
            "test": imagenes[n_train + n_val :],
        }

        for split, lista_imgs in asignaciones.items():
            carpeta_destino = SPLITS_DIR / split / clase_es
            for img_path in lista_imgs:
                shutil.copy2(img_path, carpeta_destino / img_path.name)

        print(
            f"      {clase_es:<8} -> train: {len(asignaciones['train'])}, "
            f"val: {len(asignaciones['val'])}, test: {len(asignaciones['test'])}"
        )


def copiar_ejemplos_demo():
    print("[3/4] Copiando imágenes de ejemplo para la demo...")
    EJEMPLOS_DIR.mkdir(parents=True, exist_ok=True)
    clases = list(CLASES_MAP.values())
    for idx, clase_es in enumerate(clases[:2], start=1):
        candidatas = sorted((SPLITS_DIR / "test" / clase_es).glob("*.jpg"))
        if candidatas:
            shutil.copy2(candidatas[0], EJEMPLOS_DIR / f"ejemplo_{idx}.jpg")


def exportar_coco():
    print("[4/4] Exportando anotaciones en formato estándar COCO...")
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    clases = list(CLASES_MAP.values())

    coco = {
        "info": {
            "description": "Dataset de Clasificación de 3 Hortalizas (Lechuga, Brócoli, Col)",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "categories": [
            {"id": idx, "name": nombre, "supercategory": "hortaliza"}
            for idx, nombre in enumerate(clases)
        ],
        "images": [],
        "annotations": [],
    }

    img_id = 1
    ann_id = 1
    for split in ["train", "val", "test"]:
        for cat_id, clase_es in enumerate(clases):
            for img_path in sorted((SPLITS_DIR / split / clase_es).glob("*.jpg")):
                with Image.open(img_path) as im:
                    w, h = im.size
                rel_path = f"splits/{split}/{clase_es}/{img_path.name}"
                coco["images"].append(
                    {"id": img_id, "file_name": rel_path, "width": w, "height": h, "split": split}
                )
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [0, 0, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                img_id += 1
                ann_id += 1

    salida_json = ANNOTATIONS_DIR / "dataset_coco.json"
    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"      Archivo generado: {salida_json}")


if __name__ == "__main__":
    descargar_imagenes()
    repartir_splits()
    copiar_ejemplos_demo()
    exportar_coco()
    print("\n[OK] Dataset listo y estructurado para las Unidades 1, 2, 3 y 4.")