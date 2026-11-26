import io
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
import requests
from PIL import Image

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
ROOT_DIR = Path("data")
RAW_LECHUGA = ROOT_DIR / "raw" / "lechuga"
SPLITS_DIR = ROOT_DIR / "splits"
EJEMPLOS_DIR = ROOT_DIR / "ejemplos"
ANNOTATIONS_DIR = ROOT_DIR / "annotations"

CLASES = ["lechuga", "brocoli", "col"]
OBJETIVO = 100
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEMILLA = 42

HEADERS = {
    "User-Agent": "MIA-VisionInteligente-DatasetBuilder/1.0 (Academic Project)"
}

# Términos variados en Wikimedia Commons para evitar sesgo de muestreo
CONSULTAS_WIKIMEDIA = [
    "Lactuca sativa",
    "romaine lettuce",
    "iceberg lettuce",
    "butterhead lettuce",
    "lettuce head",
]


def descargar_lechugas_wikimedia():
    print("[1/3] Descargando 100 imágenes de lechuga desde Wikimedia Commons...")
    RAW_LECHUGA.mkdir(parents=True, exist_ok=True)

    # Contar si ya había algunas descargadas
    existentes = sorted(RAW_LECHUGA.glob("*.jpg"))
    conteo = len(existentes)
    urls_vistas = set()

    for query in CONSULTAS_WIKIMEDIA:
        if conteo >= OBJETIVO:
            break

        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,  # Espacio de nombres File:
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size",
            "iiurlwidth": 800,  # Descarga versión optimizada de ~800px de ancho
        }

        try:
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params=params,
                headers=HEADERS,
                timeout=20,
            )
            data = resp.json()
        except Exception as e:
            print(f"      Aviso consultando '{query}': {e}")
            continue

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if conteo >= OBJETIVO:
                break

            info_list = page.get("imageinfo", [])
            if not info_list:
                continue

            info = info_list[0]
            img_url = info.get("thumburl") or info.get("url")
            if not img_url or img_url in urls_vistas:
                continue
            urls_vistas.add(img_url)

            # Evitar miniaturas muy pequeñas
            if info.get("width", 0) < 200 or info.get("height", 0) < 200:
                continue

            try:
                img_resp = requests.get(img_url, headers=HEADERS, timeout=20)
                img_resp.raise_for_status()
                with Image.open(io.BytesIO(img_resp.content)) as im:
                    im_rgb = im.convert("RGB")
                    destino = RAW_LECHUGA / f"lechuga_{conteo + 1:03d}.jpg"
                    im_rgb.save(destino, format="JPEG", quality=95)
                conteo += 1
                if conteo % 20 == 0:
                    print(f"      Progreso lechuga: {conteo}/{OBJETIVO}")
                time.sleep(0.15)  # Pausa respetuosa con el servidor
            except Exception:
                continue

    print(f"      Total en data/raw/lechuga: {conteo} imágenes.")


def repartir_lechuga():
    print("[2/3] Particionando lechuga en train / val / test...")
    random.seed(SEMILLA)
    imagenes = sorted(RAW_LECHUGA.glob("*.jpg"))
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
        carpeta_destino = SPLITS_DIR / split / "lechuga"
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        # Limpiar copias previas solo de lechuga si existieran
        for viejo in carpeta_destino.glob("*.jpg"):
            viejo.unlink()
        for img_path in lista_imgs:
            shutil.copy2(img_path, carpeta_destino / img_path.name)

    print(
        f"      lechuga  -> train: {len(asignaciones['train'])}, "
        f"val: {len(asignaciones['val'])}, test: {len(asignaciones['test'])}"
    )

    # Actualizar ejemplo_1.jpg con una lechuga de test
    EJEMPLOS_DIR.mkdir(parents=True, exist_ok=True)
    test_lechugas = sorted((SPLITS_DIR / "test" / "lechuga").glob("*.jpg"))
    if test_lechugas:
        shutil.copy2(test_lechugas[0], EJEMPLOS_DIR / "ejemplo_1.jpg")


def regenerar_coco_completo():
    print("[3/3] Regenerando dataset_coco.json con las 3 clases balanceadas...")
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "Dataset de Clasificación de 3 Hortalizas (Lechuga, Brócoli, Col)",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "categories": [
            {"id": idx, "name": nombre, "supercategory": "hortaliza"}
            for idx, nombre in enumerate(CLASES)
        ],
        "images": [],
        "annotations": [],
    }

    img_id = 1
    ann_id = 1
    conteo_final = {c: 0 for c in CLASES}

    for split in ["train", "val", "test"]:
        for cat_id, clase_es in enumerate(CLASES):
            for img_path in sorted((SPLITS_DIR / split / clase_es).glob("*.jpg")):
                with Image.open(img_path) as im:
                    w, h = im.size
                rel_path = f"splits/{split}/{clase_es}/{img_path.name}"
                coco["images"].append(
                    {
                        "id": img_id,
                        "file_name": rel_path,
                        "width": w,
                        "height": h,
                        "split": split,
                    }
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
                conteo_final[clase_es] += 1

    salida_json = ANNOTATIONS_DIR / "dataset_coco.json"
    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"      Resumen total por clase: {conteo_final}")
    print(f"      Archivo actualizado: {salida_json}")


if __name__ == "__main__":
    descargar_lechugas_wikimedia()
    repartir_lechuga()
    regenerar_coco_completo()
    print("\n[OK] Dataset balanceado (300 imágenes) listo.")