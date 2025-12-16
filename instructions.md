# 🧠 Grounding DINO + SAM2 – Pipeline de Detección y Segmentación Guiada por Texto

## 📌 Descripción general

Este proyecto define y evoluciona un **pipeline de visión por computador** basado en modelos fundacionales que permite:

- Detectar objetos en imágenes (y posteriormente video)
- Usar **texto libre (prompt)** para definir qué detectar
- Segmentar con precisión los objetos detectados

El flujo combina:

- **Grounding DINO** → detección open-vocabulary guiada por texto
- **SAM2 (Segment Anything Model v2)** → segmentación precisa basada en bounding boxes

Todo el sistema está pensado para ser **modular, extensible y escalable**, con soporte futuro para video.

---

## 🎯 Objetivo del proyecto

Diseñar y refactorizar un flujo de detección y segmentación que permita:

- Cambiar dinámicamente las clases de detección mediante texto
- Integrar una interfaz gráfica para cargar imágenes (y luego video)
- Migrar de SAM (v1) a **SAM2**, anticipando segmentación temporal
- Mantener un contrato de datos común mediante la clase `Detections`

---

## 🧠 Contexto actual

El flujo actual del sistema es:

1. Definición manual (hardcodeada) de clases
2. Carga de imagen
3. Inferencia con **Grounding DINO**
4. Conversión de resultados a un objeto `Detections`
5. Segmentación usando **SAM (v1)**
6. Generación de imágenes segmentadas

### 📦 Estructura clave: `Detections`

Se utiliza una clase `Detections` que encapsula:

- Bounding boxes (`xyxy`)
- Máscaras de segmentación (`mask`)
- Confianza de detección (`confidence`)
- Identificador de clase (`class_id`)
- Metadatos opcionales (`tracker_id`)

👉 **Aquí debe insertarse el código actual de la clase `Detections` y sus utilidades**, sin modificar su interfaz pública salvo necesidad estricta por SAM2 o video.

---

## ❌ Limitaciones actuales

- Las clases de detección están **hardcodeadas**
- No es posible cambiar el prompt durante la ejecución
- Se utiliza **SAM v1**, que no soporta segmentación temporal en video
- El sistema está limitado a imágenes estáticas

---

## ✅ Nuevos requerimientos

### 1️⃣ Migración a SAM2

- Reemplazar SAM v1 por **SAM2**
- Diseñar el pipeline considerando soporte futuro para video
- SAM2 recibirá únicamente bounding boxes, no texto

📌 **Nota importante:** SAM2 no infiere clases ni entiende texto, solo segmenta.

---

### 2️⃣ Detección guiada por texto (open-vocabulary)

- El usuario debe poder escribir libremente qué desea detectar
- El texto ingresado alimenta directamente a **Grounding DINO**
- No deben existir clases hardcodeadas en el código

**Ejemplo de uso:**

```text
"person wearing helmet, motorcycle"
```

Grounding DINO:
- Interpreta el texto
- Detecta los objetos correspondientes

SAM2:
- Segmenta únicamente las detecciones resultantes

---

### 3️⃣ Interfaz gráfica (GUI)

La interfaz debe permitir:

- Subir una imagen
- Ingresar un prompt de texto
- Ejecutar el pipeline completo
- Visualizar:
  - bounding boxes
  - máscaras segmentadas

📌 El soporte para video se implementará en una etapa posterior.

---

## 🔄 Flujo esperado (nuevo diseño)

1. Usuario carga imagen desde la GUI
2. Usuario ingresa texto de detección
3. **Grounding DINO**:
   - recibe imagen + texto
   - genera detecciones
4. Conversión a objeto `Detections`
5. Aplicación opcional de **Non-Maximum Suppression (NMS)**
6. **SAM2**:
   - recibe `Detections.xyxy`
   - genera máscaras
7. Construcción de un nuevo objeto `Detections` con máscaras
8. Visualización del resultado final

---

## 🧩 Responsabilidades de cada modelo

### 🟦 Grounding DINO

- Interpreta lenguaje natural
- Decide **qué** objetos existen
- Produce bounding boxes y scores

### 🟩 SAM2

- No entiende texto
- No clasifica
- Solo refina espacialmente regiones indicadas

---

## 🏗️ Consideraciones de diseño

- Arquitectura modular
- Separación clara entre:
  - detección
  - segmentación
  - visualización
- El objeto `Detections` actúa como contrato entre módulos
- El diseño debe permitir extender a video sin reescribir el pipeline

---

## 📈 Estado del proyecto

- ✔️ Detección con Grounding DINO
- ✔️ Segmentación con SAM (v1)
- ⏳ Migración a SAM2
- ⏳ GUI interactiva con prompt dinámico
- ⏳ Soporte para video

---

## 📌 Próximos pasos sugeridos

- Diseñar arquitectura de carpetas
- Adaptar `Detections` para segmentación temporal
- Definir estrategia de tracking para video
- Evaluar GUI (Tkinter vs PySide vs Web)

---

## 📜 Licencia

Uso experimental / académico. Ajustar según necesidades del proyecto.


### Codigo actual:
```python

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import Model
from supervision import Detections
from segment_anything import sam_model_registry, SamPredictor

DETECTION_CATEGORIES = ["car", "cat", "yellow school bus"]  # yellow school bus

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
DEVICE = "cpu"

MODEL_PATH = BASE_DIR / "groundingdino_swint_ogc.pth"
CONFIG_PATH = BASE_DIR / "GroundingDINO_SwinT_OGC.py"
SAM_ENCODER_VERSION = "vit_b"
SAM_CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"


OUTPUT_ANNOTATED = BASE_DIR / "annotated"
OUTPUT_SEGMENTED = BASE_DIR / "segmented"
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25

# Create output directories for each category (including unknown)
OUTPUT_ANNOTATED.mkdir(exist_ok=True)
for category in DETECTION_CATEGORIES + ["unknown"]:
    (OUTPUT_ANNOTATED / category).mkdir(exist_ok=True)

OUTPUT_SEGMENTED.mkdir(exist_ok=True)
for category in DETECTION_CATEGORIES + ["unknown"]:
    (OUTPUT_SEGMENTED / category).mkdir(exist_ok=True)

# Force PyTorch to work ONLY on CPU and load model
torch.cuda.is_available = lambda: False

gdino_model = Model(
    model_config_path=str(CONFIG_PATH),
    model_checkpoint_path=str(MODEL_PATH),
    device=DEVICE,
)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


def classify_image(image: cv2.typing.MatLike) -> Detections:
    return gdino_model.predict_with_classes(
        image=image,
        classes=DETECTION_CATEGORIES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )


def apply_sam_segmentation(image: np.ndarray, detections: Detections) -> Detections:
    sam_predictor.set_image(image)

    masks = []
    for box in detections.xyxy:
        mask, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        masks.append(mask[np.argmax(scores)])

    detections.mask = np.array(masks)
    return detections


def select_image():
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
    )

    if not file_path:
        return

    file_path = Path(file_path)

    image = cv2.imread(str(file_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = classify_image(image_rgb)

    if len(detections.xyxy) == 0:
        category = "unknown"
        shutil.copy(file_path, OUTPUT_ANNOTATED / category / file_path.name)
        shutil.copy(file_path, OUTPUT_SEGMENTED / category / file_path.name)

        messagebox.showinfo("Clasificado", "No se detectaron objetos")
        return

    # =====================
    # ANOTACIÓN (BOXES)
    # =====================
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{DETECTION_CATEGORIES[class_id]} {conf:.2f}"
        for conf, class_id in zip(detections.confidence, detections.class_id)
    ]

    annotated_frame = box_annotator.annotate(
        scene=image.copy(), detections=detections, labels=labels
    )

    category = DETECTION_CATEGORIES[detections.class_id[0]]
    cv2.imwrite(str(OUTPUT_ANNOTATED / category / file_path.name), annotated_frame)

    # =====================
    # SEGMENTACIÓN (SAM)
    # =====================
    detections = apply_sam_segmentation(image_rgb, detections)

    mask_annotator = sv.MaskAnnotator()
    segmented_frame = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # HACEMOS B-BOX PARA LAS SEGMENTACIONES
    segmented_frame = box_annotator.annotate(
        scene=segmented_frame, detections=detections, labels=labels
    )

    cv2.imwrite(str(OUTPUT_SEGMENTED / category / file_path.name), segmented_frame)

    messagebox.showinfo(
        "Clasificado",
        f"Imagen: {file_path.name}\n"
        f"Objetos detectados: {len(detections.xyxy)}\n"
        f"Categoría principal: {category.upper()}",
    )


# GUI
root = tk.Tk()
root.title("Classificador con Grounding DINO")
root.geometry("350x160")

label = tk.Label(
    root, text="Selecciona una imagen para clasificar:", font=("Arial", 12)
)
label.pack(pady=20)

button = tk.Button(
    root,
    text="Abrir Imagen",
    command=select_image,
    font=("Arial", 12),
    padx=20,
    pady=10,
)
button.pack()

root.mainloop()
```

### Clase Detections:
```python
@dataclass
class Detections:
    """
    Data class containing information about the detections in a video frame.
    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        mask: (Optional[np.ndarray]): An array of shape `(n, W, H)` containing the segmentation masks.
        confidence (Optional[np.ndarray]): An array of shape `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape `(n,)` containing the tracker ids of the detections.
    """

    xyxy: np.ndarray
    mask: np.Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
```
### Tecnologias
Obviamente usaremos python, adicional usaremos:
- flask
- html, css, javascript
- boostrap para facilidad de UI