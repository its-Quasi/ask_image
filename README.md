# 🧠 Grounding DINO + SAM2 - Text-Guided Detection & Segmentation Pipeline

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Un pipeline modular de visión por computador que combina **Grounding DINO** (detección guiada por texto) y **SAM2** (segmentación precisa) con una interfaz web moderna construida con Flask y Bootstrap.

## 🎯 Características

- ✅ **Detección Open-Vocabulary**: Detecta cualquier objeto usando descripciones en lenguaje natural
- ✅ **Segmentación Precisa**: SAM2 genera máscaras de segmentación de alta calidad
- ✅ **Prompts Dinámicos**: Sin clases hardcodeadas - define qué detectar en tiempo real
- ✅ **Interfaz Web Moderna**: UI responsive con Bootstrap 5
- ✅ **Arquitectura Modular**: Componentes separados y reutilizables
- ✅ **Extensible a Video**: Diseñado para soportar video en el futuro

## 🏗️ Arquitectura

```
ask_images/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Configuración centralizada
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py            # Grounding DINO wrapper
│   │   └── segmenter.py           # SAM2 wrapper
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── processor.py           # Pipeline principal
│   └── visualization/
│       ├── __init__.py
│       └── annotator.py           # Visualización de resultados
├── static/
│   ├── css/
│   │   └── style.css              # Estilos personalizados
│   ├── js/
│   │   └── app.js                 # Lógica del frontend
│   └── uploads/                   # Imágenes subidas por usuarios
├── templates/
│   └── index.html                 # Interfaz principal
├── models/                        # Checkpoints de modelos (descargar)
├── outputs/
│   ├── annotated/                 # Imágenes con bounding boxes
│   └── segmented/                 # Imágenes segmentadas
├── app.py                         # Aplicación Flask
├── requirements.txt               # Dependencias Python
├── install_dependencies.sh        # Script de instalación
└── README.md
```

## 🚀 Instalación

### Prerequisitos

- Python 3.8 o superior
- pip
- Git

### Paso 1: Clonar el repositorio

```bash
cd ask_images
```

### Paso 2: Instalar dependencias

**Opción A: Script automático (recomendado)**

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

**Opción B: Instalación manual**

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Para GPU con CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Instalar dependencias base
pip install -r requirements.txt

# Instalar Grounding DINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Instalar SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Paso 3: Descargar modelos

**IMPORTANTE**: Debes descargar los siguientes modelos y colocarlos en la carpeta `models/`:

#### Grounding DINO

1. **Checkpoint**: `groundingdino_swint_ogc.pth`
   - Descarga: [Grounding DINO Releases](https://github.com/IDEA-Research/GroundingDINO/releases)
   - O usa: `wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`

2. **Config**: `GroundingDINO_SwinT_OGC.py`
   - Descarga desde: [Grounding DINO Config](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py)

#### SAM2

1. **Checkpoint**: `sam2_hiera_large.pt`
   - Descarga: [SAM2 Checkpoints](https://github.com/facebookresearch/segment-anything-2#model-checkpoints)
   - O usa: `wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt`

**Estructura de modelos esperada:**

```
models/
├── groundingdino_swint_ogc.pth
├── GroundingDINO_SwinT_OGC.py
└── sam2_hiera_large.pt
```

## 🎮 Uso

### Iniciar la aplicación web

```bash
python app.py
```

Abre tu navegador en: **http://localhost:5000**

### Interfaz Web

1. **Cargar imagen**: Haz clic en "Selecciona una imagen" y elige un archivo
2. **Escribir prompt**: Ingresa los objetos que quieres detectar (ej: "person, car, dog")
3. **Ajustar parámetros** (opcional):
   - Box Threshold: Umbral de confianza para detecciones
   - Text Threshold: Umbral de coincidencia de texto
   - NMS: Supresión de no máximos
4. **Procesar**: Haz clic en "Procesar Imagen"
5. **Ver resultados**: Las imágenes con detecciones y segmentaciones aparecerán abajo

### Ejemplos de Prompts

```text
# Objetos comunes
"person, car, dog, cat"

# Colores específicos
"red apple, green apple, yellow banana"

# Descripciones complejas
"person wearing helmet, motorcycle, traffic light"

# Múltiples categorías
"laptop, mouse, keyboard, monitor, phone"
```

## 🔧 Configuración

Puedes ajustar la configuración en `app/core/config.py`:

```python
# Thresholds de detección
BOX_THRESHOLD = 0.35      # Aumenta para menos detecciones, más precisas
TEXT_THRESHOLD = 0.25     # Umbral de coincidencia de texto
NMS_THRESHOLD = 0.8       # Supresión de detecciones superpuestas

# Device
DEVICE = "cpu"            # Cambia a "cuda" si tienes GPU
```

## 🧪 Uso Programático

También puedes usar el pipeline directamente en Python:

```python
from app.pipeline.processor import ImageProcessor

# Inicializar processor
processor = ImageProcessor()

# Procesar imagen
result = processor.process_image(
    image_path="path/to/image.jpg",
    text_prompt="person, car, dog",
    box_threshold=0.35,
    text_threshold=0.25,
)

print(f"Detectados: {result['num_detections']} objetos")
print(f"Clases: {result['class_names']}")
```

## 📊 Flujo del Pipeline

```
1. Usuario carga imagen + prompt de texto
         ↓
2. Grounding DINO
   - Recibe: imagen RGB + texto
   - Genera: bounding boxes + confianza + class_id
         ↓
3. NMS (Non-Maximum Suppression)
   - Elimina detecciones superpuestas
         ↓
4. SAM2
   - Recibe: imagen + bounding boxes
   - Genera: máscaras de segmentación precisas
         ↓
5. Visualización
   - Anotaciones con bounding boxes
   - Segmentación con máscaras coloreadas
         ↓
6. Guardado de resultados
   - outputs/annotated/
   - outputs/segmented/
```

## 🔍 Componentes Principales

### 1. Detector (Grounding DINO)

```python
from app.models.detector import GroundingDINODetector

detector = GroundingDINODetector()
detections = detector.detect(image, "person, car")
```

### 2. Segmentador (SAM2)

```python
from app.models.segmenter import SAM2Segmenter

segmenter = SAM2Segmenter()
detections = segmenter.segment(image, detections)
```

### 3. Visualizador

```python
from app.visualization.annotator import ImageAnnotator

annotator = ImageAnnotator()
annotated = annotator.annotate_masks(image, detections, labels)
```

## 🐛 Solución de Problemas

### Error: "Model checkpoint not found"

**Solución**: Asegúrate de haber descargado los modelos y colocado en `models/`

### Error: "CUDA out of memory"

**Solución**: En `app/core/config.py`, cambia `DEVICE = "cpu"`

### Error: "No module named 'groundingdino'"

**Solución**: Instala Grounding DINO:
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### No detecta objetos

**Solución**:
- Reduce `box_threshold` y `text_threshold`
- Usa palabras más simples en el prompt
- Verifica que los objetos sean claramente visibles

## 🚀 Extensiones Futuras

- [ ] Soporte para video (tracking temporal)
- [ ] Batch processing de múltiples imágenes
- [ ] API REST para integración
- [ ] Exportación de máscaras en formatos COCO/YOLO
- [ ] Integración con bases de datos
- [ ] Autenticación de usuarios

## 📝 Notas Técnicas

### Separación de Responsabilidades

- **Grounding DINO**: Entiende texto, decide QUÉ objetos existen
- **SAM2**: No entiende texto, solo refina DÓNDE están los objetos

### Clase `Detections`

El pipeline usa `supervision.Detections` como contrato de datos:

```python
@dataclass
class Detections:
    xyxy: np.ndarray           # Bounding boxes [x1, y1, x2, y2]
    mask: Optional[np.ndarray] # Máscaras de segmentación
    confidence: Optional[np.ndarray]  # Scores de confianza
    class_id: Optional[np.ndarray]    # IDs de clase
    tracker_id: Optional[np.ndarray]  # IDs de tracking (futuro)
```

## 📄 Licencia

Este proyecto es de uso experimental/académico. Ajusta según las necesidades de tu proyecto.

## 🙏 Créditos

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/segment-anything-2)
- [Supervision](https://github.com/roboflow/supervision)

---

**Desarrollado con ❤️ usando Flask, Bootstrap, Grounding DINO y SAM2**
