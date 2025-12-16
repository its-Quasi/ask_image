# 📂 Estructura del Proyecto

## Árbol de Directorios

```
ask_images/
│
├── 📱 APLICACIÓN PRINCIPAL
│   ├── app.py                          # Servidor Flask (punto de entrada)
│   └── app/                            # Paquete principal de la aplicación
│       ├── __init__.py
│       │
│       ├── core/                       # Configuración y utilidades core
│       │   ├── __init__.py
│       │   └── config.py               # Configuración centralizada (paths, thresholds, device)
│       │
│       ├── models/                     # Modelos de ML (detección y segmentación)
│       │   ├── __init__.py
│       │   ├── detector.py             # Grounding DINO wrapper (detección guiada por texto)
│       │   └── segmenter.py            # SAM2 wrapper (segmentación precisa)
│       │
│       ├── pipeline/                   # Pipeline de procesamiento
│       │   ├── __init__.py
│       │   └── processor.py            # Orquestador del flujo completo
│       │
│       └── visualization/              # Visualización y anotaciones
│           ├── __init__.py
│           └── annotator.py            # Dibuja boxes, máscaras y labels
│
├── 🌐 FRONTEND (Web Interface)
│   ├── templates/
│   │   └── index.html                  # Interfaz principal (Bootstrap 5)
│   │
│   └── static/
│       ├── css/
│       │   └── style.css               # Estilos personalizados
│       ├── js/
│       │   └── app.js                  # Lógica del cliente (AJAX, preview, etc.)
│       └── uploads/                    # Imágenes subidas por usuarios
│           └── .gitkeep
│
├── 🤖 MODELOS (Checkpoints - debes descargarlos)
│   └── models/
│       ├── .gitkeep
│       ├── groundingdino_swint_ogc.pth         # (descargar ~700 MB)
│       ├── GroundingDINO_SwinT_OGC.py          # (descargar ~5 KB)
│       └── sam2_hiera_large.pt                 # (descargar ~900 MB)
│
├── 📊 OUTPUTS (Resultados generados)
│   └── outputs/
│       ├── annotated/                  # Imágenes con bounding boxes
│       │   └── .gitkeep
│       └── segmented/                  # Imágenes con máscaras de segmentación
│           └── .gitkeep
│
├── 🔧 SCRIPTS DE INSTALACIÓN Y UTILIDADES
│   ├── install_dependencies.sh         # Instalar todas las dependencias
│   ├── download_models.sh              # Descargar checkpoints de modelos
│   ├── verify_models.py                # Verificar que modelos están instalados
│   └── example_usage.py                # Ejemplo de uso programático
│
├── 📚 DOCUMENTACIÓN
│   ├── README.md                       # Documentación completa del proyecto
│   ├── QUICKSTART.md                   # Guía de inicio rápido (3 pasos)
│   ├── MODELS_SETUP.md                 # Guía detallada de descarga de modelos
│   ├── PROJECT_STRUCTURE.md            # Este archivo (estructura del proyecto)
│   └── instructions.md                 # Instrucciones originales del proyecto
│
├── 📦 CONFIGURACIÓN DE PROYECTO
│   ├── requirements.txt                # Dependencias de Python
│   └── .gitignore                      # Archivos ignorados por Git
│
└── 🗄️ OTROS
    └── .claude/                        # Configuración de Claude Code (metadata)
```

## 🎯 Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Web UI)                        │
│                     templates/index.html                         │
│                     static/js/app.js                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ 1. Usuario sube imagen + prompt
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (Flask Server)                        │
│                         app.py                                   │
│                  Endpoint: /process                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ 2. Llama al pipeline
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PIPELINE (Procesamiento)                       │
│                app/pipeline/processor.py                         │
└─────┬─────────────────────────────────────────────────┬─────────┘
      │                                                   │
      │ 3. Detección                                    │ 5. Visualización
      ▼                                                   ▼
┌──────────────────────┐                      ┌─────────────────────┐
│  GROUNDING DINO      │                      │   ANNOTATOR         │
│ app/models/          │                      │ app/visualization/  │
│   detector.py        │                      │   annotator.py      │
│                      │                      │                     │
│ Input: imagen + texto│                      │ Dibuja boxes        │
│ Output: Detections   │                      │ Dibuja máscaras     │
└──────────┬───────────┘                      │ Dibuja labels       │
           │                                  └─────────────────────┘
           │ 4. Segmentación                           │
           ▼                                            │
┌──────────────────────┐                               │
│       SAM2           │                               │
│ app/models/          │                               │
│   segmenter.py       │                               │
│                      │                               │
│ Input: Detections    │                               │
│ Output: Máscaras     │───────────────────────────────┘
└──────────────────────┘
           │
           │ 6. Guardar resultados
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                  │
│               outputs/annotated/                                 │
│               outputs/segmented/                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔑 Archivos Clave

### Backend (Python)

| Archivo | Descripción | Responsabilidad |
|---------|-------------|-----------------|
| `app.py` | Servidor Flask principal | Rutas HTTP, carga de archivos, respuestas JSON |
| `app/core/config.py` | Configuración centralizada | Paths, thresholds, device (CPU/GPU) |
| `app/models/detector.py` | Grounding DINO wrapper | Detección open-vocabulary con prompts |
| `app/models/segmenter.py` | SAM2 wrapper | Segmentación basada en bounding boxes |
| `app/pipeline/processor.py` | Pipeline completo | Orquesta detector → NMS → segmentador → visualizador |
| `app/visualization/annotator.py` | Visualización | Dibuja boxes, máscaras, labels |

### Frontend (Web)

| Archivo | Descripción | Tecnologías |
|---------|-------------|-------------|
| `templates/index.html` | Interfaz de usuario | HTML5, Bootstrap 5, Font Awesome |
| `static/css/style.css` | Estilos personalizados | CSS3, gradientes, animaciones |
| `static/js/app.js` | Lógica del cliente | JavaScript (Vanilla), Fetch API |

### Utilidades

| Archivo | Descripción | Uso |
|---------|-------------|-----|
| `install_dependencies.sh` | Instalación automática | `./install_dependencies.sh` |
| `download_models.sh` | Descarga de modelos | `./download_models.sh` |
| `verify_models.py` | Verificación de setup | `python verify_models.py` |
| `example_usage.py` | Ejemplo programático | `python example_usage.py` |

## 📐 Diseño Modular

El proyecto sigue una arquitectura modular con separación clara de responsabilidades:

### 1️⃣ Capa de Presentación (Frontend)
- **Responsabilidad**: Interfaz de usuario
- **Tecnologías**: HTML, CSS, JavaScript, Bootstrap
- **Archivos**: `templates/`, `static/`

### 2️⃣ Capa de Aplicación (Backend)
- **Responsabilidad**: API REST, manejo de requests
- **Tecnologías**: Flask, Werkzeug
- **Archivos**: `app.py`

### 3️⃣ Capa de Lógica de Negocio (Pipeline)
- **Responsabilidad**: Orquestación del flujo de procesamiento
- **Tecnologías**: Python
- **Archivos**: `app/pipeline/processor.py`

### 4️⃣ Capa de Modelos (ML Models)
- **Responsabilidad**: Inferencia de modelos de ML
- **Tecnologías**: PyTorch, Grounding DINO, SAM2
- **Archivos**: `app/models/`

### 5️⃣ Capa de Visualización
- **Responsabilidad**: Renderizado de resultados
- **Tecnologías**: OpenCV, Supervision
- **Archivos**: `app/visualization/`

### 6️⃣ Capa de Configuración
- **Responsabilidad**: Configuración centralizada
- **Archivos**: `app/core/config.py`

## 🔄 Extensibilidad

El diseño permite fácil extensión para:

- ✅ **Video processing**: SAM2 ya soporta tracking temporal
- ✅ **Batch processing**: Procesar múltiples imágenes
- ✅ **API REST**: Separar frontend del backend
- ✅ **Custom models**: Reemplazar Grounding DINO o SAM2
- ✅ **Database integration**: Guardar resultados en BD
- ✅ **User authentication**: Añadir login/registro

## 📝 Convenciones de Código

- **Módulos**: PascalCase para clases, snake_case para funciones/variables
- **Docstrings**: Formato Google para documentación
- **Type hints**: Usado en funciones principales
- **Imports**: Agrupados (stdlib, third-party, local)

---

**Última actualización**: 2025-12-15
