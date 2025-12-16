# 📦 Guía de Descarga e Instalación de Modelos

Esta guía te ayudará a descargar y configurar los modelos necesarios para el pipeline.

## 📋 Modelos Requeridos

| Modelo | Archivo | Tamaño | Propósito |
|--------|---------|--------|-----------|
| Grounding DINO | `groundingdino_swint_ogc.pth` | ~700 MB | Detección guiada por texto |
| Grounding DINO Config | `GroundingDINO_SwinT_OGC.py` | ~5 KB | Configuración del modelo |
| SAM2 | `sam2_hiera_large.pt` | ~900 MB | Segmentación precisa |

**Espacio total requerido**: ~1.6 GB

## 🚀 Descarga Rápida (Script Automático)

Guarda este script como `download_models.sh` y ejecútalo:

```bash
#!/bin/bash

# Crear directorio de modelos
mkdir -p models
cd models

# Descargar Grounding DINO checkpoint
echo "Descargando Grounding DINO checkpoint..."
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Descargar Grounding DINO config
echo "Descargando Grounding DINO config..."
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# Descargar SAM2 checkpoint
echo "Descargando SAM2 checkpoint..."
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

echo "¡Descarga completada!"
cd ..
```

Ejecuta:
```bash
chmod +x download_models.sh
./download_models.sh
```

## 📥 Descarga Manual

### 1. Grounding DINO

#### Checkpoint (groundingdino_swint_ogc.pth)

**Opción A: Desde GitHub Releases**
1. Ve a: https://github.com/IDEA-Research/GroundingDINO/releases
2. Descarga: `groundingdino_swint_ogc.pth` (~700 MB)
3. Mueve el archivo a: `models/groundingdino_swint_ogc.pth`

**Opción B: Usando wget**
```bash
cd models
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Opción C: Desde Hugging Face**
```bash
# Requiere huggingface-hub instalado
pip install huggingface-hub
```

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="ShilongLiu/GroundingDINO",
    filename="groundingdino_swint_ogc.pth",
    local_dir="./models"
)
```

#### Config (GroundingDINO_SwinT_OGC.py)

**Descarga el archivo de configuración:**
```bash
cd models
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

### 2. SAM2 (Segment Anything Model 2)

#### Checkpoint (sam2_hiera_large.pt)

**Opción A: Descarga directa**
```bash
cd models
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

**Opción B: Desde GitHub**
1. Ve a: https://github.com/facebookresearch/segment-anything-2#model-checkpoints
2. Elige el modelo que prefieras:

| Modelo | Tamaño | Archivo |
|--------|--------|---------|
| SAM2 Hiera Large (recomendado) | ~900 MB | `sam2_hiera_large.pt` |
| SAM2 Hiera Base+ | ~270 MB | `sam2_hiera_base_plus.pt` |
| SAM2 Hiera Small | ~180 MB | `sam2_hiera_small.pt` |
| SAM2 Hiera Tiny | ~158 MB | `sam2_hiera_tiny.pt` |

**Para modelos más pequeños** (si tienes limitaciones de memoria):
```bash
cd models
# Base+ (más ligero pero menos preciso)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Actualiza en app/core/config.py:
# SAM2_CHECKPOINT = MODELS_DIR / "sam2_hiera_base_plus.pt"
# SAM2_CONFIG = "sam2_hiera_b+.yaml"
```

## ✅ Verificación de Instalación

Después de descargar, tu carpeta `models/` debe verse así:

```
models/
├── groundingdino_swint_ogc.pth     (~700 MB)
├── GroundingDINO_SwinT_OGC.py      (~5 KB)
└── sam2_hiera_large.pt             (~900 MB)
```

### Script de Verificación

Ejecuta este script para verificar que los modelos están correctamente instalados:

```python
# verify_models.py
from pathlib import Path

models_dir = Path("models")
required_files = [
    "groundingdino_swint_ogc.pth",
    "GroundingDINO_SwinT_OGC.py",
    "sam2_hiera_large.pt"
]

print("Verificando modelos...")
print("-" * 50)

all_present = True
for filename in required_files:
    filepath = models_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"✗ {filename} - NO ENCONTRADO")
        all_present = False

print("-" * 50)
if all_present:
    print("✓ Todos los modelos están instalados correctamente!")
else:
    print("✗ Faltan algunos modelos. Por favor descárgalos.")
```

Ejecuta:
```bash
python verify_models.py
```

## 🔧 Configuración Alternativa

Si descargas modelos con nombres diferentes o quieres usar variantes, actualiza `app/core/config.py`:

```python
# app/core/config.py

# Cambiar checkpoint de Grounding DINO
GROUNDING_DINO_CHECKPOINT = MODELS_DIR / "tu_modelo_custom.pth"

# Cambiar modelo de SAM2
SAM2_CHECKPOINT = MODELS_DIR / "sam2_hiera_base_plus.pt"
SAM2_CONFIG = "sam2_hiera_b+.yaml"  # Debe coincidir con el modelo
```

## 🌐 Descarga desde China

Si estás en China y tienes problemas con GitHub/Meta CDN:

**Usa mirrors o VPN:**
```bash
# Mirror de GitHub (ejemplo)
git config --global url."https://github.com.cnpmjs.org/".insteadOf "https://github.com/"

# O usa un espejo específico para modelos
# Busca en gitee.com o equivalentes
```

## 🐛 Solución de Problemas

### Error: "Cannot download from GitHub"
- Usa un VPN
- Descarga manualmente desde navegador
- Usa Hugging Face como alternativa

### Error: "File corrupted" o "Checkpoint loading failed"
- Verifica el tamaño del archivo descargado
- Vuelve a descargar el modelo
- Verifica el hash MD5/SHA256 si está disponible

### Error: "Out of disk space"
- Los modelos necesitan ~1.6 GB
- Libera espacio en disco
- Usa modelos más pequeños (SAM2 tiny/small)

## 📊 Comparación de Modelos SAM2

| Modelo | Tamaño | RAM GPU | Velocidad | Precisión |
|--------|--------|---------|-----------|-----------|
| Tiny | 158 MB | 2 GB | Más rápido | Menor |
| Small | 180 MB | 3 GB | Rápido | Buena |
| Base+ | 270 MB | 4 GB | Medio | Muy buena |
| Large | 900 MB | 6 GB | Más lento | Mejor |

**Recomendación**:
- CPU: Usa `tiny` o `small`
- GPU con 4-6 GB: Usa `base+`
- GPU con 8+ GB: Usa `large`

## 🔗 Enlaces Útiles

- [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)
- [SAM2 Paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)

---

**¿Necesitas ayuda?** Abre un issue en el repositorio del proyecto.
