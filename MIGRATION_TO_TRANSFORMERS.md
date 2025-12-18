# Migración de groundingdino-py a Transformers

## Resumen

Este proyecto ahora soporta **dos implementaciones** de Grounding DINO:

1. **`detector.py`** - Implementación antigua usando `groundingdino-py` (deprecated, incompatible con supervision>=0.27.0)
2. **`detector_t.py`** - Implementación nueva usando HuggingFace Transformers (recomendada)

## Por qué migrar

- ✅ Compatible con `supervision==0.27.0` (versión más reciente)
- ✅ Mantenido oficialmente por HuggingFace
- ✅ API más estable y moderna
- ✅ Mejor rendimiento
- ✅ Fácil de instalar (sin necesidad de compilar extensiones C++)

## Cambios necesarios

### 1. Actualizar imports en `processor.py`

**Antes:**
```python
from app.models.detector import GroundingDINODetector
```

**Después:**
```python
from app.models.detector_t import GroundingDINODetector
```

### 2. Actualizar dependencias

**Instalar nueva versión de supervision:**
```bash
pip install supervision==0.27.0
```

**Desinstalar groundingdino-py (opcional):**
```bash
pip uninstall groundingdino-py
```

### 3. Instalar dependencias faltantes

Si no tienes transformers instalado:
```bash
pip install transformers
```

## API Compatible

La nueva implementación mantiene **100% de compatibilidad** con la API anterior:

### Inicialización

**Antes (detector.py):**
```python
detector = GroundingDINODetector(
    config_path="/path/to/config.py",
    checkpoint_path="/path/to/weights.pth",
    device="cuda"
)
```

**Después (detector_t.py):**
```python
# Opción 1: Usar modelo por defecto (grounding-dino-tiny)
detector = GroundingDINODetector(device="cuda")

# Opción 2: Especificar modelo de HuggingFace
detector = GroundingDINODetector(
    model_id="IDEA-Research/grounding-dino-base",  # o "grounding-dino-tiny"
    device="cuda"
)

# Los parámetros config_path y checkpoint_path son ignorados (para compatibilidad)
```

### Detección con clases

Ambas implementaciones aceptan el mismo formato:

```python
# Funciona en ambas
detections = detector.detect(
    image=image_rgb,
    classes=["person", "car", "dog"],  # Lista
    box_threshold=0.35,
    text_threshold=0.25,
)

# También funciona
detections = detector.detect(
    image=image_rgb,
    classes="person. car. dog.",  # String
    box_threshold=0.35,
    text_threshold=0.25,
)
```

### Detección con caption

```python
# Funciona en ambas
detections = detector.detect_with_caption(
    image=image_rgb,
    caption="a red car parked near a tree",
    box_threshold=0.35,
    text_threshold=0.25,
)
```

## Modelos disponibles en HuggingFace

- `IDEA-Research/grounding-dino-tiny` (recomendado para velocidad)
- `IDEA-Research/grounding-dino-base` (mejor precisión)

## Diferencias internas

| Aspecto | detector.py | detector_t.py |
|---------|------------|---------------|
| Biblioteca | groundingdino-py | transformers |
| Supervision | Requiere ==0.6.0 | Compatible con >=0.27.0 |
| Instalación | Requiere compilación | Instalación directa |
| Pesos | Descarga manual | Descarga automática de HF |
| Mantenimiento | Limitado | Activo (HuggingFace) |

## Verificación de compatibilidad

Para verificar que todo funciona correctamente:

```bash
# Reinstalar dependencias
pip install -r requirements.txt

# Ejecutar tests (si existen)
pytest tests/

# Probar manualmente
python -c "from app.models.detector_t import GroundingDINODetector; print('✓ Import OK')"
```

## Troubleshooting

### Error: "No module named 'groundingdino'"

✅ **Solución**: Estás usando `detector_t.py` correctamente. Ignora este error si viene de `detector.py`.

### Error: "Supervision version incompatible"

✅ **Solución**: Asegúrate de usar `detector_t.py` y tener `supervision==0.27.0` instalado.

### Primera ejecución lenta

✅ **Normal**: El modelo se descarga automáticamente de HuggingFace en la primera ejecución (~300MB).

## Rollback

Si necesitas volver a la versión antigua:

1. Cambiar import en `processor.py`:
   ```python
   from app.models.detector import GroundingDINODetector
   ```

2. Reinstalar groundingdino-py:
   ```bash
   pip install groundingdino-py==0.4.0
   pip install supervision==0.6.0
   ```

## Soporte

- HuggingFace Docs: https://huggingface.co/docs/transformers/model_doc/grounding-dino
- Supervision Docs: https://supervision.roboflow.com/
