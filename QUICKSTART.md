# ⚡ Guía de Inicio Rápido

Esta guía te permite poner en marcha el proyecto en menos de 10 minutos.

## 🚀 Instalación Rápida (3 pasos)

### 1️⃣ Instalar Dependencias

```bash
# Dar permisos de ejecución
chmod +x install_dependencies.sh

# Ejecutar instalación
./install_dependencies.sh
```

Esto instalará:
- PyTorch (CPU version)
- Flask
- Supervision
- Grounding DINO
- SAM2
- Otras dependencias

**Tiempo estimado**: 5-10 minutos (dependiendo de tu conexión)

### 2️⃣ Descargar Modelos

```bash
# Dar permisos de ejecución
chmod +x download_models.sh

# Ejecutar descarga
./download_models.sh
```

Descargará automáticamente:
- Grounding DINO checkpoint (~700 MB)
- Grounding DINO config (~5 KB)
- SAM2 checkpoint (~900 MB)

**Tiempo estimado**: 3-5 minutos (dependiendo de tu conexión)

### 3️⃣ Verificar y Ejecutar

```bash
# Verificar que todo está instalado correctamente
python verify_models.py

# Si todo está bien, iniciar la aplicación
python app.py
```

Abre tu navegador en: **http://localhost:5000**

## 🎯 Uso Básico

1. **Carga una imagen**: Haz clic en "Selecciona una imagen"
2. **Escribe tu prompt**: Por ejemplo: `person, car, dog`
3. **Procesa**: Haz clic en "Procesar Imagen"
4. **Ve los resultados**: Verás las detecciones y segmentaciones

## 📝 Ejemplos de Prompts

```text
Básico:
  person, car, dog, cat

Colores:
  red apple, green apple

Específico:
  person wearing helmet, motorcycle

Múltiples objetos:
  laptop, mouse, keyboard, monitor
```

## ⚙️ Configuración GPU (Opcional)

Si tienes GPU NVIDIA:

1. Instala PyTorch con soporte CUDA:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. Actualiza `app/core/config.py`:
```python
DEVICE = "cuda"  # Cambiar de "cpu" a "cuda"
```

## 🐛 Solución Rápida de Problemas

### No encuentra los modelos
```bash
# Verifica que están en la carpeta correcta
ls -lh models/

# Deberías ver:
# groundingdino_swint_ogc.pth
# GroundingDINO_SwinT_OGC.py
# sam2_hiera_large.pt
```

### Error al importar módulos
```bash
# Reinstala las dependencias
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Puerto 5000 ocupado
```bash
# Edita app.py y cambia el puerto:
app.run(debug=True, host='0.0.0.0', port=8080)  # Usa 8080 en vez de 5000
```

## 📚 Siguiente Paso

- Lee el [README.md](README.md) completo para más detalles
- Revisa [MODELS_SETUP.md](MODELS_SETUP.md) para opciones avanzadas de modelos
- Mira [example_usage.py](example_usage.py) para uso programático

## ❓ ¿Necesitas Ayuda?

1. Revisa el README.md completo
2. Ejecuta `python verify_models.py` para diagnosticar problemas
3. Abre un issue en el repositorio

---

**¡Listo! Ahora puedes empezar a detectar y segmentar objetos con texto libre.**
