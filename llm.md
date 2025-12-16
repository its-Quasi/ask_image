# Solicitud de Refactorización Técnica: Pipeline LLM + Grounding DINO

## 1. Contexto del Sistema
Estoy desarrollando una aplicación de respuesta a preguntas sobre imágenes. El flujo actual es el siguiente:
1. El usuario envía un `text_prompt`.
2. El archivo `llm.py` procesa la consulta y devuelve un JSON.
3. **Problema:** La salida actual (JSON) no es compatible con la entrada que requiere **Grounding DINO** (un modelo open-set que espera frases de texto separadas por puntos para detectar objetos).

## 2. Archivos a Modificar
- `/home/quasi/Documents/work/ask_images/app/models/llm.py`
- `/home/quasi/Documents/work/ask_images/app/core/config.py`
- `/home/quasi/Documents/work/ask_images/app/pipeline/processor.py`

## 3. Requerimientos de Implementación

### A. Mejora del Esquema y Prompt de Sistema (config.py)
Actualiza el `SYSTEM_PROMPT` para que el modelo no solo extraiga el JSON, sino que genere una cadena optimizada para detección.
- Debe transformar atributos y objetos en frases descriptivas. 
- *Ejemplo:* Si el usuario pregunta "¿Cuántas sillas rojas hay?", el LLM debe proveer un campo `dino_prompt` con el valor `"red chair ."` (formato de frase de Grounding DINO).

### B. Lógica de Parseo y Pipeline (llm.py y processor.py)
Implementa una función de procesamiento adicional que realice lo siguiente:
1. **Validación:** Asegurar que la respuesta del LLM cumple con el esquema esperado usando `Pydantic` o manejo de errores robusto.
2. **Preparación para DINO:** Extraer el `class_names` o `dino_prompt` de la respuesta del LLM.
3. **Consumo de Detecciones:** Una vez que DINO devuelva el objeto de detecciones (bounding boxes, labels, scores), el pipeline debe ser capaz de procesar esta información.

### C. Generación de Respuesta Final (Cierre del Loop)
Modifica el `processor.py` para que, tras obtener las detecciones de DINO, se realice una **segunda llamada al LLM** (o se use una lógica de post-procesamiento) que combine:
- La pregunta original.
- El conteo/análisis de las detecciones reales de DINO.
- **Resultado esperado:** Una respuesta natural como: "He detectado 3 sillas rojas en la imagen según tu solicitud".

## 4. Ejemplo de Estructura Deseada para el Parser

Estos schemas se encuentran en: /home/quasi/Documents/work/ask_images/app/core/config.py , en la variable SYSTEM_PROMPT

El modelo debe ser capaz de transformar esto:
```python
# Entrada
text_prompt: "check if there are any scratched blue cars"

# Salida del LLM (Intermedia)
{
  "intent": "detection",
  "dino_prompt": "scratched blue car .",
  "objects": ["car"],
  "attributes": {"color": "blue", "condition": "scratched"}
}