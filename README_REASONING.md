# LLM + Grounding DINO Reasoning Pipeline

This document explains the natural language question-answering feature built on top of Grounding DINO and SAM2.

## 🎯 Overview

The reasoning pipeline allows you to ask natural language questions about images and receive intelligent answers. Instead of manually crafting detection prompts, you can simply ask:

- "How many red cars are there?"
- "Are there more people than dogs?"
- "Is there a blue bike in the image?"

The system uses an LLM to:
1. **Parse** your question into a structured format
2. **Generate** optimal prompts for Grounding DINO
3. **Count** and analyze detections
4. **Answer** your question in natural language

## 🏗️ Architecture

```
User Question
    ↓
┌─────────────────────────────────────┐
│ LlmModel.parse_query()              │
│ - Parses natural language           │
│ - Generates dino_prompt             │
│ - Validates with Pydantic schemas   │
└─────────────────────────────────────┘
    ↓
    {"intent": "count", "dino_prompt": "red car .", ...}
    ↓
┌─────────────────────────────────────┐
│ GroundingDINODetector.detect()      │
│ - Uses generated dino_prompt        │
│ - Returns bounding boxes            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ SAM2Segmenter.segment()             │
│ - Generates precise masks           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ LlmModel.generate_answer()          │
│ - Analyzes detection results        │
│ - Generates natural language answer │
└─────────────────────────────────────┘
    ↓
Natural Language Answer
```

## 📦 Components

### 1. **Pydantic Schemas** (`app/models/schemas.py`)

Defines structured data models for:
- **SimpleQueryResult**: detect/count/exists queries
- **CompareQueryResult**: comparison queries ("more X than Y")
- **SpatialQueryResult**: spatial relations (future)
- **DetectionResult**: DINO detection outputs
- **AnswerResponse**: Final answer format

### 2. **LLM Model** (`app/models/llm.py`)

Main class: `LlmModel`

**Methods:**
- `parse_query(question: str)`: Converts question → structured format
- `generate_answer(...)`: Converts detections → natural answer

**Features:**
- JSON extraction from LLM responses
- Pydantic validation
- Error handling with fallback answers
- Works with Ollama local LLM

### 3. **Reasoning Processor** (`app/pipeline/processor.py`)

Main class: `ReasoningImageProcessor`

**Methods:**
- `answer_question(image_path, question)`: Full pipeline for file-based processing
- `answer_question_array(image, question)`: For web interface (in-memory)

**Pipeline steps:**
1. Parse question with LLM
2. Load and prepare image
3. Execute DINO detection with generated prompt
4. Apply NMS (Non-Maximum Suppression)
5. Run SAM2 segmentation
6. Build detection results
7. Generate natural language answer
8. Create visualizations
9. Save results

### 4. **Flask Endpoint** (`app.py`)

New endpoint: `POST /ask`

**Request:**
```json
{
  "image": "<file>",
  "question": "How many red cars?",
  "box_threshold": 0.35,  // optional
  "text_threshold": 0.25,  // optional
  "apply_nms": true        // optional
}
```

**Response:**
```json
{
  "success": true,
  "question": "How many red cars?",
  "answer": "I detected 3 red cars in the image.",
  "intent": "count",
  "num_detections": 3,
  "class_names": ["car"],
  "annotated_image": "data:image/jpeg;base64,...",
  "segmented_image": "data:image/jpeg;base64,..."
}
```

## 🚀 Usage

### Prerequisites

1. **Install Ollama**:
```bash
# On Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows, download from ollama.ai
```

2. **Pull the LLM model**:
```bash
ollama pull qwen3-vl:235b-cloud
# Or use a different model by updating MODEL in config.py
```

3. **Ensure models are downloaded**:
```bash
python verify_models.py
```

### Python API

```python
from app.pipeline.processor import ReasoningImageProcessor

# Initialize processor
processor = ReasoningImageProcessor()

# Ask a question
result = processor.answer_question(
    image_path="path/to/image.jpg",
    question="How many red cars are there?",
    box_threshold=0.35,
    text_threshold=0.25,
)

# Access results
print(f"Question: {result.question}")
print(f"Answer: {result.answer}")
print(f"Intent: {result.intent}")
print(f"Detections: {result.detections.count}")
print(f"Classes: {result.detections.class_names}")
```

### CLI Example

```bash
python example_reasoning.py
```

### Web API (cURL)

```bash
curl -X POST http://localhost:5000/ask \
  -F "image=@path/to/image.jpg" \
  -F "question=How many red cars are there?"
```

### Web API (Python requests)

```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/ask",
        files={"image": f},
        data={"question": "How many red cars are there?"}
    )

result = response.json()
print(result["answer"])
```

## 🔧 Configuration

### LLM Settings (`app/core/config.py`)

```python
# Model name (must be available in Ollama)
MODEL = "qwen3-vl:235b-cloud"

# System prompt that guides LLM behavior
SYSTEM_PROMPT = """..."""
```

### Supported Intents

The system currently supports these question types:

1. **detect**: "Show me all cars"
2. **count**: "How many cars are there?"
3. **exists**: "Is there a red car?"
4. **compare_count**: "Are there more cars than bikes?"
5. **spatial_relation**: "Is the car near the building?" (future)

### Customizing the Prompt

Edit `SYSTEM_PROMPT` in `config.py` to:
- Add new intents
- Change output format
- Add more attributes (size, condition, etc.)

## 📊 Example Questions

### Count Questions
```python
"How many red cars?"
"Count the yellow chairs"
"How many people are in the image?"
```

### Existence Questions
```python
"Is there a blue bike?"
"Are there any scratched cars?"
"Is there a dog in the picture?"
```

### Comparison Questions
```python
"Are there more cars than motorcycles?"
"Is the number of people greater than bikes?"
```

### Detection Questions
```python
"Show me all red vehicles"
"Detect blue cars"
"Find all scratched objects"
```

## 🧪 Testing

Run tests with different questions:

```python
questions = [
    "How many cars?",
    "Are there any red cars?",
    "Are there more people than dogs?",
    "Is there a blue bike?",
]

processor = ReasoningImageProcessor()
for question in questions:
    result = processor.answer_question("image.jpg", question)
    print(f"Q: {question}")
    print(f"A: {result.answer}\n")
```

## 🐛 Troubleshooting

### Error: "Failed to parse query"

**Cause**: LLM didn't return valid JSON or schema is wrong

**Solutions**:
- Check Ollama is running: `ollama list`
- Try a more capable model
- Review LLM response in logs
- Adjust `SYSTEM_PROMPT`

### Error: "Could not understand question"

**Cause**: Question format not recognized by LLM

**Solutions**:
- Rephrase the question more clearly
- Add examples to `SYSTEM_PROMPT`
- Check if intent is supported

### Poor Detection Results

**Solutions**:
- Adjust `box_threshold` (lower = more detections)
- Adjust `text_threshold` (lower = more lenient matching)
- Check if object is in DINO's training data
- Verify dino_prompt generation in logs

### LLM Generates Wrong dino_prompt

**Solutions**:
- Add more examples to `SYSTEM_PROMPT`
- Use a more capable LLM model
- Check logs to see what prompt was generated
- Manually test prompt with ImageProcessor

## 📝 Code Style

All new code follows these principles:

✅ **Type hints** on all functions
✅ **Docstrings** with Args/Returns/Raises
✅ **Pydantic** for data validation
✅ **Logging** for debugging
✅ **Error handling** with fallbacks
✅ **Modular design** (single responsibility)

## 🔮 Future Enhancements

- [ ] Support for spatial relation queries
- [ ] Attribute filtering (size, condition)
- [ ] Multi-turn conversations
- [ ] Confidence scores in answers
- [ ] Visual grounding (highlight objects in answer)
- [ ] Support for video queries

## 📚 Related Files

- `app/models/llm.py` - LLM wrapper
- `app/models/schemas.py` - Pydantic data models
- `app/pipeline/processor.py` - Main reasoning pipeline
- `app/core/config.py` - Configuration and prompts
- `app.py` - Flask web server
- `example_reasoning.py` - CLI example

## 🤝 Contributing

When adding new features:

1. Add new intent to `SYSTEM_PROMPT`
2. Create Pydantic schema in `schemas.py`
3. Update `LlmModel._validate_response()`
4. Test with `example_reasoning.py`
5. Document in this README

---

**Need help?** Check logs with `logging.DEBUG` level for detailed pipeline execution traces.
