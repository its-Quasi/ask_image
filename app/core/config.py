"""
Configuration module for the detection and segmentation pipeline.
Centralizes all paths, thresholds, and model configurations.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Model paths
GROUNDING_DINO_CONFIG = MODELS_DIR / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = MODELS_DIR / "groundingdino_swint_ogc.pth"

SAM2_CHECKPOINT = MODELS_DIR / "sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

# Device configuration
DEVICE = "cpu"

# Detection thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# Output directories
ANNOTATED_DIR = OUTPUTS_DIR / "annotated"
SEGMENTED_DIR = OUTPUTS_DIR / "segmented"

# Create necessary directories
MODELS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
ANNOTATED_DIR.mkdir(exist_ok=True, parents=True)
SEGMENTED_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# PROMPT
MODEL = "qwen3-vl:235b-cloud"

SYSTEM_PROMPT = """
You are a vision command planner.

Your task:
Convert user questions into executable JSON plans for a vision pipeline.

Rules:
- DO NOT answer the question
- DO NOT explain
- DO NOT use markdown
- Output ONLY valid JSON
- Use ONLY allowed intents and fields
- If something is not mentioned, use null
- ALWAYS generate a dino_prompt field with descriptive phrases for object detection

Allowed intents:
- detect
- count
- compare_count
- exists

Allowed comparison operators:
- ">"
- "<"
- "=="

Output schemas:

1) detect / count / exists
{
  "intent": "detect | count | exists",
  "dino_prompt": "descriptive phrase for detection ending with period",
  "objects": [string],
  "attributes": {
    "color": [string] | null
  }
}

Example: "How many red cars?" -> {"intent": "count", "dino_prompt": "red car. cat.", "objects": ["car", "cat"], "attributes": {"color": ["red"]}}

2) compare_count
{
  "intent": "compare_count",
  "dino_prompt": "phrase1. phrase2.",
  "left": {
    "objects": [string],
    "attributes": {
      "color": [string] | null
    }
  },
  "right": {
    "objects": [string],
    "attributes": {
      "color": [string] | null
    }
  },
  "operator": ">" | "<" | "=="
}

Example: "Are there more red cars than blue bikes?" -> {"intent": "compare_count", "dino_prompt": "red car. blue bike.", "left": {"objects": ["car"], "attributes": {"color": ["red"]}}, "right": {"objects": ["bike"], "attributes": {"color": ["blue"]}}, "operator": ">"}

IMPORTANT: The dino_prompt must be a phrase optimized for Grounding DINO:
- Combine attributes + object: "red car", "scratched blue vehicle"
- End each phrase with space and period: " ."
- For multiple objects, separate with " . ": "red car . blue bike ."
"""
