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
NMS_THRESHOLD = 0.8

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
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
