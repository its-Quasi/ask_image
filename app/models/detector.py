"""
Grounding DINO detection module.
Handles text-guided object detection using open-vocabulary prompts.
"""

import numpy as np
import torch
from groundingdino.util.inference import Model
from supervision import Detections

from app.core.config import (
    DEVICE,
    GROUNDING_DINO_CONFIG,
    GROUNDING_DINO_CHECKPOINT,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
)


class GroundingDINODetector:
    """
    Wrapper for Grounding DINO model that performs text-guided object detection.

    This class enables open-vocabulary detection where the user can specify
    what to detect using natural language prompts.
    """

    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = None,
    ):
        """
        Initialize the Grounding DINO detector.

        Args:
            config_path: Path to model config file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.config_path = config_path or str(GROUNDING_DINO_CONFIG)
        self.checkpoint_path = checkpoint_path or str(GROUNDING_DINO_CHECKPOINT)
        self.device = device or DEVICE

        # Force CPU if needed
        if self.device == "cpu":
            torch.cuda.is_available = lambda: False

        self.model = Model(
            model_config_path=self.config_path,
            model_checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> Detections:
        """
        Detect objects in an image based on a text prompt.

        Args:
            image: Input image (RGB format, numpy array)
            text_prompt: Text description of objects to detect (e.g., "person, car, dog")
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            Detections object containing bounding boxes, confidence scores, and class IDs
        """
        # Parse classes from text prompt
        classes = [cls.strip().lower() for cls in text_prompt.split(".") if cls.strip()]

        # Run detection
        detections = self.model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        return detections

    def detect_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> Detections:
        """
        Detect objects using free-form caption instead of comma-separated classes.

        Args:
            image: Input image (RGB format)
            caption: Free-form text description
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            Detections object
        """
        detections = self.model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        return detections
