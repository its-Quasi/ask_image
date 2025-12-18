"""
Grounding DINO detection module using HuggingFace Transformers.
Handles text-guided object detection using open-vocabulary prompts.
Compatible with supervision>=0.27.0
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, GroundingDinoProcessor
from supervision import Detections

from app.core.config import (
    DEVICE,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
)


class GroundingDINODetector:
    """
    Wrapper for Grounding DINO model using HuggingFace Transformers.

    This class enables open-vocabulary detection where the user can specify
    what to detect using natural language prompts.

    Compatible with supervision>=0.27.0 and uses the official Transformers implementation.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = None,
        config_path: str = None,  # For API compatibility, not used
        checkpoint_path: str = None,  # For API compatibility, not used
    ):
        """
        Initialize the Grounding DINO detector using Transformers.

        Args:
            model_id: HuggingFace model ID (default: grounding-dino-tiny)
            device: Device to run inference on ('cpu' or 'cuda')
            config_path: (Ignored) For API compatibility with old detector
            checkpoint_path: (Ignored) For API compatibility with old detector
        """
        self.device = device or DEVICE
        self.model_id = model_id

        # Load processor and model from HuggingFace
        print(f"Loading Grounding DINO from Transformers: {model_id}")
        self.processor: GroundingDinoProcessor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        # # Move model to device
        # self.model.to(self.device)
        # self.model.eval()

    def detect(
        self,
        image: np.ndarray,
        classes: list[str] | str = None,
        text_prompt: str = None,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> Detections:
        """
        Detect objects in an image based on a text prompt.

        Args:
            image: Input image (RGB format, numpy array)
            classes: List of class names or comma-separated string (e.g., ["person", "car"] or "person, car")
            text_prompt: Alternative to classes - direct text prompt (for backwards compatibility)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching (used as overall threshold)

        Returns:
            Detections object containing bounding boxes, confidence scores, and class IDs
        """
        # Support both 'classes' and 'text_prompt' parameters for compatibility
        if text_prompt is not None:
            prompt = text_prompt
        elif classes is not None:
            # Convert classes to text prompt
            if isinstance(classes, list):
                # Handle list format: ["person", "car"] or ["person.", "car."]
                prompt = ". ".join([cls.strip().rstrip(".") for cls in classes]) + "."
            else:
                # Handle string format
                prompt = classes
        else:
            raise ValueError("Either 'classes' or 'text_prompt' must be provided")

        return self._run_inference(image, prompt, box_threshold, text_threshold)

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
        return self._run_inference(image, caption, box_threshold, text_threshold)

    def _run_inference(
        self,
        image: np.ndarray,
        text: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Detections:
        """
        Internal method to run Grounding DINO inference.

        Args:
            image: RGB numpy array
            text: Text prompt
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold (used as overall threshold)

        Returns:
            supervision.Detections object
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Prepare inputs
        inputs = self.processor(images=pil_image, text=text, return_tensors="pt").to(
            self.device
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=torch.tensor([pil_image.size[::-1]]).to(self.device),
            threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # results = processor.post_process_grounded_object_detection(
        #     outputs,
        #     inputs.input_ids,
        #     box_threshold=0.4,
        #     text_threshold=0.3,
        #     target_sizes=[image.size[::-1]],
        # )

        print(results)
        # Convert to supervision Detections format
        detections = self._to_supervision_detections(results)

        return detections

    def _to_supervision_detections(self, results: dict) -> Detections:
        """
        Convert Transformers output to supervision.Detections format.

        Args:
            results: Dictionary with 'boxes', 'scores', 'labels' from post_process

        Returns:
            supervision.Detections object
        """
        # Extract boxes, scores, and labels
        boxes = results["boxes"].cpu().numpy()  # [N, 4] in xyxy format
        scores = results["scores"].cpu().numpy()  # [N]

        # Handle labels - transformers v4.51.0+ returns integer ids in 'labels'
        # and text labels in 'text_labels' (if available)
        if "text_labels" in results:
            labels = results["text_labels"]  # List of strings
        else:
            # Fallback for older versions
            labels = results.get("labels", [])
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()

        # Create class_id array
        # Map each unique label to an integer ID
        unique_labels = list(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        class_ids = np.array([label_to_id[label] for label in labels])

        # Create supervision Detections object
        # supervision.Detections expects:
        # - xyxy: bounding boxes
        # - confidence: scores
        # - class_id: class identifiers
        # - data: optional dict with additional info
        detections = Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids,
            data={
                "class_name": np.array(labels),  # Store text labels
            },
        )

        return detections
