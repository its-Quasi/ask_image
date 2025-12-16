"""
SAM2 segmentation module.
Handles precise segmentation of detected objects using bounding boxes.
"""

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision import Detections

from app.core.config import DEVICE, SAM2_CHECKPOINT, SAM2_CONFIG


class SAM2Segmenter:
    """
    Wrapper for SAM2 (Segment Anything Model v2).

    SAM2 performs precise segmentation based on bounding box prompts.
    It does not understand text or classify objects - it only refines spatial regions.
    """

    def __init__(
        self,
        config: str = None,
        checkpoint_path: str = None,
        device: str = None,
    ):
        """
        Initialize the SAM2 segmenter.

        Args:
            config: SAM2 model configuration name
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.config = config or SAM2_CONFIG
        self.checkpoint_path = checkpoint_path or str(SAM2_CHECKPOINT)
        self.device = device or DEVICE

        # Build SAM2 model
        sam2_model = build_sam2(
            self.config,
            self.checkpoint_path,
            device=self.device,
        )

        # Create predictor
        self.predictor = SAM2ImagePredictor(sam2_model)

    def segment(
        self,
        image: np.ndarray,
        detections: Detections,
        multimask_output: bool = True,
    ) -> Detections:
        """
        Segment objects in an image based on detected bounding boxes.

        Args:
            image: Input image (RGB format, numpy array)
            detections: Detection results containing bounding boxes
            multimask_output: Whether to generate multiple mask predictions per box

        Returns:
            Updated Detections object with segmentation masks added
        """
        if len(detections.xyxy) == 0:
            return detections

        # Set image for SAM2
        self.predictor.set_image(image)

        masks = []

        # Process each bounding box
        for box in detections.xyxy:
            # SAM2 expects box in xyxy format
            mask, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=multimask_output,
            )

            # Select best mask (highest score)
            best_mask_idx = np.argmax(scores)
            masks.append(mask[best_mask_idx])

        # Update detections with masks
        detections.mask = np.array(masks)

        return detections

    def segment_with_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> tuple:
        """
        Segment using point prompts (for future extensions).

        Args:
            image: Input image
            points: Point coordinates
            labels: Point labels (1=foreground, 0=background)

        Returns:
            Tuple of (masks, scores, logits)
        """
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        return masks, scores, logits
