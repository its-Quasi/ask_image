"""
Main processing pipeline that orchestrates detection and segmentation.
Integrates Grounding DINO and SAM2 for end-to-end processing.
"""

import cv2
import numpy as np
from pathlib import Path

from app.models.detector import GroundingDINODetector
from app.models.segmenter import SAM2Segmenter
from app.visualization.annotator import ImageAnnotator
from app.core.config import NMS_THRESHOLD, ANNOTATED_DIR, SEGMENTED_DIR


class ImageProcessor:
    """
    Main pipeline for processing images with detection and segmentation.

    This class orchestrates the complete workflow:
    1. Load image
    2. Detect objects with Grounding DINO (text-guided)
    3. Apply NMS to reduce overlapping detections
    4. Segment objects with SAM2
    5. Visualize and save results
    """

    def __init__(self):
        """Initialize the processing pipeline components."""
        self.detector = GroundingDINODetector()
        self.segmenter = SAM2Segmenter()
        self.annotator = ImageAnnotator()

    def process_image(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = None,
        text_threshold: float = None,
        apply_nms: bool = True,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> dict:
        """
        Process a single image through the complete pipeline.

        Args:
            image_path: Path to input image
            text_prompt: Text description of objects to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            apply_nms: Whether to apply Non-Maximum Suppression
            nms_threshold: IoU threshold for NMS

        Returns:
            Dictionary containing:
                - detections: Detections object
                - annotated_path: Path to saved annotated image
                - segmented_path: Path to saved segmented image
                - class_names: List of detected class names
                - num_detections: Number of objects detected
        """
        # Load image
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Parse class names from prompt
        class_names = [cls.strip() for cls in text_prompt.split(",")]

        # Step 1: Detection with Grounding DINO
        detections = self.detector.detect(
            image=image_rgb,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Check if any objects were detected
        if len(detections.xyxy) == 0:
            return {
                "detections": detections,
                "annotated_path": None,
                "segmented_path": None,
                "class_names": class_names,
                "num_detections": 0,
                "message": "No objects detected",
            }

        # Step 2: Apply NMS to reduce overlapping boxes
        if apply_nms and len(detections.xyxy) > 0:
            detections = detections.with_nms(threshold=nms_threshold)

        # Step 3: Segmentation with SAM2
        detections = self.segmenter.segment(
            image=image_rgb,
            detections=detections,
        )

        # Step 4: Create labels
        labels = self.annotator.create_labels(detections, class_names)

        # Step 5: Generate visualizations
        # Annotated image (boxes only)
        annotated_image = self.annotator.annotate_boxes(
            image=image_bgr,
            detections=detections,
            labels=labels,
        )

        # Segmented image (masks + boxes)
        segmented_image = self.annotator.annotate_masks(
            image=image_bgr,
            detections=detections,
            labels=labels,
        )

        # Step 6: Save results
        image_name = Path(image_path).name
        annotated_path = ANNOTATED_DIR / image_name
        segmented_path = SEGMENTED_DIR / image_name

        cv2.imwrite(str(annotated_path), annotated_image)
        cv2.imwrite(str(segmented_path), segmented_image)

        return {
            "detections": detections,
            "annotated_path": str(annotated_path),
            "segmented_path": str(segmented_path),
            "class_names": class_names,
            "num_detections": len(detections.xyxy),
            "message": "Success",
        }

    def process_image_array(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = None,
        text_threshold: float = None,
        apply_nms: bool = True,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> dict:
        """
        Process an image array (useful for web interface).

        Args:
            image: Image as numpy array (BGR format)
            text_prompt: Text description of objects to detect
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            apply_nms: Whether to apply NMS
            nms_threshold: IoU threshold for NMS

        Returns:
            Dictionary with processing results including annotated and segmented images
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_names = [cls.strip() for cls in text_prompt.split(",")]

        # Detection
        detections = self.detector.detect(
            image=image_rgb,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(detections.xyxy) == 0:
            return {
                "detections": detections,
                "annotated_image": image,
                "segmented_image": image,
                "class_names": class_names,
                "num_detections": 0,
                "message": "No objects detected",
            }

        # NMS
        if apply_nms and len(detections.xyxy) > 0:
            detections = detections.with_nms(threshold=nms_threshold)

        # Segmentation
        detections = self.segmenter.segment(image=image_rgb, detections=detections)

        # Visualization
        labels = self.annotator.create_labels(detections, class_names)
        annotated_image = self.annotator.annotate_boxes(image, detections, labels)
        segmented_image = self.annotator.annotate_masks(image, detections, labels)

        return {
            "detections": detections,
            "annotated_image": annotated_image,
            "segmented_image": segmented_image,
            "class_names": class_names,
            "num_detections": len(detections.xyxy),
            "message": "Success",
        }
