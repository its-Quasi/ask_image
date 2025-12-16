"""
Visualization module for annotations and segmentation masks.
Handles rendering of detection and segmentation results on images.
"""

import numpy as np
import supervision as sv
from supervision import Detections


class ImageAnnotator:
    """
    Handles visualization of detection and segmentation results.

    Provides methods to annotate images with bounding boxes, masks, and labels.
    """

    def __init__(self):
        """Initialize annotators."""
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

    def upload_classes(prompt):
        pass

    def annotate_boxes(
        self,
        image: np.ndarray,
        detections: Detections,
        labels: list = None,
    ) -> np.ndarray:
        """
        Annotate image with bounding boxes.

        Args:
            image: Input image (BGR format for OpenCV)
            detections: Detection results
            labels: Optional list of label strings for each detection

        Returns:
            Annotated image
        """

        if len(detections.xyxy) == 0:
            return image

        annotated = self.box_annotator.annotate(
            scene=image.copy(), detections=detections, labels=labels
        )

        return annotated

    def annotate_masks(
        self,
        image: np.ndarray,
        detections: Detections,
        labels: list = None,
    ) -> np.ndarray:
        """
        Annotate image with segmentation masks and bounding boxes.

        Args:
            image: Input image (BGR format for OpenCV)
            detections: Detection results with masks
            labels: Optional list of label strings for each detection

        Returns:
            Annotated image with masks and boxes
        """

        self.box_annotator.annotate()
        if detections.mask is None:
            return self.annotate_boxes(image, detections, labels)

        annotated = image.copy()

        # Draw masks first
        annotated = self.mask_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        # Then draw boxes on top
        annotated = self.box_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )
        
        return annotated

    @staticmethod
    def create_labels(detections: Detections, class_names: list) -> list:
        """
        Create formatted labels for detections.

        Args:
            detections: Detection results
            class_names: List of class names corresponding to class IDs

        Returns:
            List of formatted label strings
        """
        labels = []

        for conf, class_id in zip(detections.confidence, detections.class_id):
            if class_id is not None and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = "unknown"

            label = f"{class_name} {conf:.2f}"
            labels.append(label)

        return labels
