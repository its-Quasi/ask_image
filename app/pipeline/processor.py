"""
Main processing pipeline that orchestrates detection and segmentation.
Integrates Grounding DINO and SAM2 for end-to-end processing.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from supervision import Detections
from app.models.detector import GroundingDINODetector
from app.models.segmenter import SAM2Segmenter
from app.models.llm import LlmModel
from app.models.schemas import (
    SimpleQueryResult,
    CompareQueryResult,
    DetectionResult,
    AnswerResponse,
)
from app.visualization.annotator import ImageAnnotator
from app.core.config import NMS_THRESHOLD, ANNOTATED_DIR, SEGMENTED_DIR

logger = logging.getLogger(__name__)


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


class ReasoningImageProcessor:
    """
    Enhanced pipeline that integrates LLM reasoning with vision detection.

    Complete workflow:
    1. User asks natural language question
    2. LLM parses question into structured format with dino_prompt
    3. Execute detection with Grounding DINO using generated prompt
    4. Apply segmentation with SAM2
    5. LLM generates natural language answer from detection results
    """

    def __init__(self) -> None:
        """Initialize all pipeline components."""
        self.llm = LlmModel()
        self.detector = GroundingDINODetector()
        self.segmenter = SAM2Segmenter()
        self.annotator = ImageAnnotator()
        logger.info("ReasoningImageProcessor initialized")

    def answer_question(
        self,
        image_path: str,
        question: str,
        box_threshold: float = None,
        text_threshold: float = None,
        apply_nms: bool = True,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> AnswerResponse:
        """
        Process a natural language question about an image.

        Args:
            image_path: Path to input image
            question: Natural language question (e.g., "How many red cars?")
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            apply_nms: Whether to apply Non-Maximum Suppression
            nms_threshold: IoU threshold for NMS

        Returns:
            AnswerResponse with question, natural language answer, and detection details

        Raises:
            ValueError: If question cannot be parsed or image cannot be loaded
        """
        logger.info(f"Processing question: '{question}' for image: {image_path}")

        # Step 1: Parse question with LLM
        try:
            parsed_query = self.llm.parse_query(question)
            logger.debug(f"Parsed query: {parsed_query}")
        except ValueError as e:
            logger.error(f"Failed to parse question: {e}")
            raise ValueError(f"Could not understand question: {e}")

        # Step 2: Load image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Could not load image from: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Step 3: Execute detection with DINO using generated prompt
        dino_prompt = parsed_query.dino_prompt
        logger.info(f"Executing DINO with prompt: '{dino_prompt}'")

        # Extract class names from dino_prompt
        classes = [cls.strip().lower() for cls in dino_prompt.split(".") if cls.strip()]

        detections = self.detector.detect(
            image=image_rgb,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Step 4: Apply NMS if requested
        if apply_nms and len(detections.xyxy) > 0:
            detections = detections.with_nms(threshold=nms_threshold)
            logger.debug(f"Applied NMS, remaining detections: {len(detections.xyxy)}")

        # Step 5: Segmentation with SAM2
        if len(detections.xyxy) > 0:
            detections = self.segmenter.segment(image=image_rgb, detections=detections)
            logger.debug("Segmentation complete")

        # Step 6: Build detection result
        detection_result = self._build_detection_result(detections, parsed_query)

        # Step 7: Generate natural language answer
        answer = self.llm.generate_answer(
            question=question,
            intent=parsed_query.intent,
            detections=detections,
            parsed_query=parsed_query,
            class_names=classes,
        )
        logger.info(f"Generated answer: '{answer}'")

        # Step 7: Create visualizations
        annotated_image, segmented_image = self._create_visualizations(
            image_bgr=image_bgr,
            detections=detections,
            parsed_query=parsed_query,
        )

        # Step 8: Save results
        self._save_results(
            image_path=image_path,
            annotated_image=annotated_image,
            segmented_image=segmented_image,
        )

        return AnswerResponse(
            question=question,
            answer=answer,
            detections=detection_result,
            intent=parsed_query.intent,
        )

    def answer_question_array(
        self,
        image: np.ndarray,
        image_path: str,
        question: str,
        box_threshold: float = None,
        text_threshold: float = None,
        apply_nms: bool = True,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> dict[str, any]:
        """
        Process a natural language question about an image array (for web interface).

        Args:
            image: Image as numpy array (BGR format)
            question: Natural language question
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            apply_nms: Whether to apply NMS
            nms_threshold: IoU threshold for NMS

        Returns:
            Dictionary with answer, images, and detection details
        """
        logger.info(f"Processing question (array mode): '{question}'")

        # Step 1: Parse question
        try:
            parsed_query = self.llm.parse_query(question)
        except ValueError as e:
            return {
                "success": False,
                "error": f"Could not understand question: {e}",
                "question": question,
            }

        # Step 2: Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 3: Detection
        dino_input = parsed_query.dino_prompt
        classes = [cls.strip().lower() for cls in dino_input.split(".") if cls.strip()]

        detections = self.detector.detect(
            image=image_rgb,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Step 4: NMS
        # if apply_nms and len(detections.xyxy) > 0:
        #     detections = detections.with_nms(threshold=nms_threshold)

        # Step 5: Segmentation
        if len(detections.xyxy) > 0:
            detections = self.segmenter.segment(image=image_rgb, detections=detections)

        # # Step 6: Build result
        # detection_result = self._build_detection_result(detections, parsed_query)

        # Step 7: Generate answer
        answer = self.llm.generate_answer(
            question=question,
            intent=parsed_query.intent,
            detections=detections,
            parsed_query=parsed_query,
            class_names=classes,
        )

        # Step 8: Visualize
        annotated_image, segmented_image = self._create_visualizations(
            image_bgr=image,
            detections=detections,
            class_names=classes,
        )

        # Step 9: Save results
        self._save_results(
            image_path=image_path,
            annotated_image=annotated_image,
            segmented_image=segmented_image,
        )

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "intent": parsed_query.intent,
            "num_detections": len(detections.xyxy),
            "class_names": classes,
            "annotated_image": annotated_image,
            "segmented_image": segmented_image,
        }

    def _build_detection_result(
        self,
        detections: Detections,
        parsed_query: SimpleQueryResult | CompareQueryResult,
    ) -> DetectionResult:
        """
        Build structured detection result from DINO detections.

        Args:
            detections: supervision.Detections object
            parsed_query: Parsed query with object information

        Returns:
            DetectionResult schema
        """
        count = len(detections.xyxy) if detections is not None else 0

        # Extract class names from parsed query
        if isinstance(parsed_query, SimpleQueryResult):
            class_names = parsed_query.objects
        elif isinstance(parsed_query, CompareQueryResult):
            class_names = parsed_query.left.objects + parsed_query.right.objects
        else:
            class_names = []

        # Extract confidences if available
        confidences = (
            detections.confidence.tolist()
            if detections is not None
            and hasattr(detections, "confidence")
            and detections.confidence is not None
            else []
        )

        # Extract bounding boxes
        bounding_boxes = (
            detections.xyxy.tolist()
            if detections is not None
            and hasattr(detections, "xyxy")
            and detections.xyxy is not None
            else []
        )

        return DetectionResult(
            count=count,
            class_names=class_names,
            confidences=confidences,
            bounding_boxes=bounding_boxes,
        )

    def _create_visualizations(
        self,
        image_bgr: np.ndarray,
        detections: Detections,
        class_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create annotated and segmented visualizations.

        Args:
            image_bgr: Original image in BGR format
            detections: Detection results
            parsed_query: Parsed query with object information

        Returns:
            Tuple of (annotated_image, segmented_image)
        """
        if detections is None or len(detections.xyxy) == 0:
            # No detections, return original image
            return image_bgr, image_bgr

        # Create labels
        labels = self.annotator.create_labels(detections, class_names)

        # Generate visualizations
        annotated_image = self.annotator.annotate_boxes(
            image=image_bgr,
            detections=detections,
            labels=labels,
        )

        segmented_image = self.annotator.annotate_masks(
            image=image_bgr,
            detections=detections,
            labels=labels,
        )

        return annotated_image, segmented_image

    def _save_results(
        self,
        image_path: str,
        annotated_image: np.ndarray,
        segmented_image: np.ndarray,
    ) -> dict[str, str]:
        """
        Save annotated and segmented images to disk.

        Args:
            image_path: Original image path (used for naming)
            annotated_image: Image with bounding boxes
            segmented_image: Image with segmentation masks

        Returns:
            Dictionary with saved file paths
        """
        image_name = Path(image_path).name
        annotated_path = ANNOTATED_DIR / f"reasoning_{image_name}"
        segmented_path = SEGMENTED_DIR / f"reasoning_{image_name}"

        cv2.imwrite(str(annotated_path), annotated_image)
        cv2.imwrite(str(segmented_path), segmented_image)

        logger.info(f"Saved results: {annotated_path}, {segmented_path}")

        return {
            "annotated_path": str(annotated_path),
            "segmented_path": str(segmented_path),
        }
