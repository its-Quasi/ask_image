"""
Example usage of ReasoningImageProcessor for natural language question answering.

This script demonstrates how to use the LLM + DINO pipeline to answer
questions about images in natural language.
"""

import logging
from pathlib import Path
from app.pipeline.processor import ReasoningImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run example questions on images."""

    # Initialize the reasoning processor
    logger.info("Initializing ReasoningImageProcessor...")
    processor = ReasoningImageProcessor()

    # Example questions to test
    examples = [
        {
            "image": "path/to/your/image.jpg",  # Replace with actual image path
            "questions": [
                "How many cars are in the image?",
                "Are there any red cars?",
                "How many people are there?",
                "Are there more cars than motorcycles?",
            ]
        }
    ]

    for example in examples:
        image_path = example["image"]

        # Check if image exists
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}. Skipping...")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing image: {image_path}")
        logger.info(f"{'='*60}\n")

        for question in example["questions"]:
            logger.info(f"Question: {question}")

            try:
                # Process the question
                result = processor.answer_question(
                    image_path=image_path,
                    question=question,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    apply_nms=True,
                )

                # Display results
                logger.info(f"Intent: {result.intent}")
                logger.info(f"Answer: {result.answer}")
                logger.info(f"Detections: {result.detections.count} object(s)")
                logger.info(f"Classes: {', '.join(result.detections.class_names)}\n")

            except Exception as e:
                logger.error(f"Error processing question: {e}\n")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║  LLM + Grounding DINO Reasoning Pipeline Example          ║
╚════════════════════════════════════════════════════════════╝

This example demonstrates natural language question answering
about images using LLM-guided object detection.

Examples of questions you can ask:
• "How many red cars are there?"
• "Are there more people than dogs?"
• "Is there a blue car in the image?"
• "Count the yellow chairs"

Make sure to:
1. Update the image path in the script
2. Have Ollama running with the configured model
3. Have the DINO and SAM2 models downloaded

Press Ctrl+C to stop.
    """)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
