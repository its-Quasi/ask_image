"""
Example script demonstrating programmatic usage of the detection pipeline.

This script shows how to use the ImageProcessor directly without the web interface.
"""

from pathlib import Path
from app.pipeline.processor import ImageProcessor
from app.core.config import UPLOADS_DIR


def main():
    print("=" * 60)
    print("Grounding DINO + SAM2 Pipeline - Example Usage")
    print("=" * 60)

    # Initialize the processor
    print("\nInitializing pipeline...")
    processor = ImageProcessor()
    print("✓ Pipeline initialized successfully!")

    # Example 1: Process a single image
    print("\n" + "=" * 60)
    print("Example 1: Process single image")
    print("=" * 60)

    # You need to have an image file to test
    # Replace this path with your actual image
    image_path = UPLOADS_DIR / "test_image.jpg"

    if not image_path.exists():
        print(f"⚠ Warning: Image not found at {image_path}")
        print("Please place a test image in the static/uploads/ folder")
        print("and update the image_path variable in this script.")
        return

    # Define what to detect
    text_prompt = "person, car, dog, cat"

    print(f"\nImage: {image_path.name}")
    print(f"Prompt: {text_prompt}")
    print("\nProcessing...")

    # Process the image
    result = processor.process_image(
        image_path=str(image_path),
        text_prompt=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
        apply_nms=True,
        nms_threshold=0.8,
    )

    # Display results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Status: {result['message']}")
    print(f"Objects detected: {result['num_detections']}")

    if result['num_detections'] > 0:
        print(f"\nDetected classes: {', '.join(result['class_names'])}")
        print(f"Annotated image saved: {result['annotated_path']}")
        print(f"Segmented image saved: {result['segmented_path']}")

        # Print individual detections
        print("\nDetection details:")
        for i, (conf, class_id) in enumerate(zip(
            result['detections'].confidence,
            result['detections'].class_id
        )):
            class_name = result['class_names'][class_id]
            print(f"  #{i+1}: {class_name} (confidence: {conf:.2%})")
    else:
        print("\nNo objects detected. Try:")
        print("  - Lowering the box_threshold")
        print("  - Using more general keywords")
        print("  - Checking if the objects are clearly visible")

    # Example 2: Different prompts on the same image
    print("\n" + "=" * 60)
    print("Example 2: Try different prompts")
    print("=" * 60)

    prompts_to_try = [
        "person, car",
        "dog, cat, bird",
        "bottle, cup, laptop",
        "tree, building, sky",
    ]

    for prompt in prompts_to_try:
        print(f"\nTrying prompt: '{prompt}'")

        result = processor.process_image(
            image_path=str(image_path),
            text_prompt=prompt,
            box_threshold=0.3,  # Slightly lower threshold
        )

        if result['num_detections'] > 0:
            detected_classes = [
                result['class_names'][cid]
                for cid in result['detections'].class_id
            ]
            print(f"  ✓ Found {result['num_detections']} objects: {', '.join(set(detected_classes))}")
        else:
            print(f"  ✗ No objects found")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
