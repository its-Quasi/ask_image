"""
Flask web application for image detection and segmentation.
Provides a web interface for uploading images and detecting objects using text prompts.
"""

import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.pipeline.processor import ImageProcessor, ReasoningImageProcessor
from app.core.config import UPLOADS_DIR, is_allowed_file, BASE_DIR

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)

# Initialize processors (lazy loading to avoid model loading on import)
processor = None
reasoning_processor = None


def get_processor():
    """Get or initialize the standard image processor."""
    global processor
    if processor is None:
        processor = ImageProcessor()
    return processor


def get_reasoning_processor():
    """Get or initialize the reasoning image processor."""
    global reasoning_processor
    if reasoning_processor is None:
        reasoning_processor = ReasoningImageProcessor()
    return reasoning_processor


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    """
    Process uploaded image with text prompt.

    Expected form data:
        - image: Image file
        - prompt: Text prompt for detection

    Returns:
        JSON with:
            - success: Boolean
            - message: Status message
            - num_detections: Number of objects detected
            - annotated_image: Base64 encoded annotated image
            - segmented_image: Base64 encoded segmented image
    """
    try:
        # Validate request
        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image file provided"}), 400

        if "prompt" not in request.form:
            return jsonify(
                {"success": False, "message": "No text prompt provided"}
            ), 400

        file = request.files["image"]
        text_prompt = request.form["prompt"].strip()

        # Validate file
        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected"}), 400

        if not is_allowed_file(file.filename):
            return jsonify(
                {
                    "success": False,
                    "message": "Invalid file type. Allowed: jpg, jpeg, png, bmp, webp",
                }
            ), 400

        # Validate prompt
        if not text_prompt:
            return jsonify(
                {"success": False, "message": "Text prompt cannot be empty"}
            ), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(str(filepath))

        # Get thresholds from request (optional)
        box_threshold = float(request.form.get("box_threshold", 0.35))
        text_threshold = float(request.form.get("text_threshold", 0.25))
        apply_nms = request.form.get("apply_nms", "true").lower() == "true"

        # Process image
        proc = get_processor()

        result = proc.process_image(
            image_path=str(filepath),
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            apply_nms=apply_nms,
        )

        # Handle no detections
        if result["num_detections"] == 0:
            return jsonify(
                {
                    "success": True,
                    "message": "No objects detected. Try adjusting thresholds or using different keywords.",
                    "num_detections": 0,
                    "annotated_image": None,
                    "segmented_image": None,
                }
            )

        # Encode images to base64 for web display
        with open(result["annotated_path"], "rb") as f:
            annotated_b64 = base64.b64encode(f.read()).decode("utf-8")

        with open(result["segmented_path"], "rb") as f:
            segmented_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Prepare detection info
        detections_info = []
        for i, (conf, class_id) in enumerate(
            zip(result["detections"].confidence, result["detections"].class_id)
        ):
            class_name = (
                result["class_names"][class_id]
                if class_id < len(result["class_names"])
                else "unknown"
            )
            detections_info.append(
                {
                    "id": i + 1,
                    "class": class_name,
                    "confidence": f"{conf:.2%}",
                }
            )

        return jsonify(
            {
                "success": True,
                "message": f"Successfully detected {result['num_detections']} object(s)",
                "num_detections": result["num_detections"],
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                "segmented_image": f"data:image/jpeg;base64,{segmented_b64}",
                "detections": detections_info,
            }
        )

    except Exception as e:
        return jsonify(
            {"success": False, "message": f"Error processing image: {str(e)}"}
        ), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Process either class-based detection or question-based analysis.

    Expected form data:
        - image: Image file
        - mode: 'classes' or 'question'
        - classes: JSON array of class names (if mode='classes')
        - question: Natural language question (if mode='question')
        - box_threshold: Detection threshold (optional)
        - text_threshold: Text matching threshold (optional)
        - apply_nms: Whether to apply NMS (optional)
        - nms_threshold: NMS IoU threshold (optional)

    Returns:
        JSON with detection results and visualizations
    """
    try:
        # Validate request
        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image file provided"}), 400

        if "mode" not in request.form:
            return jsonify({"success": False, "message": "No mode specified"}), 400

        file = request.files["image"]
        mode = request.form["mode"]

        # Validate file
        if file.filename == "":
            return jsonify({"success": False, "message": "No file selected"}), 400

        if not is_allowed_file(file.filename):
            return jsonify(
                {
                    "success": False,
                    "message": "Invalid file type. Allowed: jpg, jpeg, png, bmp, webp",
                }
            ), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(str(filepath))

        # Get thresholds from request
        box_threshold = float(request.form.get("box_threshold", 0.35))
        text_threshold = float(request.form.get("text_threshold", 0.25))
        apply_nms = request.form.get("apply_nms", "true").lower() == "true"
        nms_threshold = float(request.form.get("nms_threshold", 0.45))

        # Load image
        import cv2
        import numpy as np

        image = cv2.imread(str(filepath))

        if image is None:
            return jsonify(
                {"success": False, "message": "Could not load image file"}
            ), 400

        def encode_image(img_array: np.ndarray) -> str:
            """Encode numpy array image to base64."""
            _, buffer = cv2.imencode(".jpg", img_array)
            return base64.b64encode(buffer).decode("utf-8")

        # Process based on mode
        if mode == "classes":
            # Class-based detection mode
            import json

            if "classes" not in request.form:
                return jsonify(
                    {"success": False, "message": "No classes provided"}
                ), 400

            classes_json = request.form["classes"]
            classes = json.loads(classes_json)

            if not classes or len(classes) == 0:
                return jsonify(
                    {"success": False, "message": "No classes provided"}
                ), 400

            # Get LLM model for validation
            proc = get_reasoning_processor()

            # Validate and normalize classes
            validation_result = proc.llm.analyze_classes(classes)

            if not validation_result["success"]:
                return jsonify(
                    {"success": False, "message": validation_result["error"]}
                ), 400

            # Get normalized classes and DINO prompt
            normalized_classes = validation_result["normalized_classes"]
            dino_prompt = validation_result["dino_prompt"]

            # Perform detection using the base processor
            base_proc = get_processor()

            result = base_proc.process_image_array(
                image=image,
                text_prompt=dino_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                apply_nms=apply_nms,
                nms_threshold=nms_threshold,
            )

            # Encode images
            annotated_b64 = encode_image(result["annotated_image"])
            segmented_b64 = encode_image(result["segmented_image"])

            # Build response message
            message = f"Se detectaron {result['num_detections']} objeto(s)"
            if (
                validation_result.get("discarded")
                and len(validation_result["discarded"]) > 0
            ):
                message += (
                    f". Clases descartadas: {', '.join(validation_result['discarded'])}"
                )

            return jsonify(
                {
                    "success": True,
                    "message": message,
                    "num_detections": result["num_detections"],
                    "class_names": normalized_classes,
                    "detected_classes": result.get("detected_classes"),
                    "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                    "segmented_image": f"data:image/jpeg;base64,{segmented_b64}",
                    "answer": message,
                }
            )

        elif mode == "question":
            # Question-based analysis mode
            if "question" not in request.form:
                return jsonify(
                    {"success": False, "message": "No question provided"}
                ), 400

            question = request.form["question"].strip()

            if not question:
                return jsonify(
                    {"success": False, "message": "Question cannot be empty"}
                ), 400

            # Get reasoning processor
            proc = get_reasoning_processor()

            # Validate question
            validation_result = proc.llm.validate_question(question)

            if not validation_result["valid"]:
                return jsonify(
                    {"success": False, "message": validation_result["error"]}
                ), 400

            # Process question
            result = proc.answer_question_array(
                image=image,
                image_path=str(filepath),
                question=question,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                apply_nms=apply_nms,
                nms_threshold=nms_threshold,
            )

            # Check if processing failed
            if not result.get("success", False):
                return jsonify(
                    {
                        "success": False,
                        "message": result.get("error", "Unknown error occurred"),
                        "question": question,
                    }
                ), 400

            # Encode images
            annotated_b64 = encode_image(result["annotated_image"])
            segmented_b64 = encode_image(result["segmented_image"])

            return jsonify(
                {
                    "success": True,
                    "question": result["question"],
                    "answer": result["answer"],
                    "intent": result["intent"],
                    "num_detections": result["num_detections"],
                    "class_names": result["class_names"],
                    "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                    "segmented_image": f"data:image/jpeg;base64,{segmented_b64}",
                }
            )

        else:
            return jsonify(
                {
                    "success": False,
                    "message": f'Invalid mode: {mode}. Must be "classes" or "question"',
                }
            ), 400

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(
            {"success": False, "message": f"Error processing request: {str(e)}"}
        ), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "processor_loaded": processor is not None,
            "reasoning_processor_loaded": reasoning_processor is not None,
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Grounding DINO + SAM2 Web Application")
    print("=" * 60)
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📂 Upload folder: {UPLOADS_DIR}")
    print("=" * 60)
    print("🌐 Open your browser and navigate to: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host="0.0.0.0", port=5000)
