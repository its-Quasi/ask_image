"""
Flask web application for image detection and segmentation.
Provides a web interface for uploading images and detecting objects using text prompts.
"""

import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.pipeline.processor import ImageProcessor
from app.core.config import UPLOADS_DIR, is_allowed_file, BASE_DIR

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = str(UPLOADS_DIR)

# Initialize processor (lazy loading to avoid model loading on import)
processor = None


def get_processor():
    """Get or initialize the image processor."""
    global processor
    if processor is None:
        processor = ImageProcessor()
    return processor


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
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
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided'
            }), 400

        if 'prompt' not in request.form:
            return jsonify({
                'success': False,
                'message': 'No text prompt provided'
            }), 400

        file = request.files['image']
        text_prompt = request.form['prompt'].strip()

        # Validate file
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400

        if not is_allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Allowed: jpg, jpeg, png, bmp, webp'
            }), 400

        # Validate prompt
        if not text_prompt:
            return jsonify({
                'success': False,
                'message': 'Text prompt cannot be empty'
            }), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(filepath))

        # Get thresholds from request (optional)
        box_threshold = float(request.form.get('box_threshold', 0.35))
        text_threshold = float(request.form.get('text_threshold', 0.25))
        apply_nms = request.form.get('apply_nms', 'true').lower() == 'true'

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
        if result['num_detections'] == 0:
            return jsonify({
                'success': True,
                'message': 'No objects detected. Try adjusting thresholds or using different keywords.',
                'num_detections': 0,
                'annotated_image': None,
                'segmented_image': None,
            })

        # Encode images to base64 for web display
        with open(result['annotated_path'], 'rb') as f:
            annotated_b64 = base64.b64encode(f.read()).decode('utf-8')

        with open(result['segmented_path'], 'rb') as f:
            segmented_b64 = base64.b64encode(f.read()).decode('utf-8')

        # Prepare detection info
        detections_info = []
        for i, (conf, class_id) in enumerate(zip(
            result['detections'].confidence,
            result['detections'].class_id
        )):
            class_name = result['class_names'][class_id] if class_id < len(result['class_names']) else 'unknown'
            detections_info.append({
                'id': i + 1,
                'class': class_name,
                'confidence': f"{conf:.2%}",
            })

        return jsonify({
            'success': True,
            'message': f'Successfully detected {result["num_detections"]} object(s)',
            'num_detections': result['num_detections'],
            'annotated_image': f'data:image/jpeg;base64,{annotated_b64}',
            'segmented_image': f'data:image/jpeg;base64,{segmented_b64}',
            'detections': detections_info,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': processor is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Grounding DINO + SAM2 Web Application")
    print("=" * 60)
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📂 Upload folder: {UPLOADS_DIR}")
    print("=" * 60)
    print("🌐 Open your browser and navigate to: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
