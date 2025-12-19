// Main JavaScript for Grounding DINO + SAM2 Web Interface

document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const processBtn = document.getElementById('processBtn');
    const progressContainer = document.getElementById('progressContainer');
    const alertContainer = document.getElementById('alertContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const outputImagesContainer = document.getElementById('outputImagesContainer');
    const annotatedImage = document.getElementById('annotatedImage');
    const segmentedImage = document.getElementById('segmentedImage');

    // Input mode elements
    const modeClasses = document.getElementById('modeClasses');
    const modeQuestion = document.getElementById('modeQuestion');
    const classesContainer = document.getElementById('classesContainer');
    const questionContainer = document.getElementById('questionContainer');
    const classInput = document.getElementById('classInput');
    const addClassBtn = document.getElementById('addClassBtn');
    const classTags = document.getElementById('classTags');
    const questionInput = document.getElementById('questionInput');

    // Threshold sliders
    const boxThreshold = document.getElementById('boxThreshold');
    const textThreshold = document.getElementById('textThreshold');
    const nmsThreshold = document.getElementById('nmsThreshold');
    const boxThresholdValue = document.getElementById('boxThresholdValue');
    const textThresholdValue = document.getElementById('textThresholdValue');
    const nmsThresholdValue = document.getElementById('nmsThresholdValue');
    const applyNMS = document.getElementById('applyNMS');

    // State
    let classes = [];
    let currentMode = 'classes';

    // Update threshold display values
    boxThreshold.addEventListener('input', function () {
        boxThresholdValue.textContent = this.value;
    });

    textThreshold.addEventListener('input', function () {
        textThresholdValue.textContent = this.value;
    });

    nmsThreshold.addEventListener('input', function () {
        nmsThresholdValue.textContent = this.value;
    });

    // Mode switching
    modeClasses.addEventListener('change', function () {
        if (this.checked) {
            currentMode = 'classes';
            classesContainer.style.display = 'block';
            questionContainer.style.display = 'none';
        }
    });

    modeQuestion.addEventListener('change', function () {
        if (this.checked) {
            currentMode = 'question';
            classesContainer.style.display = 'none';
            questionContainer.style.display = 'block';
        }
    });

    // Add class functionality
    function addClass() {
        const value = classInput.value.trim();
        if (value && !classes.includes(value.toLowerCase())) {
            classes.push(value.toLowerCase());
            renderTags();
            classInput.value = '';
        }
    }

    addClassBtn.addEventListener('click', addClass);

    classInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            addClass();
        }
    });

    // Render tags
    function renderTags() {
        classTags.innerHTML = '';
        classes.forEach((className, index) => {
            const tag = document.createElement('div');
            tag.className = 'class-tag';
            tag.innerHTML = `
                <span>${className}</span>
                <span class="remove-tag" data-index="${index}">
                    <i class="fas fa-times"></i>
                </span>
            `;
            classTags.appendChild(tag);
        });

        // Add click handlers for remove buttons
        document.querySelectorAll('.remove-tag').forEach(btn => {
            btn.addEventListener('click', function () {
                const index = parseInt(this.getAttribute('data-index'));
                classes.splice(index, 1);
                renderTags();
            });
        });
    }

    // Image preview on file selection
    imageInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            // Validate file size (16MB max)
            if (file.size > 16 * 1024 * 1024) {
                showAlert('El archivo es demasiado grande. Máximo 16MB.', 'danger');
                imageInput.value = '';
                return;
            }

            // Validate file type
            const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp'];
            if (!validTypes.includes(file.type)) {
                showAlert('Tipo de archivo no válido. Use JPG, PNG, BMP o WEBP.', 'danger');
                imageInput.value = '';
                return;
            }

            // Show preview
            const reader = new FileReader();
            reader.onload = function (event) {
                imagePreview.src = event.target.result;
                previewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            previewContainer.style.display = 'none';
        }
    });

    // Form submission
    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Validate inputs
        if (!imageInput.files[0]) {
            showAlert('Por favor selecciona una imagen.', 'warning');
            return;
        }

        // Validate based on mode
        if (currentMode === 'classes' && classes.length === 0) {
            showAlert('Por favor agrega al menos una clase para detectar.', 'warning');
            return;
        }

        if (currentMode === 'question' && !questionInput.value.trim()) {
            showAlert('Por favor escribe una pregunta.', 'warning');
            return;
        }

        // Prepare form data
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        formData.append('mode', currentMode);

        if (currentMode === 'classes') {
            formData.append('classes', JSON.stringify(classes));
        } else {
            formData.append('question', questionInput.value.trim());
        }

        formData.append('box_threshold', boxThreshold.value);
        formData.append('text_threshold', textThreshold.value);
        formData.append('apply_nms', applyNMS.checked);
        formData.append('nms_threshold', nmsThreshold.value);

        // Show loading state
        processBtn.disabled = true;
        progressContainer.style.display = 'block';
        alertContainer.innerHTML = '';
        outputImagesContainer.style.display = 'none';

        try {
            // Send request
            const response = await fetch('/ask', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            // Hide loading
            progressContainer.style.display = 'none';
            processBtn.disabled = false;

            if (result.success) {
                if (result.num_detections === 0) {
                    // No detections found
                    showAlert(result.message || result.answer || 'No se detectaron objetos.', 'info');
                    showNoDetectionsResults();
                } else {
                    // Success with detections
                    showAlert(result.message || 'Procesamiento completado exitosamente.', 'success');
                    displayResults(result);
                }
            } else {
                // Error
                showAlert(result.message || 'Error al procesar la imagen.', 'danger');
                showNoDetectionsResults();
            }

        } catch (error) {
            progressContainer.style.display = 'none';
            processBtn.disabled = false;
            showAlert('Error de conexión: ' + error.message, 'danger');
            console.error('Error:', error);
        }
    });

    // Display results
    function displayResults(result) {
        // Show detection info
        let html = `
            <div class="alert alert-success">
                <h6 class="mb-2">
                    <i class="fas fa-check-circle"></i>
                    ${result.num_detections} objeto(s) detectado(s)
                </h6>
            </div>
        `;

        if (result.answer) {
            html += `
            <div class="alert alert-info mt-3">
                <h6 class="mb-2">
                    <i class="fas fa-brain"></i> Respuesta
                </h6>
                <p class="mb-0">${result.answer}</p>
            </div>
        `;
        }

        if (result.detected_classes && result.detected_classes.length > 0) {
            html += '<div class="mt-3"><h6>Clases detectadas:</h6>';
            html += '<div class="d-flex flex-wrap gap-2">';
            result.detected_classes.forEach(className => {
                html += `<span class="badge bg-primary">${className}</span>`;
            });
            html += '</div></div>';
        }

        resultsContainer.innerHTML = html;

        // Show output images
        if (result.annotated_image && result.segmented_image) {
            annotatedImage.src = result.annotated_image;
            segmentedImage.src = result.segmented_image;
            outputImagesContainer.style.display = 'flex';

            // Smooth scroll to results
            setTimeout(() => {
                outputImagesContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        }
    }

    // Show no detections state
    function showNoDetectionsResults() {
        resultsContainer.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-search fa-3x mb-3"></i>
                <p class="mb-2"><strong>No se detectaron objetos</strong></p>
                <p class="small">Intenta ajustar los thresholds o usar diferentes palabras clave.</p>
            </div>
        `;
        outputImagesContainer.style.display = 'none';
    }

    // Show alert message
    function showAlert(message, type) {
        const alertHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                <i class="fas fa-${getAlertIcon(type)}"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        alertContainer.innerHTML = alertHTML;

        // Auto-dismiss success alerts after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                const alert = alertContainer.querySelector('.alert');
                if (alert) {
                    const bsAlert = new bootstrap.Alert(alert);
                    bsAlert.close();
                }
            }, 5000);
        }
    }

    // Get icon for alert type
    function getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-circle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
});
