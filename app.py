"""
S3C-Smart-Canteen: Food Waste Analysis API
===========================================
Flask backend untuk analisis sisa makanan menggunakan YOLO instance segmentation.

Endpoints:
    POST /api/analyze       — Upload gambar, jalankan analisis, return estimasi berat
    GET  /api/health        — Health check
    GET  /api/classes       — Daftar class makanan yang didukung
    GET  /api/model-info    — Informasi model yang digunakan

Usage:
    python app.py                           # Run development server
    gunicorn -w 2 -b 0.0.0.0:8080 app:app  # Run production server
"""

import os
import time
import logging
from datetime import datetime

from flask import Flask, request
from flask_cors import CORS

import config
from core.detector import FoodDetector
from core.plate_calibrator import PlateCalibrator
from core.weight_estimator import WeightEstimator
from utils.image_utils import (
    allowed_file,
    generate_image_id,
    save_upload,
    load_image,
    load_image_from_bytes,
    preprocess_image,
    cleanup_upload,
    create_annotated_image,
    encode_image_to_base64,
)
from utils.response_builder import (
    success_response,
    error_response,
    build_analysis_response,
)

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Flask App Initialization
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = config.SECRET_KEY

# Enable CORS for all routes (untuk integrasi frontend S3C-Smart-Canteen)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Ensure upload directory exists
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────
# Initialize Core Components
# ──────────────────────────────────────────────
logger.info("=" * 60)
logger.info("S3C-Smart-Canteen: Food Waste Analysis API")
logger.info("=" * 60)

logger.info("Initializing YOLO Food Detector...")
detector = FoodDetector()

logger.info("Initializing Plate Calibrator...")
calibrator = PlateCalibrator()

logger.info("Initializing Weight Estimator...")
estimator = WeightEstimator()

logger.info("All components initialized successfully!")
logger.info("=" * 60)


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze_food_waste():
    """
    Endpoint utama: Analisis sisa makanan dari foto piring.

    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: file (image), optional: include_annotated (bool)

    Response:
        JSON dengan daftar makanan terdeteksi dan estimasi berat.
    """
    total_start = time.time()

    # ── Validasi Input ──
    if "file" not in request.files:
        return error_response(
            message="No file uploaded. Please send an image file with key 'file'.",
            error_code="NO_FILE",
            status_code=400,
        )

    file = request.files["file"]
    if file.filename == "":
        return error_response(
            message="Empty filename. Please select an image file.",
            error_code="EMPTY_FILENAME",
            status_code=400,
        )

    if not allowed_file(file.filename):
        return error_response(
            message=f"File type not allowed. Supported: {', '.join(config.ALLOWED_EXTENSIONS)}",
            error_code="INVALID_FILE_TYPE",
            status_code=400,
        )

    # Option: apakah include annotated image di response
    include_annotated = request.form.get("include_annotated", "false").lower() == "true"

    # ── Process Image ──
    image_id = generate_image_id()
    filepath = None

    try:
        # Save uploaded file
        filepath = save_upload(file, image_id)
        logger.info(f"Processing image: {image_id}")

        # Load and preprocess
        image = load_image(filepath)
        image = preprocess_image(image)
        h, w = image.shape[:2]
        logger.info(f"Image loaded: {w}x{h}")

        # ── Step 1: Plate Detection & Calibration ──
        plate_info = calibrator.detect_plate(image)
        logger.info(
            f"Plate detection: detected={plate_info['detected']}, "
            f"method={plate_info['method']}, "
            f"scale={plate_info['scale_factor']:.5f} cm/px"
        )

        # ── Step 2: YOLO Inference ──
        detection_result = detector.detect(image)
        detections = detection_result["detections"]
        logger.info(
            f"YOLO inference: {detection_result['num_detections']} objects detected "
            f"in {detection_result['processing_time_ms']:.1f}ms"
        )

        if not detections:
            total_time = (time.time() - total_start) * 1000
            data = build_analysis_response(
                image_id=image_id,
                plate_info=plate_info,
                estimations=[],
                total_weight=0.0,
                weight_by_category={},
                processing_time_ms=total_time,
                model_type=detection_result["model_type"],
            )
            return success_response(
                data=data,
                message="No food waste detected in the image.",
            )

        # ── Step 3: Weight Estimation ──
        estimations = estimator.estimate(
            detections=detections,
            scale_factor=plate_info["scale_factor"],
            plate_area_px=plate_info["plate_area_px"],
            is_custom_model=detector.is_custom_model,
        )

        total_weight = estimator.get_total_weight(estimations)
        weight_by_category = estimator.get_weight_by_category(estimations)

        logger.info(
            f"Weight estimation complete: "
            f"{len(estimations)} items, total={total_weight}g"
        )

        # ── Step 4: Annotated Image (optional) ──
        annotated_b64 = None
        if include_annotated:
            annotated_img = create_annotated_image(image, detections, estimations)
            annotated_b64 = encode_image_to_base64(annotated_img)

        # ── Build Response ──
        total_time = (time.time() - total_start) * 1000
        data = build_analysis_response(
            image_id=image_id,
            plate_info=plate_info,
            estimations=estimations,
            total_weight=total_weight,
            weight_by_category=weight_by_category,
            processing_time_ms=total_time,
            model_type=detection_result["model_type"],
            annotated_image_base64=annotated_b64,
        )

        return success_response(
            data=data,
            message=f"Analysis complete. Detected {len(estimations)} food item(s) "
                    f"with total estimated waste of {total_weight}g.",
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return error_response(
            message=str(e),
            error_code="VALIDATION_ERROR",
            status_code=400,
        )
    except Exception as e:
        logger.exception(f"Unexpected error processing image {image_id}")
        return error_response(
            message="An internal error occurred while processing the image.",
            error_code="INTERNAL_ERROR",
            status_code=500,
            details={"error": str(e)} if config.DEBUG else None,
        )
    finally:
        # Cleanup uploaded file
        if filepath:
            cleanup_upload(filepath)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint untuk monitoring dan GCP load balancer."""
    return success_response(
        data={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_loaded": detector.model is not None,
            "model_type": "custom" if detector.is_custom_model else "coco_pretrained",
        },
        message="S3C-Smart-Canteen Food Waste Analysis API is running.",
    )


@app.route("/", methods=["GET"])
def index():
    """Root endpoint untuk mengecek status API secara cepat dari browser."""
    return success_response(
        data={
            "endpoints": {
                "analyze": "POST /api/analyze",
                "health": "GET /api/health",
                "classes": "GET /api/classes",
                "model_info": "GET /api/model-info"
            }
        },
        message="Welcome to S3C-Smart-Canteen Food Waste Analysis API!",
    )


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Daftar class makanan yang didukung oleh model."""
    from core.food_constants import FOOD_CLASSES, FOOD_LABELS, FOOD_DENSITY, FOOD_HEIGHT

    classes = []
    for class_id, class_name in FOOD_CLASSES.items():
        classes.append({
            "class_id": class_id,
            "class_name": class_name,
            "label": FOOD_LABELS.get(class_name, class_name),
            "density_g_per_cm3": FOOD_DENSITY.get(class_name, 0.7),
            "estimated_height_cm": FOOD_HEIGHT.get(class_name, 1.5),
        })

    return success_response(
        data={
            "classes": classes,
            "total_classes": len(classes),
            "model_type": "custom" if detector.is_custom_model else "coco_pretrained",
        },
        message="Supported food waste classes.",
    )


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Informasi tentang model YOLO yang sedang digunakan."""
    return success_response(
        data={"model_info": detector.get_model_info()},
        message="Model information.",
    )


# ──────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(error):
    max_mb = config.MAX_CONTENT_LENGTH / (1024 * 1024)
    return error_response(
        message=f"File too large. Maximum size is {max_mb:.0f} MB.",
        error_code="FILE_TOO_LARGE",
        status_code=413,
    )


@app.errorhandler(404)
def not_found(error):
    return error_response(
        message="Endpoint not found.",
        error_code="NOT_FOUND",
        status_code=404,
    )


@app.errorhandler(405)
def method_not_allowed(error):
    return error_response(
        message="Method not allowed.",
        error_code="METHOD_NOT_ALLOWED",
        status_code=405,
    )


@app.errorhandler(500)
def internal_error(error):
    return error_response(
        message="Internal server error.",
        error_code="INTERNAL_ERROR",
        status_code=500,
    )


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting Flask server on {config.HOST}:{config.PORT}")
    logger.info(f"Debug mode: {config.DEBUG}")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
    )
