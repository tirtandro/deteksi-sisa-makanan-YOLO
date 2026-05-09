"""
S3C-Smart-Canteen: Food Waste Analysis System
Global Configuration
"""

import os

# ──────────────────────────────────────────────
# Flask Settings
# ──────────────────────────────────────────────
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_PORT", 8080))
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "s3c-smart-canteen-secret-key-change-me")

# ──────────────────────────────────────────────
# File Upload Settings
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# ──────────────────────────────────────────────
# YOLO Model Settings
# ──────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "food_seg_model.pt"),
)
# Fallback to pre-trained YOLOv8 segmentation model if custom model not found
FALLBACK_MODEL = "yolo11n-seg.pt"

# Inference settings
CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", 0.35))
IOU_THRESHOLD = float(os.environ.get("YOLO_IOU_THRESHOLD", 0.45))
INPUT_IMAGE_SIZE = int(os.environ.get("YOLO_IMG_SIZE", 640))

# ──────────────────────────────────────────────
# Plate Calibration Settings
# ──────────────────────────────────────────────
# Standard plate diameter in cm (piring standar kantin)
PLATE_REAL_DIAMETER_CM = float(os.environ.get("PLATE_DIAMETER_CM", 26.0))

# Fallback scale factor if plate is not detected (cm/pixel)
# Calculated assuming typical phone camera at ~30cm distance capturing a 640px wide image
# covering approximately 30cm of real-world width
FALLBACK_SCALE_FACTOR = float(os.environ.get("FALLBACK_SCALE_FACTOR", 0.047))

# Hough Circle Detection parameters
PLATE_DETECT_DP = 1.2         # Inverse ratio of accumulator resolution
PLATE_DETECT_MIN_DIST = 100   # Minimum distance between circle centers
PLATE_DETECT_PARAM1 = 50      # Upper Canny edge threshold
PLATE_DETECT_PARAM2 = 30      # Accumulator threshold for circle centers
PLATE_MIN_RADIUS_RATIO = 0.2  # Min plate radius as ratio of image width
PLATE_MAX_RADIUS_RATIO = 0.45 # Max plate radius as ratio of image width

# ──────────────────────────────────────────────
# GCP / Deployment Settings
# ──────────────────────────────────────────────
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_REGION = os.environ.get("GCP_REGION", "asia-southeast2")  # Jakarta

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
