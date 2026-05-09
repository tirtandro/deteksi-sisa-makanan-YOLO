"""
S3C-Smart-Canteen: Image Utilities
Helper functions untuk preprocessing gambar sebelum inferensi.
"""

import os
import uuid
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage

import config

logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """Cek apakah file memiliki ekstensi yang diizinkan."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def generate_image_id() -> str:
    """Generate unique image ID."""
    return str(uuid.uuid4())


def save_upload(file: FileStorage, image_id: str) -> str:
    """
    Simpan file upload ke disk.

    Args:
        file: Werkzeug FileStorage object
        image_id: Unique identifier untuk file

    Returns:
        Path ke file yang disimpan
    """
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

    ext = file.filename.rsplit(".", 1)[1].lower() if file.filename else "jpg"
    filename = f"{image_id}.{ext}"
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)

    file.save(filepath)
    logger.info(f"Image saved: {filepath}")
    return filepath


def load_image(filepath: str) -> np.ndarray:
    """
    Load gambar dari file path sebagai numpy array (BGR).

    Args:
        filepath: Path ke file gambar

    Returns:
        numpy array (BGR format)

    Raises:
        ValueError: Jika file tidak bisa dibaca
    """
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Cannot read image from: {filepath}")
    return image


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Load gambar dari bytes sebagai numpy array (BGR).

    Args:
        file_bytes: Raw bytes dari file gambar

    Returns:
        numpy array (BGR format)

    Raises:
        ValueError: Jika bytes tidak bisa di-decode
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image from bytes")
    return image


def preprocess_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """
    Preprocessing gambar: resize jika terlalu besar, normalize.

    Args:
        image: Input image (BGR, numpy array)
        max_size: Maximum dimension (width or height)

    Returns:
        Preprocessed image
    """
    h, w = image.shape[:2]

    # Resize jika lebih besar dari max_size
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Image resized from {w}x{h} to {new_w}x{new_h}")

    return image


def cleanup_upload(filepath: str):
    """Hapus file upload setelah diproses."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Cleaned up: {filepath}")
    except OSError as e:
        logger.warning(f"Failed to cleanup {filepath}: {e}")


def create_annotated_image(
    image: np.ndarray,
    detections: list,
    estimations: list,
) -> np.ndarray:
    """
    Buat gambar dengan anotasi bounding box, mask, dan label berat.

    Args:
        image: Original image (BGR)
        detections: List of detection dicts
        estimations: List of estimation dicts

    Returns:
        Annotated image (BGR)
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Warna untuk setiap kategori makanan (sesuai model custom)
    colors = {
        "buah": (255, 0, 255),       # Magenta
        "karbo": (0, 165, 255),      # Orange
        "nasi": (0, 255, 255),       # Yellow
        "protein": (255, 128, 0),    # Blue-ish
        "sayur": (0, 255, 0),        # Green
        "susu": (255, 255, 0),       # Cyan
        "lainnya": (128, 128, 128),  # Gray
    }

    for det, est in zip(detections, estimations):
        food_cat = est["food_category"]
        color = colors.get(food_cat, (255, 255, 255))
        weight = est["estimated_weight_grams"]
        conf = det["confidence"]

        # Draw mask overlay
        if det.get("mask_binary") is not None:
            mask = det["mask_binary"]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h))
            colored_mask = np.zeros_like(annotated)
            colored_mask[mask > 0.5] = color
            annotated = cv2.addWeighted(annotated, 1, colored_mask, 0.3, 0)

        # Draw bounding box
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{food_cat}: {weight:.0f}g ({conf:.0%})"
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Label background
        cv2.rectangle(
            annotated,
            (x1, y1 - th - 8),
            (x1 + tw + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return annotated


def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode numpy image ke base64 string (JPEG)."""
    import base64
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")
