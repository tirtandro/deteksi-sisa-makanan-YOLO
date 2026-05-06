"""
S3C-Smart-Canteen: YOLO Detector
Wrapper untuk menjalankan inferensi YOLO instance segmentation.
Mendukung model custom (fine-tuned) dan fallback ke model pre-trained COCO.
"""

import os
import time
import logging
import numpy as np
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Lazy import ultralytics to speed up module loading
_YOLO = None


def _get_yolo_class():
    """Lazy import YOLO class from ultralytics."""
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO
    return _YOLO


class FoodDetector:
    """
    YOLO-based food waste detector with instance segmentation.

    Supports:
    - Custom fine-tuned model for specific food categories
    - Fallback to pre-trained YOLOv8-seg (COCO) model
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO model file (.pt).
                        If None, uses config.MODEL_PATH.
                        If that doesn't exist, falls back to pre-trained.
        """
        self.model = None
        self.model_path = model_path or config.MODEL_PATH
        self.is_custom_model = False
        self._load_model()

    def _load_model(self):
        """Load YOLO model with fallback logic."""
        YOLO = _get_yolo_class()

        if os.path.exists(self.model_path):
            logger.info(f"Loading custom YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.is_custom_model = True
        else:
            logger.warning(
                f"Custom model not found at '{self.model_path}'. "
                f"Falling back to pre-trained model: {config.FALLBACK_MODEL}"
            )
            self.model = YOLO(config.FALLBACK_MODEL)
            self.is_custom_model = False

        logger.info(
            f"Model loaded successfully. "
            f"Type: {'custom' if self.is_custom_model else 'pre-trained COCO'}. "
            f"Classes: {self.model.names}"
        )

    def detect(self, image: np.ndarray) -> dict:
        """
        Run YOLO instance segmentation on an image.

        Args:
            image: Input image (BGR format, numpy array)

        Returns:
            dict with keys:
                - detections: list of detection dicts
                - num_detections: int
                - processing_time_ms: float
                - model_type: str ('custom' or 'coco_pretrained')
        """
        start_time = time.time()

        # Run inference
        results = self.model.predict(
            source=image,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INPUT_IMAGE_SIZE,
            verbose=False,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Parse results
        detections = self._parse_results(results, image.shape)

        return {
            "detections": detections,
            "num_detections": len(detections),
            "processing_time_ms": round(processing_time_ms, 2),
            "model_type": "custom" if self.is_custom_model else "coco_pretrained",
        }

    def _parse_results(self, results, image_shape: tuple) -> list:
        """
        Parse YOLO results into a structured list of detections.

        Each detection contains:
        - class_id, class_name, confidence
        - bbox (x1, y1, x2, y2)
        - mask_polygon (list of (x, y) points)
        - mask_area_pixels
        """
        detections = []
        h, w = image_shape[:2]

        if not results or len(results) == 0:
            return detections

        result = results[0]

        # Check if segmentation masks are available
        has_masks = result.masks is not None
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            class_name = result.names.get(class_id, f"class_{class_id}")

            # Bounding box (xyxy format)
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            bbox = [round(v, 1) for v in bbox]

            # Mask data
            mask_polygon = []
            mask_area_pixels = 0
            mask_binary = None

            if has_masks and i < len(result.masks):
                mask_data = result.masks[i]

                # Get polygon points
                if mask_data.xy is not None and len(mask_data.xy) > 0:
                    mask_polygon = mask_data.xy[0].tolist() if len(mask_data.xy) > 0 else []

                # Get binary mask and compute area
                if mask_data.data is not None:
                    mask_tensor = mask_data.data[0]
                    mask_binary = mask_tensor.cpu().numpy()
                    # Resize mask to original image size
                    if mask_binary.shape != (h, w):
                        import cv2
                        mask_binary = cv2.resize(
                            mask_binary.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(np.float32)
                    mask_area_pixels = int(np.count_nonzero(mask_binary))

            # If no mask, estimate area from bounding box
            if mask_area_pixels == 0:
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                # Approximate mask area as ~70% of bbox area
                mask_area_pixels = int(bbox_area * 0.7)

            # Filter: only include food-related classes for COCO model
            if not self.is_custom_model:
                from core.food_constants import COCO_FOOD_CLASS_IDS
                if class_id not in COCO_FOOD_CLASS_IDS:
                    continue

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "mask_polygon": mask_polygon,
                "mask_area_pixels": mask_area_pixels,
                "mask_binary": mask_binary,
            }
            detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def get_class_names(self) -> dict:
        """Return the class names supported by the loaded model."""
        if self.model:
            return dict(self.model.names)
        return {}

    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        return {
            "model_path": self.model_path,
            "is_custom_model": self.is_custom_model,
            "fallback_model": config.FALLBACK_MODEL,
            "class_names": self.get_class_names(),
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "iou_threshold": config.IOU_THRESHOLD,
            "input_size": config.INPUT_IMAGE_SIZE,
        }
