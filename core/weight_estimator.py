"""
S3C-Smart-Canteen: Weight Estimator
Mengestimasi berat sisa makanan berdasarkan segmentation mask,
scale factor dari plate calibration, dan konstanta densitas makanan.

Formula:
    Berat (g) = Luas_Real (cm²) × Tinggi_Estimasi (cm) × Densitas (g/cm³) × Height_Adjustment

Dimana:
    - Luas_Real = mask_area_pixels × (scale_factor)²
    - Tinggi_Estimasi = lookup dari FOOD_HEIGHT table
    - Densitas = lookup dari FOOD_DENSITY table
    - Height_Adjustment = faktor koreksi berdasarkan rasio mask/plate area
"""

import logging
import numpy as np
from typing import Optional

from core.food_constants import (
    FOOD_DENSITY,
    FOOD_HEIGHT,
    FOOD_CLASSES,
    FOOD_LABELS,
    COCO_TO_FOOD_MAPPING,
    get_height_adjustment,
)

logger = logging.getLogger(__name__)


class WeightEstimator:
    """Mengestimasi berat makanan dari data segmentation + plate calibration."""

    def __init__(self):
        self.food_density = FOOD_DENSITY
        self.food_height = FOOD_HEIGHT

    def estimate(
        self,
        detections: list,
        scale_factor: float,
        plate_area_px: float,
        is_custom_model: bool = True,
    ) -> list:
        """
        Estimasi berat untuk setiap deteksi.

        Args:
            detections: List of detection dicts dari FoodDetector
            scale_factor: cm per pixel dari PlateCalibrator
            plate_area_px: Luas piring dalam pixel² (untuk height adjustment)
            is_custom_model: Apakah menggunakan model custom

        Returns:
            List of dicts dengan informasi estimasi berat
        """
        results = []

        for det in detections:
            result = self._estimate_single(
                detection=det,
                scale_factor=scale_factor,
                plate_area_px=plate_area_px,
                is_custom_model=is_custom_model,
            )
            results.append(result)

        return results

    def _estimate_single(
        self,
        detection: dict,
        scale_factor: float,
        plate_area_px: float,
        is_custom_model: bool,
    ) -> dict:
        """
        Estimasi berat untuk satu deteksi tunggal.

        Returns:
            dict dengan semua informasi estimasi
        """
        class_id = detection["class_id"]
        class_name = detection["class_name"]
        mask_area_pixels = detection["mask_area_pixels"]

        # Map class ke food category
        food_category = self._get_food_category(
            class_id, class_name, is_custom_model
        )

        # Step 1: Konversi pixel area ke real area (cm²)
        area_cm2 = mask_area_pixels * (scale_factor ** 2)

        # Step 2: Dapatkan estimasi tinggi makanan
        base_height = self.food_height.get(food_category, 1.5)

        # Step 3: Hitung height adjustment berdasarkan rasio mask/plate
        mask_area_ratio = mask_area_pixels / plate_area_px if plate_area_px > 0 else 0.2
        height_adjustment = get_height_adjustment(mask_area_ratio)
        adjusted_height = base_height * height_adjustment

        # Step 4: Estimasi volume (cm³)
        volume_cm3 = area_cm2 * adjusted_height

        # Step 5: Dapatkan densitas makanan
        density = self.food_density.get(food_category, 0.70)

        # Step 6: Hitung berat (gram)
        weight_grams = volume_cm3 * density

        # Minimum weight threshold (jika terdeteksi, minimal 1 gram)
        weight_grams = max(weight_grams, 1.0)

        # Build result
        result = {
            "class_id": class_id,
            "class_name": class_name,
            "food_category": food_category,
            "food_label": FOOD_LABELS.get(food_category, food_category),
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "mask_area_pixels": mask_area_pixels,
            "area_cm2": round(area_cm2, 2),
            "estimated_height_cm": round(adjusted_height, 2),
            "estimated_volume_cm3": round(volume_cm3, 2),
            "estimated_weight_grams": round(weight_grams, 1),
            "density_used": density,
            "height_base": base_height,
            "height_adjustment_factor": round(height_adjustment, 2),
            "mask_area_ratio": round(mask_area_ratio, 4),
        }

        logger.debug(
            f"Estimated weight for {food_category}: "
            f"area={area_cm2:.2f}cm², height={adjusted_height:.2f}cm, "
            f"volume={volume_cm3:.2f}cm³, weight={weight_grams:.1f}g"
        )

        return result

    def _get_food_category(
        self, class_id: int, class_name: str, is_custom_model: bool
    ) -> str:
        """
        Map model class ke food category kita.

        Args:
            class_id: YOLO class ID
            class_name: YOLO class name
            is_custom_model: True jika model custom

        Returns:
            Food category string (e.g., 'nasi', 'daging', etc.)
        """
        if is_custom_model:
            # Model custom menggunakan mapping langsung dari FOOD_CLASSES
            return FOOD_CLASSES.get(class_id, "lainnya")
        else:
            # Model COCO → map ke food category kita
            return COCO_TO_FOOD_MAPPING.get(class_id, "lainnya")

    def get_total_weight(self, estimations: list) -> float:
        """Hitung total berat semua deteksi."""
        return round(
            sum(e["estimated_weight_grams"] for e in estimations), 1
        )

    def get_weight_by_category(self, estimations: list) -> dict:
        """Kelompokkan berat berdasarkan kategori makanan."""
        by_category = {}
        for e in estimations:
            cat = e["food_category"]
            if cat not in by_category:
                by_category[cat] = {
                    "food_label": e["food_label"],
                    "total_weight_grams": 0,
                    "count": 0,
                }
            by_category[cat]["total_weight_grams"] += e["estimated_weight_grams"]
            by_category[cat]["count"] += 1

        # Round totals
        for cat in by_category:
            by_category[cat]["total_weight_grams"] = round(
                by_category[cat]["total_weight_grams"], 1
            )

        return by_category
