"""
S3C-Smart-Canteen: Plate Calibrator
Mendeteksi piring dalam gambar dan menghitung scale factor (cm/pixel)
untuk mengkonversi ukuran piksel ke ukuran dunia nyata.

Pendekatan:
1. Deteksi lingkaran (piring) menggunakan Hough Circle Transform
2. Jika gagal, coba ellipse fitting pada kontur terbesar
3. Jika keduanya gagal, gunakan fallback scale factor
"""

import cv2
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)


class PlateCalibrator:
    """Mendeteksi piring dan menghitung scale factor pixel → cm."""

    def __init__(self):
        self.plate_real_diameter_cm = config.PLATE_REAL_DIAMETER_CM
        self.fallback_scale = config.FALLBACK_SCALE_FACTOR

    def detect_plate(self, image: np.ndarray) -> dict:
        """
        Mendeteksi piring dalam gambar dan menghitung scale factor.

        Args:
            image: Input image (BGR format, numpy array)

        Returns:
            dict dengan keys:
                - detected (bool): Apakah piring terdeteksi
                - scale_factor (float): cm per pixel
                - plate_center (tuple): (x, y) center piring
                - plate_radius_px (float): radius piring dalam pixel
                - plate_area_px (float): luas piring dalam pixel²
                - method (str): Metode deteksi yang berhasil
        """
        h, w = image.shape[:2]

        # Coba metode 1: Hough Circle Transform
        result = self._detect_hough_circles(image, w, h)
        if result["detected"]:
            logger.info(
                f"Plate detected via Hough Circle: "
                f"center=({result['plate_center'][0]}, {result['plate_center'][1]}), "
                f"radius={result['plate_radius_px']:.0f}px, "
                f"scale={result['scale_factor']:.5f} cm/px"
            )
            return result

        # Coba metode 2: Ellipse Fitting
        result = self._detect_ellipse(image, w, h)
        if result["detected"]:
            logger.info(
                f"Plate detected via Ellipse Fitting: "
                f"center=({result['plate_center'][0]}, {result['plate_center'][1]}), "
                f"radius={result['plate_radius_px']:.0f}px, "
                f"scale={result['scale_factor']:.5f} cm/px"
            )
            return result

        # Fallback: gunakan scale factor default
        logger.warning(
            "Plate not detected. Using fallback scale factor: "
            f"{self.fallback_scale} cm/px"
        )
        return {
            "detected": False,
            "scale_factor": self.fallback_scale,
            "plate_center": (w // 2, h // 2),
            "plate_radius_px": min(w, h) * 0.35,
            "plate_area_px": np.pi * (min(w, h) * 0.35) ** 2,
            "method": "fallback",
        }

    def _detect_hough_circles(self, image: np.ndarray, w: int, h: int) -> dict:
        """Deteksi piring menggunakan Hough Circle Transform."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        min_radius = int(w * config.PLATE_MIN_RADIUS_RATIO)
        max_radius = int(w * config.PLATE_MAX_RADIUS_RATIO)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=config.PLATE_DETECT_DP,
            minDist=config.PLATE_DETECT_MIN_DIST,
            param1=config.PLATE_DETECT_PARAM1,
            param2=config.PLATE_DETECT_PARAM2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Pilih lingkaran terbesar (kemungkinan besar piring)
            best = max(circles, key=lambda c: c[2])
            cx, cy, radius = int(best[0]), int(best[1]), int(best[2])

            diameter_px = radius * 2
            scale_factor = self.plate_real_diameter_cm / diameter_px

            return {
                "detected": True,
                "scale_factor": scale_factor,
                "plate_center": (cx, cy),
                "plate_radius_px": float(radius),
                "plate_area_px": np.pi * radius**2,
                "method": "hough_circle",
            }

        return {"detected": False}

    def _detect_ellipse(self, image: np.ndarray, w: int, h: int) -> dict:
        """Deteksi piring menggunakan ellipse fitting pada kontur terbesar."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding untuk menangani variasi pencahayaan
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations untuk membersihkan noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"detected": False}

        # Cari kontur terbesar yang menyerupai ellipse
        min_area = w * h * 0.05  # Minimal 5% dari gambar
        max_area = w * h * 0.85  # Maksimal 85% dari gambar

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            if len(contour) < 5:  # fitEllipse butuh minimal 5 titik
                continue

            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (ma, MA), angle = ellipse

            # Cek circularity (piring biasanya cukup bulat)
            aspect_ratio = min(ma, MA) / max(ma, MA) if max(ma, MA) > 0 else 0
            if aspect_ratio > 0.6:  # Toleransi perspektif
                candidates.append({
                    "contour": contour,
                    "ellipse": ellipse,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "center": (int(cx), int(cy)),
                    "axes": (ma, MA),
                })

        if not candidates:
            return {"detected": False}

        # Pilih kandidat dengan area terbesar
        best = max(candidates, key=lambda c: c["area"])
        cx, cy = best["center"]
        # Gunakan rata-rata kedua sumbu sebagai "diameter"
        avg_diameter_px = (best["axes"][0] + best["axes"][1]) / 2
        radius_px = avg_diameter_px / 2
        scale_factor = self.plate_real_diameter_cm / avg_diameter_px

        return {
            "detected": True,
            "scale_factor": scale_factor,
            "plate_center": (cx, cy),
            "plate_radius_px": float(radius_px),
            "plate_area_px": np.pi * radius_px**2,
            "method": "ellipse_fitting",
        }

    def get_plate_area_cm2(self, plate_info: dict) -> float:
        """Menghitung luas piring dalam cm²."""
        if plate_info["detected"]:
            radius_cm = plate_info["plate_radius_px"] * plate_info["scale_factor"]
            return np.pi * radius_cm**2
        else:
            # Fallback: luas piring standar
            radius_cm = self.plate_real_diameter_cm / 2
            return np.pi * radius_cm**2
