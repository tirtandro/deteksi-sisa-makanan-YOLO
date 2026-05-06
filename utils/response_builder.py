"""
S3C-Smart-Canteen: Response Builder
Membangun response JSON yang terstandarisasi untuk semua endpoint API.
"""

from flask import jsonify
from typing import Optional


def success_response(
    data: dict,
    message: str = "Success",
    status_code: int = 200,
):
    """
    Build a successful JSON response.

    Args:
        data: Response data dictionary
        message: Success message
        status_code: HTTP status code

    Returns:
        Flask JSON response
    """
    response = {
        "success": True,
        "message": message,
        **data,
    }
    return jsonify(response), status_code


def error_response(
    message: str,
    status_code: int = 400,
    error_code: Optional[str] = None,
    details: Optional[dict] = None,
):
    """
    Build an error JSON response.

    Args:
        message: Error message
        status_code: HTTP status code
        error_code: Machine-readable error code
        details: Additional error details

    Returns:
        Flask JSON response
    """
    response = {
        "success": False,
        "message": message,
    }
    if error_code:
        response["error_code"] = error_code
    if details:
        response["details"] = details

    return jsonify(response), status_code


def build_analysis_response(
    image_id: str,
    plate_info: dict,
    estimations: list,
    total_weight: float,
    weight_by_category: dict,
    processing_time_ms: float,
    model_type: str,
    annotated_image_base64: Optional[str] = None,
) -> dict:
    """
    Build the full analysis response data.

    Args:
        image_id: Unique image identifier
        plate_info: Plate detection info
        estimations: List of weight estimations
        total_weight: Total waste weight in grams
        weight_by_category: Weight grouped by food category
        processing_time_ms: Total processing time
        model_type: Model type used
        annotated_image_base64: Optional annotated image

    Returns:
        dict ready to be passed to success_response()
    """
    # Clean up estimations for response (remove binary mask data)
    clean_estimations = []
    for est in estimations:
        clean = {k: v for k, v in est.items() if k != "mask_binary"}
        # Convert bbox values to serializable format
        if "bbox" in clean:
            clean["bbox"] = [round(v, 1) for v in clean["bbox"]]
        clean_estimations.append(clean)

    data = {
        "image_id": image_id,
        "plate_detected": plate_info["detected"],
        "plate_detection_method": plate_info["method"],
        "scale_factor_cm_per_pixel": round(plate_info["scale_factor"], 6),
        "detections": clean_estimations,
        "summary": {
            "total_items_detected": len(clean_estimations),
            "total_waste_weight_grams": total_weight,
            "weight_by_category": weight_by_category,
        },
        "model_type": model_type,
        "processing_time_ms": round(processing_time_ms, 2),
    }

    if annotated_image_base64:
        data["annotated_image_base64"] = annotated_image_base64

    return data
