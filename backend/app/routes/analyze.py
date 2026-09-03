"""
analyze.py - Single-object AI/CV understanding and N-way geometric division endpoints.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import numpy as np

from ..cv.segmentation import isolate_primary_object, SegmentationError
from ..cv.orientation import normalize_posture
from ..cv.shape import extract_shape_properties
from ..cv.dimensions import extract_dimensions
from ..cv.color import extract_color_analysis
from ..cv.texture import extract_texture_analysis
from ..cv.features import extract_visual_features
from ..cv.contour import extract_edge_analysis
from ..cv.slicing import divide_normalized_object
from ..ai.object_detection import identify_object
from ..config import DISCLAIMER_TEXT
from ..utils.imaging import decode_upload_bytes, encode_bgr_to_data_url, encode_rgba_to_data_url, InvalidImageError

router = APIRouter(prefix="/api")


def process_single_object_pipeline(image_bgr: np.ndarray, parts_count: int = 4):
    """Full CV/AI pipeline for understanding a single object."""
    # Step 1: Detect & Isolate primary object
    seg_obj, resized = isolate_primary_object(image_bgr)
    frame_area = float(resized.shape[0] * resized.shape[1])

    # Step 2: Posture Normalization (Before -> After)
    orientation_res = normalize_posture(resized, seg_obj)

    # Step 3: Extract Visual Properties
    shape_props = extract_shape_properties(orientation_res.normalized_contour, orientation_res.normalized_mask)
    dim_metrics = extract_dimensions(orientation_res.normalized_contour, frame_area)
    color_analysis = extract_color_analysis(orientation_res.normalized_bgr, orientation_res.normalized_mask)
    texture_analysis = extract_texture_analysis(orientation_res.normalized_bgr, orientation_res.normalized_mask)
    edge_analysis = extract_edge_analysis(orientation_res.normalized_bgr, orientation_res.normalized_mask)
    visual_features = extract_visual_features(orientation_res.normalized_bgr, orientation_res.normalized_mask)

    # Step 4: Identify Object
    id_result = identify_object(shape_props, dim_metrics, color_analysis, texture_analysis, edge_analysis)

    # Step 5: N-way Equal Geometric Division along principal axis
    slicing_res = divide_normalized_object(
        orientation_res.normalized_bgr,
        orientation_res.normalized_mask,
        orientation_res.normalized_contour,
        parts_count=parts_count,
    )

    return {
        "success": True,
        "object": {
            "id": 1,
            "detected_type": id_result.detected_type,
            "category": id_result.category,
            "confidence": id_result.confidence,
            "confidence_pct": id_result.confidence_pct,
            "description": id_result.description,
            "related_categories": id_result.related_categories,
            "orientation": {
                "detected_angle_deg": orientation_res.detected_angle_deg,
                "correction_angle_deg": orientation_res.correction_angle_deg,
                "is_vertical": orientation_res.is_vertical,
            },
            "shape": shape_props.to_dict(),
            "dimensions": dim_metrics.to_dict(),
            "color": color_analysis.to_dict(),
            "texture": texture_analysis.to_dict(),
            "edges": edge_analysis.to_dict(),
            "features": visual_features.to_dict(),
        },
        "visuals": {
            "original_annotated": encode_bgr_to_data_url(orientation_res.before_annotated_bgr),
            "normalized_corrected": encode_bgr_to_data_url(orientation_res.normalized_bgr),
            "isolated_crop": encode_rgba_to_data_url(seg_obj.isolated_rgba),
            "divided_image": encode_bgr_to_data_url(slicing_res.divided_image_bgr),
        },
        "division": slicing_res.to_dict(),
        "disclaimer": DISCLAIMER_TEXT,
    }


@router.post("/analyze-object")
async def analyze_object(file: UploadFile = File(...), parts: int = Form(4)):
    try:
        data = await file.read()
        image_bgr = decode_upload_bytes(data)
    except InvalidImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPG, PNG, or WebP photo.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

    try:
        result = process_single_object_pipeline(image_bgr, parts_count=parts)
        return result
    except SegmentationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"We encountered an issue analyzing the object: {str(e)}. Try a clearer photo with better contrast.",
        )


@router.post("/divide-object")
async def divide_object(file: UploadFile = File(...), parts: int = Form(4)):
    """Re-slices an object into a specified number of parts."""
    try:
        data = await file.read()
        image_bgr = decode_upload_bytes(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        result = process_single_object_pipeline(image_bgr, parts_count=parts)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
