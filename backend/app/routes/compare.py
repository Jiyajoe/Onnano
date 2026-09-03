"""
compare.py - Multi-object AI/CV understanding and pairwise comparison endpoints.
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np

from ..cv.segmentation import isolate_primary_object, SegmentationError
from ..cv.orientation import normalize_posture
from ..cv.shape import extract_shape_properties
from ..cv.dimensions import extract_dimensions
from ..cv.color import extract_color_analysis
from ..cv.texture import extract_texture_analysis
from ..cv.features import extract_visual_features
from ..cv.contour import extract_edge_analysis
from ..cv.similarity import compare_two_objects, build_similarity_matrix, PairwiseComparison
from ..ai.object_detection import identify_object
from ..ai.verdict import classify_relationship
from ..config import DISCLAIMER_TEXT
from ..utils.imaging import decode_upload_bytes, encode_bgr_to_data_url, encode_rgba_to_data_url, InvalidImageError

router = APIRouter(prefix="/api")


@router.post("/compare-objects")
async def compare_objects(files: List[UploadFile] = File(...)):
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 object images are required for multi-object comparison.",
        )

    analyzed_objects = []
    internal_data = []

    # Process each uploaded object image through the full CV/AI pipeline
    for idx, f in enumerate(files):
        obj_id = idx + 1
        try:
            data = await f.read()
            image_bgr = decode_upload_bytes(data)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Image #{obj_id} could not be decoded. Please upload a valid image file.",
            )

        try:
            seg_obj, resized = isolate_primary_object(image_bgr)
        except SegmentationError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Object #{obj_id}: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Object #{obj_id} detection failed: {str(e)}",
            )

        frame_area = float(resized.shape[0] * resized.shape[1])
        orientation_res = normalize_posture(resized, seg_obj)
        norm_contour = orientation_res.normalized_contour
        norm_mask = orientation_res.normalized_mask
        norm_bgr = orientation_res.normalized_bgr

        shape_props = extract_shape_properties(norm_contour, norm_mask)
        dim_metrics = extract_dimensions(norm_contour, frame_area)
        color_analysis = extract_color_analysis(norm_bgr, norm_mask)
        texture_analysis = extract_texture_analysis(norm_bgr, norm_mask)
        edge_analysis = extract_edge_analysis(norm_bgr, norm_mask)
        visual_features = extract_visual_features(norm_bgr, norm_mask)

        id_res = identify_object(shape_props, dim_metrics, color_analysis, texture_analysis, edge_analysis)

        obj_info = {
            "id": obj_id,
            "label": f"Object {obj_id}",
            "detected_type": id_res.detected_type,
            "category": id_res.category,
            "confidence": id_res.confidence,
            "confidence_pct": id_res.confidence_pct,
            "description": id_res.description,
            "orientation": {
                "detected_angle_deg": orientation_res.detected_angle_deg,
                "correction_angle_deg": orientation_res.correction_angle_deg,
            },
            "shape": shape_props.to_dict(),
            "dimensions": dim_metrics.to_dict(),
            "color": color_analysis.to_dict(),
            "texture": texture_analysis.to_dict(),
            "edges": edge_analysis.to_dict(),
            "features": visual_features.to_dict(),
            "visuals": {
                "original_annotated": encode_bgr_to_data_url(orientation_res.before_annotated_bgr),
                "normalized_corrected": encode_bgr_to_data_url(orientation_res.normalized_bgr),
                "isolated_crop": encode_rgba_to_data_url(seg_obj.isolated_rgba),
            },
        }

        analyzed_objects.append(obj_info)
        internal_data.append({
            "id": obj_id,
            "shape": shape_props,
            "dim": dim_metrics,
            "color": color_analysis,
            "texture": texture_analysis,
            "edges": edge_analysis,
            "feat": visual_features,
            "contour": norm_contour,
            "mask": norm_mask,
            "bgr": norm_bgr,
            "type": id_res.detected_type,
            "category": id_res.category,
        })

    # Pairwise comparison across all pairs
    n = len(analyzed_objects)
    comparisons: List[PairwiseComparison] = []

    for i in range(n):
        for j in range(i + 1, n):
            d1 = internal_data[i]
            d2 = internal_data[j]
            pair_comp = compare_two_objects(
                id1=d1["id"],
                id2=d2["id"],
                shape1=d1["shape"],
                shape2=d2["shape"],
                contour1=d1["contour"],
                contour2=d2["contour"],
                dim1=d1["dim"],
                dim2=d2["dim"],
                color1=d1["color"],
                color2=d2["color"],
                img1=d1["bgr"],
                mask1=d1["mask"],
                img2=d2["bgr"],
                mask2=d2["mask"],
                tex1=d1["texture"],
                tex2=d2["texture"],
                feat1=d1["feat"],
                feat2=d2["feat"],
                edge1=d1["edges"],
                edge2=d2["edges"],
            )
            comparisons.append(pair_comp)

    # Build NxN similarity matrix
    sim_matrix = build_similarity_matrix(comparisons, n)

    # Average similarity across all pairs
    if comparisons:
        overall_avg = sum(c.overall_similarity for c in comparisons) / float(len(comparisons))
    else:
        overall_avg = 100.0

    # Determine category match
    types = [d["type"] for d in internal_data]
    categories = [d["category"] for d in internal_data]
    same_type = len(set(types)) == 1
    same_category = len(set(categories)) == 1

    # Classify relationship and generate funny Malayalam verdict
    first_comp = comparisons[0] if comparisons else None
    verdict_info = classify_relationship(
        avg_score=overall_avg,
        same_category=same_category,
        same_type=same_type,
        type1=types[0],
        type2=types[1] if len(types) > 1 else types[0],
        avg_comp=first_comp,
    )

    # Build Comparison Table rows
    table_rows = [
        {
            "feature": "Detected Object",
            "values": [f"{obj['detected_type']} ({obj['confidence_pct']}%)" for obj in analyzed_objects],
        },
        {
            "feature": "Height",
            "values": [f"{obj['dimensions']['pixel_height']} px" for obj in analyzed_objects],
        },
        {
            "feature": "Width",
            "values": [f"{obj['dimensions']['pixel_width']} px" for obj in analyzed_objects],
        },
        {
            "feature": "Aspect Ratio",
            "values": [f"{obj['dimensions']['aspect_ratio']}" for obj in analyzed_objects],
        },
        {
            "feature": "Dominant Color",
            "values": [obj['color']['dominant_name'] for obj in analyzed_objects],
            "colors": [obj['color']['dominant_hex'] for obj in analyzed_objects],
        },
        {
            "feature": "Pixel Area",
            "values": [f"{obj['dimensions']['pixel_area']:,} px" for obj in analyzed_objects],
        },
        {
            "feature": "Shape Class",
            "values": [obj['shape']['shape_type'] for obj in analyzed_objects],
        },
        {
            "feature": "Circularity",
            "values": [f"{obj['shape']['circularity']}" for obj in analyzed_objects],
        },
        {
            "feature": "Texture Profile",
            "values": [obj['texture']['descriptor'] for obj in analyzed_objects],
        },
        {
            "feature": "Edge Density",
            "values": [f"{obj['edges']['edge_density_pct']}%" for obj in analyzed_objects],
        },
    ]

    return {
        "success": True,
        "object_count": n,
        "objects": analyzed_objects,
        "comparisons": [c.to_dict() for c in comparisons],
        "similarity_matrix": sim_matrix,
        "comparison_table": table_rows,
        "overall_similarity": round(overall_avg, 1),
        "relationship": verdict_info.to_dict(),
        "disclaimer": DISCLAIMER_TEXT,
    }
