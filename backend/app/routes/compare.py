"""
compare.py - Multi-object AI/CV understanding, RANSAC matching, explainable verdicts, and pairwise comparison endpoints.
"""

from typing import List, Optional
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
from ..cv.similarity import compare_two_objects, build_similarity_matrix, PairwiseComparison
from ..ai.object_detection import identify_object
from ..ai.verdict import classify_relationship
from ..config import DISCLAIMER_TEXT
from ..utils.imaging import decode_upload_bytes, encode_bgr_to_data_url, encode_rgba_to_data_url, InvalidImageError

router = APIRouter(prefix="/api")


@router.post("/compare-objects")
async def compare_objects(
    files: List[UploadFile] = File(...),
    reference_cm: Optional[float] = Form(None),
):
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
        aligned_bgr = orientation_res.ai_aligned_bgr

        dim_metrics = extract_dimensions(norm_contour, frame_area, reference_length_cm=reference_cm)
        shape_props = extract_shape_properties(norm_contour, norm_mask)
        color_analysis = extract_color_analysis(aligned_bgr, norm_mask)
        texture_analysis = extract_texture_analysis(aligned_bgr, norm_mask)
        edge_analysis = extract_edge_analysis(aligned_bgr, norm_mask)
        visual_features = extract_visual_features(aligned_bgr, norm_mask)

        # Each object is identified independently via its own AI vision call.
        # Object 1's classification does NOT influence Object 2.
        id_res = identify_object(
            shape_props, dim_metrics, color_analysis, texture_analysis, edge_analysis,
            normalized_mask=norm_mask,
            original_image_bgr=resized,
        )

        obj_info = {
            "id": obj_id,
            "label": f"Object {obj_id}",
            "detected_type": id_res.name,
            "specific_type": id_res.specific_type,
            "category": id_res.category,
            "brand": id_res.brand,
            "confidence": id_res.confidence,
            "confidence_pct": id_res.confidence_pct,
            "characteristics": id_res.characteristics,
            "image_quality": seg_obj.quality_report.to_dict(),
            "orientation": orientation_res.to_dict(),
            "selection": seg_obj.to_selection_dict(),
            "shape": shape_props.to_dict(),
            "dimensions": dim_metrics.to_dict(),
            "color": color_analysis.to_dict(),
            "texture": texture_analysis.to_dict(),
            "edges": edge_analysis.to_dict(),
            "features": visual_features.to_dict(),
            "visuals": {
                "original": encode_bgr_to_data_url(orientation_res.original_bgr),
                "ai_selected": encode_bgr_to_data_url(orientation_res.ai_selected_bgr),
                "ai_aligned": encode_rgba_to_data_url(orientation_res.ai_aligned_rgba),
            },
            "ai_identification": id_res.ai_identification,
            "validation_info": id_res.validation_info,
            "pipeline_source": id_res.pipeline_source,
            "debug_log": id_res.debug_log,
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
            "bgr": aligned_bgr,
            "type": id_res.name,
            "category": id_res.category,
            "mask_qual": seg_obj.mask_quality,
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
                mask_qual1=d1["mask_qual"],
                mask_qual2=d2["mask_qual"],
                type1=d1["type"],
                type2=d2["type"],
                cat1=d1["category"],
                cat2=d2["category"],
            )
            comparisons.append(pair_comp)

    # Build NxN similarity matrix
    sim_matrix = build_similarity_matrix(comparisons, n)

    # Average similarity across all pairs
    if comparisons:
        overall_avg = sum(c.overall_similarity for c in comparisons) / float(len(comparisons))
        overall_conf = sum(c.overall_confidence for c in comparisons) / float(len(comparisons))
    else:
        overall_avg = 100.0
        overall_conf = 95.0

    # Determine category match
    types = [d["type"] for d in internal_data]
    categories = [d["category"] for d in internal_data]
    same_type = len(set(types)) == 1
    same_category = len(set(categories)) == 1

    # Classify relationship and generate Malayalam verdict
    first_comp = comparisons[0] if comparisons else None
    verdict_info = classify_relationship(
        avg_score=overall_avg,
        same_category=same_category,
        same_type=same_type,
        type1=types[0],
        type2=types[1] if len(types) > 1 else types[0],
        avg_comp=first_comp,
    )

    # Primary "Why This Result?" breakdown from first pair or aggregated
    primary_why = first_comp.why_explanation.to_dict() if first_comp else {
        "verdict_summary": "All objects analyzed.",
        "positive_factors": ["High similarity across measured features"],
        "differing_factors": [],
        "final_verdict": verdict_info.tier_name,
    }

    # Primary Technical Analysis for Demo/Judges
    primary_tech = first_comp.technical_analysis.to_dict() if first_comp else {}

    # Build Comparison Table rows
    table_rows = [
        {
            "feature": "Detected Object",
            "values": [f"{obj['detected_type']} ({obj['confidence_pct']}%)" for obj in analyzed_objects],
        },
        {
            "feature": "Specific Subtype",
            "values": [obj['specific_type'] for obj in analyzed_objects],
        },
        {
            "feature": "Category",
            "values": [obj['category'] for obj in analyzed_objects],
        },
        {
            "feature": "Brand / Model",
            "values": [obj['brand'] for obj in analyzed_objects],
        },
        {
            "feature": "Pixel Height",
            "values": [f"{obj['dimensions']['pixel_height']} px" for obj in analyzed_objects],
        },
        {
            "feature": "Pixel Width",
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
            "values": [f"{obj['dimensions']['pixel_area']:,} px²" for obj in analyzed_objects],
        },
        {
            "feature": "Perimeter",
            "values": [f"{obj['dimensions']['perimeter']} px" for obj in analyzed_objects],
        },
        {
            "feature": "Shape Profile",
            "values": [obj['shape']['shape_type'] for obj in analyzed_objects],
        },
        {
            "feature": "Circularity / Solidity",
            "values": [f"{obj['shape']['circularity']} / {obj['shape']['solidity']}" for obj in analyzed_objects],
        },
        {
            "feature": "Texture Profile",
            "values": [obj['texture']['descriptor'] for obj in analyzed_objects],
        },
        {
            "feature": "Keypoints Detected",
            "values": [f"{obj['features']['keypoints_count']} ({obj['features'].get('detector_type', 'SIFT/ORB')})" for obj in analyzed_objects],
        },
        {
            "feature": "Selection Confidence",
            "values": [f"{obj['selection']['mask_confidence_pct']}%" for obj in analyzed_objects],
        },
        {
            "feature": "Physical Dimensions",
            "values": [
                f"{obj['dimensions']['physical_height_cm']:.1f} × {obj['dimensions']['physical_width_cm']:.1f} cm"
                if obj['dimensions']['is_calibrated']
                else "unavailable (no reference)"
                for obj in analyzed_objects
            ],
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
        "overall_confidence": round(overall_conf, 1),
        "relationship": verdict_info.to_dict(),
        "why_explanation": primary_why,
        "technical_analysis": primary_tech,
        "disclaimer": DISCLAIMER_TEXT,
    }
