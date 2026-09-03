"""
analyze.py

POST /analyze/single    - Mode A: find the fairest cutting line for one object
POST /analyze/multiple  - Mode B: detect N objects and split them fairly
"""

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..vision.object_detection import find_largest_object, find_objects, DetectionError
from ..vision.segmentation import object_mask
from ..vision.measurement import measure_contour, normalized_value
from ..vision.cutting import find_equal_split_line
from ..algorithms.fair_split import fair_split
from ..algorithms.scoring import fairness_score, classify
from ..utils.verdicts import get_verdict, get_error_message, VISUAL_MEASUREMENT_DISCLAIMER
from ..utils.imaging import decode_upload_bytes, encode_bgr_to_data_url, InvalidImageError

router = APIRouter()

MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MB safety cap


async def _read_and_decode(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=get_error_message("invalid_image"))
    try:
        return decode_upload_bytes(data)
    except InvalidImageError:
        raise HTTPException(status_code=400, detail=get_error_message("invalid_image"))


def _fail(code: str) -> HTTPException:
    return HTTPException(status_code=422, detail=get_error_message(code))


@router.post("/analyze/single")
async def analyze_single(file: UploadFile = File(...)):
    image = await _read_and_decode(file)

    try:
        obj, resized, mask_all = find_largest_object(image)
    except DetectionError as e:
        raise _fail(str(e))

    measurements = measure_contour(obj.contour)
    mask = object_mask(resized.shape, obj)

    cut_line = find_equal_split_line(mask)
    if cut_line is None:
        raise _fail("no_cutting_line")

    score = fairness_score(cut_line.area1, cut_line.area2)
    tier = classify(score)
    verdict = get_verdict(tier.tier_index)

    # Annotated preview image: contour + cutting line drawn on the resized frame
    annotated = resized.copy()
    cv2.drawContours(annotated, [obj.contour], -1, (0, 200, 90), 2)
    p1 = (int(round(cut_line.x1)), int(round(cut_line.y1)))
    p2 = (int(round(cut_line.x2)), int(round(cut_line.y2)))
    cv2.line(annotated, p1, p2, (255, 90, 90), 3)

    return {
        "success": True,
        "object": measurements.to_dict(),
        "cut_line": cut_line.to_dict(),
        "piece1_percentage": round(cut_line.piece1_percentage, 2),
        "piece2_percentage": round(cut_line.piece2_percentage, 2),
        "fairness_score": round(score, 2),
        "classification": {"label": tier.label, "emoji": tier.emoji},
        "verdict": verdict,
        "annotated_image": encode_bgr_to_data_url(annotated),
        "disclaimer": VISUAL_MEASUREMENT_DISCLAIMER,
    }


@router.post("/analyze/multiple")
async def analyze_multiple(file: UploadFile = File(...), max_objects: int = Form(20)):
    image = await _read_and_decode(file)
    max_objects = max(2, min(60, max_objects))

    try:
        objects, resized, mask_all = find_objects(image, max_objects=max_objects)
    except DetectionError as e:
        raise _fail(str(e))

    if len(objects) < 2:
        raise _fail("too_many_objects" if len(objects) == 0 else "no_object")

    reference_area = max(o.area for o in objects)
    ids = [o.id for o in objects]
    values = [normalized_value(o.area, reference_area) for o in objects]

    result = fair_split(ids, values)

    score = fairness_score(result.group_a_value, result.group_b_value)
    tier = classify(score)
    verdict = get_verdict(tier.tier_index)

    id_to_obj = {o.id: o for o in objects}

    def describe(ids_list):
        return [
            {
                "id": oid,
                "area": round(id_to_obj[oid].area, 1),
                "bounding_rect": {
                    "x": id_to_obj[oid].bounding_rect[0], "y": id_to_obj[oid].bounding_rect[1],
                    "w": id_to_obj[oid].bounding_rect[2], "h": id_to_obj[oid].bounding_rect[3],
                },
            }
            for oid in ids_list
        ]

    annotated = resized.copy()
    colors = {oid: (60, 180, 255) for oid in result.group_a_ids}
    colors.update({oid: (255, 120, 200) for oid in result.group_b_ids})
    for obj in objects:
        cv2.drawContours(annotated, [obj.contour], -1, colors.get(obj.id, (0, 200, 90)), 2)
        x, y, w, h = obj.bounding_rect
        cv2.putText(annotated, f"#{obj.id}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(obj.id, (0, 200, 90)), 2)

    return {
        "success": True,
        "object_count": len(objects),
        "strategy": result.strategy,
        "group_a": describe(result.group_a_ids),
        "group_b": describe(result.group_b_ids),
        "group_a_value": round(result.group_a_value, 2),
        "group_b_value": round(result.group_b_value, 2),
        "group_a_percentage": round(result.group_a_percentage, 2),
        "group_b_percentage": round(result.group_b_percentage, 2),
        "fairness_score": round(score, 2),
        "classification": {"label": tier.label, "emoji": tier.emoji},
        "verdict": verdict,
        "annotated_image": encode_bgr_to_data_url(annotated),
        "disclaimer": VISUAL_MEASUREMENT_DISCLAIMER,
    }
