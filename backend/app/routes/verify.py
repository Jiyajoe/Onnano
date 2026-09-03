"""
verify.py

POST /verify/single    - rescan the two cut pieces, score actual fairness
POST /verify/multiple  - rescan the two distributed piles, compare to plan
"""

import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..vision.object_detection import find_objects, DetectionError
from ..vision.verification import verify_single_cut
from ..algorithms.scoring import fairness_score, classify
from ..utils.verdicts import get_verdict, get_error_message, VISUAL_MEASUREMENT_DISCLAIMER
from ..utils.imaging import decode_upload_bytes, encode_bgr_to_data_url, InvalidImageError

router = APIRouter()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


async def _read_and_decode(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=get_error_message("invalid_image"))
    try:
        return decode_upload_bytes(data)
    except InvalidImageError:
        raise HTTPException(status_code=400, detail=get_error_message("invalid_image"))


def _fail(code: str) -> HTTPException:
    return HTTPException(status_code=422, detail=get_error_message(code))


@router.post("/verify/single")
async def verify_single(file: UploadFile = File(...)):
    image = await _read_and_decode(file)

    try:
        pct1, pct2, pieces, resized, mask = verify_single_cut(image)
    except DetectionError as e:
        raise _fail(str(e))

    score = fairness_score(pct1, pct2)
    tier = classify(score)
    verdict = get_verdict(tier.tier_index)

    annotated = resized.copy()
    for i, obj in enumerate(pieces):
        color = (60, 180, 255) if i == 0 else (255, 120, 200)
        cv2.drawContours(annotated, [obj.contour], -1, color, 2)

    return {
        "success": True,
        "piece1_percentage": round(pct1, 2),
        "piece2_percentage": round(pct2, 2),
        "fairness_score": round(score, 2),
        "classification": {"label": tier.label, "emoji": tier.emoji},
        "verdict": verdict,
        "annotated_image": encode_bgr_to_data_url(annotated),
        "disclaimer": VISUAL_MEASUREMENT_DISCLAIMER,
    }


@router.post("/verify/multiple")
async def verify_multiple(
    file: UploadFile = File(...),
    expected_group_a: int = Form(0),
    expected_group_b: int = Form(0),
):
    image = await _read_and_decode(file)

    try:
        objects, resized, mask = find_objects(image, max_objects=60)
    except DetectionError as e:
        raise _fail(str(e))

    total_area = sum(o.area for o in objects)
    actual_count = len(objects)

    # Best-effort: split detected objects into two groups by spatial position
    # (left half / right half of frame) so a single "both piles in frame" shot
    # can still be scored, without assuming any particular arrangement.
    frame_w = resized.shape[1]
    left = [o for o in objects if o.centroid[0] < frame_w / 2]
    right = [o for o in objects if o.centroid[0] >= frame_w / 2]

    val_left = sum(o.area for o in left)
    val_right = sum(o.area for o in right)

    score = fairness_score(val_left, val_right) if (val_left + val_right) > 0 else 0.0
    tier = classify(score)
    verdict = get_verdict(tier.tier_index)

    annotated = resized.copy()
    for obj in left:
        cv2.drawContours(annotated, [obj.contour], -1, (60, 180, 255), 2)
    for obj in right:
        cv2.drawContours(annotated, [obj.contour], -1, (255, 120, 200), 2)
    cv2.line(annotated, (frame_w // 2, 0), (frame_w // 2, resized.shape[0]), (255, 255, 255), 1)

    return {
        "success": True,
        "actual_object_count": actual_count,
        "expected_group_a": expected_group_a,
        "expected_group_b": expected_group_b,
        "group_a_count": len(left),
        "group_b_count": len(right),
        "group_a_percentage": round(100.0 * val_left / total_area, 2) if total_area else 0.0,
        "group_b_percentage": round(100.0 * val_right / total_area, 2) if total_area else 0.0,
        "fairness_score": round(score, 2),
        "classification": {"label": tier.label, "emoji": tier.emoji},
        "verdict": verdict,
        "annotated_image": encode_bgr_to_data_url(annotated),
        "disclaimer": VISUAL_MEASUREMENT_DISCLAIMER,
        "note": "Place both piles side by side (left = Sibling A, right = Sibling B) in one frame for verification.",
    }
