"""
test_algorithms.py

Tests the mathematical core independently of FastAPI/React, per spec
section 26: single-object cutting on various shapes, multi-object fair
division, and error cases.

Run with:  python -m pytest tests/ -v
(or, without pytest installed:  python tests/test_algorithms.py)
"""

import sys
import os
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.vision.cutting import find_equal_split_line
from app.vision.object_detection import find_objects, find_largest_object, DetectionError
from app.algorithms.fair_split import fair_split
from app.algorithms.scoring import fairness_score, classify


# ---------------------------------------------------------------------------
# helpers to synthesize test masks/images
# ---------------------------------------------------------------------------

def rectangle_mask(w=400, h=400, rw=200, rh=100, angle=0):
    mask = np.zeros((h, w), dtype=np.uint8)
    rect = ((w / 2, h / 2), (rw, rh), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], 255)
    return mask


def circle_mask(w=400, h=400, r=120):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return mask


def irregular_polygon_mask(w=400, h=400):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [80, 300], [120, 90], [220, 40], [320, 120],
        [300, 260], [230, 340], [140, 360],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def synthetic_object_image(shape_masks_bgr_color=(30, 30, 30)):
    """Plain white background with black shapes drawn on it, for detection tests."""
    img = np.full((400, 400, 3), 245, dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# Single-object equal-area split
# ---------------------------------------------------------------------------

def _assert_balanced(mask, label, tolerance_pct=2.0):
    line = find_equal_split_line(mask)
    assert line is not None, f"{label}: no cut line found"
    total = np.count_nonzero(mask)
    diff_pct = 100.0 * abs(line.area1 - line.area2) / total
    assert diff_pct <= tolerance_pct, f"{label}: split off by {diff_pct:.2f}% (>{tolerance_pct}%)"
    print(f"  OK  {label}: {line.piece1_percentage:.2f}% / {line.piece2_percentage:.2f}%  (angle={line.angle_deg:.1f} deg)")


def test_rectangle():
    _assert_balanced(rectangle_mask(rw=240, rh=120, angle=0), "axis-aligned rectangle")


def test_square():
    _assert_balanced(rectangle_mask(rw=180, rh=180, angle=0), "square")


def test_rotated_rectangle():
    _assert_balanced(rectangle_mask(rw=220, rh=100, angle=37), "rotated rectangle (37 deg)")


def test_irregular_polygon():
    _assert_balanced(irregular_polygon_mask(), "irregular polygon")


def test_circle():
    _assert_balanced(circle_mask(r=130), "circle")


# ---------------------------------------------------------------------------
# Multi-object fair division
# ---------------------------------------------------------------------------

def test_equal_sized_objects():
    ids = list(range(1, 11))
    values = [100.0] * 10  # 10 identical chocolates
    result = fair_split(ids, values)
    assert len(result.group_a_ids) == 5 and len(result.group_b_ids) == 5
    score = fairness_score(result.group_a_value, result.group_b_value)
    assert score >= 99.0, f"equal objects should score ~100, got {score}"
    print(f"  OK  10 equal objects -> {len(result.group_a_ids)}v{len(result.group_b_ids)}, score={score:.2f}")


def test_different_sized_objects():
    ids = [1, 2, 3, 4, 5, 6]
    values = [100, 90, 80, 70, 60, 50]  # matches spec example
    result = fair_split(ids, values)
    score = fairness_score(result.group_a_value, result.group_b_value)
    assert score >= 95.0, f"expected near-perfect balance, got {score}"
    print(f"  OK  differently sized -> A={result.group_a_value} B={result.group_b_value}, score={score:.2f}")


def test_odd_number_of_objects():
    ids = [1, 2, 3, 4, 5]
    values = [50, 50, 50, 50, 50]
    result = fair_split(ids, values)
    assert len(result.group_a_ids) + len(result.group_b_ids) == 5
    diff = abs(len(result.group_a_ids) - len(result.group_b_ids))
    assert diff == 1
    print(f"  OK  odd count (5) -> {len(result.group_a_ids)}v{len(result.group_b_ids)}")


def test_highly_unequal_sizes():
    ids = [1, 2, 3]
    values = [500, 10, 5]
    result = fair_split(ids, values)
    score = fairness_score(result.group_a_value, result.group_b_value)
    # With one huge outlier, perfect balance is impossible - just sanity check it runs.
    assert 0 <= score <= 100
    print(f"  OK  highly unequal sizes -> score={score:.2f} (strategy={result.strategy})")


# ---------------------------------------------------------------------------
# Scoring / classification
# ---------------------------------------------------------------------------

def test_scoring_classification_tiers():
    assert classify(99).label == "PERFECTLY FAIR"
    assert classify(96).label == "ALMOST PERFECT"
    assert classify(91).label == "PRETTY FAIR"
    assert classify(60).label == "SIBLING FIGHT WARNING"
    print("  OK  classification tiers map correctly")


def test_fairness_score_bounds():
    assert fairness_score(0, 0) == 0.0
    assert math.isclose(fairness_score(50, 50), 100.0)
    assert math.isclose(fairness_score(100, 0), 0.0)
    print("  OK  fairness score bounds correct")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_empty_image_raises():
    blank = np.full((400, 400, 3), 128, dtype=np.uint8)  # uniform gray, no contrast
    try:
        find_objects(blank)
        raised = False
    except DetectionError:
        raised = True
    assert raised, "a flat, contrast-less image should raise DetectionError"
    print("  OK  empty/flat image correctly raises DetectionError")


def test_no_object_on_plain_background():
    img = np.full((300, 300, 3), 250, dtype=np.uint8)
    try:
        find_objects(img)
        raised = False
    except DetectionError:
        raised = True
    assert raised
    print("  OK  plain background with nothing on it correctly raises DetectionError")


def test_multiple_objects_detected_on_real_image():
    img = np.full((400, 400, 3), 250, dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, (30, 30, 30), -1)
    cv2.circle(img, (300, 100), 40, (30, 30, 30), -1)
    cv2.rectangle(img, (150, 250), (250, 320), (30, 30, 30), -1)
    objects, resized, mask = find_objects(img)
    assert len(objects) == 3
    print(f"  OK  3 distinct shapes on plain background -> detected {len(objects)}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {e!r}")
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
