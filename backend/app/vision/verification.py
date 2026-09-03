"""
verification.py

After the human actually cuts the object (Mode A) or distributes the
objects into two piles (Mode B), the app re-scans and checks how close
the real-world result came to the recommendation.
"""

from typing import List
import numpy as np

from .object_detection import find_objects, DetectionError, DetectedObject


def verify_single_cut(image_bgr: np.ndarray):
    """
    Expects exactly two pieces visible in frame after cutting.
    Returns (piece1_pct, piece2_pct, objects) or raises DetectionError.
    """
    objects, resized, mask = find_objects(image_bgr, max_objects=10)

    if len(objects) < 2:
        raise DetectionError("less_than_two_pieces")

    # Take the two largest contours as "the two pieces" (spec section 16.4)
    top_two = sorted(objects, key=lambda o: o.area, reverse=True)[:2]
    total = sum(o.area for o in top_two)
    if total <= 0:
        raise DetectionError("no_object")

    pct1 = 100.0 * top_two[0].area / total
    pct2 = 100.0 * top_two[1].area / total
    return pct1, pct2, top_two, resized, mask


def verify_multiple_distribution(image_bgr: np.ndarray, expected_group_a: int, expected_group_b: int):
    """
    Expects the whole scene (both piles) in frame, OR is called twice - once
    per pile from the frontend. Here we simply count+measure whatever objects
    are visible and let the route layer decide how to compare against the
    expected allocation.
    """
    objects, resized, mask = find_objects(image_bgr, max_objects=40)
    return objects, resized, mask
