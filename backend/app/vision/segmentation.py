"""
segmentation.py

Turns a DetectedObject's contour into a clean, filled binary mask that
downstream modules (measurement, cutting) operate on. Kept separate from
object_detection so the "find candidate blobs" step and the
"produce an analysis-ready mask for one object" step can evolve independently
(e.g. swapping in GrabCut or a learned segmenter later without touching
detection code).
"""

from typing import Tuple
import cv2
import numpy as np

from .object_detection import DetectedObject


def object_mask(image_shape: Tuple[int, int], obj: DetectedObject) -> np.ndarray:
    """Binary (0/255) filled mask for a single object's contour, full frame size."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [obj.contour], -1, 255, thickness=cv2.FILLED)
    return mask


def refine_with_grabcut(image_bgr: np.ndarray, obj: DetectedObject, iterations: int = 3) -> np.ndarray:
    """
    Optional refinement pass: seeds GrabCut with the contour's bounding box
    to pull the mask tighter to the true object edges. Falls back to the
    plain contour mask if GrabCut fails for any reason (e.g. degenerate rect).
    """
    x, y, w, h = obj.bounding_rect
    h_img, w_img = image_bgr.shape[:2]

    pad = int(0.05 * max(w, h)) + 2
    rect = (
        max(0, x - pad),
        max(0, y - pad),
        min(w_img - 1, w + 2 * pad),
        min(h_img - 1, h + 2 * pad),
    )

    if rect[2] <= 1 or rect[3] <= 1:
        return object_mask(image_bgr.shape, obj)

    try:
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Sanity check: refined mask shouldn't collapse to near-nothing
        if np.count_nonzero(refined) < 0.2 * obj.area:
            return object_mask(image_bgr.shape, obj)
        return refined
    except cv2.error:
        return object_mask(image_bgr.shape, obj)
