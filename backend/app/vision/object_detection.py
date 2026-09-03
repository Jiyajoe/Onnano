"""
object_detection.py

Generic object detection from a camera frame using classical OpenCV
techniques: resize -> blur -> color conversion -> adaptive threshold ->
morphology -> contour extraction.

Deliberately NOT specialized to any one object class (no "chocolate"
logic). Works on whatever visually-distinguishable blobs appear against
a reasonably plain/contrasting background.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np

# ---- Tunable configuration (kept here so behaviour is easy to reason about) ----

MAX_DIMENSION = 900          # working resolution objects are resized to
MIN_OBJECT_AREA_RATIO = 0.001   # an object smaller than this fraction of the
                                 # frame area is treated as noise
MAX_OBJECT_COUNT_DEFAULT = 20    # safeguard against runaway detection / DP cost
BLUR_KERNEL = (5, 5)
MORPH_KERNEL_SIZE = 5


@dataclass
class DetectedObject:
    contour: np.ndarray
    area: float
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float]
    perimeter: float
    id: int = 0

    def to_dict(self):
        x, y, w, h = self.bounding_rect
        return {
            "id": self.id,
            "area": round(float(self.area), 2),
            "bounding_rect": {"x": x, "y": y, "w": w, "h": h},
            "centroid": {"x": round(self.centroid[0], 1), "y": round(self.centroid[1], 1)},
            "perimeter": round(float(self.perimeter), 2),
        }


class DetectionError(Exception):
    """Raised for user-recoverable detection problems (bad lighting, no object, etc.)"""


def _resize_keep_aspect(image: np.ndarray, max_dim: int = MAX_DIMENSION) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def _check_lighting_and_contrast(gray: np.ndarray) -> None:
    """Cheap sanity checks -> friendly errors instead of garbage detections."""
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if mean_brightness < 25:
        raise DetectionError("too_dark")
    if mean_brightness > 245:
        raise DetectionError("too_bright")
    if contrast < 8:
        raise DetectionError("poor_contrast")


def _blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian - a standard, cheap blur metric."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (resized_bgr, binary_mask, scale_factor).
    Raises DetectionError on unusable input.
    """
    resized, scale = _resize_keep_aspect(image_bgr)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _check_lighting_and_contrast(gray)

    if _blur_score(gray) < 15:
        raise DetectionError("blurry_image")

    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    # Otsu gives a solid automatic global threshold for plain-background shots;
    # adaptive threshold is used as a fallback when Otsu produces a mask that is
    # almost entirely foreground or background (a sign of uneven lighting).
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    fg_ratio = float(np.count_nonzero(otsu_mask)) / otsu_mask.size
    if fg_ratio < 0.005 or fg_ratio > 0.85:
        adaptive_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 5,
        )
        mask = adaptive_mask
    else:
        mask = otsu_mask

    # Morphological cleanup: close small gaps, remove speckle noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return resized, mask, scale


def find_objects(image_bgr: np.ndarray, max_objects: int = MAX_OBJECT_COUNT_DEFAULT) -> Tuple[List[DetectedObject], np.ndarray, np.ndarray]:
    """
    Detect all plausible objects in the frame.
    Returns (objects, resized_image, mask).
    """
    resized, mask, _scale = preprocess(image_bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = resized.shape[0] * resized.shape[1]
    min_area = frame_area * MIN_OBJECT_AREA_RATIO

    objects: List[DetectedObject] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        perim = cv2.arcLength(c, True)
        objects.append(DetectedObject(
            contour=c, area=area, bounding_rect=(x, y, w, h),
            centroid=(cx, cy), perimeter=perim,
        ))

    if not objects:
        raise DetectionError("no_object")

    # Largest first, then assign stable ids
    objects.sort(key=lambda o: o.area, reverse=True)
    for i, obj in enumerate(objects):
        obj.id = i + 1

    if len(objects) > max_objects:
        raise DetectionError("too_many_objects")

    return objects, resized, mask


def find_largest_object(image_bgr: np.ndarray) -> Tuple[DetectedObject, np.ndarray, np.ndarray]:
    """Convenience wrapper for Mode A (single object) - returns the largest blob only."""
    objects, resized, mask = find_objects(image_bgr, max_objects=MAX_OBJECT_COUNT_DEFAULT)
    return objects[0], resized, mask


def draw_contours(image_bgr: np.ndarray, objects: List[DetectedObject], color=(0, 200, 90), thickness=2) -> np.ndarray:
    out = image_bgr.copy()
    for obj in objects:
        cv2.drawContours(out, [obj.contour], -1, color, thickness)
        x, y, w, h = obj.bounding_rect
        cv2.putText(out, f"#{obj.id}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
