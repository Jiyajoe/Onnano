"""
segmentation.py - Foreground isolation and primary object detection.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np

from ..config import (
    MAX_IMAGE_DIMENSION,
    MIN_OBJECT_AREA_RATIO,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_CONTRAST,
    MIN_LAPLACIAN_VAR,
)


class SegmentationError(Exception):
    """User-facing segmentation error."""
    pass


@dataclass
class SegmentedObject:
    contour: np.ndarray
    area: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float]
    mask: np.ndarray                 # 2D uint8 binary mask (255 inside object)
    isolated_rgba: np.ndarray        # BGRA crop with transparent background
    cropped_bgr: np.ndarray          # BGR crop with white/neutral background
    id: int = 1


def resize_keep_aspect(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def validate_image_quality(gray: np.ndarray) -> None:
    mean_b = float(np.mean(gray))
    std_b = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if mean_b < MIN_BRIGHTNESS:
        raise SegmentationError("The image is too dark. Please take the photo in better lighting.")
    if mean_b > MAX_BRIGHTNESS:
        raise SegmentationError("The image is overexposed. Please reduce direct glare or bright backlight.")
    if std_b < MIN_CONTRAST:
        raise SegmentationError("Poor contrast detected. Please place the object against a contrasting background.")
    if lap_var < MIN_LAPLACIAN_VAR:
        raise SegmentationError("The photo appears blurry. Please hold the camera steady and try again.")


def isolate_primary_object(image_bgr: np.ndarray) -> Tuple[SegmentedObject, np.ndarray]:
    """
    Isolates the single primary object from the image.
    Returns (SegmentedObject, resized_image_bgr).
    """
    resized, _ = resize_keep_aspect(image_bgr)
    h, w = resized.shape[:2]
    frame_area = float(h * w)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    validate_image_quality(gray)

    # Multi-method segmentation:
    # 1. Color distance from border (detecting background color from corner patches)
    corners = np.vstack([
        resized[0:15, 0:15].reshape(-1, 3),
        resized[0:15, -15:].reshape(-1, 3),
        resized[-15:, 0:15].reshape(-1, 3),
        resized[-15:, -15:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)
    color_dist = np.linalg.norm(resized.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    dist_thresh = np.percentile(color_dist, 70) * 0.45
    color_mask = (color_dist > max(18.0, dist_thresh)).astype(np.uint8) * 255

    # 2. Otsu thresholding with bilateral filtering to preserve edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu_inv = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu_norm = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Adaptive thresholding fallback
    adaptive = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
    )

    # Select best initial mask based on reasonable foreground ratio (5% - 85%)
    candidates = [color_mask, otsu_inv, otsu_norm, adaptive]
    best_mask = color_mask
    best_score = -1.0

    for m in candidates:
        fg_ratio = np.count_nonzero(m) / float(frame_area)
        if 0.03 <= fg_ratio <= 0.85:
            # Score candidate by compactness and connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
            if num_labels > 1:
                largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
                score = largest_area / float(frame_area)
                if score > best_score:
                    best_score = score
                    best_mask = m

    # Morphological cleaning: close gaps, remove salt-and-pepper noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = frame_area * MIN_OBJECT_AREA_RATIO

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            valid_contours.append((c, area))

    if not valid_contours:
        raise SegmentationError("We couldn't clearly detect the object. Try placing it on a plain, contrasting background.")

    # Sort by area descending
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    primary_contour, primary_area = valid_contours[0]

    # Create refined single-object mask
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(obj_mask, [primary_contour], -1, 255, -1)

    # Fill small holes inside the primary contour
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=2)

    # Re-extract clean contour from smoothed mask
    refined_contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if refined_contours:
        primary_contour = max(refined_contours, key=cv2.contourArea)
        primary_area = cv2.contourArea(primary_contour)

    # Bounding rect & centroid
    x, y, bw, bh = cv2.boundingRect(primary_contour)
    M = cv2.moments(primary_contour)
    cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + bw / 2.0
    cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + bh / 2.0

    # Extract isolated RGBA with transparency
    b, g, r = cv2.split(resized)
    rgba = cv2.merge([b, g, r, obj_mask])

    # Crop with small margin
    pad = 12
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)

    isolated_crop = rgba[y0:y1, x0:x1]
    
    # BGR crop with white background
    crop_bgr = resized[y0:y1, x0:x1].copy()
    crop_mask = obj_mask[y0:y1, x0:x1]
    crop_bgr[crop_mask == 0] = [248, 248, 250]

    seg_obj = SegmentedObject(
        contour=primary_contour,
        area=float(primary_area),
        bbox=(x, y, bw, bh),
        centroid=(float(cx), float(cy)),
        mask=obj_mask,
        isolated_rgba=isolated_crop,
        cropped_bgr=crop_bgr,
        id=1,
    )

    return seg_obj, resized
