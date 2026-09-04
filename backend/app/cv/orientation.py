"""
orientation.py - Object mask-based PCA orientation estimation and adaptive canonical alignment.
Rotates ONLY the extracted physical object on a transparent background without rotating the original photo.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from .segmentation import SegmentedObject


@dataclass
class OrientationResult:
    detected_angle_deg: float          # Raw angle relative to horizontal
    correction_angle_deg: float        # Rotation applied to normalize upright
    alignment_method: str              # "principal_component_analysis", "rotational_symmetry_invariant"
    is_symmetric: bool                 # Whether object is rotationally symmetric
    orientation_significance: str      # "HIGH", "MODERATE", "LOW"
    original_bgr: np.ndarray           # Unmodified original photo
    ai_selected_bgr: np.ndarray        # Original photo with glowing selection overlay
    ai_aligned_rgba: np.ndarray        # Normalized upright object-only BGRA image (transparent background)
    ai_aligned_bgr: np.ndarray         # Normalized upright object BGR (for internal legacy CV passes)
    normalized_mask: np.ndarray        # Upright binary mask
    normalized_contour: np.ndarray     # Upright contour

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_angle_deg": round(self.detected_angle_deg, 1),
            "correction_angle_deg": round(self.correction_angle_deg, 1),
            "alignment_method": self.alignment_method,
            "is_symmetric": self.is_symmetric,
            "orientation_significance": self.orientation_significance,
        }


def estimate_mask_orientation(mask: np.ndarray, contour: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, bool, str]:
    """
    Computes orientation strictly from the object mask and contour points using covariance PCA.
    Returns (angle_deg, centroid, principal_axis_unit_vector, is_circular_symmetric, significance).
    """
    area = cv2.contourArea(contour) if contour is not None and len(contour) >= 3 else 0.0
    perimeter = cv2.arcLength(contour, True) if contour is not None and len(contour) >= 3 else 0.0
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    min_rect = cv2.minAreaRect(contour) if contour is not None and len(contour) >= 3 else ((0, 0), (1, 1), 0)
    rw, rh = min_rect[1]
    aspect_ratio = max(rw, rh) / max(1.0, min(rw, rh))

    if circularity > 0.82 and aspect_ratio < 1.25:
        # Rotational symmetry: orientation is invariant (e.g. round coin, plate, circular cup top)
        M = cv2.moments(contour)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else min_rect[0][0]
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else min_rect[0][1]
        return 90.0, np.array([cx, cy]), np.array([0.0, 1.0]), True, "LOW"

    # Extract foreground pixel coordinates (y, x)
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        pts = contour.reshape(-1, 2).astype(np.float64)
    else:
        pts = np.column_stack((xs, ys)).astype(np.float64)

    # PCA on foreground coordinates
    mean, eigenvectors = cv2.PCACompute(pts, mean=np.empty((0)))
    center = mean[0]
    principal_axis = eigenvectors[0]

    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = float(np.degrees(angle_rad))

    significance = "HIGH" if aspect_ratio > 2.0 else "MODERATE"
    return angle_deg, center, principal_axis, False, significance


def normalize_posture(image_bgr: np.ndarray, seg_obj: SegmentedObject) -> OrientationResult:
    """
    Performs object alignment based strictly on the object mask.
    Rotates ONLY the extracted object (with transparent background) and never the original photo canvas.
    """
    h, w = image_bgr.shape[:2]
    contour = seg_obj.contour
    mask = seg_obj.mask

    angle_deg, center, axis, is_symmetric, significance = estimate_mask_orientation(mask, contour)
    cx, cy = float(center[0]), float(center[1])

    if is_symmetric:
        correction_angle = 0.0
        method = "rotational_symmetry_invariant"
    else:
        # Target canonical upright vertical orientation (90 degrees)
        correction_angle = 90.0 - angle_deg
        while correction_angle > 90.0:
            correction_angle -= 180.0
        while correction_angle < -90.0:
            correction_angle += 180.0
        method = "principal_component_analysis"

    # Prepare extracted object RGBA (channel 0-2 = BGR, channel 3 = Alpha)
    obj_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    obj_rgba[:, :, :3] = seg_obj.original_bgr
    obj_rgba[:, :, 3] = mask

    # Compute rotation matrix around object center of mass
    M = cv2.getRotationMatrix2D((cx, cy), -correction_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # Rotate ONLY the extracted object RGBA with transparent constant border
    rotated_rgba = cv2.warpAffine(
        obj_rgba, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0, 0],
    )
    rotated_mask = cv2.warpAffine(
        mask, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Re-extract contour in the rotated coordinate frame
    rot_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if rot_contours:
        norm_contour = max(rot_contours, key=cv2.contourArea)
    else:
        norm_contour = contour

    # Find tight bounding box around the rotated object mask
    rx, ry, rw, rh = cv2.boundingRect(norm_contour)
    pad = 14
    x0 = max(0, rx - pad)
    y0 = max(0, ry - pad)
    x1 = min(new_w, rx + rw + pad)
    y1 = min(new_h, ry + rh + pad)

    # Tight crop containing only the object on transparent background
    normalized_crop_rgba = rotated_rgba[y0:y1, x0:x1].copy()
    crop_mask = rotated_mask[y0:y1, x0:x1].copy()
    shifted_contour = norm_contour - np.array([x0, y0])

    # Internal BGR representation (with black background for legacy CV features if needed)
    crop_bgr = normalized_crop_rgba[:, :, :3].copy()

    return OrientationResult(
        detected_angle_deg=round(float(angle_deg), 1),
        correction_angle_deg=round(float(correction_angle), 1),
        alignment_method=method,
        is_symmetric=is_symmetric,
        orientation_significance=significance,
        original_bgr=seg_obj.original_bgr,
        ai_selected_bgr=seg_obj.ai_selected_bgr,
        ai_aligned_rgba=normalized_crop_rgba,
        ai_aligned_bgr=crop_bgr,
        normalized_mask=crop_mask,
        normalized_contour=shifted_contour,
    )
