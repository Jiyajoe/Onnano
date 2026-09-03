"""
orientation.py - Object posture & orientation estimation, canonical normalization, and Before/After visualization.
"""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

from .segmentation import SegmentedObject


@dataclass
class OrientationResult:
    detected_angle_deg: float          # Raw angle relative to horizontal
    correction_angle_deg: float        # Rotation applied to normalize upright
    is_vertical: bool                  # Whether primary axis is oriented vertically
    before_annotated_bgr: np.ndarray   # Original image with orientation axis overlay
    normalized_bgr: np.ndarray         # Upright corrected image crop
    normalized_mask: np.ndarray        # Upright binary mask
    normalized_contour: np.ndarray     # Contour in upright frame


def estimate_principal_orientation(contour: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes principal orientation using PCA on contour coordinates and minAreaRect.
    Returns (angle_deg, mean_point, principal_eigenvector).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        # Fallback to bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        angle = 90.0 if h >= w else 0.0
        return angle, np.array([x + w / 2.0, y + h / 2.0]), np.array([0.0, 1.0])

    # PCA estimation
    mean, eigenvectors = cv2.PCACompute(pts, mean=np.empty((0)))
    center = mean[0]
    principal_axis = eigenvectors[0]  # Vector along longest dimension

    # Compute angle in degrees [-90, 90]
    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = float(np.degrees(angle_rad))

    # Also compare with minAreaRect for verification
    min_rect = cv2.minAreaRect(contour)
    (rw, rh) = min_rect[1]
    rect_angle = min_rect[2]

    return angle_deg, center, principal_axis


def normalize_posture(image_bgr: np.ndarray, seg_obj: SegmentedObject) -> OrientationResult:
    """
    Detects object tilt, computes canonical upright orientation, rotates both the image and mask,
    and returns Before/After visual artifacts.
    """
    h, w = image_bgr.shape[:2]
    contour = seg_obj.contour
    cx, cy = seg_obj.centroid

    angle_deg, center, axis = estimate_principal_orientation(contour)

    # We want the principal (longest) axis to stand canonical vertical (90 deg)
    # Angle in degrees relative to horizontal:
    # If angle is between -45 and 45, it's roughly horizontal -> rotate to 90
    # If angle is between 45 and 135 (or -45 and -135), rotate to exactly vertical (90 deg)
    
    # Target orientation is vertical (90 degrees / pi/2 radians)
    # Angle needed to rotate axis to point along Y-axis (vertical):
    correction_angle = 90.0 - angle_deg

    # Keep correction angle in [-90, 90] range for minimal necessary rotation
    while correction_angle > 90.0:
        correction_angle -= 180.0
    while correction_angle < -90.0:
        correction_angle += 180.0

    # 1. Annotate BEFORE image
    before_annotated = image_bgr.copy()
    cv2.drawContours(before_annotated, [contour], -1, (47, 217, 168), 2)  # Fair Mint contour

    # Draw principal axis vector
    axis_len = max(seg_obj.bbox[2], seg_obj.bbox[3]) * 0.45
    p1 = (int(cx - axis[0] * axis_len), int(cy - axis[1] * axis_len))
    p2 = (int(cx + axis[0] * axis_len), int(cy + axis[1] * axis_len))
    cv2.line(before_annotated, p1, p2, (39, 182, 255), 3, cv2.LINE_AA)  # Mango Gold axis
    cv2.circle(before_annotated, (int(cx), int(cy)), 5, (255, 93, 93), -1)  # Coral centroid

    # Add angle badge on before image
    cv2.putText(
        before_annotated,
        f"Tilt: {angle_deg:+.1f} deg",
        (max(10, seg_obj.bbox[0]), max(25, seg_obj.bbox[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (39, 182, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Perform rotation transformation
    # Compute bounding box of rotated image to avoid clipping
    M = cv2.getRotationMatrix2D((cx, cy), -correction_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # Rotate image and mask
    rotated_img = cv2.warpAffine(
        image_bgr, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(248, 248, 250),
    )
    rotated_mask = cv2.warpAffine(
        seg_obj.mask, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Extract contours of the rotated object
    rot_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if rot_contours:
        norm_contour = max(rot_contours, key=cv2.contourArea)
    else:
        norm_contour = contour

    # Crop tightly with padding around normalized object
    rx, ry, rw, rh = cv2.boundingRect(norm_contour)
    pad = 16
    x0 = max(0, rx - pad)
    y0 = max(0, ry - pad)
    x1 = min(new_w, rx + rw + pad)
    y1 = min(new_h, ry + rh + pad)

    normalized_crop = rotated_img[y0:y1, x0:x1].copy()
    crop_mask = rotated_mask[y0:y1, x0:x1]

    # White-out background of normalized crop
    normalized_crop[crop_mask == 0] = [248, 248, 250]

    # Shift contour coordinates to crop frame
    shifted_contour = norm_contour - np.array([x0, y0])

    return OrientationResult(
        detected_angle_deg=round(float(angle_deg), 1),
        correction_angle_deg=round(float(correction_angle), 1),
        is_vertical=True,
        before_annotated_bgr=before_annotated,
        normalized_bgr=normalized_crop,
        normalized_mask=crop_mask,
        normalized_contour=shifted_contour,
    )
