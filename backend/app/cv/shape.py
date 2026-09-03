"""
shape.py - Comprehensive geometric shape analysis, symmetry, and descriptors.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import cv2
import numpy as np


@dataclass
class ShapeProperties:
    shape_type: str            # e.g., "Elongated / Rod", "Rectangular / Box", "Circular / Round"
    contour_points: int        # Number of vertices in simplified contour
    circularity: float         # 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
    aspect_ratio: float        # Height / Width (along principal axes)
    solidity: float            # Area / Convex Hull Area (1.0 for convex shapes)
    rectangularity: float      # Area / MinAreaRect Area
    symmetry_score: float      # 0 to 100% bilateral symmetry
    hu_moments: List[float]    # 7 Hu invariant moments (log scaled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape_type": self.shape_type,
            "contour_points": self.contour_points,
            "circularity": round(self.circularity, 3),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "solidity": round(self.solidity, 3),
            "rectangularity": round(self.rectangularity, 3),
            "symmetry_score": round(self.symmetry_score, 1),
            "hu_moments": [round(m, 4) for m in self.hu_moments],
        }


def calculate_symmetry(mask: np.ndarray) -> float:
    """Calculates bilateral symmetry percentage around the vertical center of the mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0

    cx = int(round(np.mean(xs)))
    # Flip mask horizontally around center cx
    h, w = mask.shape[:2]
    # Crop symmetric window around cx
    left_dist = cx
    right_dist = w - cx
    max_d = min(left_dist, right_dist)

    if max_d < 5:
        return 50.0

    left_half = mask[:, cx - max_d:cx]
    right_half = mask[:, cx:cx + max_d]
    flipped_right = cv2.flip(right_half, 1)

    intersection = np.count_nonzero(np.logical_and(left_half, flipped_right))
    union = np.count_nonzero(np.logical_or(left_half, flipped_right))

    if union == 0:
        return 100.0
    iou = float(intersection) / float(union)
    return float(iou * 100.0)


def extract_shape_properties(contour: np.ndarray, mask: np.ndarray) -> ShapeProperties:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0 or area <= 0:
        return ShapeProperties(
            shape_type="Undetermined",
            contour_points=0,
            circularity=0.0,
            aspect_ratio=1.0,
            solidity=0.0,
            rectangularity=0.0,
            symmetry_score=0.0,
            hu_moments=[0.0] * 7,
        )

    # 1. Circularity: 4*pi*Area / P^2
    circularity = min(1.0, (4.0 * np.pi * area) / (perimeter ** 2))

    # 2. Minimum Area Rectangle and Aspect Ratio
    min_rect = cv2.minAreaRect(contour)
    rw, rh = min_rect[1]
    longer = max(rw, rh)
    shorter = max(1.0, min(rw, rh))
    aspect_ratio = longer / shorter
    min_rect_area = rw * rh

    # 3. Solidity: Area / ConvexHullArea
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = min(1.0, area / hull_area) if hull_area > 0 else 0.0

    # 4. Rectangularity: Area / MinAreaRectArea
    rectangularity = min(1.0, area / min_rect_area) if min_rect_area > 0 else 0.0

    # 5. Simplified contour points
    epsilon = 0.015 * perimeter
    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
    contour_points = len(approx_poly)

    # 6. Symmetry Score
    symmetry = calculate_symmetry(mask)

    # 7. Hu Moments (Log scale)
    moments = cv2.moments(contour)
    raw_hu = cv2.HuMoments(moments).flatten()
    log_hu = []
    for h in raw_hu:
        if abs(h) > 1e-12:
            log_hu.append(-1.0 * np.sign(h) * np.log10(abs(h)))
        else:
            log_hu.append(0.0)

    # 8. Shape Classification
    if circularity > 0.82 and aspect_ratio < 1.3:
        shape_type = "Circular / Round"
    elif aspect_ratio > 3.8:
        shape_type = "Elongated / Rod"
    elif rectangularity > 0.78 and solidity > 0.90:
        shape_type = "Rectangular / Box"
    elif aspect_ratio > 1.8:
        shape_type = "Oblong / Cylindrical"
    elif solidity < 0.75:
        shape_type = "Complex / Concave"
    else:
        shape_type = "Organic / Polygon"

    return ShapeProperties(
        shape_type=shape_type,
        contour_points=contour_points,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        solidity=solidity,
        rectangularity=rectangularity,
        symmetry_score=symmetry,
        hu_moments=log_hu,
    )
