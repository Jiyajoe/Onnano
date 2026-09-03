"""
measurement.py

Geometric measurements derived from a contour/mask: area, bounding rect,
minimum-area (rotated) rectangle, orientation, width/height, centroid.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ShapeMeasurements:
    area: float
    perimeter: float
    bbox: tuple            # x, y, w, h (axis-aligned)
    centroid: tuple        # cx, cy
    min_area_rect: tuple   # ((cx, cy), (w, h), angle) - rotated rect
    orientation_deg: float
    width: float
    height: float
    circularity: float     # 4*pi*area / perimeter^2 -> 1.0 for a perfect circle

    def to_dict(self):
        (rcx, rcy), (rw, rh), angle = self.min_area_rect
        return {
            "area": round(self.area, 2),
            "perimeter": round(self.perimeter, 2),
            "bbox": {"x": self.bbox[0], "y": self.bbox[1], "w": self.bbox[2], "h": self.bbox[3]},
            "centroid": {"x": round(self.centroid[0], 1), "y": round(self.centroid[1], 1)},
            "min_area_rect": {
                "center": {"x": round(rcx, 1), "y": round(rcy, 1)},
                "size": {"w": round(rw, 1), "h": round(rh, 1)},
                "angle": round(angle, 1),
            },
            "orientation_deg": round(self.orientation_deg, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "circularity": round(self.circularity, 3),
        }


def measure_contour(contour: np.ndarray) -> ShapeMeasurements:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    bbox = cv2.boundingRect(contour)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    else:
        x, y, w, h = bbox
        centroid = (x + w / 2, y + h / 2)

    min_rect = cv2.minAreaRect(contour)
    (_rw, _rh), angle = min_rect[1], min_rect[2]
    width, height = min_rect[1]

    # orientation of the object's principal axis (0-180 deg)
    orientation = angle if width >= height else angle + 90
    orientation = orientation % 180

    circularity = 0.0
    if perimeter > 0:
        circularity = float(4 * np.pi * area / (perimeter ** 2))

    return ShapeMeasurements(
        area=area,
        perimeter=perimeter,
        bbox=bbox,
        centroid=centroid,
        min_area_rect=min_rect,
        orientation_deg=orientation,
        width=width,
        height=height,
        circularity=circularity,
    )


def normalized_value(area: float, reference_area: float) -> float:
    """Object 'value' used by the multi-object fairness optimizer, normalized to 0-100."""
    if reference_area <= 0:
        return 0.0
    return float(area / reference_area * 100.0)
