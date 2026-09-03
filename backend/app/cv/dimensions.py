"""
dimensions.py - Relative image-based dimensions and bounding box metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any
import cv2
import numpy as np


@dataclass
class DimensionMetrics:
    pixel_width: int
    pixel_height: int
    aspect_ratio: float
    bounding_box: Dict[str, int]  # x, y, width, height
    pixel_area: int
    relative_frame_area_pct: float
    unit_label: str = "px (Relative / Image-based)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "bounding_box": self.bounding_box,
            "pixel_area": self.pixel_area,
            "relative_frame_area_pct": round(self.relative_frame_area_pct, 2),
            "unit_label": self.unit_label,
        }


def extract_dimensions(contour: np.ndarray, total_frame_area: float) -> DimensionMetrics:
    x, y, w, h = cv2.boundingRect(contour)
    pixel_area = int(cv2.contourArea(contour))
    aspect_ratio = float(h) / float(max(1, w))
    relative_pct = (float(pixel_area) / float(max(1.0, total_frame_area))) * 100.0

    return DimensionMetrics(
        pixel_width=w,
        pixel_height=h,
        aspect_ratio=aspect_ratio,
        bounding_box={"x": x, "y": y, "width": w, "height": h},
        pixel_area=pixel_area,
        relative_frame_area_pct=relative_pct,
        unit_label="px (Relative / Image-based)",
    )
