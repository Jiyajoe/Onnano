"""
dimensions.py - Relative image-based dimensions and physical calibration support.
Distinguishes visible pixel measurements from uncalibrated physical units.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import cv2
import numpy as np


@dataclass
class DimensionMetrics:
    # Visible raw pixel measurements (always accurate)
    pixel_width: int
    pixel_height: int
    contour_width: int
    contour_height: int
    aspect_ratio: float
    pixel_area: int
    perimeter: float
    extent: float
    bounding_box: Dict[str, int]  # x, y, width, height
    relative_frame_area_pct: float
    unit_label: str = "px (Image-based pixel measurement)"

    # Physical calibration fields (strictly gated on real reference)
    is_calibrated: bool = False
    pixels_per_cm: Optional[float] = None
    physical_height_cm: Optional[float] = None
    physical_width_cm: Optional[float] = None
    physical_area_cm2: Optional[float] = None
    calibration_status: str = "Physical measurement: unavailable (no calibration reference)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "contour_width": self.contour_width,
            "contour_height": self.contour_height,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "pixel_area": self.pixel_area,
            "perimeter": round(self.perimeter, 1),
            "extent": round(self.extent, 3),
            "bounding_box": self.bounding_box,
            "relative_frame_area_pct": round(self.relative_frame_area_pct, 2),
            "unit_label": self.unit_label,
            "is_calibrated": self.is_calibrated,
            "pixels_per_cm": round(self.pixels_per_cm, 2) if self.pixels_per_cm else None,
            "physical_height_cm": round(self.physical_height_cm, 2) if self.physical_height_cm else None,
            "physical_width_cm": round(self.physical_width_cm, 2) if self.physical_width_cm else None,
            "physical_area_cm2": round(self.physical_area_cm2, 2) if self.physical_area_cm2 else None,
            "calibration_status": self.calibration_status,
        }


def extract_dimensions(
    contour: np.ndarray,
    total_frame_area: float,
    reference_length_cm: Optional[float] = None,
    reference_pixel_span: Optional[float] = None,
) -> DimensionMetrics:
    """
    Measures visible pixel dimensions and optionally applies calibrated physical conversion.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pixel_area = int(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    # Min area bounding rect gives rotated contour dimensions
    min_rect = cv2.minAreaRect(contour)
    rw, rh = min_rect[1]
    longer = max(rw, rh)
    shorter = max(1.0, min(rw, rh))

    aspect_ratio = float(longer) / float(shorter)
    bbox_area = float(max(1, w * h))
    extent = float(pixel_area) / bbox_area
    relative_pct = (float(pixel_area) / float(max(1.0, total_frame_area))) * 100.0

    # Calibration logic: ONLY when explicit real-world reference is supplied
    is_calibrated = False
    px_per_cm = None
    phys_h = None
    phys_w = None
    phys_area = None
    cal_status = "Physical measurement: unavailable (no calibration reference)"

    if reference_length_cm and reference_length_cm > 0:
        span = reference_pixel_span if (reference_pixel_span and reference_pixel_span > 0) else float(longer)
        if span > 10:
            px_per_cm = span / float(reference_length_cm)
            phys_h = float(longer) / px_per_cm
            phys_w = float(shorter) / px_per_cm
            phys_area = float(pixel_area) / (px_per_cm ** 2)
            is_calibrated = True
            cal_status = f"Calibrated with reference scale ({px_per_cm:.1f} px/cm)"

    return DimensionMetrics(
        pixel_width=w,
        pixel_height=h,
        contour_width=int(round(shorter)),
        contour_height=int(round(longer)),
        aspect_ratio=aspect_ratio,
        pixel_area=pixel_area,
        perimeter=perimeter,
        extent=extent,
        bounding_box={"x": x, "y": y, "width": w, "height": h},
        relative_frame_area_pct=relative_pct,
        unit_label="px (Image-based pixel measurement)",
        is_calibrated=is_calibrated,
        pixels_per_cm=px_per_cm,
        physical_height_cm=phys_h,
        physical_width_cm=phys_w,
        physical_area_cm2=phys_area,
        calibration_status=cal_status,
    )
