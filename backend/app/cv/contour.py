"""
contour.py - Canny edge maps, edge density, and contour outline similarity.
"""

from dataclasses import dataclass
from typing import Dict, Any
import cv2
import numpy as np


@dataclass
class EdgeAnalysis:
    edge_density_pct: float
    total_edge_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_density_pct": round(self.edge_density_pct, 2),
            "total_edge_pixels": self.total_edge_pixels,
        }


def extract_edge_analysis(image_bgr: np.ndarray, mask: np.ndarray) -> EdgeAnalysis:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    fg_edges = np.logical_and(edges > 0, mask > 0)
    edge_pixels = int(np.count_nonzero(fg_edges))
    fg_area = int(np.count_nonzero(mask))

    density = (float(edge_pixels) / float(max(1, fg_area))) * 100.0

    return EdgeAnalysis(
        edge_density_pct=density,
        total_edge_pixels=edge_pixels,
    )


def compare_contour_shapes(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Compares two contours using cv2.matchShapes (Hu Moments, method I1).
    Lower distance = more similar. Converts to 0 - 100% similarity score.
    """
    if len(c1) < 4 or len(c2) < 4:
        return 50.0
    dist = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
    # Distance typically ranges 0.0 (identical) to 2.0+ (different)
    score = max(0.0, 100.0 * np.exp(-1.8 * dist))
    return round(float(score), 1)
