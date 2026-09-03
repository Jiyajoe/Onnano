"""
cutting.py

The equal-area split search described in the spec (section 8):

  1. Work from a binary mask of the segmented object.
  2. Generate candidate straight cutting lines at many orientations.
  3. For each orientation, find the *optimal offset* directly: project every
     foreground pixel onto the line's normal vector, sort the projections,
     and pick the split point whose cumulative pixel count is closest to
     half the object's area. This effectively performs the "try many
     offsets, keep the best" search for that orientation in O(n log n)
     instead of a nested brute-force loop.
  4. Across all orientations, keep the line(s) with the smallest area
     difference; break ties by preferring the physically shortest cut
     (simplest to actually make with a knife/scissors).

Works for rectangles, rotated rectangles, circles and irregular/organic
contours alike, since it never assumes a fixed shape model - it only
operates on the pixel mask.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CutLine:
    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float
    area1: float
    area2: float
    total_area: float

    @property
    def piece1_percentage(self) -> float:
        return 100.0 * self.area1 / self.total_area if self.total_area else 0.0

    @property
    def piece2_percentage(self) -> float:
        return 100.0 * self.area2 / self.total_area if self.total_area else 0.0

    def to_dict(self):
        return {
            "x1": round(self.x1, 1), "y1": round(self.y1, 1),
            "x2": round(self.x2, 1), "y2": round(self.y2, 1),
            "angle_deg": round(self.angle_deg, 1),
            "piece1_area": round(self.area1, 1),
            "piece2_area": round(self.area2, 1),
            "piece1_percentage": round(self.piece1_percentage, 2),
            "piece2_percentage": round(self.piece2_percentage, 2),
        }


def _clip_line_to_box(point: np.ndarray, direction: np.ndarray, box, pad: float = 0.0):
    """
    Clip the infinite line {point + t*direction} against an axis-aligned box
    (x_min, y_min, x_max, y_max), returning the two boundary intersection
    points that bound the visible segment. Uses the Liang-Barsky approach.
    """
    x_min, y_min, x_max, y_max = box
    x_min -= pad; y_min -= pad; x_max += pad; y_max += pad

    t0, t1 = -1e9, 1e9
    px, py = point
    dx, dy = direction

    for p, q in ((-dx, px - x_min), (dx, x_max - px), (-dy, py - y_min), (dy, y_max - py)):
        if abs(p) < 1e-9:
            if q < 0:
                return None  # parallel and outside -> no intersection
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)

    if t0 > t1:
        return None

    p_start = point + t0 * direction
    p_end = point + t1 * direction
    return p_start, p_end


def find_equal_split_line(mask: np.ndarray, angle_step_deg: float = 3.0) -> Optional[CutLine]:
    """
    Search over candidate orientations/offsets and return the best straight
    line dividing the object mask into two ~equal-area pieces.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    points = np.column_stack((xs, ys)).astype(np.float64)
    total = len(points)
    half = total / 2.0

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    box = (x_min, y_min, x_max, y_max)

    best: Optional[Tuple[float, float, np.ndarray, np.ndarray, float, float, float]] = None
    # tuple: (diff, cut_length, p_start, p_end, angle_deg, area1, area2)

    angles = np.arange(0.0, 180.0, angle_step_deg)

    for angle_deg in angles:
        theta = np.radians(angle_deg)
        direction = np.array([np.cos(theta), np.sin(theta)])   # line runs along this
        normal = np.array([-np.sin(theta), np.cos(theta)])     # split axis

        proj = points @ normal
        sorted_proj = np.sort(proj)

        # index k => first k points (by projection) go to region 1.
        # Splitting exactly at the midpoint of the sorted projections gives
        # the offset with the smallest achievable area difference for this
        # orientation (this is the "try many offsets" search, solved directly).
        k = int(round(half))
        k = max(1, min(total - 1, k))
        threshold = sorted_proj[k]

        area1 = k
        area2 = total - k
        diff = abs(area1 - area2)

        # point on the line closest to the origin (since normal is unit length)
        line_point = normal * threshold

        clipped = _clip_line_to_box(line_point, direction, box, pad=1.0)
        if clipped is None:
            continue
        p_start, p_end = clipped
        cut_length = float(np.linalg.norm(p_end - p_start))

        candidate = (diff, cut_length, p_start, p_end, angle_deg, area1, area2)

        if best is None:
            best = candidate
        else:
            best_diff, best_len = best[0], best[1]
            # Prefer smaller area difference; within a small tolerance (2% of area)
            # prefer the shorter/simpler cut.
            tolerance = max(1.0, 0.02 * total)
            if diff < best_diff - tolerance:
                best = candidate
            elif abs(diff - best_diff) <= tolerance and cut_length < best_len:
                best = candidate

    if best is None:
        return None

    diff, cut_length, p_start, p_end, angle_deg, area1, area2 = best
    return CutLine(
        x1=p_start[0], y1=p_start[1], x2=p_end[0], y2=p_end[1],
        angle_deg=float(angle_deg), area1=float(area1), area2=float(area2),
        total_area=float(total),
    )
