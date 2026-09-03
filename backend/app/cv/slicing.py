"""
slicing.py - N-way geometric equal division along the normalized principal axis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

# Distinct vibrant colors for visual part segments (BGR)
PART_COLORS_BGR = [
    (39, 182, 255),   # Mango Gold
    (214, 107, 125),  # Soft Purple / Indigo
    (168, 217, 47),   # Fair Mint
    (93, 93, 255),    # Referee Coral
    (230, 110, 45),   # Cobalt Blue
    (45, 220, 240),   # Yellow
    (195, 35, 200),   # Pink
    (40, 180, 95),    # Emerald Green
]

PART_COLORS_HEX = [
    "#ffb627",
    "#7d6bd6",
    "#2fd9a8",
    "#ff5d5d",
    "#2d6ee6",
    "#f0dc2d",
    "#c823c3",
    "#28b45f",
]


@dataclass
class SlicedPart:
    index: int
    label: str
    pixel_height: int
    pixel_width: int
    pixel_area: int
    percentage: float
    color_hex: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "pixel_height": self.pixel_height,
            "pixel_width": self.pixel_width,
            "pixel_area": self.pixel_area,
            "percentage": round(self.percentage, 2),
            "color_hex": self.color_hex,
        }


@dataclass
class SlicingResult:
    parts_count: int
    parts: List[SlicedPart]
    divided_image_bgr: np.ndarray
    is_equal_split: bool
    division_axis: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parts_count": self.parts_count,
            "parts": [p.to_dict() for p in self.parts],
            "is_equal_split": self.is_equal_split,
            "division_axis": self.division_axis,
        }


def divide_normalized_object(
    normalized_bgr: np.ndarray,
    normalized_mask: np.ndarray,
    normalized_contour: np.ndarray,
    parts_count: int = 4,
) -> SlicingResult:
    """
    Divides the posture-normalized object into N equal geometric parts along its principal vertical axis.
    """
    parts_count = max(2, min(12, int(parts_count)))
    h, w = normalized_bgr.shape[:2]

    # Find the vertical bounds of the object mask
    ys, xs = np.nonzero(normalized_mask)
    if len(ys) == 0:
        # Fallback if empty
        return SlicingResult(
            parts_count=parts_count,
            parts=[],
            divided_image_bgr=normalized_bgr.copy(),
            is_equal_split=False,
            division_axis="Vertical",
        )

    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    total_fg_pixels = len(ys)
    target_part_area = float(total_fg_pixels) / float(parts_count)

    # Calculate cumulative pixel count along Y axis to find equal-area cut planes
    row_counts = np.sum(normalized_mask > 0, axis=1)  # shape (h,)
    cum_counts = np.cumsum(row_counts)

    # Find the y-indices where cumulative count crosses (k * target_part_area)
    cut_y_indices: List[int] = [y_min]
    for k in range(1, parts_count):
        target_val = k * target_part_area
        # Find index in cum_counts closest to target_val
        idx = int(np.searchsorted(cum_counts, target_val))
        idx = max(y_min + 1, min(y_max - 1, idx))
        cut_y_indices.append(idx)
    cut_y_indices.append(y_max + 1)

    # Create visualization
    annotated = normalized_bgr.copy()
    overlay = annotated.copy()

    parts: List[SlicedPart] = []

    for i in range(parts_count):
        y_start = cut_y_indices[i]
        y_end = cut_y_indices[i + 1]

        # Extract mask for this slice
        part_mask = np.zeros_like(normalized_mask)
        part_mask[y_start:y_end, :] = normalized_mask[y_start:y_end, :]

        p_ys, p_xs = np.nonzero(part_mask)
        part_area = len(p_ys)
        pct = (float(part_area) / float(max(1, total_fg_pixels))) * 100.0

        p_width = int(np.max(p_xs) - np.min(p_xs) + 1) if len(p_xs) > 0 else 0
        p_height = int(y_end - y_start)

        color_bgr = PART_COLORS_BGR[i % len(PART_COLORS_BGR)]
        color_hex = PART_COLORS_HEX[i % len(PART_COLORS_HEX)]

        # Tint the section in the overlay
        overlay[part_mask > 0] = (
            0.65 * overlay[part_mask > 0] + 0.35 * np.array(color_bgr, dtype=np.float64)
        ).astype(np.uint8)

        parts.append(
            SlicedPart(
                index=i + 1,
                label=f"Part {i + 1}",
                pixel_height=p_height,
                pixel_width=p_width,
                pixel_area=part_area,
                percentage=pct,
                color_hex=color_hex,
            )
        )

    # Blend colored tint overlay
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Draw object boundary contour
    cv2.drawContours(annotated, [normalized_contour], -1, (40, 40, 50), 2, cv2.LINE_AA)

    # Draw division cut lines and labels
    for i in range(1, parts_count):
        cut_y = cut_y_indices[i]
        # Find horizontal extents of object at cut_y
        row_mask = normalized_mask[cut_y, :]
        row_xs = np.nonzero(row_mask)[0]
        if len(row_xs) > 0:
            rx1 = max(0, int(np.min(row_xs)) - 4)
            rx2 = min(w - 1, int(np.max(row_xs)) + 4)
            # Draw glowing cut line
            cv2.line(annotated, (rx1, cut_y), (rx2, cut_y), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(annotated, (rx1, cut_y), (rx2, cut_y), (255, 93, 93), 2, cv2.LINE_AA)  # Referee Coral line

    # Draw part label pills
    for i in range(parts_count):
        y_start = cut_y_indices[i]
        y_end = cut_y_indices[i + 1]
        mid_y = int((y_start + y_end) / 2)

        part_mask = np.zeros_like(normalized_mask)
        part_mask[y_start:y_end, :] = normalized_mask[y_start:y_end, :]
        p_ys, p_xs = np.nonzero(part_mask)

        if len(p_xs) > 0:
            mid_x = int(np.mean(p_xs))
            color_bgr = PART_COLORS_BGR[i % len(PART_COLORS_BGR)]
            label_text = f"P{i + 1}: {parts[i].percentage:.1f}%"

            # Draw small background pill
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            px1 = max(2, mid_x - int(tw / 2) - 6)
            py1 = max(2, mid_y - int(th / 2) - 4)
            px2 = min(w - 2, mid_x + int(tw / 2) + 6)
            py2 = min(h - 2, mid_y + int(th / 2) + 4)

            cv2.rectangle(annotated, (px1, py1), (px2, py2), (20, 22, 59), -1)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color_bgr, 1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                label_text,
                (px1 + 4, py2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return SlicingResult(
        parts_count=parts_count,
        parts=parts,
        divided_image_bgr=annotated,
        is_equal_split=True,
        division_axis="Principal Axis (Vertical)",
    )
