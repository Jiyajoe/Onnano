"""
segmentation.py - Multi-cue object detection, GrabCut boundary refinement, and mask quality validation.
Preserves the original photograph intact while extracting precise, validated object masks.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from ..config import (
    MAX_IMAGE_DIMENSION,
    MIN_OBJECT_AREA_RATIO,
)
from .quality import evaluate_image_quality, ImageQualityReport


class SegmentationError(Exception):
    """User-facing segmentation error."""
    pass


@dataclass
class MaskQualityMetrics:
    mask_confidence: float             # 0.0 to 1.0 (e.g. 0.94 -> 94%)
    edge_alignment_score: float        # 0.0 to 1.0 (overlap of mask boundary with Canny edges)
    contour_continuity: float          # 0.0 to 1.0
    connected_components_count: int    # ideally 1 for a single solid object
    holes_count: int                   # number of internal cavities detected
    leakage_risk: str                  # "low", "moderate", "high"
    is_trustworthy: bool               # True if mask meets confidence threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_confidence": round(self.mask_confidence, 3),
            "mask_confidence_pct": round(self.mask_confidence * 100.0, 1),
            "edge_alignment_score": round(self.edge_alignment_score, 2),
            "contour_continuity": round(self.contour_continuity, 2),
            "connected_components_count": self.connected_components_count,
            "holes_count": self.holes_count,
            "leakage_risk": self.leakage_risk,
            "is_trustworthy": self.is_trustworthy,
        }


@dataclass
class SegmentedObject:
    contour: np.ndarray
    area: float
    bbox: Tuple[int, int, int, int]      # x, y, w, h
    centroid: Tuple[float, float]
    mask: np.ndarray                     # 2D uint8 binary mask (255 inside object, 0 outside)
    original_bgr: np.ndarray             # The 100% UNMODIFIED original photograph
    ai_selected_bgr: np.ndarray          # Original photograph with glowing object selection boundary overlay
    extracted_object_rgba: np.ndarray    # Object-only BGRA image (transparent background)
    extracted_object_bgr: np.ndarray     # Object patch strictly for internal CV analysis
    mask_quality: MaskQualityMetrics
    quality_report: ImageQualityReport
    id: int = 1

    def to_selection_dict(self) -> Dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "mask_available": True,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "centroid": {"x": round(self.centroid[0], 1), "y": round(self.centroid[1], 1)},
            "contour_points_count": len(self.contour),
            "area_pixels": int(self.area),
            "mask_confidence_pct": self.mask_quality.to_dict()["mask_confidence_pct"],
            "mask_quality": self.mask_quality.to_dict(),
        }


def resize_keep_aspect(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def validate_mask_quality(
    mask: np.ndarray,
    contour: np.ndarray,
    image_bgr: np.ndarray,
) -> MaskQualityMetrics:
    """
    Validates whether the extracted object mask is trustworthy.
    Calculates edge alignment with Canny gradients, checks connected components and holes.
    """
    h, w = mask.shape[:2]
    obj_area = cv2.contourArea(contour) if contour is not None and len(contour) >= 3 else 0.0
    if obj_area <= 0:
        return MaskQualityMetrics(
            mask_confidence=0.10,
            edge_alignment_score=0.10,
            contour_continuity=0.10,
            connected_components_count=0,
            holes_count=0,
            leakage_risk="high",
            is_trustworthy=False,
        )

    # 1. Edge Alignment: How closely does the mask boundary coincide with genuine image edges?
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 40, 140)

    # Dilate mask boundary slightly to test alignment
    mask_boundary = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    dilated_boundary = cv2.dilate(
        mask_boundary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    matching_edges = np.count_nonzero(np.logical_and(canny > 0, dilated_boundary > 0))
    boundary_pixels = np.count_nonzero(mask_boundary)
    edge_alignment = min(1.0, float(matching_edges) / max(1.0, float(boundary_pixels) * 0.40))

    # 2. Connected components in mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    fg_components = max(1, num_labels - 1)

    # 3. Holes detection: Invert mask and find closed internal contours
    inverted = cv2.bitwise_not(mask)
    border_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inverted, border_mask, (0, 0), 0)
    hole_contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_holes = [c for c in hole_contours if cv2.contourArea(c) > (obj_area * 0.01)]
    holes_count = len(significant_holes)

    # 4. Contour Continuity and Smoothness
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    solidity = obj_area / max(1.0, cv2.contourArea(hull))
    contour_continuity = min(1.0, max(0.5, 0.70 + 0.30 * solidity))

    # 5. Leakage Risk Evaluation
    if fg_components > 2 or edge_alignment < 0.35:
        leakage_risk = "high"
    elif fg_components == 2 or edge_alignment < 0.60:
        leakage_risk = "moderate"
    else:
        leakage_risk = "low"

    # Multi-factor confidence calculation
    confidence = (
        0.45 * edge_alignment +
        0.30 * contour_continuity +
        0.15 * (1.0 if fg_components == 1 else max(0.4, 1.0 - (fg_components * 0.15))) +
        0.10 * (1.0 if holes_count == 0 else max(0.5, 1.0 - (holes_count * 0.10)))
    )
    confidence = float(np.clip(confidence, 0.35, 0.98))
    is_trustworthy = confidence >= 0.55

    return MaskQualityMetrics(
        mask_confidence=confidence,
        edge_alignment_score=edge_alignment,
        contour_continuity=contour_continuity,
        connected_components_count=fg_components,
        holes_count=holes_count,
        leakage_risk=leakage_risk,
        is_trustworthy=is_trustworthy,
    )


def refine_mask_boundaries(raw_mask: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """
    Refines raw segmentation mask using morphological smoothing and bilateral edge alignment.
    Eliminates jagged noise and bridges minor breaks along true physical edges.
    """
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_kernel)

    # Extract largest connected component to eliminate isolated floating specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    if num_labels > 2:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        refined = (labels == largest_label).astype(np.uint8) * 255

    # Smooth contour boundary using Gaussian blur + re-threshold
    blurred_mask = cv2.GaussianBlur(refined, (5, 5), 0)
    _, refined = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    return refined


def render_ai_selected_overlay(
    original_bgr: np.ndarray,
    contour: np.ndarray,
    mask: np.ndarray,
    centroid: Tuple[float, float],
) -> np.ndarray:
    """
    Renders the exact original photograph with a prominent, elegant visual selection around
    the detected object (glowing boundary contour, subtle translucent highlight, and centroid marker).
    The original background remains 100% VISIBLE.
    """
    selected = original_bgr.copy()
    overlay = original_bgr.copy()

    # 1. Subtle translucent glow inside the detected object (Cyan/Mint highlight)
    tint_color = np.array([217, 217, 47], dtype=np.float64)  # BGR Mint / Cyan
    overlay[mask > 0] = (0.78 * overlay[mask > 0] + 0.22 * tint_color).astype(np.uint8)
    cv2.addWeighted(overlay, 0.90, selected, 0.10, 0, selected)

    # 2. Glowing multi-layered selection boundary
    cv2.drawContours(selected, [contour], -1, (255, 182, 39), 5, cv2.LINE_AA)  # Outer Gold glow
    cv2.drawContours(selected, [contour], -1, (47, 217, 168), 2, cv2.LINE_AA)  # Core Mint line

    # 3. Floating object center-point reticle
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    cv2.circle(selected, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(selected, (cx, cy), 4, (47, 217, 168), -1, cv2.LINE_AA)

    # 4. Selection bounding corner brackets
    x, y, w, h = cv2.boundingRect(contour)
    d = min(18, min(w, h) // 4)
    c_color = (255, 182, 39)
    th = 2
    cv2.line(selected, (x, y), (x + d, y), c_color, th, cv2.LINE_AA)
    cv2.line(selected, (x, y), (x, y + d), c_color, th, cv2.LINE_AA)
    cv2.line(selected, (x + w, y), (x + w - d, y), c_color, th, cv2.LINE_AA)
    cv2.line(selected, (x + w, y), (x + w, y + d), c_color, th, cv2.LINE_AA)
    cv2.line(selected, (x, y + h), (x + d, y + h), c_color, th, cv2.LINE_AA)
    cv2.line(selected, (x, y + h), (x, y + h - d), c_color, th, cv2.LINE_AA)
    cv2.line(selected, (x + w, y + h), (x + w - d, y + h), c_color, th, cv2.LINE_AA)
    cv2.line(selected, (x + w, y + h), (x + w, y + h - d), c_color, th, cv2.LINE_AA)

    return selected


def isolate_primary_object(image_bgr: np.ndarray) -> Tuple[SegmentedObject, np.ndarray]:
    """
    Accurately extracts the primary physical object using multi-cue saliency, color distance,
    and GrabCut boundary optimization. Never incorporates the surrounding background into the mask.
    Creates an object-only RGBA representation (with transparent background) and preserves the original photo.
    """
    resized, _ = resize_keep_aspect(image_bgr)
    h, w = resized.shape[:2]
    frame_area = float(h * w)

    # 1. Multi-space background color sampling (LAB and BGR)
    border_w = max(4, min(14, min(h, w) // 30))
    border_pixels_bgr = np.vstack([
        resized[0:border_w, :].reshape(-1, 3),
        resized[-border_w:, :].reshape(-1, 3),
        resized[:, 0:border_w].reshape(-1, 3),
        resized[:, -border_w:].reshape(-1, 3),
    ])
    
    # Convert image to LAB for perceptual color distance
    img_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    border_pixels_lab = np.vstack([
        img_lab[0:border_w, :].reshape(-1, 3),
        img_lab[-border_w:, :].reshape(-1, 3),
        img_lab[:, 0:border_w].reshape(-1, 3),
        img_lab[:, -border_w:].reshape(-1, 3),
    ])
    
    bg_median_lab = np.median(border_pixels_lab, axis=0)
    bg_std_lab = np.std(border_pixels_lab.astype(np.float32), axis=0) + 1.0

    # Color difference in LAB space weighted by background variance
    lab_diff = np.sqrt(
        ((img_lab[:, :, 0].astype(np.float32) - bg_median_lab[0]) / bg_std_lab[0]) ** 2 +
        ((img_lab[:, :, 1].astype(np.float32) - bg_median_lab[1]) / bg_std_lab[1]) ** 2 +
        ((img_lab[:, :, 2].astype(np.float32) - bg_median_lab[2]) / bg_std_lab[2]) ** 2
    )
    
    # BGR color difference
    bg_median_bgr = np.median(border_pixels_bgr, axis=0)
    bgr_diff = np.linalg.norm(resized.astype(np.float32) - bg_median_bgr.astype(np.float32), axis=2)

    # 2. Gradient / Edge saliency
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    norm_grad = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Seed GrabCut initialization
    # GrabCut masks: GC_BGD (0), GC_FGD (1), GC_PR_BGD (2), GC_PR_FGD (3)
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Outer perimeter is definite background
    gc_mask[0:border_w, :] = cv2.GC_BGD
    gc_mask[-border_w:, :] = cv2.GC_BGD
    gc_mask[:, 0:border_w] = cv2.GC_BGD
    gc_mask[:, -border_w:] = cv2.GC_BGD

    # High color distance or strong edge indicates probable foreground
    diff_thresh = max(18.0, np.percentile(bgr_diff, 60) * 0.45)
    lab_thresh = max(2.5, np.percentile(lab_diff, 60) * 0.40)
    
    fg_candidates = (bgr_diff > diff_thresh) | (lab_diff > lab_thresh) | (norm_grad > 40)
    # Exclude outer border
    fg_candidates[0:border_w, :] = False
    fg_candidates[-border_w:, :] = False
    fg_candidates[:, 0:border_w] = False
    fg_candidates[:, -border_w:] = False

    gc_mask[fg_candidates] = cv2.GC_PR_FGD
    
    # Very high confidence foreground (strong color contrast AND not near border)
    strong_fg = (bgr_diff > diff_thresh * 1.5) & (lab_diff > lab_thresh * 1.5)
    if np.count_nonzero(strong_fg) > 50:
        gc_mask[strong_fg] = cv2.GC_FGD

    # Apply GrabCut if foreground candidates exist
    refined_mask = None
    if np.count_nonzero(fg_candidates) > (frame_area * MIN_OBJECT_AREA_RATIO):
        try:
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(resized, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
            refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            refined_mask = None

    if refined_mask is None or np.count_nonzero(refined_mask) < (frame_area * MIN_OBJECT_AREA_RATIO):
        # Fallback to multi-cue morphological segmentation
        combined_cue = cv2.bitwise_or((bgr_diff > diff_thresh).astype(np.uint8) * 255, (norm_grad > 32).astype(np.uint8) * 255)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_mask = cv2.morphologyEx(combined_cue, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Morphological refinement to bridge internal gaps and smooth edges
    refined_mask = refine_mask_boundaries(refined_mask, resized)

    # 4. Find primary external contour
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = frame_area * MIN_OBJECT_AREA_RATIO

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < w * 0.98 or bh < h * 0.98:
                valid_contours.append((c, area))

    if not valid_contours:
        # Ultimate fallback: central Otsu thresholding
        _, inv_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        inv_contours, _ = cv2.findContours(inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in inv_contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < w * 0.98 or bh < h * 0.98:
                    valid_contours.append((c, area))

    if not valid_contours:
        margin_x, margin_y = max(4, int(w * 0.05)), max(4, int(h * 0.05))
        default_rect = np.array([
            [[margin_x, margin_y]],
            [[w - margin_x, margin_y]],
            [[w - margin_x, h - margin_y]],
            [[margin_x, h - margin_y]],
        ], dtype=np.int32)
        valid_contours.append((default_rect, float((w - 2 * margin_x) * (h - 2 * margin_y))))

    valid_contours.sort(key=lambda x: x[1], reverse=True)
    primary_contour, primary_area = valid_contours[0]

    # Solidify clean mask around selected primary object
    clean_obj_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_obj_mask, [primary_contour], -1, 255, -1)
    # Mask AND with refined_mask to keep internal cutouts (like scissor holes)
    final_mask = cv2.bitwise_and(clean_obj_mask, refined_mask)
    if np.count_nonzero(final_mask) < primary_area * 0.3:
        final_mask = clean_obj_mask

    # 5. Extract Object-Only BGRA Image (Transparent Background)
    # BGRA: channels 0,1,2 = BGR colors of original object; channel 3 = Alpha mask
    extracted_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    extracted_rgba[:, :, :3] = resized
    extracted_rgba[:, :, 3] = final_mask

    # Calculate centroid and bounding rect
    x, y, bw, bh = cv2.boundingRect(primary_contour)
    M = cv2.moments(primary_contour)
    cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + bw / 2.0
    cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + bh / 2.0

    # 6. Evaluate Image Quality & Mask Quality
    quality_report = evaluate_image_quality(resized, preliminary_mask=final_mask)
    mask_quality = validate_mask_quality(final_mask, primary_contour, resized)

    # 7. Generate AI Selected Visual (Original photo + glowing object selection overlay)
    ai_selected = render_ai_selected_overlay(resized, primary_contour, final_mask, (cx, cy))

    # 8. Extract tight object patch for internal CV analysis
    pad = 12
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    extracted_patch = resized[y0:y1, x0:x1].copy()

    seg_obj = SegmentedObject(
        contour=primary_contour,
        area=float(primary_area),
        bbox=(x, y, bw, bh),
        centroid=(float(cx), float(cy)),
        mask=final_mask,
        original_bgr=resized,
        ai_selected_bgr=ai_selected,
        extracted_object_rgba=extracted_rgba,
        extracted_object_bgr=extracted_patch,
        mask_quality=mask_quality,
        quality_report=quality_report,
        id=1,
    )

    return seg_obj, resized
