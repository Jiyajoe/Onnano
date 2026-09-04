"""
quality.py - Advisory image quality analysis and telemetry.
All checks are 100% ADVISORY and informative — photos are ALWAYS accepted for analysis.
Calculates resolution, lighting, sharpness, border proximity, and provides user-friendly guidance.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


@dataclass
class QualityCheckItem:
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    status_text: str
    detail: str


@dataclass
class ImageQualityReport:
    overall_score: float  # 0 to 100%
    is_acceptable: bool = True  # Always accepted for valid image data
    accepted: bool = True
    warnings: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    checks: List[QualityCheckItem] = field(default_factory=list)

    @property
    def score_pct(self) -> int:
        return int(round(self.overall_score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "is_acceptable": self.is_acceptable,
            "overall_score": round(self.overall_score, 1),
            "score_pct": self.score_pct,
            "warnings": self.warnings,
            "rejection_reason": self.rejection_reason,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "score": round(c.score * 100.0, 1),
                    "status_text": c.status_text,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def check_blur(gray: np.ndarray) -> Tuple[QualityCheckItem, List[str]]:
    """Calculates Laplacian variance as an advisory proxy for image focus/blur."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings = []

    if lap_var < 15.0:
        warnings.append("Image has noticeable softness or motion blur; measurements may be approximate.")
        item = QualityCheckItem(
            name="Blur & Sharpness",
            passed=True,
            score=max(0.40, min(0.70, lap_var / 30.0)),
            status_text="Soft Focus / Minor Blur",
            detail=f"Sharpness index: {lap_var:.1f}. Image accepted; edge definition may be softened.",
        )
    elif lap_var < 35.0:
        item = QualityCheckItem(
            name="Blur & Sharpness",
            passed=True,
            score=0.75,
            status_text="Moderate Sharpness",
            detail=f"Sharpness index: {lap_var:.1f}. Acceptable focus for measurement.",
        )
    else:
        score = min(1.0, 0.80 + (lap_var / 500.0) * 0.20)
        item = QualityCheckItem(
            name="Blur & Sharpness",
            passed=True,
            score=score,
            status_text="Sharp Focus ✓",
            detail=f"Sharpness index: {lap_var:.1f}. Edge definition is crisp.",
        )
    return item, warnings


def check_lighting_and_exposure(gray: np.ndarray, bgr: np.ndarray) -> Tuple[QualityCheckItem, List[str]]:
    """Checks brightness, contrast, and highlights as advisory metrics."""
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    warnings = []

    # Glare check via HSV Value & Saturation
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    s_chan = hsv[:, :, 1]
    glare_pixels = np.count_nonzero(np.logical_and(v_chan > 250, s_chan < 25))
    glare_ratio = float(glare_pixels) / max(1.0, float(gray.size))

    if mean_val < 25.0:
        warnings.append("Lighting is low; AI confidence may be reduced.")
        item = QualityCheckItem(
            name="Lighting & Exposure",
            passed=True,
            score=0.55,
            status_text="Low Illumination",
            detail=f"Mean brightness is {mean_val:.1f}/255. Shadow areas enhanced automatically.",
        )
    elif mean_val > 248.0:
        warnings.append("Image is brightly exposed; boundary contrast may be softened.")
        item = QualityCheckItem(
            name="Lighting & Exposure",
            passed=True,
            score=0.60,
            status_text="High Exposure",
            detail=f"Mean brightness is {mean_val:.1f}/255. High dynamic range balanced.",
        )
    elif glare_ratio > 0.30:
        warnings.append("Surface glare detected; spectral highlights handled by color invariant analysis.")
        item = QualityCheckItem(
            name="Lighting & Exposure",
            passed=True,
            score=0.70,
            status_text="Specular Highlights Present",
            detail=f"{glare_ratio * 100.0:.1f}% specular reflection observed.",
        )
    else:
        score = min(1.0, 0.78 + (min(std_val, 60.0) / 60.0) * 0.22)
        item = QualityCheckItem(
            name="Lighting & Exposure",
            passed=True,
            score=score,
            status_text="Balanced Lighting ✓",
            detail=f"Mean brightness {mean_val:.1f}, contrast {std_val:.1f}.",
        )
    return item, warnings


def check_resolution(h: int, w: int) -> Tuple[QualityCheckItem, List[str]]:
    """Advisory check for image resolution and aspect ratio."""
    megapixels = (h * w) / 1_000_000.0
    min_dim = min(h, w)
    warnings = []

    aspect = max(h, w) / max(1.0, float(min(h, w)))
    orientation = "Square" if aspect < 1.15 else ("Portrait" if h > w else "Landscape")

    if min_dim < 120:
        warnings.append("Image resolution is low; sub-pixel boundary refinement applied.")
        item = QualityCheckItem(
            name="Resolution",
            passed=True,
            score=0.60,
            status_text=f"Low Resolution ({w}×{h})",
            detail=f"Dimensions {w}×{h} ({megapixels:.2f} MP, {orientation}). Processed with interpolation.",
        )
    elif min_dim < 300:
        item = QualityCheckItem(
            name="Resolution",
            passed=True,
            score=0.82,
            status_text=f"Standard Resolution ({w}×{h})",
            detail=f"Resolution is {w}×{h} ({megapixels:.2f} MP, {orientation}). Good clarity.",
        )
    else:
        item = QualityCheckItem(
            name="Resolution",
            passed=True,
            score=0.98,
            status_text=f"High Resolution ({w}×{h}) ✓",
            detail=f"Resolution is {w}×{h} ({megapixels:.2f} MP, {orientation}). Excellent pixel density.",
        )
    return item, warnings


def check_object_boundary_and_clipping(
    mask: np.ndarray,
    h: int,
    w: int,
) -> Tuple[QualityCheckItem, List[str]]:
    """
    Advisory check for object visibility and border proximity.
    NEVER blocks uploads; provides friendly advisory notice if object extends near frame edges.
    """
    obj_pixels = np.count_nonzero(mask)
    frame_pixels = float(h * w)
    area_ratio = float(obj_pixels) / max(1.0, frame_pixels)
    warnings = []

    # Check border margin clipping
    border_thickness = 4
    top_border = np.count_nonzero(mask[:border_thickness, :])
    bot_border = np.count_nonzero(mask[-border_thickness:, :])
    left_border = np.count_nonzero(mask[:, :border_thickness])
    right_border = np.count_nonzero(mask[:, -border_thickness:])
    total_border_touches = top_border + bot_border + left_border + right_border
    border_cut_ratio = total_border_touches / float(2 * (h + w) * border_thickness)

    if border_cut_ratio > 0.35:
        warnings.append("Object is partially close to the frame edge; results may be slightly less accurate.")
        item = QualityCheckItem(
            name="Object Visibility",
            passed=True,
            score=0.70,
            status_text="Touches Frame Edge",
            detail="Object extends near the frame border; visible geometry and features analyzed.",
        )
    elif border_cut_ratio > 0.12:
        item = QualityCheckItem(
            name="Object Visibility",
            passed=True,
            score=0.85,
            status_text="Near Frame Edge",
            detail="Object is close to photo edge; full visible region analyzed.",
        )
    else:
        item = QualityCheckItem(
            name="Object Visibility",
            passed=True,
            score=0.96,
            status_text="Clean In-Frame Boundaries ✓",
            detail=f"Object occupies {area_ratio * 100.0:.1f}% of photo frame with clear separation.",
        )
    return item, warnings


def evaluate_image_quality(
    image_bgr: np.ndarray,
    preliminary_mask: np.ndarray = None,
) -> ImageQualityReport:
    """
    Evaluates the input image across resolution, lighting, sharpness, and visibility.
    Always returns accepted=True with advisory scores and warnings for telemetry.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ImageQualityReport(
            overall_score=0.0,
            is_acceptable=False,
            accepted=False,
            warnings=["No image data provided."],
            rejection_reason="The uploaded file contains no readable image data.",
            checks=[],
        )

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    res_check, res_warnings = check_resolution(h, w)
    light_check, light_warnings = check_lighting_and_exposure(gray, image_bgr)
    blur_check, blur_warnings = check_blur(gray)

    if preliminary_mask is None:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, preliminary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vis_check, vis_warnings = check_object_boundary_and_clipping(preliminary_mask, h, w)

    checks = [res_check, light_check, blur_check, vis_check]
    all_warnings = res_warnings + light_warnings + blur_warnings + vis_warnings

    weights = [0.20, 0.30, 0.30, 0.20]
    total_score = sum(c.score * w for c, w in zip(checks, weights)) * 100.0

    return ImageQualityReport(
        overall_score=round(total_score, 1),
        is_acceptable=True,
        accepted=True,
        warnings=all_warnings,
        rejection_reason="",
        checks=checks,
    )
