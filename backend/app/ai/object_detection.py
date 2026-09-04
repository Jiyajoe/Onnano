"""
object_detection.py - AI Vision + CV Validation object identification pipeline.

Architecture:
    Original Image → Gemini Vision API → Semantic Identification
                                              ↓
    CV Shape/Dimensions → Validation → Final Object Identity

AI semantic understanding is the PRIMARY source of object identity.
CV measurements SUPPORT the AI — they do NOT override it.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np

from ..cv.shape import ShapeProperties
from ..cv.dimensions import DimensionMetrics
from ..cv.color import ColorAnalysis
from ..cv.texture import TextureAnalysis
from ..cv.contour import EdgeAnalysis

from .vision_identifier import identify_with_vision_model, VisionIdentificationResult
from .validation import validate_ai_identification, ValidationResult

logger = logging.getLogger(__name__)


# ---- Category hierarchy for known objects ----
CATEGORY_HIERARCHY = {
    "pen": ("Pen", "Writing Instrument"),
    "ballpoint pen": ("Ballpoint Pen", "Writing Instrument"),
    "pencil": ("Pencil", "Writing Instrument"),
    "marker": ("Marker", "Writing Instrument"),
    "scissors": ("Scissors", "Cutting Tool"),
    "banana": ("Banana", "Fruit"),
    "apple": ("Apple", "Fruit"),
    "bottle": ("Bottle", "Drinkware & Bottles"),
    "water bottle": ("Water Bottle", "Drinkware & Bottles"),
    "cup": ("Cup", "Drinkware & Bottles"),
    "mug": ("Mug", "Drinkware & Bottles"),
    "spoon": ("Spoon", "Cutlery & Tableware"),
    "fork": ("Fork", "Cutlery & Tableware"),
    "knife": ("Knife", "Cutlery & Tableware"),
    "book": ("Book", "Stationery & Books"),
    "notebook": ("Notebook", "Stationery & Books"),
    "ruler": ("Ruler", "Stationery & Books"),
    "phone": ("Phone", "Personal Electronics"),
    "smartphone": ("Smartphone", "Personal Electronics"),
    "mobile phone": ("Mobile Phone", "Personal Electronics"),
}


@dataclass
class IdentificationResult:
    name: str                  # e.g. "Pen", "Scissors", "Banana"
    specific_type: str         # e.g. "Ballpoint Pen", "Craft Scissors"
    category: str              # e.g. "Writing Instrument", "Cutting Tool"
    brand: str                 # "Not reliably identifiable" unless visual logo/text detected
    confidence: float          # 0.0 to 1.0
    characteristics: str       # Visible physical characteristics
    related_categories: List[str]
    # New fields for AI pipeline transparency
    ai_identification: Optional[Dict[str, Any]] = None
    validation_info: Optional[Dict[str, Any]] = None
    pipeline_source: str = "ai_vision"  # "ai_vision" | "cv_fallback"
    debug_log: List[str] = field(default_factory=list)

    @property
    def confidence_pct(self) -> float:
        return round(self.confidence * 100.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "specific_type": self.specific_type,
            "category": self.category,
            "brand": self.brand,
            "confidence": round(self.confidence, 2),
            "confidence_pct": self.confidence_pct,
            "characteristics": self.characteristics,
            "related_categories": self.related_categories,
            "pipeline_source": self.pipeline_source,
        }
        if self.ai_identification:
            result["ai_identification"] = self.ai_identification
        if self.validation_info:
            result["validation_info"] = self.validation_info
        if self.debug_log:
            result["debug_log"] = self.debug_log
        return result


def _title_case(name: str) -> str:
    """Capitalize object name properly."""
    return " ".join(word.capitalize() for word in name.strip().split())


def _get_related_categories(category: str) -> List[str]:
    """Get related categories for a given category."""
    category_groups = {
        "Writing Instrument": ["Pen", "Pencil", "Marker", "Stylus"],
        "Cutting Tool": ["Scissors", "Knife", "Cutter"],
        "Fruit": ["Banana", "Apple", "Orange"],
        "Drinkware & Bottles": ["Bottle", "Cup", "Mug", "Glass"],
        "Cutlery & Tableware": ["Spoon", "Fork", "Knife"],
        "Stationery & Books": ["Book", "Notebook", "Ruler"],
        "Personal Electronics": ["Phone", "Tablet", "Laptop"],
    }
    return category_groups.get(category, ["Physical Object"])


def _build_cv_characteristics(
    shape: ShapeProperties,
    color: ColorAnalysis,
    texture: TextureAnalysis,
) -> str:
    """Build a human-readable characteristics string from CV measurements."""
    parts = []
    if shape.aspect_ratio > 3.5:
        parts.append("Elongated form")
    elif shape.aspect_ratio < 1.3:
        parts.append("Compact/rounded form")
    else:
        parts.append(f"Moderate proportions (AR={shape.aspect_ratio:.1f})")

    if shape.circularity > 0.8:
        parts.append("highly circular profile")
    elif shape.solidity < 0.7:
        parts.append("complex/articulated structure")
    elif shape.rectangularity > 0.85:
        parts.append("rectangular profile")

    if hasattr(color, 'dominant_name') and color.dominant_name:
        parts.append(f"{color.dominant_name} dominant color")

    if hasattr(texture, 'descriptor') and texture.descriptor:
        parts.append(f"{texture.descriptor.lower()} surface")

    return ". ".join(parts) + "." if parts else "Physical object with measurable geometric properties."


def _cv_fallback_identification(
    shape: ShapeProperties,
    dim: DimensionMetrics,
    color: ColorAnalysis,
    texture: TextureAnalysis,
) -> IdentificationResult:
    """
    Fallback identification using CV shape heuristics.
    Only used when the AI vision model is unavailable (no API key, API error, etc.)
    This is the DEGRADED mode — less accurate than the AI pipeline.
    """
    ar = shape.aspect_ratio
    circ = shape.circularity
    sol = shape.solidity
    rect = shape.rectangularity

    debug = ["[FALLBACK] AI vision unavailable — using CV shape heuristics (degraded mode)"]

    if ar > 4.5 and sol > 0.75:
        name = "Elongated Object"
        stype = "Elongated Cylindrical Instrument"
        cat = "Writing Instrument"
        chars = f"High aspect ratio ({ar:.1f}:1) elongated form. Could be pen, pencil, or similar instrument."
    elif circ > 0.78 and ar < 1.3:
        name = "Round Object"
        stype = "Spherical/Circular Object"
        cat = "Miscellaneous"
        chars = f"High circularity ({circ:.2f}) with compact proportions."
    elif rect > 0.88 and sol > 0.90:
        name = "Rectangular Object"
        stype = "Flat Rectangular Object"
        cat = "Stationery & Books"
        chars = f"High rectangularity ({rect:.2f}) with orthogonal edges."
    elif sol < 0.70:
        name = "Complex Object"
        stype = "Articulated/Complex Shape"
        cat = "Miscellaneous"
        chars = f"Low solidity ({sol:.2f}) suggesting articulated or concave structure."
    else:
        name = "Physical Object"
        stype = f"{shape.shape_type} Object"
        cat = "Miscellaneous"
        chars = f"Geometric profile: {shape.shape_type} (AR={ar:.2f}, circ={circ:.2f})."

    debug.append(f"[FALLBACK] CV classification: {name} ({cat})")

    return IdentificationResult(
        name=name,
        specific_type=stype,
        category=cat,
        brand="Not reliably identifiable (no visible manufacturer marks)",
        confidence=0.55,  # Low confidence for CV-only classification
        characteristics=chars,
        related_categories=_get_related_categories(cat),
        pipeline_source="cv_fallback",
        debug_log=debug,
    )


def identify_object(
    shape: ShapeProperties,
    dim: DimensionMetrics,
    color: ColorAnalysis,
    texture: TextureAnalysis,
    edges: EdgeAnalysis,
    normalized_mask: np.ndarray = None,
    original_image_bgr: np.ndarray = None,
) -> IdentificationResult:
    """
    Full AI Vision + CV Validation identification pipeline.

    Pipeline:
        1. Send ORIGINAL image to Gemini Vision → get semantic identification
        2. Cross-validate AI result against CV measurements
        3. Adjust confidence based on validation
        4. Return final identification with full debug info

    If AI vision is unavailable (no API key, error), falls back to CV heuristics.
    """

    debug_log: List[str] = []
    brand_str = "Not reliably identifiable (no visible manufacturer marks)"

    # ---- Stage 1: AI Vision Identification ----
    debug_log.append("[PIPELINE] Stage 1: AI Vision Identification")

    if original_image_bgr is not None:
        vision_result = identify_with_vision_model(original_image_bgr)
        debug_log.append(f"[AI INPUT] Original image sent to vision model ({original_image_bgr.shape})")
        debug_log.append(f"[AI OUTPUT] Detected object = {vision_result.object_name}")
        debug_log.append(f"[AI OUTPUT] Category = {vision_result.category}")
        debug_log.append(f"[AI OUTPUT] Confidence = {vision_result.confidence:.2f}")
        debug_log.append(f"[AI OUTPUT] Source = {vision_result.source}")
        if vision_result.reasoning:
            debug_log.append(f"[AI OUTPUT] Reasoning = {vision_result.reasoning}")
        if vision_result.error:
            debug_log.append(f"[AI ERROR] {vision_result.error}")
    else:
        debug_log.append("[AI INPUT] No original image provided — cannot call vision model")
        vision_result = VisionIdentificationResult(
            object_name="", category="", confidence=0.0,
            source="error_fallback", error="No original image provided.",
        )

    # ---- Check if AI succeeded ----
    if vision_result.source == "error_fallback" or not vision_result.object_name:
        debug_log.append("[PIPELINE] AI vision unavailable — falling back to CV heuristics")
        fallback = _cv_fallback_identification(shape, dim, color, texture)
        fallback.debug_log = debug_log + fallback.debug_log
        fallback.ai_identification = vision_result.to_dict() if vision_result else None
        return fallback

    # ---- Stage 2: CV Validation ----
    debug_log.append("[PIPELINE] Stage 2: CV Validation")

    validation = validate_ai_identification(vision_result, shape)

    debug_log.append(f"[VALIDATION] Status = {validation.validation_status}")
    debug_log.append(f"[VALIDATION] Confidence adjustment = {validation.confidence_adjustment:.2f}")
    debug_log.append(f"[VALIDATION] Supporting evidence = {validation.supporting_count}")
    debug_log.append(f"[VALIDATION] Contradicting evidence = {validation.contradicting_count}")
    if validation.notes:
        debug_log.append(f"[VALIDATION] Notes = {validation.notes}")

    # ---- Stage 3: Final Identification ----
    debug_log.append("[PIPELINE] Stage 3: Final Identification")

    # AI provides the identity — CV validation adjusts confidence
    raw_name = vision_result.object_name
    display_name = _title_case(raw_name)

    # Look up in category hierarchy for standardized naming
    lookup_key = raw_name.lower().strip()
    if lookup_key in CATEGORY_HIERARCHY:
        display_name, category = CATEGORY_HIERARCHY[lookup_key]
    else:
        category = vision_result.category or "Miscellaneous"

    # Adjust confidence based on validation
    final_confidence = min(0.98, vision_result.confidence * validation.confidence_adjustment)
    final_confidence = max(0.10, final_confidence)

    # Build characteristics from both AI reasoning and CV measurements
    ai_chars = vision_result.reasoning or ""
    cv_chars = _build_cv_characteristics(shape, color, texture)
    characteristics = ai_chars if ai_chars else cv_chars

    debug_log.append(f"[FINAL] Object = {display_name}")
    debug_log.append(f"[FINAL] Category = {category}")
    debug_log.append(f"[FINAL] Confidence = {final_confidence:.2f} (AI={vision_result.confidence:.2f} × validation={validation.confidence_adjustment:.2f})")

    # Log the full pipeline summary
    logger.info(
        f"[PIPELINE COMPLETE] "
        f"AI={vision_result.object_name} → "
        f"Validation={validation.validation_status} → "
        f"Final={display_name} ({category}) "
        f"conf={final_confidence:.2f}"
    )

    return IdentificationResult(
        name=display_name,
        specific_type=vision_result.object_name,
        category=category,
        brand=brand_str,
        confidence=final_confidence,
        characteristics=characteristics,
        related_categories=_get_related_categories(category),
        ai_identification=vision_result.to_dict(),
        validation_info=validation.to_dict(),
        pipeline_source="ai_vision",
        debug_log=debug_log,
    )
