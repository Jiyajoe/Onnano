"""
validation.py - CV-based validation layer for AI vision identification results.

Cross-references the AI classification against measured CV properties (aspect ratio,
solidity, etc.) to catch obvious contradictions. CV evidence SUPPORTS the AI — it does
NOT override it. Only clear contradictions trigger a flag.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..config import (
    OBJECT_FEATURE_EXPECTATIONS,
    VALIDATION_CONFIDENCE_THRESHOLD,
    VALIDATION_CONTRADICTION_THRESHOLD,
)
from ..cv.shape import ShapeProperties
from .vision_identifier import VisionIdentificationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationEvidence:
    """Individual piece of supporting or contradicting evidence."""
    feature: str
    expected: str
    actual: str
    supports: bool  # True if this evidence supports the AI classification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "expected": self.expected,
            "actual": self.actual,
            "supports": self.supports,
        }


@dataclass
class ValidationResult:
    """Result of cross-referencing AI identification against CV evidence."""
    is_validated: bool  # True if CV evidence generally supports AI classification
    validation_status: str  # "confirmed", "plausible", "uncertain", "contradicted"
    confidence_adjustment: float  # Multiplier applied to AI confidence (0.5 to 1.1)
    evidence: List[ValidationEvidence] = field(default_factory=list)
    supporting_count: int = 0
    contradicting_count: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_validated": self.is_validated,
            "validation_status": self.validation_status,
            "confidence_adjustment": round(self.confidence_adjustment, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "supporting_count": self.supporting_count,
            "contradicting_count": self.contradicting_count,
            "notes": self.notes,
        }


def _normalize_object_name(name: str) -> str:
    """Normalize an object name for lookup in the feature expectations table."""
    name = name.lower().strip()
    # Handle common variations
    aliases = {
        "ballpoint pen": "pen",
        "ball pen": "pen",
        "marker": "pen",
        "felt-tip pen": "pen",
        "gel pen": "pen",
        "fountain pen": "pen",
        "mechanical pencil": "pencil",
        "colored pencil": "pencil",
        "craft scissors": "scissors",
        "kitchen scissors": "scissors",
        "shears": "scissors",
        "water bottle": "bottle",
        "glass bottle": "bottle",
        "plastic bottle": "bottle",
        "flask": "bottle",
        "thermos": "bottle",
        "coffee mug": "cup",
        "mug": "cup",
        "tea cup": "cup",
        "ceramic cup": "cup",
        "glass": "cup",
        "tumbler": "cup",
        "notebook": "book",
        "textbook": "book",
        "novel": "book",
        "diary": "book",
        "smartphone": "phone",
        "mobile phone": "phone",
        "cell phone": "phone",
        "mobile": "phone",
        "iphone": "phone",
        "android phone": "phone",
        "table spoon": "spoon",
        "tablespoon": "spoon",
        "teaspoon": "spoon",
        "dessert spoon": "spoon",
        "dinner fork": "fork",
        "salad fork": "fork",
        "fresh apple": "apple",
        "red apple": "apple",
        "green apple": "apple",
        "geometry ruler": "ruler",
        "scale": "ruler",
        "straightedge": "ruler",
        "fresh banana": "banana",
        "yellow banana": "banana",
    }
    return aliases.get(name, name)


def _check_range(value: float, expected_range: tuple) -> bool:
    """Check if a value falls within an expected range (with 20% tolerance)."""
    low, high = expected_range
    tolerance = (high - low) * 0.20
    return (low - tolerance) <= value <= (high + tolerance)


def validate_ai_identification(
    vision_result: VisionIdentificationResult,
    shape_props: ShapeProperties,
) -> ValidationResult:
    """
    Cross-references AI vision result against measured CV properties.

    Rules:
    - CV evidence SUPPORTS but does NOT override AI classification
    - Only clear, obvious contradictions reduce confidence
    - Unknown objects (not in feature table) get a neutral validation
    """

    # If AI didn't return a result, nothing to validate
    if not vision_result.object_name or vision_result.source == "error_fallback":
        return ValidationResult(
            is_validated=False,
            validation_status="no_ai_result",
            confidence_adjustment=1.0,
            notes="No AI identification available to validate.",
        )

    normalized_name = _normalize_object_name(vision_result.object_name)
    expectations = OBJECT_FEATURE_EXPECTATIONS.get(normalized_name)

    evidence_list: List[ValidationEvidence] = []
    supporting = 0
    contradicting = 0

    if expectations is None:
        # Object not in our feature table — AI identification is accepted as-is
        logger.info(
            f"[VALIDATION] Object '{vision_result.object_name}' (normalized: '{normalized_name}') "
            f"not in feature expectations table — accepting AI result as-is"
        )
        return ValidationResult(
            is_validated=True,
            validation_status="plausible",
            confidence_adjustment=1.0,
            notes=f"Object '{vision_result.object_name}' not in validation table; AI result accepted.",
        )

    # Check aspect ratio
    ar = shape_props.aspect_ratio
    ar_range = expectations["expected_aspect_ratio"]
    ar_ok = _check_range(ar, ar_range)
    evidence_list.append(ValidationEvidence(
        feature="Aspect Ratio",
        expected=f"{ar_range[0]:.1f} – {ar_range[1]:.1f}",
        actual=f"{ar:.2f}",
        supports=ar_ok,
    ))
    if ar_ok:
        supporting += 1
    else:
        contradicting += 1

    # Check solidity
    sol = shape_props.solidity
    sol_range = expectations["expected_solidity"]
    sol_ok = _check_range(sol, sol_range)
    evidence_list.append(ValidationEvidence(
        feature="Solidity",
        expected=f"{sol_range[0]:.2f} – {sol_range[1]:.2f}",
        actual=f"{sol:.3f}",
        supports=sol_ok,
    ))
    if sol_ok:
        supporting += 1
    else:
        contradicting += 1

    # Determine validation status
    total_checks = supporting + contradicting
    support_ratio = supporting / max(1, total_checks)

    if support_ratio >= 0.75:
        status = "confirmed"
        adjustment = 1.05  # Slightly boost confidence
        is_valid = True
        notes = f"CV evidence strongly supports AI identification of '{vision_result.object_name}'."
    elif support_ratio >= 0.50:
        status = "plausible"
        adjustment = 1.0  # No change
        is_valid = True
        notes = f"CV evidence is consistent with AI identification of '{vision_result.object_name}'."
    elif support_ratio >= 0.25:
        status = "uncertain"
        adjustment = 0.85  # Moderate reduction
        is_valid = True
        notes = (
            f"Some CV features don't match expected profile for '{vision_result.object_name}'. "
            f"AI identification retained but confidence reduced."
        )
    else:
        status = "contradicted"
        adjustment = 0.70  # Significant reduction, but AI still wins
        is_valid = False
        notes = (
            f"CV evidence largely contradicts AI identification of '{vision_result.object_name}'. "
            f"AI identification retained with reduced confidence. "
            f"Consider alternatives: {[a.name for a in vision_result.alternatives]}."
        )

    logger.info(
        f"[VALIDATION] Status={status} | "
        f"Supporting={supporting}/{total_checks} | "
        f"Adjustment={adjustment:.2f} | "
        f"AI={vision_result.object_name}"
    )

    return ValidationResult(
        is_validated=is_valid,
        validation_status=status,
        confidence_adjustment=adjustment,
        evidence=evidence_list,
        supporting_count=supporting,
        contradicting_count=contradicting,
        notes=notes,
    )
