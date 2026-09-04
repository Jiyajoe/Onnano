"""
vision_identifier.py - Gemini Vision API integration for semantic object identification.

Sends the ORIGINAL image (not processed/thresholded) to Google Gemini for genuine
semantic understanding. Returns structured identification with confidence and alternatives.
"""

import json
import logging
import re
import base64
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

from ..config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SEC,
    GEMINI_MAX_RETRIES,
    VISION_IDENTIFICATION_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class VisionAlternative:
    name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "confidence": round(self.confidence, 3)}


@dataclass
class VisionIdentificationResult:
    """Result from the AI vision model."""
    object_name: str
    category: str
    confidence: float
    alternatives: List[VisionAlternative] = field(default_factory=list)
    reasoning: str = ""
    source: str = "gemini_vision"  # "gemini_vision" | "cv_fallback" | "error_fallback"
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_name": self.object_name,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "reasoning": self.reasoning,
            "source": self.source,
            "error": self.error if self.error else None,
        }


def _encode_image_for_api(image_bgr: np.ndarray) -> bytes:
    """
    Encode the ORIGINAL image as JPEG bytes for the Gemini API.
    Preserves full visual information — no thresholding, no edge detection,
    no grayscale conversion. Just a clean JPEG of the original photograph.
    """
    # Encode at high quality to preserve semantic detail
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    success, buffer = cv2.imencode('.jpg', image_bgr, encode_params)
    if not success:
        raise ValueError("Failed to encode image for vision API")
    return buffer.tobytes()


def _parse_gemini_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse the Gemini response text into structured JSON.
    Handles cases where the model wraps JSON in markdown fences.
    """
    text = raw_text.strip()

    # Remove markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from Gemini response: {text[:200]}")


def identify_with_vision_model(image_bgr: np.ndarray) -> VisionIdentificationResult:
    """
    Send the ORIGINAL image to Gemini Vision for semantic identification.

    Key design decisions:
    - Original image is sent (not processed/thresholded/edge-detected)
    - Structured prompt prevents color-only or shape-only guessing
    - Returns uncertainty rather than fabricating a confident wrong answer
    - Gracefully falls back on API errors
    """

    # Check if API key is configured
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
        logger.warning("[VISION] No GEMINI_API_KEY configured — skipping AI vision identification")
        return VisionIdentificationResult(
            object_name="",
            category="",
            confidence=0.0,
            source="error_fallback",
            error="No GEMINI_API_KEY configured. Set it in .env file.",
        )

    try:
        # Import here so the rest of the app works even without the package installed
        from google import genai

        # Initialize the client
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Encode the original image
        image_bytes = _encode_image_for_api(image_bgr)

        logger.info(f"[VISION] Sending original image ({len(image_bytes)} bytes) to {GEMINI_MODEL}")

        # Send to Gemini with the original image
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {
                    "parts": [
                        {"text": VISION_IDENTIFICATION_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                            }
                        },
                    ]
                }
            ],
        )

        raw_text = response.text
        logger.info(f"[VISION] Gemini raw response: {raw_text[:300]}")

        # Parse the structured JSON response
        parsed = _parse_gemini_response(raw_text)

        # Extract alternatives
        alternatives = []
        for alt in parsed.get("alternatives", []):
            if isinstance(alt, dict) and "name" in alt:
                alternatives.append(VisionAlternative(
                    name=alt["name"],
                    confidence=float(alt.get("confidence", 0.0)),
                ))

        result = VisionIdentificationResult(
            object_name=parsed.get("object_name", "Unknown"),
            category=parsed.get("category", "Unknown"),
            confidence=float(parsed.get("confidence", 0.5)),
            alternatives=alternatives,
            reasoning=parsed.get("reasoning", ""),
            source="gemini_vision",
            raw_response=raw_text[:500],
        )

        logger.info(
            f"[VISION] AI Identification: {result.object_name} "
            f"(category={result.category}, confidence={result.confidence:.2f})"
        )

        return result

    except ImportError:
        logger.error("[VISION] google-genai package not installed. Run: pip install google-genai")
        return VisionIdentificationResult(
            object_name="",
            category="",
            confidence=0.0,
            source="error_fallback",
            error="google-genai package not installed.",
        )

    except Exception as e:
        logger.error(f"[VISION] Gemini API error: {traceback.format_exc()}")
        return VisionIdentificationResult(
            object_name="",
            category="",
            confidence=0.0,
            source="error_fallback",
            error=f"Vision API error: {str(e)}",
        )
