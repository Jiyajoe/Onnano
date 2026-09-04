"""
config.py - Centralized configuration for AI/CV pipeline, weights, and classification rules.
"""

# Similarity Calculation Weights (Must sum to 1.0)
SIMILARITY_WEIGHTS = {
    "shape": 0.25,
    "dimensions": 0.15,
    "color": 0.15,
    "texture": 0.15,
    "features": 0.20,
    "edges": 0.10,
}

# Image Processing Configurations
MAX_IMAGE_DIMENSION = 1000
MIN_OBJECT_AREA_RATIO = 0.002
MAX_OBJECTS_ALLOWED = 10

# Illumination & Image Quality Thresholds
MIN_BRIGHTNESS = 15
MAX_BRIGHTNESS = 252
MIN_CONTRAST = 4
MIN_LAPLACIAN_VAR = 8

# Relationship Classification Rules
RELATIONSHIP_TIERS = [
    {
        "id": "twin",
        "name": "Twin-like",
        "emoji": "👯",
        "min_score": 85.0,
        "requires_category_match": True,
        "description": "Very high visual similarity and same category.",
    },
    {
        "id": "related",
        "name": "Related",
        "emoji": "👨‍👩‍👧",
        "min_score": 70.0,
        "requires_category_match": True,
        "description": "High similarity / same category but noticeable differences.",
    },
    {
        "id": "distantly_related",
        "name": "Distantly Related",
        "emoji": "🤝",
        "min_score": 50.0,
        "requires_category_match": False,
        "description": "Moderate similarity or related object categories.",
    },
    {
        "id": "barely_related",
        "name": "Barely Related",
        "emoji": "👀",
        "min_score": 30.0,
        "requires_category_match": False,
        "description": "Low similarity but some shared visual properties.",
    },
    {
        "id": "strangers",
        "name": "Strangers",
        "emoji": "💀",
        "min_score": 0.0,
        "requires_category_match": False,
        "description": "Very low similarity / unrelated objects.",
    },
]

DISCLAIMER_TEXT = (
    "All dimensions are relative image-based pixel measurements and not calibrated physical units. "
    "Similarity percentages are derived from algorithmic CV & ML feature extraction."
)

# ---- AI Vision Model Configuration ----
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MAX_OBJECT_COUNT = int(os.environ.get("MAX_OBJECT_COUNT", "10"))
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TIMEOUT_SEC = 15
GEMINI_MAX_RETRIES = 2

# Vision Identification Prompt (structured, anti-hallucination)
VISION_IDENTIFICATION_PROMPT = """Analyze this image carefully.

Identify the primary physical object visible in the image.

Return the most specific object name that can be supported by the visual evidence.

Do not guess based only on color or geometric shape.

Consider:
- overall appearance
- object structure
- recognizable parts
- semantic characteristics
- texture
- shape
- context

If the object is a pen, return "pen".
If it is a pencil, return "pencil".
If it is scissors, return "scissors".
If it is a banana, return "banana".

Do not choose a category merely because the object has a similar shape.

If uncertain, return an uncertainty state rather than confidently inventing an unrelated object.

Return ONLY a JSON object with this exact structure (no markdown fences, no extra text):
{
  "object_name": "<most specific name>",
  "category": "<general category>",
  "confidence": <0.0 to 1.0>,
  "alternatives": [
    {"name": "<alternative1>", "confidence": <0.0 to 1.0>},
    {"name": "<alternative2>", "confidence": <0.0 to 1.0>}
  ],
  "reasoning": "<brief explanation of why this identification>"
}"""

# Validation: Object feature knowledge base for cross-referencing AI results
OBJECT_FEATURE_EXPECTATIONS = {
    "pen": {
        "expected_aspect_ratio": (3.5, 22.0),
        "expected_solidity": (0.70, 1.0),
        "structural_cues": ["elongated", "cylindrical", "clip", "tip", "cap"],
        "category": "Writing Instrument",
    },
    "pencil": {
        "expected_aspect_ratio": (4.0, 22.0),
        "expected_solidity": (0.80, 1.0),
        "structural_cues": ["elongated", "hexagonal", "tapered_tip", "eraser"],
        "category": "Writing Instrument",
    },
    "scissors": {
        "expected_aspect_ratio": (1.2, 4.5),
        "expected_solidity": (0.35, 0.80),
        "structural_cues": ["dual_blades", "pivot", "finger_loops", "articulated"],
        "category": "Cutting Tool",
    },
    "banana": {
        "expected_aspect_ratio": (2.0, 6.0),
        "expected_solidity": (0.65, 0.92),
        "structural_cues": ["curved", "organic", "tapered_ends", "stem"],
        "category": "Fruit",
    },
    "bottle": {
        "expected_aspect_ratio": (1.8, 5.5),
        "expected_solidity": (0.80, 1.0),
        "structural_cues": ["cylindrical", "neck", "cap", "base"],
        "category": "Drinkware & Bottles",
    },
    "cup": {
        "expected_aspect_ratio": (0.6, 1.8),
        "expected_solidity": (0.70, 1.0),
        "structural_cues": ["cylindrical_vessel", "handle", "open_rim"],
        "category": "Drinkware & Bottles",
    },
    "book": {
        "expected_aspect_ratio": (1.0, 2.0),
        "expected_solidity": (0.88, 1.0),
        "structural_cues": ["rectangular", "flat", "spine", "pages"],
        "category": "Stationery & Books",
    },
    "phone": {
        "expected_aspect_ratio": (1.6, 2.5),
        "expected_solidity": (0.92, 1.0),
        "structural_cues": ["rectangular", "flat_slab", "rounded_corners", "screen"],
        "category": "Personal Electronics",
    },
    "spoon": {
        "expected_aspect_ratio": (2.5, 7.0),
        "expected_solidity": (0.65, 0.96),
        "structural_cues": ["handle", "bowl_head", "metallic"],
        "category": "Cutlery & Tableware",
    },
    "fork": {
        "expected_aspect_ratio": (2.5, 7.5),
        "expected_solidity": (0.50, 0.90),
        "structural_cues": ["handle", "tines", "prongs", "metallic"],
        "category": "Cutlery & Tableware",
    },
    "apple": {
        "expected_aspect_ratio": (0.8, 1.3),
        "expected_solidity": (0.88, 1.0),
        "structural_cues": ["spherical", "stem", "organic"],
        "category": "Fruit",
    },
    "ruler": {
        "expected_aspect_ratio": (4.0, 18.0),
        "expected_solidity": (0.90, 1.0),
        "structural_cues": ["flat_strip", "straight_edges", "markings"],
        "category": "Stationery & Books",
    },
}

# Validation thresholds
VALIDATION_CONFIDENCE_THRESHOLD = 0.50  # Below this, flag as low confidence
VALIDATION_CONTRADICTION_THRESHOLD = 0.30  # If CV evidence contradicts AI this strongly, re-evaluate
