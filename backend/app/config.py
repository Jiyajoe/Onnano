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
