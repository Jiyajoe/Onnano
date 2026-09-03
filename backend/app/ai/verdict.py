"""
verdict.py - Dynamic relationship classification and humorous Malayalam AI verdicts.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import random

from ..config import RELATIONSHIP_TIERS
from ..cv.similarity import PairwiseComparison


@dataclass
class RelationshipVerdict:
    tier_id: str
    tier_name: str
    tier_emoji: str
    description: str
    malayalam_verdict: str
    english_translation: str
    category_match: bool
    is_twin: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "tier_name": self.tier_name,
            "tier_emoji": self.tier_emoji,
            "description": self.description,
            "malayalam_verdict": self.malayalam_verdict,
            "english_translation": self.english_translation,
            "category_match": self.category_match,
            "is_twin": self.is_twin,
        }


# Dynamic Malayalam verdict templates conditioned on relationship tier and feature strengths
VERDICT_TEMPLATES = {
    "twin": [
        (
            "Randum {type1} aanu... 100% pixel-level TWINS aanu! 👯🏆 Ini aarum fight cheyyanda!",
            "Both are {type1}... 100% pixel-level twins! No more sibling fights!",
        ),
        (
            "Sherikkum twins thanne! Shape-um dimension-um kandal aarum confuse aavum! 😎✨",
            "True twins! Seeing the shape and dimensions, anyone would get confused!",
        ),
        (
            "Case closed! Two {type1} objects with uncanny twin precision! 🏆⚖️",
            "Case closed! Two {type1} objects with uncanny twin precision!",
        ),
    ],
    "related": [
        (
            "Randum {type1} aanu... pakshe twins alla, close siblings aanu! 👀😂",
            "Both are {type1}... but not twins, they are close siblings!",
        ),
        (
            "Same category aanu, but dimensions nokkumpol oru cheriyath difference und! 📏🤝",
            "Same category, but looking at dimensions there is a slight noticeable difference!",
        ),
        (
            "Almost identical aanu, but AI found subtle differences in texture and color! 👨‍👩‍👧",
            "Almost identical, but AI found subtle differences in texture and color!",
        ),
    ],
    "distantly_related": [
        (
            "Same family-il aanu... pakshe cousins aanu! Oru pole thonnum pakshe alla! 🤝😭",
            "In the same family tree... but they are cousins! Looks alike at first glance, but not identical!",
        ),
        (
            "Shape nokkiyal oru pole... pakshe colour-um feature-um kandille? Distant relatives! 🤝",
            "Looking at shape they look similar... but did you see the color and features? Distant relatives!",
        ),
        (
            "Oru family WhatsApp group-il undenkilum, direct match alla ketto! 😂",
            "Even if they share the same family group, this is definitely not a direct match!",
        ),
    ],
    "barely_related": [
        (
            "Evideyokkeyo oru similarity und... pakshe total different category aanu! 👀💀",
            "There's some vague visual resemblance... but totally different categories!",
        ),
        (
            "Kandal oru cheriya match thonnum, pakshe AI scan nokkiyal full mismatch! 😂",
            "Looks like a slight match from afar, but the AI scan reveals a clear mismatch!",
        ),
    ],
    "strangers": [
        (
            "Ithu randum thammil oru bandhavum illa! Total strangers aanu ketto! 💀😂",
            "These two have absolutely no relationship! Total strangers!",
        ),
        (
            "Object 1: {type1} vs Object 2: {type2}... Pole apart! Sibling dispute dismissed! 🙅‍♂️⚖️",
            "Object 1: {type1} vs Object 2: {type2}... Poles apart! Sibling dispute dismissed!",
        ),
    ],
}


def classify_relationship(
    avg_score: float,
    same_category: bool,
    same_type: bool,
    type1: str = "Object",
    type2: str = "Object",
    avg_comp: PairwiseComparison = None,
) -> RelationshipVerdict:
    """
    Classifies relationship using multi-factor criteria (not just overall score).
    """
    if same_type and avg_score >= 84.0 and (avg_comp is None or avg_comp.shape_similarity >= 80.0):
        tier_id = "twin"
    elif same_category and avg_score >= 68.0:
        tier_id = "related"
    elif avg_score >= 48.0 or (same_category and avg_score >= 42.0):
        tier_id = "distantly_related"
    elif avg_score >= 28.0:
        tier_id = "barely_related"
    else:
        tier_id = "strangers"

    # Find matching tier config
    tier_info = next((t for t in RELATIONSHIP_TIERS if t["id"] == tier_id), RELATIONSHIP_TIERS[-1])

    # Select funny verdict template
    templates = VERDICT_TEMPLATES.get(tier_id, VERDICT_TEMPLATES["strangers"])
    mal_template, eng_template = random.choice(templates)

    mal_verdict = mal_template.format(type1=type1, type2=type2)
    eng_verdict = eng_template.format(type1=type1, type2=type2)

    return RelationshipVerdict(
        tier_id=tier_info["id"],
        tier_name=tier_info["name"],
        tier_emoji=tier_info["emoji"],
        description=tier_info["description"],
        malayalam_verdict=mal_verdict,
        english_translation=eng_verdict,
        category_match=same_category,
        is_twin=(tier_id == "twin"),
    )
