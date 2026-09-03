"""
scoring.py

One consistent fairness score used everywhere in the app (spec section 10),
plus the classification tiers (section 11).
"""

from dataclasses import dataclass

# Classification thresholds - kept as simple module-level constants so
# they're easy to tune without hunting through logic.
CLASSIFICATION_THRESHOLDS = [
    (98.0, "PERFECTLY FAIR", "🏆"),
    (95.0, "ALMOST PERFECT", "😎"),
    (90.0, "PRETTY FAIR", "👍"),
    (0.0, "SIBLING FIGHT WARNING", "🚨😂"),
]


def fairness_score(value1: float, value2: float) -> float:
    """
    balance_error = |v1 - v2| / (v1 + v2)
    fairness_score = (1 - balance_error) * 100, clamped to [0, 100]
    """
    total = value1 + value2
    if total <= 0:
        return 0.0
    balance_error = abs(value1 - value2) / total
    score = (1 - balance_error) * 100.0
    return max(0.0, min(100.0, score))


@dataclass
class Classification:
    label: str
    emoji: str
    tier_index: int


def classify(score: float) -> Classification:
    for i, (threshold, label, emoji) in enumerate(CLASSIFICATION_THRESHOLDS):
        if score >= threshold:
            return Classification(label=label, emoji=emoji, tier_index=i)
    # Should be unreachable since the last threshold is 0.0
    label_last = CLASSIFICATION_THRESHOLDS[-1]
    return Classification(label=label_last[1], emoji=label_last[2], tier_index=len(CLASSIFICATION_THRESHOLDS) - 1)
