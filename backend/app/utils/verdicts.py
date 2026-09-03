"""
verdicts.py

Configurable verdict templates (spec section 12) plus friendly, translated
error/warning strings (section 24) so the frontend never has to show a
raw exception to the user.
"""

import random

VERDICT_TEMPLATES = {
    "tier_0": [  # PERFECTLY FAIR
        "50:50! Ini aarum complain cheyyanda! 🏆",
        "Pixel-perfect aanu ketto. Case closed! 🏆",
    ],
    "tier_1": [  # ALMOST PERFECT
        "Athu poli! Randu perkkum almost same share aanu! 😎⚖️",
        "Ini fair aanu! Fight cancelled. 😂",
    ],
    "tier_2": [  # PRETTY FAIR
        "Almost fair aanu… pakshe oru side kurachu valuthaanu 👀",
        "Pretty fair aanu, but AI is still watching. 👀⚖️",
    ],
    "tier_3": [  # SIBLING FIGHT WARNING
        "Ithu fair alla ketto… sibling fight officially started! 😂",
        "Technically fair alla. AI evidence prakaram oral kurachu extra kitti! 😭",
    ],
}

PERSONALITY_LINES = [
    "AI has entered the family dispute.",
    "Please remain calm. I am calculating.",
    "Sibling complaint detected.",
    "No bias. Only pixels.",
    "Decision loading…",
]

LOADING_STAGES = [
    "🔍 Detecting objects...",
    "📐 Measuring...",
    "⚖️ Comparing...",
    "🧮 Finding fairest division...",
    "🤖 Preparing verdict...",
]

VISUAL_MEASUREMENT_DISCLAIMER = (
    "This fairness score is based on visual measurements. For true "
    "weight-based fairness, connect a weighing sensor in a future version."
)

ERROR_MESSAGES = {
    "camera_permission_denied": "Camera permission was denied. Please allow camera access to scan objects.",
    "no_camera": "No camera was found on this device.",
    "no_object": "No object was detected. Try placing it on a plain, contrasting background.",
    "too_many_objects": "Too many objects detected. Please reduce the number of items or increase the max-object limit.",
    "object_too_small": "The object looks too small in the frame. Try moving the camera closer.",
    "too_dark": "The image is too dark. Try scanning in better lighting.",
    "too_bright": "The image is too bright / overexposed. Try reducing glare or direct light.",
    "poor_contrast": "The object doesn't stand out enough from the background. Try a plainer, more contrasting surface.",
    "blurry_image": "The image looks blurry. Hold the camera steady and try again.",
    "no_cutting_line": "Could not determine a fair cutting line for this object. Try a clearer, unobstructed shot.",
    "less_than_two_pieces": "Only one piece was detected during verification. Make sure both pieces are visible.",
    "backend_unavailable": "Couldn't reach the AI backend. Please check your connection and try again.",
    "invalid_image": "That doesn't look like a valid image. Please try scanning again.",
}


def get_verdict(tier_index: int) -> str:
    key = f"tier_{min(tier_index, 3)}"
    return random.choice(VERDICT_TEMPLATES[key])


def get_personality_line() -> str:
    return random.choice(PERSONALITY_LINES)


def get_error_message(code: str) -> str:
    return ERROR_MESSAGES.get(code, "Something went wrong while processing the image. Please try again.")
