"""
color.py - Dominant color clustering, HSV/CIELAB color space statistics, and illumination invariance.
Confined strictly to foreground object mask pixels.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class ColorPaletteItem:
    hex: str
    rgb: List[int]
    percentage: float
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hex": self.hex,
            "rgb": self.rgb,
            "percentage": round(self.percentage, 1),
            "name": self.name,
        }


@dataclass
class ColorAnalysis:
    dominant_name: str
    dominant_hex: str
    dominant_rgb: List[int]
    average_rgb: List[int]
    cielab: Dict[str, float]            # L* (0-100), a* (-128 to 127), b* (-128 to 127)
    hsv: Dict[str, float]               # Hue (0-360), Saturation (0-100), Brightness (0-100)
    palette: List[ColorPaletteItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_name": self.dominant_name,
            "dominant_hex": self.dominant_hex,
            "dominant_rgb": self.dominant_rgb,
            "average_rgb": self.average_rgb,
            "cielab": self.cielab,
            "hsv": self.hsv,
            "palette": [p.to_dict() for p in self.palette],
        }


COLOR_MAP = [
    {"name": "Pitch Black", "rgb": (20, 20, 25), "hsv_range": (0, 360, 0, 100, 0, 20)},
    {"name": "Pure White", "rgb": (245, 245, 245), "hsv_range": (0, 360, 0, 15, 85, 100)},
    {"name": "Slate Gray", "rgb": (128, 130, 138), "hsv_range": (0, 360, 0, 20, 20, 85)},
    {"name": "Crimson Red", "rgb": (220, 45, 55), "hsv_range": (345, 15, 40, 100, 30, 100)},
    {"name": "Amber Gold / Orange", "rgb": (245, 160, 30), "hsv_range": (15, 45, 40, 100, 40, 100)},
    {"name": "Lemon Yellow", "rgb": (240, 220, 40), "hsv_range": (45, 68, 40, 100, 50, 100)},
    {"name": "Emerald Green", "rgb": (40, 180, 95), "hsv_range": (68, 160, 30, 100, 30, 100)},
    {"name": "Cyan / Teal", "rgb": (35, 195, 200), "hsv_range": (160, 195, 30, 100, 30, 100)},
    {"name": "Cobalt Blue", "rgb": (45, 110, 230), "hsv_range": (195, 255, 30, 100, 30, 100)},
    {"name": "Royal Purple", "rgb": (145, 65, 210), "hsv_range": (255, 310, 30, 100, 30, 100)},
    {"name": "Rose Pink", "rgb": (235, 105, 165), "hsv_range": (310, 345, 30, 100, 40, 100)},
    {"name": "Earth Brown", "rgb": (140, 85, 45), "hsv_range": (15, 45, 30, 90, 20, 60)},
]


def name_color(r: int, g: int, b: int) -> str:
    rgb_arr = np.uint8([[[b, g, r]]])
    hsv_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2HSV)[0][0]
    h = float(hsv_arr[0]) * 2.0
    s = (float(hsv_arr[1]) / 255.0) * 100.0
    v = (float(hsv_arr[2]) / 255.0) * 100.0

    if v < 18:
        return "Pitch Black"
    if s < 12 and v > 82:
        return "Pure White"
    if s < 18:
        return "Silver / Gray"

    for entry in COLOR_MAP:
        hmin, hmax, smin, smax, vmin, vmax = entry["hsv_range"]
        if hmin > hmax:
            h_match = (h >= hmin or h <= hmax)
        else:
            h_match = (hmin <= h <= hmax)

        if h_match and smin <= s <= smax and vmin <= v <= vmax:
            return entry["name"]

    closest_name = "Neutral"
    min_dist = 1e9
    for entry in COLOR_MAP:
        er, eg, eb = entry["rgb"]
        d = ((r - er) ** 2 + (g - eg) ** 2 + (b - eb) ** 2) ** 0.5
        if d < min_dist:
            min_dist = d
            closest_name = entry["name"]
    return closest_name


def extract_color_analysis(image_bgr: np.ndarray, mask: np.ndarray, k_clusters: int = 5) -> ColorAnalysis:
    fg_pixels = image_bgr[mask > 0]
    if len(fg_pixels) == 0:
        return ColorAnalysis(
            dominant_name="Unknown",
            dominant_hex="#888888",
            dominant_rgb=[136, 136, 136],
            average_rgb=[136, 136, 136],
            cielab={"L": 50.0, "a": 0.0, "b": 0.0},
            hsv={"hue": 0.0, "saturation": 0.0, "brightness": 50.0},
            palette=[],
        )

    # Convert to RGB
    fg_rgb = cv2.cvtColor(fg_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # Subsample if large for fast clustering
    if len(fg_rgb) > 25000:
        indices = np.random.choice(len(fg_rgb), 25000, replace=False)
        sample_rgb = fg_rgb[indices]
    else:
        sample_rgb = fg_rgb

    # Average RGB color
    avg_rgb = [int(round(c)) for c in np.mean(fg_rgb, axis=0)]

    # KMeans clustering for dominant palette
    k = min(k_clusters, max(1, len(sample_rgb)))
    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
    kmeans.fit(sample_rgb)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=k)
    total_pts = float(len(labels))

    sorted_indices = np.argsort(counts)[::-1]

    palette: List[ColorPaletteItem] = []
    for idx in sorted_indices:
        r, g, b = [int(round(max(0, min(255, c)))) for c in centers[idx]]
        hex_val = f"#{r:02x}{g:02x}{b:02x}"
        pct = (counts[idx] / total_pts) * 100.0
        cname = name_color(r, g, b)
        palette.append(ColorPaletteItem(hex=hex_val, rgb=[r, g, b], percentage=pct, name=cname))

    dominant = palette[0]

    # Calculate average HSV
    avg_bgr_pixel = np.uint8([[[avg_rgb[2], avg_rgb[1], avg_rgb[0]]]])
    avg_hsv_pixel = cv2.cvtColor(avg_bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
    hsv_dict = {
        "hue": round(float(avg_hsv_pixel[0]) * 2.0, 1),
        "saturation": round((float(avg_hsv_pixel[1]) / 255.0) * 100.0, 1),
        "brightness": round((float(avg_hsv_pixel[2]) / 255.0) * 100.0, 1),
    }

    # Calculate average CIELAB for perceptual color consistency
    avg_lab_pixel = cv2.cvtColor(avg_bgr_pixel, cv2.COLOR_BGR2LAB)[0][0]
    # L: 0-255 -> scale to 0-100; a, b: 0-255 -> offset to -128..127
    lab_dict = {
        "L": round(float(avg_lab_pixel[0]) * (100.0 / 255.0), 1),
        "a": round(float(avg_lab_pixel[1]) - 128.0, 1),
        "b": round(float(avg_lab_pixel[2]) - 128.0, 1),
    }

    return ColorAnalysis(
        dominant_name=dominant.name,
        dominant_hex=dominant.hex,
        dominant_rgb=dominant.rgb,
        average_rgb=avg_rgb,
        cielab=lab_dict,
        hsv=hsv_dict,
        palette=palette,
    )
