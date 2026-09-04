"""
texture.py - Surface variation, Local Binary Patterns (LBP), Shannon entropy, and texture descriptors.
Identifies smooth objects with low texture information to prevent manufactured texture scores.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import cv2
import numpy as np


@dataclass
class TextureAnalysis:
    laplacian_variance: float
    gradient_mean: float
    gradient_std: float
    entropy: float
    lbp_uniformity: float
    descriptor: str
    roughness_score: float         # 0 to 100
    is_informative: bool           # False if smooth surface lacks visible texture
    information_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "laplacian_variance": round(self.laplacian_variance, 1),
            "gradient_mean": round(self.gradient_mean, 2),
            "gradient_std": round(self.gradient_std, 2),
            "entropy": round(self.entropy, 2),
            "lbp_uniformity": round(self.lbp_uniformity, 3),
            "descriptor": self.descriptor,
            "roughness_score": round(self.roughness_score, 1),
            "is_informative": self.is_informative,
            "information_note": self.information_note,
        }


def compute_shannon_entropy(gray_pixels: np.ndarray) -> float:
    if len(gray_pixels) == 0:
        return 0.0
    hist, _ = np.histogram(gray_pixels, bins=256, range=(0, 256))
    prob = hist / float(len(gray_pixels))
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def compute_simplified_lbp(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Computes 8-neighbor Local Binary Pattern uniformity over foreground pixels.
    Uniformity close to 1.0 indicates very smooth/flat texture.
    """
    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return 1.0

    # Kernel comparisons for 8 neighbors
    center = gray[1:h-1, 1:w-1]
    neighbors = [
        gray[0:h-2, 0:w-2], gray[0:h-2, 1:w-1], gray[0:h-2, 2:w],
        gray[1:h-1, 2:w], gray[2:h, 2:w], gray[2:h, 1:w-1],
        gray[2:h, 0:w-2], gray[1:h-1, 0:w-2]
    ]

    inner_mask = mask[1:h-1, 1:w-1] > 0
    if np.count_nonzero(inner_mask) == 0:
        return 1.0

    # Count how many neighbors differ by more than threshold 4
    diff_sum = np.zeros(center.shape, dtype=np.float32)
    for n in neighbors:
        diff_sum += (np.abs(n.astype(np.float32) - center.astype(np.float32)) > 4.0).astype(np.float32)

    fg_diff = diff_sum[inner_mask]
    uniform_pixels = np.count_nonzero(fg_diff <= 2)
    uniformity = float(uniform_pixels) / max(1.0, float(len(fg_diff)))
    return uniformity


def extract_texture_analysis(image_bgr: np.ndarray, mask: np.ndarray) -> TextureAnalysis:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    fg_gray = gray[mask > 0]

    if len(fg_gray) == 0:
        return TextureAnalysis(
            laplacian_variance=0.0,
            gradient_mean=0.0,
            gradient_std=0.0,
            entropy=0.0,
            lbp_uniformity=1.0,
            descriptor="Smooth / Uniform",
            roughness_score=0.0,
            is_informative=False,
            information_note="Texture analysis: Low information (no foreground pixels).",
        )

    # 1. Laplacian Variance (edge frequency)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    fg_lap = laplacian[mask > 0]
    lap_var = float(np.var(fg_lap)) if len(fg_lap) > 0 else 0.0

    # 2. Sobel Gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    fg_mag = mag[mask > 0]

    grad_mean = float(np.mean(fg_mag)) if len(fg_mag) > 0 else 0.0
    grad_std = float(np.std(fg_mag)) if len(fg_mag) > 0 else 0.0

    # 3. Shannon Entropy
    entropy = compute_shannon_entropy(fg_gray)

    # 4. LBP Uniformity
    lbp_uniformity = compute_simplified_lbp(gray, mask)

    # 5. Normalized Roughness Score (0 - 100)
    raw_score = (min(1000.0, lap_var) / 1000.0) * 50.0 + (min(60.0, grad_mean) / 60.0) * 50.0
    roughness = max(0.0, min(100.0, raw_score))

    # 6. Informative vs Low-Information determination
    is_smooth = roughness < 14.0 or (lap_var < 18.0 and entropy < 4.6 and lbp_uniformity > 0.85)
    is_informative = not is_smooth

    if is_smooth:
        descriptor = "Smooth / Matte Surface"
        info_note = "Texture analysis: Low information (surface is predominantly smooth/uniform)."
    elif roughness < 35.0:
        descriptor = "Subtle Microtexture"
        info_note = "Visible low-frequency surface texture present."
    elif roughness < 60.0:
        descriptor = "Brushed Surface / Satin Grain"
        info_note = "Distinct directional or satin grain detected."
    elif roughness < 80.0:
        descriptor = "Coarse / Grainy Texture"
        info_note = "High-contrast surface relief and granulations."
    else:
        descriptor = "Patterned / High-Relief"
        info_note = "Dense textural pattern and micro-features present."

    return TextureAnalysis(
        laplacian_variance=lap_var,
        gradient_mean=grad_mean,
        gradient_std=grad_std,
        entropy=entropy,
        lbp_uniformity=lbp_uniformity,
        descriptor=descriptor,
        roughness_score=roughness,
        is_informative=is_informative,
        information_note=info_note,
    )
