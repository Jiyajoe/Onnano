"""
texture.py - Surface variation, gradient metrics, Shannon entropy, and texture descriptors.
"""

from dataclasses import dataclass
from typing import Dict, Any
import cv2
import numpy as np


@dataclass
class TextureAnalysis:
    laplacian_variance: float
    gradient_mean: float
    gradient_std: float
    entropy: float
    descriptor: str
    roughness_score: float     # 0 to 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "laplacian_variance": round(self.laplacian_variance, 1),
            "gradient_mean": round(self.gradient_mean, 2),
            "gradient_std": round(self.gradient_std, 2),
            "entropy": round(self.entropy, 2),
            "descriptor": self.descriptor,
            "roughness_score": round(self.roughness_score, 1),
        }


def compute_shannon_entropy(gray_pixels: np.ndarray) -> float:
    if len(gray_pixels) == 0:
        return 0.0
    hist, _ = np.histogram(gray_pixels, bins=256, range=(0, 256))
    prob = hist / float(len(gray_pixels))
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def extract_texture_analysis(image_bgr: np.ndarray, mask: np.ndarray) -> TextureAnalysis:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    fg_gray = gray[mask > 0]

    if len(fg_gray) == 0:
        return TextureAnalysis(
            laplacian_variance=0.0,
            gradient_mean=0.0,
            gradient_std=0.0,
            entropy=0.0,
            descriptor="Smooth / Uniform",
            roughness_score=0.0,
        )

    # 1. Laplacian Variance
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

    # 3. Entropy
    entropy = compute_shannon_entropy(fg_gray)

    # 4. Normalized Roughness Score (0 - 100)
    # Combining laplacian variance and gradient mean
    raw_score = (min(1000.0, lap_var) / 1000.0) * 50.0 + (min(60.0, grad_mean) / 60.0) * 50.0
    roughness = max(0.0, min(100.0, raw_score))

    # 5. Texture Descriptor
    if roughness < 15.0 and entropy < 4.5:
        descriptor = "Ultra Smooth / Uniform"
    elif roughness < 35.0:
        descriptor = "Slightly Textured / Matte"
    elif roughness < 60.0:
        descriptor = "Brushed Surface / Satin"
    elif roughness < 80.0:
        descriptor = "Coarse / Grainy"
    else:
        descriptor = "High Detail / Patterned"

    return TextureAnalysis(
        laplacian_variance=lap_var,
        gradient_mean=grad_mean,
        gradient_std=grad_std,
        entropy=entropy,
        descriptor=descriptor,
        roughness_score=roughness,
    )
