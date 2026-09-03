"""
features.py - Real ORB/AKAZE keypoint detection, descriptor extraction, and pairwise matching.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


@dataclass
class VisualFeatures:
    keypoints_count: int
    descriptors: Optional[np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keypoints_count": self.keypoints_count,
            "has_descriptors": self.descriptors is not None,
        }


def extract_visual_features(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> VisualFeatures:
    """Extracts ORB keypoints and descriptors confined to foreground mask."""
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=12)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    kps, descs = orb.detectAndCompute(gray, mask=mask)
    if descs is None:
        # Fallback to AKAZE if ORB returns no descriptors
        akaze = cv2.AKAZE_create()
        kps, descs = akaze.detectAndCompute(gray, mask=mask)

    kp_count = len(kps) if kps else 0
    return VisualFeatures(keypoints_count=kp_count, descriptors=descs)


def match_feature_similarity(feat1: VisualFeatures, feat2: VisualFeatures) -> float:
    """
    Computes pairwise feature similarity score (0 - 100%) using BFMatcher with Lowe's ratio test.
    """
    d1 = feat1.descriptors
    d2 = feat2.descriptors

    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = bf.knnMatch(d1, d2, k=2)
    except Exception:
        return 0.0

    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.78 * n.distance:
                good_matches.append(m)

    min_pts = min(len(d1), len(d2))
    if min_pts == 0:
        return 0.0

    match_ratio = float(len(good_matches)) / float(min_pts)
    similarity = min(100.0, match_ratio * 160.0)  # Scale realistic ratio to percentage
    return round(float(similarity), 1)
