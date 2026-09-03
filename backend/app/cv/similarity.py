"""
similarity.py - Pairwise CV/ML similarity calculations, weighted scores, and NxN matrix computation.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from ..config import SIMILARITY_WEIGHTS
from .shape import ShapeProperties
from .dimensions import DimensionMetrics
from .color import ColorAnalysis
from .texture import TextureAnalysis
from .features import VisualFeatures, match_feature_similarity
from .contour import EdgeAnalysis, compare_contour_shapes


@dataclass
class PairwiseComparison:
    obj1_id: int
    obj2_id: int
    shape_similarity: float
    dimension_similarity: float
    color_similarity: float
    texture_similarity: float
    feature_similarity: float
    edge_similarity: float
    area_similarity: float
    overall_similarity: float

    @property
    def pair_label(self) -> str:
        return f"Object {self.obj1_id} ↔ Object {self.obj2_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obj1_id": self.obj1_id,
            "obj2_id": self.obj2_id,
            "pair_label": self.pair_label,
            "shape_similarity": round(self.shape_similarity, 1),
            "dimension_similarity": round(self.dimension_similarity, 1),
            "color_similarity": round(self.color_similarity, 1),
            "texture_similarity": round(self.texture_similarity, 1),
            "feature_similarity": round(self.feature_similarity, 1),
            "edge_similarity": round(self.edge_similarity, 1),
            "area_similarity": round(self.area_similarity, 1),
            "overall_similarity": round(self.overall_similarity, 1),
        }


def compute_shape_similarity(shape1: ShapeProperties, shape2: ShapeProperties, contour1: np.ndarray, contour2: np.ndarray) -> float:
    # 1. Circularity difference (0 to 1)
    circ_sim = max(0.0, 1.0 - abs(shape1.circularity - shape2.circularity)) * 100.0
    
    # 2. Aspect Ratio difference
    max_ar = max(shape1.aspect_ratio, shape2.aspect_ratio, 1.0)
    min_ar = max(0.1, min(shape1.aspect_ratio, shape2.aspect_ratio))
    ar_sim = (min_ar / max_ar) * 100.0

    # 3. Solidity difference
    sol_sim = max(0.0, 1.0 - abs(shape1.solidity - shape2.solidity)) * 100.0

    # 4. Contour shape matching (Hu moments)
    contour_sim = compare_contour_shapes(contour1, contour2)

    shape_score = 0.35 * contour_sim + 0.30 * ar_sim + 0.20 * circ_sim + 0.15 * sol_sim
    return max(0.0, min(100.0, shape_score))


def compute_dimension_similarity(dim1: DimensionMetrics, dim2: DimensionMetrics) -> float:
    # Compare aspect ratio
    max_ar = max(dim1.aspect_ratio, dim2.aspect_ratio, 0.01)
    min_ar = min(dim1.aspect_ratio, dim2.aspect_ratio)
    ar_score = (min_ar / max_ar) * 100.0

    # Compare normalized width/height proportions
    max_w = max(dim1.pixel_width, dim2.pixel_width, 1)
    min_w = min(dim1.pixel_width, dim2.pixel_width)
    w_score = (float(min_w) / float(max_w)) * 100.0

    max_h = max(dim1.pixel_height, dim2.pixel_height, 1)
    min_h = min(dim1.pixel_height, dim2.pixel_height)
    h_score = (float(min_h) / float(max_h)) * 100.0

    dim_score = 0.50 * ar_score + 0.25 * w_score + 0.25 * h_score
    return max(0.0, min(100.0, dim_score))


def compute_color_similarity(
    color1: ColorAnalysis,
    color2: ColorAnalysis,
    img1_bgr: np.ndarray,
    mask1: np.ndarray,
    img2_bgr: np.ndarray,
    mask2: np.ndarray,
) -> float:
    # 1. 2D HSV Histogram Correlation
    hsv1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([hsv1], [0, 1], mask1, [30, 32], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], mask2, [30, 32], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    hist_score = max(0.0, float(hist_corr)) * 100.0

    # 2. Dominant RGB distance
    r1, g1, b1 = color1.dominant_rgb
    r2, g2, b2 = color2.dominant_rgb
    rgb_dist = (((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5) / 441.67  # normalized 0-1
    dominant_score = max(0.0, (1.0 - rgb_dist)) * 100.0

    color_score = 0.60 * hist_score + 0.40 * dominant_score
    return max(0.0, min(100.0, color_score))


def compute_texture_similarity(tex1: TextureAnalysis, tex2: TextureAnalysis) -> float:
    # 1. Roughness difference
    rough_diff = abs(tex1.roughness_score - tex2.roughness_score)
    rough_sim = max(0.0, (100.0 - rough_diff))

    # 2. Entropy difference
    ent_diff = abs(tex1.entropy - tex2.entropy)
    ent_sim = max(0.0, (1.0 - (ent_diff / 8.0))) * 100.0

    # 3. Gradient mean difference
    max_gm = max(tex1.gradient_mean, tex2.gradient_mean, 1.0)
    min_gm = min(tex1.gradient_mean, tex2.gradient_mean)
    gm_sim = (min_gm / max_gm) * 100.0

    tex_score = 0.45 * rough_sim + 0.35 * ent_sim + 0.20 * gm_sim
    return max(0.0, min(100.0, tex_score))


def compute_edge_similarity(edge1: EdgeAnalysis, edge2: EdgeAnalysis) -> float:
    max_dens = max(edge1.edge_density_pct, edge2.edge_density_pct, 0.01)
    min_dens = min(edge1.edge_density_pct, edge2.edge_density_pct)
    return max(0.0, min(100.0, (min_dens / max_dens) * 100.0))


def compute_area_similarity(dim1: DimensionMetrics, dim2: DimensionMetrics) -> float:
    max_area = max(dim1.pixel_area, dim2.pixel_area, 1)
    min_area = min(dim1.pixel_area, dim2.pixel_area)
    return max(0.0, min(100.0, (float(min_area) / float(max_area)) * 100.0))


def compare_two_objects(
    id1: int,
    id2: int,
    shape1: ShapeProperties,
    shape2: ShapeProperties,
    contour1: np.ndarray,
    contour2: np.ndarray,
    dim1: DimensionMetrics,
    dim2: DimensionMetrics,
    color1: ColorAnalysis,
    color2: ColorAnalysis,
    img1: np.ndarray,
    mask1: np.ndarray,
    img2: np.ndarray,
    mask2: np.ndarray,
    tex1: TextureAnalysis,
    tex2: TextureAnalysis,
    feat1: VisualFeatures,
    feat2: VisualFeatures,
    edge1: EdgeAnalysis,
    edge2: EdgeAnalysis,
) -> PairwiseComparison:
    shape_sim = compute_shape_similarity(shape1, shape2, contour1, contour2)
    dim_sim = compute_dimension_similarity(dim1, dim2)
    col_sim = compute_color_similarity(color1, color2, img1, mask1, img2, mask2)
    tex_sim = compute_texture_similarity(tex1, tex2)
    feat_sim = match_feature_similarity(feat1, feat2)
    edge_sim = compute_edge_similarity(edge1, edge2)
    area_sim = compute_area_similarity(dim1, dim2)

    # Weighted Overall Similarity based on config
    w = SIMILARITY_WEIGHTS
    overall = (
        w["shape"] * shape_sim +
        w["dimensions"] * dim_sim +
        w["color"] * col_sim +
        w["texture"] * tex_sim +
        w["features"] * feat_sim +
        w["edges"] * edge_sim
    )

    return PairwiseComparison(
        obj1_id=id1,
        obj2_id=id2,
        shape_similarity=shape_sim,
        dimension_similarity=dim_sim,
        color_similarity=col_sim,
        texture_similarity=tex_sim,
        feature_similarity=feat_sim,
        edge_similarity=edge_sim,
        area_similarity=area_sim,
        overall_similarity=overall,
    )


def build_similarity_matrix(comparisons: List[PairwiseComparison], n_objects: int) -> List[List[float]]:
    """Builds an N x N symmetric similarity matrix with 100.0 on the diagonal."""
    matrix = [[100.0 for _ in range(n_objects)] for _ in range(n_objects)]
    for comp in comparisons:
        i = comp.obj1_id - 1
        j = comp.obj2_id - 1
        if 0 <= i < n_objects and 0 <= j < n_objects:
            score = round(comp.overall_similarity, 1)
            matrix[i][j] = score
            matrix[j][i] = score
    return matrix
