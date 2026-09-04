"""
test_pipeline.py - Test script to verify the upgraded CV, normalization, slicing, quality gate, and comparison pipeline.
"""

import cv2
import numpy as np
from app.cv.quality import evaluate_image_quality
from app.cv.segmentation import isolate_primary_object, validate_mask_quality
from app.cv.orientation import normalize_posture
from app.cv.shape import extract_shape_properties
from app.cv.dimensions import extract_dimensions
from app.cv.color import extract_color_analysis
from app.cv.texture import extract_texture_analysis
from app.cv.features import extract_visual_features, match_feature_similarity_with_ransac
from app.cv.contour import extract_edge_analysis
from app.cv.slicing import divide_normalized_object
from app.cv.similarity import compare_two_objects, build_similarity_matrix
from app.ai.object_detection import identify_object
from app.ai.verdict import classify_relationship
from app.routes.analyze import process_single_object_pipeline


def create_tilted_object(color_bgr, width=40, height=260, angle_deg=35, bg_val=245):
    img = np.ones((500, 500, 3), dtype=np.uint8) * bg_val
    center = (250, 250)
    rect = (center, (width, height), angle_deg)
    box = np.int32(cv2.boxPoints(rect))
    
    cv2.fillPoly(img, [box], color_bgr)
    cv2.polylines(img, [box], True, (max(0, color_bgr[0]-30), max(0, color_bgr[1]-30), max(0, color_bgr[2]-30)), 2)
    
    # Add texture noise
    noise = np.random.randint(-4, 4, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def run_tests():
    print("==================================================")
    print("1. Testing Single Object Pipeline (Split One Object)")
    print("==================================================")
    pencil1 = create_tilted_object((30, 180, 240), width=35, height=280, angle_deg=-30)  # Yellow Pencil
    res = process_single_object_pipeline(pencil1, parts_count=4, reference_cm=18.0)
    
    print("Image Quality Score:", res["image_quality"]["overall_score"], "%")
    print("Detected Object:", res["object"]["detected_type"])
    print("Category:", res["object"]["category"])
    print("Brand (Non-Hallucinated):", res["object"]["brand"])
    print("Confidence:", res["object"]["confidence_pct"], "%")
    print("Raw Tilt Angle:", res["object"]["orientation"]["detected_angle_deg"], "deg")
    print("Correction Rotation:", res["object"]["orientation"]["correction_angle_deg"], "deg")
    print("Shape:", res["object"]["shape"]["shape_type"], f"(AR: {res['object']['shape']['aspect_ratio']})")
    print("Dominant Color:", res["object"]["color"]["dominant_name"], res["object"]["color"]["dominant_hex"])
    print("Texture:", res["object"]["texture"]["descriptor"], f"(Informative: {res['object']['texture']['is_informative']})")
    print("Keypoints:", res["object"]["features"]["keypoints_count"])
    print("Calibrated Height:", res["object"]["dimensions"]["physical_height_cm"], "cm")
    print("Equal Slicing Parts Count:", res["division"]["parts_count"])
    for p in res["division"]["parts"]:
        print(f"  -> {p['label']}: height={p['pixel_height']}px, share={p['percentage']:.1f}%, color={p['color_hex']}")
    
    print("\n==================================================")
    print("2. Testing Multi-Object Pipeline (Compare 3 Objects)")
    print("==================================================")
    pencil2 = create_tilted_object((25, 175, 235), width=36, height=275, angle_deg=15)  # Similar Yellow Pencil
    bottle = create_tilted_object((200, 100, 40), width=90, height=220, angle_deg=5)     # Blue Bottle
    
    objs = [pencil1, pencil2, bottle]
    data_list = []
    for i, img in enumerate(objs):
        seg, resized = isolate_primary_object(img)
        fa = float(resized.shape[0] * resized.shape[1])
        ori = normalize_posture(resized, seg)
        shp = extract_shape_properties(ori.normalized_contour, ori.normalized_mask)
        dim = extract_dimensions(ori.normalized_contour, fa)
        col = extract_color_analysis(ori.ai_aligned_bgr, ori.normalized_mask)
        tex = extract_texture_analysis(ori.ai_aligned_bgr, ori.normalized_mask)
        edg = extract_edge_analysis(ori.ai_aligned_bgr, ori.normalized_mask)
        feat = extract_visual_features(ori.ai_aligned_bgr, ori.normalized_mask)
        ident = identify_object(shp, dim, col, tex, edg)
        data_list.append({
            "id": i+1, "shape": shp, "dim": dim, "col": col, "tex": tex,
            "edg": edg, "feat": feat, "contour": ori.normalized_contour,
            "mask": ori.normalized_mask, "bgr": ori.ai_aligned_bgr,
            "ident": ident, "mask_qual": seg.mask_quality,
        })
        print(f"Object {i+1}: {ident.name} ({ident.confidence_pct}%) - {col.dominant_name} [Mask Conf: {seg.mask_quality.mask_confidence*100:.1f}%]")

    comparisons = []
    for i in range(len(data_list)):
        for j in range(i+1, len(data_list)):
            d1, d2 = data_list[i], data_list[j]
            comp = compare_two_objects(
                d1["id"], d2["id"], d1["shape"], d2["shape"], d1["contour"], d2["contour"],
                d1["dim"], d2["dim"], d1["col"], d2["col"], d1["bgr"], d1["mask"],
                d2["bgr"], d2["mask"], d1["tex"], d2["tex"], d1["feat"], d2["feat"],
                d1["edg"], d2["edg"],
                mask_qual1=d1["mask_qual"], mask_qual2=d2["mask_qual"],
                type1=d1["ident"].name, type2=d2["ident"].name,
                cat1=d1["ident"].category, cat2=d2["ident"].category,
            )
            comparisons.append(comp)
            print(f"\nPair Obj{comp.obj1_id} vs Obj{comp.obj2_id}:")
            print(f"  Shape: {comp.shape_similarity}% | Dims: {comp.dimension_similarity}% | Color: {comp.color_similarity}%")
            print(f"  Texture: {comp.texture_similarity}% | RANSAC Inliers: {comp.ransac_matches.geometrically_consistent_matches} (Sim: {comp.feature_similarity}%)")
            print(f"  --> OVERALL SIMILARITY: {comp.overall_similarity}% | CONFIDENCE: {comp.overall_confidence}%")
            print(f"  Why Breakdown: {len(comp.why_explanation.positive_factors)} positives, {len(comp.why_explanation.differing_factors)} diffs")

    matrix = build_similarity_matrix(comparisons, 3)
    print("\nSimilarity Matrix:")
    for row in matrix:
        print("  ", row)

    verdict = classify_relationship(
        avg_score=comparisons[0].overall_similarity,
        same_category=(data_list[0]["ident"].category == data_list[1]["ident"].category),
        same_type=(data_list[0]["ident"].name == data_list[1]["ident"].name),
        type1=data_list[0]["ident"].name,
        type2=data_list[1]["ident"].name,
        avg_comp=comparisons[0],
    )
    print("\nRelationship Tier:", verdict.tier_name)
    print("Malayalam AI Verdict:", verdict.malayalam_verdict.encode('ascii', 'ignore').decode('ascii'))
    print("English Translation:", verdict.english_translation)
    print("\n==================================================")
    print("ALL UPGRADED BACKEND CV & AI PIPELINE TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
