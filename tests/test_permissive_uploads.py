"""
test_permissive_uploads.py - Comprehensive verification of permissive image acceptance.
Tests the 10 real-world image scenarios required:
1. Portrait phone photo (9:16)
2. Landscape phone photo (16:9)
3. Square photo (1:1)
4. Object near left edge
5. Object near right edge
6. Object near top/bottom edge
7. Close-up object
8. Tilted / rotated object
9. Object on colorful background
10. Object with shadows
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.cv.segmentation import isolate_primary_object
from app.cv.quality import evaluate_image_quality
from app.routes.analyze import process_single_object_pipeline


def make_blank_canvas(h: int, w: int, color=(240, 240, 240)) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = color
    return canvas


def test_portrait_photo():
    """Test 1: Portrait phone photo (9:16 aspect ratio)."""
    img = make_blank_canvas(960, 540, (230, 230, 235))
    # Draw bottle/pen in portrait center
    cv2.rectangle(img, (220, 200), (320, 760), (30, 80, 200), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    assert seg.area > 0
    res = process_single_object_pipeline(img, parts_count=4)
    assert res["object"]["detected_type"] is not None
    print("  OK [Test 1] Portrait Phone Photo (9:16) accepted & analyzed")


def test_landscape_photo():
    """Test 2: Landscape phone photo (16:9 aspect ratio)."""
    img = make_blank_canvas(720, 1280, (245, 245, 245))
    # Draw ruler/pencil across landscape
    cv2.rectangle(img, (200, 310), (1080, 410), (20, 160, 220), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    assert seg.area > 0
    res = process_single_object_pipeline(img, parts_count=4)
    assert res["object"]["detected_type"] is not None
    print("  OK [Test 2] Landscape Phone Photo (16:9) accepted & analyzed")


def test_square_photo():
    """Test 3: Square photo (1:1 aspect ratio)."""
    img = make_blank_canvas(600, 600, (240, 235, 230))
    cv2.circle(img, (300, 300), 180, (180, 50, 40), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img, parts_count=3)
    assert res["object"]["detected_type"] is not None
    print("  OK [Test 3] Square Photo (1:1) accepted & analyzed")


def test_object_near_left_edge():
    """Test 4: Object located near/touching left edge."""
    img = make_blank_canvas(600, 800, (250, 250, 250))
    # Touching x = 0..150
    cv2.rectangle(img, (0, 150), (160, 450), (50, 50, 180), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img)
    assert res["image_quality"]["accepted"] is True
    print("  OK [Test 4] Object Near Left Edge accepted & analyzed without clipping error")


def test_object_near_right_edge():
    """Test 5: Object located near/touching right edge."""
    img = make_blank_canvas(600, 800, (250, 250, 250))
    # Touching right border x = 650..800
    cv2.rectangle(img, (650, 150), (800, 450), (50, 180, 50), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img)
    assert res["image_quality"]["accepted"] is True
    print("  OK [Test 5] Object Near Right Edge accepted & analyzed without clipping error")


def test_object_near_top_bottom_edge():
    """Test 6: Object near top or bottom edge."""
    img = make_blank_canvas(800, 600, (245, 245, 245))
    # Touching top border y = 0..250
    cv2.rectangle(img, (180, 0), (420, 250), (200, 100, 20), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img)
    assert res["image_quality"]["accepted"] is True
    print("  OK [Test 6] Object Near Top/Bottom Edge accepted & analyzed without error")


def test_closeup_object():
    """Test 7: Close-up object occupying 85% of frame."""
    img = make_blank_canvas(600, 800, (220, 220, 220))
    # Large object filling almost entire view
    cv2.rectangle(img, (40, 40), (760, 560), (40, 120, 210), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img)
    assert res["image_quality"]["accepted"] is True
    print("  OK [Test 7] Close-Up Object (filling frame) accepted & analyzed")


def test_tilted_object():
    """Test 8: Rotated / tilted object."""
    img = make_blank_canvas(700, 700, (240, 240, 240))
    # Create tilted rectangle at 45 degrees
    center = (350, 350)
    size = (400, 80)
    angle = 45.0
    rect = (center, size, angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(img, [box], 0, (220, 80, 30), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img)
    assert res["object"]["orientation"]["is_symmetric"] is not None
    print("  OK [Test 8] Tilted Object (45 deg) accepted & posture normalized")


def test_colorful_background():
    """Test 9: Object placed on a textured / colorful background."""
    h, w = 600, 800
    # Colorful gradient background
    x_grad = np.tile(np.linspace(100, 220, w, dtype=np.uint8), (h, 1))
    y_grad = np.tile(np.linspace(120, 240, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    b_chan = np.full((h, w), 180, dtype=np.uint8)
    img = cv2.merge([b_chan, y_grad, x_grad])

    # Distinct physical object in middle
    cv2.rectangle(img, (250, 180), (550, 420), (10, 10, 20), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img)
    assert res["image_quality"]["accepted"] is True
    print("  OK [Test 9] Object on Colorful Background accepted & analyzed")


def test_object_with_shadows():
    """Test 10: Object with realistic cast shadows."""
    img = make_blank_canvas(600, 800, (235, 235, 235))
    # Shadow (darker gray gradient)
    cv2.ellipse(img, (430, 400), (220, 60), 15, 0, 360, (160, 160, 160), -1)
    # Primary physical object
    cv2.rectangle(img, (220, 150), (460, 380), (30, 120, 210), -1)
    seg, _ = isolate_primary_object(img)
    assert seg is not None
    res = process_single_object_pipeline(img)
    assert res["image_quality"]["accepted"] is True
    print("  OK [Test 10] Object with Cast Shadow accepted & analyzed")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PERMISSIVE IMAGE UPLOAD TEST SUITE (10 SCENARIOS)")
    print("=" * 60)

    test_portrait_photo()
    test_landscape_photo()
    test_square_photo()
    test_object_near_left_edge()
    test_object_near_right_edge()
    test_object_near_top_bottom_edge()
    test_closeup_object()
    test_tilted_object()
    test_colorful_background()
    test_object_with_shadows()

    print("=" * 60)
    print("ALL 10 PERMISSIVE IMAGE SCENARIOS PASSED WITH FLYING COLORS!")
    print("=" * 60)
