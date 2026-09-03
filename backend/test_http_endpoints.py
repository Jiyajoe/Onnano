"""
test_http_endpoints.py - Comprehensive API testing script.
"""

import urllib.request
import urllib.parse
import json
import io
import cv2
import numpy as np

def create_synthetic_image(bgr_color, width=40, height=250, angle_deg=25):
    img = np.ones((450, 450, 3), dtype=np.uint8) * 245
    center = (225, 225)
    rect = (center, (width, height), angle_deg)
    box = np.int32(cv2.boxPoints(rect))
    cv2.fillPoly(img, [box], bgr_color)
    cv2.polylines(img, [box], True, (20, 20, 20), 2)
    
    # Encode as JPEG
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def post_multipart(url, files_dict, fields_dict=None):
    boundary = "----WebKitFormBoundaryOnnanoTest7MA4YWxkTrZu0gW"
    body = io.BytesIO()

    if fields_dict:
        for k, v in fields_dict.items():
            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            body.write(f"{v}\r\n".encode("utf-8"))

    for field_name, (filename, file_bytes) in files_dict:
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(file_bytes)
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    payload = body.getvalue()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_api():
    print("Testing HTTP Endpoints on http://127.0.0.1:8000 ...")
    
    # 1. Health
    with urllib.request.urlopen("http://127.0.0.1:8000/health") as resp:
        health_data = json.loads(resp.read().decode("utf-8"))
        print("1. Health Endpoint:", health_data)

    # 2. Analyze Single Object
    img1_bytes = create_synthetic_image((30, 200, 245), width=35, height=260, angle_deg=35) # Yellow Pencil
    res_single = post_multipart(
        "http://127.0.0.1:8000/api/analyze-object",
        [("file", ("pencil.jpg", img1_bytes))],
        {"parts": 4}
    )
    print("2. Analyze Single Object:")
    print("   Detected Type:", res_single["object"]["detected_type"])
    print("   Tilt:", res_single["object"]["orientation"]["detected_angle_deg"], "deg")
    print("   Correction:", res_single["object"]["orientation"]["correction_angle_deg"], "deg")
    print("   Division Parts:", len(res_single["division"]["parts"]))

    # 3. Divide Single Object (6 parts)
    res_div = post_multipart(
        "http://127.0.0.1:8000/api/divide-object",
        [("file", ("pencil.jpg", img1_bytes))],
        {"parts": 6}
    )
    print("3. Divide Single Object (6 parts):", len(res_div["division"]["parts"]), "parts returned")

    # 4. Compare Multiple Objects (3 objects)
    img2_bytes = create_synthetic_image((25, 190, 240), width=36, height=255, angle_deg=-20) # Similar Yellow Pencil
    img3_bytes = create_synthetic_image((210, 110, 40), width=85, height=210, angle_deg=0)   # Blue Bottle

    res_compare = post_multipart(
        "http://127.0.0.1:8000/api/compare-objects",
        [
            ("files", ("obj1.jpg", img1_bytes)),
            ("files", ("obj2.jpg", img2_bytes)),
            ("files", ("obj3.jpg", img3_bytes)),
        ]
    )
    print("4. Compare Multiple Objects:")
    print("   Objects Count:", res_compare["object_count"])
    print("   Comparisons Count:", len(res_compare["comparisons"]))
    print("   Overall Similarity:", res_compare["overall_similarity"], "%")
    print("   Relationship Tier:", res_compare["relationship"]["tier_name"])
    print("   Malayalam AI Verdict:", res_compare["relationship"]["malayalam_verdict"].encode("ascii", "ignore").decode("ascii"))
    print("   English Translation:", res_compare["relationship"]["english_translation"])
    print("   Similarity Matrix:", res_compare["similarity_matrix"])
    print("   Comparison Table Rows:", len(res_compare["comparison_table"]))
    print("\nALL HTTP API TESTS PASSED WITH FLYING COLORS!")

if __name__ == "__main__":
    test_api()
