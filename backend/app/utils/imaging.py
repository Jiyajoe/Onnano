"""
imaging.py - Base64 data-URL encoding/decoding and image helper routines.
"""

import base64
import cv2
import numpy as np


class InvalidImageError(Exception):
    pass


def decode_upload_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise InvalidImageError("empty upload")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("could not decode image")
    return image


def encode_bgr_to_data_url(image_bgr: np.ndarray, quality: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InvalidImageError("could not encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def encode_rgba_to_data_url(image_rgba: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_rgba)
    if not ok:
        raise InvalidImageError("could not encode PNG image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"
