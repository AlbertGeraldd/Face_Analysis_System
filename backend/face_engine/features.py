import math
import numpy as np


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FeaturesExtractor:
    def __init__(self):
        pass

    def compute(self, pts):
        """
        Compute features from selected landmarks (pixel coordinates).
        Returned features are normalized floats useful for micro-expression analysis.
        """
        mouth_top = pts["mouth_top"]
        mouth_bottom = pts["mouth_bottom"]
        mouth_left = pts["mouth_left"]
        mouth_right = pts["mouth_right"]
        face_width = pts.get("face_width", 1.0)

        mouth_height = _dist(mouth_top, mouth_bottom)
        mouth_width = _dist(mouth_left, mouth_right) or 1.0
        mouth_open_ratio = mouth_height / mouth_width


        left_eye_top = pts["left_eye_top"]
        left_eye_bottom = pts["left_eye_bottom"]
        left_eye_width = _dist(pts["left_eye_left"], pts["left_eye_right"]) or 1.0
        left_eye_open = _dist(left_eye_top, left_eye_bottom) / left_eye_width

        right_eye_top = pts["right_eye_top"]
        right_eye_bottom = pts["right_eye_bottom"]
        right_eye_width = _dist(pts["right_eye_left"], pts["right_eye_right"]) or 1.0
        right_eye_open = _dist(right_eye_top, right_eye_bottom) / right_eye_width

        eye_openness = (left_eye_open + right_eye_open) / 2.0

        left_eyebrow_inner = pts["left_eyebrow_inner"]
        right_eyebrow_inner = pts["right_eyebrow_inner"]
        left_eyebrow_dist = abs(left_eyebrow_inner[1] - left_eye_top[1])
        right_eyebrow_dist = abs(right_eyebrow_inner[1] - right_eye_top[1])
        eyebrow_intensity = ((left_eyebrow_dist / left_eye_width) + (right_eyebrow_dist / right_eye_width)) / 2.0

        mouth_open_norm = mouth_open_ratio / (face_width / 100.0 + 1e-6)
        eyebrow_norm = eyebrow_intensity / (face_width / 100.0 + 1e-6)

        features = {
            "mouth_open_ratio": float(mouth_open_ratio),
            "mouth_open_norm": float(mouth_open_norm),
            "eye_openness": float(eye_openness),
            "eyebrow_intensity": float(eyebrow_intensity),
            "eyebrow_norm": float(eyebrow_norm),
        }
        return features
