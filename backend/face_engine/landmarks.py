import numpy as np


# Some selected FaceMesh landmark indices used for features.
# Indices are taken from MediaPipe FaceMesh 468 landmarks.
INDEX_MAP = {
    "mouth_top": 13,
    "mouth_bottom": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eye_left": 33,
    "left_eye_right": 133,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "right_eye_left": 362,
    "right_eye_right": 263,
    "left_eyebrow_inner": 46,
    "right_eyebrow_inner": 276,
}


class LandmarksExtractor:
    def __init__(self):
        pass

    def extract(self, mp_landmarks, image_shape):
        """
        Convert MediaPipe normalized landmarks to pixel coordinates and return key points.
        Returns a dict containing pixel (x,y) for selected points.
        """
        h, w = image_shape[0], image_shape[1]
        pts = {}
        for name, idx in INDEX_MAP.items():
            # mp_landmarks.landmark is a sequence indexed by landmark id
            lm = mp_landmarks.landmark[idx]
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            pts[name] = (x_px, y_px)

        # Also return some facial size normalization distances, e.g., face width
        # Use mouth_left and mouth_right to approximate face width
        ml = pts["mouth_left"]
        mr = pts["mouth_right"]
        face_width = np.linalg.norm(np.array(ml) - np.array(mr))
        pts["face_width"] = float(face_width)
        return pts


def normalize_landmarks(lm_list):
    """
    Convert absolute pixel landmark coordinates to face-relative normalized
    coordinates that are invariant to head translation and camera distance.

    Normalization strategy:
      - Origin: nose tip (approximate central reference on the face). We subtract
        the nose pixel coordinates so positional measurements become head-relative.
      - Scale: inter-ocular distance (distance between left and right eye centers).
        Using the distance between eyes compensates for camera distance and face
        size in the image.

    Why normalization is required for micro-expression analysis:
      - Micro-expressions are subtle, small deformations of facial muscles. Raw
        pixel coordinates change a lot when the subject moves the head or when
        camera distance changes. Normalizing by a stable facial reference frame
        (nose origin + eye-distance scale) removes those global transformations
        so downstream measurements focus on relative local deformations.

    Args:
      lm_list: list of landmark dicts as produced by `WebcamLandmarkStreamer._landmarks_to_list`.

    Returns:
      A list of dicts with fields `nx`, `ny`, `nz` (normalized coordinates), and
      preserves original fields `x`,`y`,`z`,`x_px`,`y_px`.
    """
    if not lm_list:
        return []

    # Define indices we use for normalization (MediaPipe FaceMesh indices)
    # These indices are selected for robustness and are commonly used.
    NOSE_TIP = 1
    LEFT_EYE_L = 33
    LEFT_EYE_R = 133
    RIGHT_EYE_L = 362
    RIGHT_EYE_R = 263

    # Defensive: ensure indices exist in lm_list
    n = len(lm_list)
    def safe_get(idx):
        if 0 <= idx < n:
            return lm_list[idx]
        return None

    nose = safe_get(NOSE_TIP) or lm_list[n // 2]

    left_l = safe_get(LEFT_EYE_L)
    left_r = safe_get(LEFT_EYE_R)
    right_l = safe_get(RIGHT_EYE_L)
    right_r = safe_get(RIGHT_EYE_R)

    # Compute eye centers as averages of two eye landmarks when available
    def center(a, b):
        if a is None and b is None:
            return None
        if a is None:
            return (b["x_px"], b["y_px"]) if isinstance(b, dict) else (b["x_px"], b["y_px"])
        if b is None:
            return (a["x_px"], a["y_px"]) if isinstance(a, dict) else (a["x_px"], a["y_px"])
        return ((a["x_px"] + b["x_px"]) / 2.0, (a["y_px"] + b["y_px"]) / 2.0)

    left_center = center(left_l, left_r)
    right_center = center(right_l, right_r)

    if left_center is None or right_center is None:
        # fall back to a rough scale: use horizontal span of all landmarks
        xs = [p["x_px"] for p in lm_list]
        eye_dist = max(xs) - min(xs) or 1.0
    else:
        dx = left_center[0] - right_center[0]
        dy = left_center[1] - right_center[1]
        eye_dist = (dx * dx + dy * dy) ** 0.5 or 1.0

    nose_x = nose["x_px"]
    nose_y = nose["y_px"]

    normalized = []
    for p in lm_list:
        nx = (p["x_px"] - nose_x) / eye_dist
        ny = (p["y_px"] - nose_y) / eye_dist
        nz = p.get("z", 0.0) / eye_dist
        out = p.copy()
        out.update({"nx": float(nx), "ny": float(ny), "nz": float(nz)})
        normalized.append(out)

    return normalized
