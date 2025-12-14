"""
Rule-based Action Unit (AU) detector using normalized landmarks.

Notes:
- Uses normalized, face-relative coordinates (nx,ny,nz) where coordinates are
  centered at the nose tip and scaled by inter-ocular distance. This makes
  measurements invariant to head translation and camera distance.
- Baseline is maintained as a slow exponential moving average (EMA). AU
  activation is computed as the relative change (delta) between current
  measurement and baseline, scaled to [0,1].
- This module is intentionally rule-based and explainable: each AU lists the
  landmarks used, the baseline measurement, and the rule for activation.
"""
from typing import List, Dict, Optional
import numpy as np


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


class AUDetector:
    """
    Detect a limited set of AUs using normalized landmarks.

    Target AUs:
      - AU1: Inner brow raise
      - AU2: Outer brow raise (approximated using available landmarks)
      - AU4: Brow lowerer
      - AU6: Cheek raise (orbicularis oculi / eye squint)
      - AU12: Lip corner puller (smile)
      - AU15: Lip corner depressor
    """

    # MediaPipe indices used (documented): use the same indices as other modules
    IND = {
        "mouth_top": 13,
        "mouth_bottom": 14,
        "mouth_left": 61,
        "mouth_right": 291,
        "left_eye_top": 159,
        "left_eye_bottom": 145,
        "right_eye_top": 386,
        "right_eye_bottom": 374,
        "left_eyebrow_inner": 46,
        "right_eyebrow_inner": 276,
    }

    def __init__(self, baseline_alpha: float = 0.01, delta_window: int = 3):
        """
        Args:
          baseline_alpha: EMA alpha for updating baseline (small -> slow update)
          delta_window: number of recent frames to consider for delta smoothing
        """
        self.baseline_alpha = float(baseline_alpha)
        self.delta_window = int(max(1, delta_window))
        self.baseline = None  # will hold per-landmark baseline Nx3 numpy array
        self.recent = []  # circular buffer of recent normalized frames (np arrays)

        # thresholds for activation (empirical, normalized units)
        self.thresh = {
            "au1": 0.06,  # inner brow raise
            "au2": 0.06,  # outer brow raise (approximated)
            "au4": 0.05,  # brow lowerer
            "au6": 0.04,  # cheek raise / eye squint (eye openness drop)
            "au12": 0.04,  # lip corner puller outward/up
            "au15": 0.03,  # lip corner depressor (down)
        }

    def reset_baseline(self):
        self.baseline = None
        self.recent = []

    def _frame_to_array(self, normalized_frame: List[Dict]) -> np.ndarray:
        return np.array([[p["nx"], p["ny"], p["nz"]] for p in normalized_frame], dtype=float)

    def _update_baseline(self, arr: np.ndarray):
        if self.baseline is None:
            self.baseline = arr.copy()
        else:
            self.baseline = (1.0 - self.baseline_alpha) * self.baseline + self.baseline_alpha * arr

    def detect(self, normalized_frame: List[Dict]) -> Dict:
        """
        Compute AU activations for a single normalized frame.

        Returns a dict mapping AU names to activation dicts {
           'score': 0..1,
           'rule': explanation,
           'components': {...}
        }
        """
        out = {}
        if not normalized_frame:
            # no face: do not update baseline
            return out

        arr = self._frame_to_array(normalized_frame)

        # maintain recent buffer for delta smoothing
        self.recent.append(arr)
        if len(self.recent) > self.delta_window:
            self.recent.pop(0)

        # compute a short-term mean (to reduce single-frame noise)
        recent_mean = np.stack(self.recent, axis=0).mean(axis=0)

        # Initialize baseline if missing (do not treat as activation)
        if self.baseline is None:
            self._update_baseline(recent_mean)
            return out

        # baseline is available; compute deltas between recent_mean and baseline
        delta = recent_mean - self.baseline  # shape (N,3)

        # helper to get scalar measures per side/landmark
        def idx(name):
            return self.IND[name]

        # AU1: inner brow raise -> eyebrow inner moves up (ny decreases)
        left_brow_idx = idx("left_eyebrow_inner")
        right_brow_idx = idx("right_eyebrow_inner")
        # positive raise = baseline_ny - current_ny
        left_delta_brow = (self.baseline[left_brow_idx, 1] - recent_mean[left_brow_idx, 1])
        right_delta_brow = (self.baseline[right_brow_idx, 1] - recent_mean[right_brow_idx, 1])
        au1_score = _clamp01(((left_delta_brow + right_delta_brow) / 2.0) / self.thresh["au1"])
        if au1_score > 0.0:
            out["AU1"] = {
                "score": float(au1_score),
                "rule": "inner eyebrow vertical raise relative to baseline",
                "components": {"left": float(left_delta_brow), "right": float(right_delta_brow)},
            }

        # AU2: outer brow raise (approximated using same inner-brow indices due
        # to limited landmark set). In a richer mapping we would use outer brow
        # indices; here we detect a broader eyebrow lift.
        au2_score = _clamp01(((left_delta_brow + right_delta_brow) / 2.0) / self.thresh["au2"])
        if au2_score > 0.0:
            out["AU2"] = {
                "score": float(au2_score),
                "rule": "approximated outer brow lift (using inner-brow movement)",
                "components": {"left": float(left_delta_brow), "right": float(right_delta_brow)},
            }

        # AU4: brow lowerer -> eyebrow moves down (ny increases)
        left_lower = (recent_mean[left_brow_idx, 1] - self.baseline[left_brow_idx, 1])
        right_lower = (recent_mean[right_brow_idx, 1] - self.baseline[right_brow_idx, 1])
        au4_score = _clamp01(((left_lower + right_lower) / 2.0) / self.thresh["au4"])
        if au4_score > 0.0:
            out["AU4"] = {
                "score": float(au4_score),
                "rule": "inner eyebrow lowering relative to baseline",
                "components": {"left": float(left_lower), "right": float(right_lower)},
            }

        # AU6: cheek raise / eye squint -> eye openness decreases
        l_eye_top = idx("left_eye_top")
        l_eye_bot = idx("left_eye_bottom")
        r_eye_top = idx("right_eye_top")
        r_eye_bot = idx("right_eye_bottom")

        left_eye_open_base = self.baseline[l_eye_top, 1] - self.baseline[l_eye_bot, 1]
        left_eye_open_now = recent_mean[l_eye_top, 1] - recent_mean[l_eye_bot, 1]
        right_eye_open_base = self.baseline[r_eye_top, 1] - self.baseline[r_eye_bot, 1]
        right_eye_open_now = recent_mean[r_eye_top, 1] - recent_mean[r_eye_bot, 1]

        # decrease in openness indicates squint/cheek raise
        left_eye_drop = (left_eye_open_base - left_eye_open_now)
        right_eye_drop = (right_eye_open_base - right_eye_open_now)
        au6_score = _clamp01(((left_eye_drop + right_eye_drop) / 2.0) / self.thresh["au6"])
        if au6_score > 0.0:
            out["AU6"] = {
                "score": float(au6_score),
                "rule": "eye openness decrease (squint) relative to baseline",
                "components": {"left_drop": float(left_eye_drop), "right_drop": float(right_eye_drop)},
            }

        # AU12: lip corner puller (smile) -> corners move outward and slightly up
        ml = idx("mouth_left")
        mr = idx("mouth_right")
        # outward = increase in absolute x distance from nose center (since normalized coords)
        left_out = abs(recent_mean[ml, 0] ) - abs(self.baseline[ml, 0])
        right_out = abs(recent_mean[mr, 0]) - abs(self.baseline[mr, 0])
        # upward = baseline_y - current_y
        left_up = (self.baseline[ml, 1] - recent_mean[ml, 1])
        right_up = (self.baseline[mr, 1] - recent_mean[mr, 1])

        # combine outward and upward; take max normalized score
        left_score = max(left_out / self.thresh["au12"], left_up / self.thresh["au12"]) if self.thresh["au12"]>0 else 0.0
        right_score = max(right_out / self.thresh["au12"], right_up / self.thresh["au12"]) if self.thresh["au12"]>0 else 0.0
        au12_score = _clamp01((left_score + right_score) / 2.0)
        if au12_score > 0.0:
            out["AU12"] = {
                "score": float(au12_score),
                "rule": "lip corner outward/up movement relative to baseline",
                "components": {"left_out": float(left_out), "right_out": float(right_out), "left_up": float(left_up), "right_up": float(right_up)},
            }

        # AU15: lip corner depressor -> corners move down
        left_down = (recent_mean[ml, 1] - self.baseline[ml, 1])
        right_down = (recent_mean[mr, 1] - self.baseline[mr, 1])
        au15_score = _clamp01(((left_down + right_down) / 2.0) / self.thresh["au15"])
        if au15_score > 0.0:
            out["AU15"] = {
                "score": float(au15_score),
                "rule": "lip corner downward movement relative to baseline",
                "components": {"left": float(left_down), "right": float(right_down)},
            }

        # After detections, update baseline slowly so the system adapts to slow changes
        # but not to short expressions. EMA with small alpha accomplishes this.
        self._update_baseline(recent_mean)

        return out
