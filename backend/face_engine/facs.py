import math


class FACSDetector:
    """
    Simple, rule-based FACS / micro-expression detector.

    This is intentionally small and explainable: each detected label includes
    a numeric score and a short explanation of which rule fired and which
    features contributed.

    Rules operate on features produced by `FeaturesExtractor` and on selected
    landmarks when needed. Audio intensity may be used only as a contextual
    modifier to boost or dampen scores (not as a primary detection signal).
    """

    def __init__(self, cfg=None):
        # thresholds are conservative defaults for a research prototype
        self.cfg = cfg or {
            "surprise_eyebrow_norm": 0.6,
            "surprise_eye_openness": 0.45,
            "surprise_mouth_open": 0.25,
            "mouth_open_thresh": 0.15,
            "eyebrow_raise_thresh": 0.35,
            "audio_boost": 0.25,  # fraction to boost score when audio intensity high
        }

    def _clamp(self, v):
        return max(0.0, min(1.0, float(v)))

    def detect(self, features: dict, landmarks: dict = None, audio_intensity: float = None):
        """
        Return a dict of micro-expression candidates with scores and rule traces.

        - `features`: dict from `FeaturesExtractor.compute`
        - `landmarks`: pixel coordinates dict (optional)
        - `audio_intensity`: optional float [0..1] used only to modulate scores
        """
        res = {}

        mouth_open = features.get("mouth_open_norm", 0.0)
        eye_open = features.get("eye_openness", 0.0)
        eyebrow = features.get("eyebrow_norm", 0.0)

        # Surprise: eyebrows raised + eyes wide + mouth open (all contribute)
        surprise_score = 0.0
        surprise_components = {}
        if eyebrow >= self.cfg["surprise_eyebrow_norm"]:
            surprise_components["eyebrow"] = eyebrow
            surprise_score += 0.4
        if eye_open >= self.cfg["surprise_eye_openness"]:
            surprise_components["eye_open"] = eye_open
            surprise_score += 0.35
        if mouth_open >= self.cfg["surprise_mouth_open"]:
            surprise_components["mouth_open"] = mouth_open
            surprise_score += 0.25

        surprise_score = self._clamp(surprise_score)

        # Mouth open (AU27-like): simple detector for mouth opening
        mouth_score = self._clamp((mouth_open - self.cfg["mouth_open_thresh"]) * 2.0)

        # Eyebrow raise / inner brow (AU1/AU2-like)
        eyebrow_score = self._clamp((eyebrow - self.cfg["eyebrow_raise_thresh"]) * 1.5)

        # Apply audio intensity contextual boost if provided (explainable and small)
        if audio_intensity is not None:
            # expect audio_intensity roughly in [0..1]; scale boost linearly
            boost = self.cfg.get("audio_boost", 0.0) * float(audio_intensity)
            surprise_score = self._clamp(surprise_score + boost)
            mouth_score = self._clamp(mouth_score + boost * 0.5)
            eyebrow_score = self._clamp(eyebrow_score + boost * 0.5)

        if surprise_score > 0.0:
            res["surprise"] = {
                "score": float(surprise_score),
                "rule": "eyebrow+eye+mouth combination",
                "components": surprise_components,
            }

        if mouth_score > 0.0:
            res["mouth_open"] = {
                "score": float(mouth_score),
                "rule": "mouth_open_norm > mouth_open_thresh",
                "value": float(mouth_open),
            }

        if eyebrow_score > 0.0:
            res["eyebrow_raise"] = {
                "score": float(eyebrow_score),
                "rule": "eyebrow_norm > eyebrow_raise_thresh",
                "value": float(eyebrow),
            }

        return res
