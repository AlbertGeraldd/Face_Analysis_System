from backend.face_engine.detector import FaceDetector
from backend.face_engine.landmarks import LandmarksExtractor
from backend.face_engine.features import FeaturesExtractor
from backend.face_engine.facs import FACSDetector
import cv2


class FaceEngine:
    """
    FaceEngine ties together detection, landmark extraction, and feature computations.
    Use FaceEngine.process_frame(img) to analyze an image (BGR numpy array).
    """

    def __init__(self):
        self.detector = FaceDetector()
        self.landmarker = LandmarksExtractor()
        self.features = FeaturesExtractor()
        self.facs = FACSDetector()

    def process_frame(self, img, audio_intensity: float = None):
        # Accepts BGR image array
        # Optionally resize large images for performance
        h, w = img.shape[:2]
        max_w = 640
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        result = {"face_detected": False}

        detections = self.detector.detect(img)
        if not detections:
            return result

        mp_results = detections[0]
        landmarks = self.landmarker.extract(mp_results, img.shape)
        feats = self.features.compute(landmarks)

        # Run rule-based FACS detector with optional audio context
        micro = self.facs.detect(feats, landmarks=landmarks, audio_intensity=audio_intensity)

        result.update({
            "face_detected": True,
            "landmarks": landmarks,
            "features": feats,
            "micro_expressions": micro,
        })
        return result

    def close(self):
        self.detector.close()
