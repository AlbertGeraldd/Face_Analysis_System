import mediapipe as mp
import cv2
import numpy as np


class FaceDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 model_complexity=0, refine_landmarks=False):
        """
        FaceDetector wraps MediaPipe FaceMesh.

        Args:
            max_num_faces: maximum faces to track
            min_detection_confidence: detection threshold
            min_tracking_confidence: tracking threshold
            model_complexity: 0,1,2 trade-off between speed and accuracy (lower= faster)
            refine_landmarks: whether to enable iris refinement (slower)
        """
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def detect(self, image):
        # image: BGR numpy array
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return []
        return results.multi_face_landmarks

    def close(self):
        self.face_mesh.close()
