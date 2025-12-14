import time
import json
import cv2
import numpy as np
import mediapipe as mp

from backend.face_engine.detector import FaceDetector
from backend.face_engine.landmarks import normalize_landmarks
from backend.face_engine.smoother import LandmarkSmoother


class WebcamLandmarkStreamer:
    """
    WebcamLandmarkStreamer provides a reusable API to capture frames from a
    webcam, run MediaPipe FaceMesh (468 landmarks) and return landmark data.

    Methods:
      - get_landmarks_from_frame(frame): returns list of landmark dicts
      - run_display(): runs a live window with landmarks drawn (for debugging)
    """

    def __init__(self, src=0, width=640, height=480, fps=30, detector_params=None):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        dp = detector_params or {}
        # Default to fast settings: low model complexity, no refine
        self.detector = FaceDetector(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=dp.get("model_complexity", 0),
            refine_landmarks=dp.get("refine_landmarks", False),
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        # Temporal smoother for normalized landmarks. Default window 5 (small)
        # to reduce jitter but preserve micro-expression spikes.
        self.smoother = LandmarkSmoother(window=5, spike_thresh=0.20)

    def _landmarks_to_list(self, mp_landmarks, image_shape):
        """
        Convert MediaPipe landmark list to a list of dicts with both
        normalized (x,y,z) and pixel (x_px, y_px) coordinates.
        """
        h, w = image_shape[0], image_shape[1]
        out = []
        for lm in mp_landmarks.landmark:
            x = float(lm.x)
            y = float(lm.y)
            z = float(lm.z)
            x_px = int(round(x * w))
            y_px = int(round(y * h))
            out.append({"x": x, "y": y, "z": z, "x_px": x_px, "y_px": y_px})
        return out

    def get_landmarks_from_frame(self, frame):
        """
        Process a BGR frame and return:
          - landmarks: list of 468 landmark dicts (or [] if none)
          - mp_landmarks: raw MediaPipe landmarks object (or None)
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return [], None
        mp_landmarks = results.multi_face_landmarks[0]
        lm_list = self._landmarks_to_list(mp_landmarks, frame.shape)
        normalized = normalize_landmarks(lm_list)
        # smoothed: list with added fields 'sx','sy','sz' (smoothed coords)
        smoothed = self.smoother.smooth(normalized)
        return lm_list, mp_landmarks, normalized, smoothed

    def draw_landmarks(self, frame, mp_landmarks):
        """Draw landmarks on BGR frame in-place for debugging display."""
        if mp_landmarks is None:
            return frame
        self.mp_draw.draw_landmarks(
            frame,
            mp_landmarks,
            self.detector.mp_face.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style(),
        )
        self.mp_draw.draw_landmarks(
            frame,
            mp_landmarks,
            self.detector.mp_face.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style(),
        )
        return frame

    def run_display(self, show_fps=True, print_json=False):
        """
        Start webcam loop, display annotated frames and return when 'q' pressed.

        If `print_json` is True, prints per-frame landmark JSON to stdout.
        """
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        prev = time.time()
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[webcam] frame read failed")
                    break

                frame_count += 1
                lm_list, mp_landmarks, normalized, smoothed = self.get_landmarks_from_frame(frame)

                # draw for debug
                self.draw_landmarks(frame, mp_landmarks)

                # fps calc
                if show_fps:
                    now = time.time()
                    dt = now - prev if now - prev > 0 else 1e-6
                    fps = 1.0 / dt
                    prev = now
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("FaceMesh Landmarks", frame)

                if print_json:
                    out = {
                        "timestamp": time.time(),
                        "landmarks": lm_list,
                        "normalized": normalized,
                        "smoothed": smoothed,
                    }
                    print(json.dumps(out))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
