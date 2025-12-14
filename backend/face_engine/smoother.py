from collections import deque
import numpy as np


class LandmarkSmoother:
    """
    Temporal smoother for normalized landmarks.

    - Maintains a rolling buffer of the last `window` normalized frames.
    - Computes a moving average but preserves sudden spikes (micro-expressions)
      by bypassing smoothing when the current frame deviates strongly from
      the recent mean.

    Trade-off (explained):
      - Smoothing reduces sensor noise and jitter (helpful for stable feature
        computation), but excessive smoothing can wash out the short, sharp
        movements that characterize micro-expressions. To balance this, the
        algorithm uses a small window (3-5 frames) and a spike threshold.
        When the current normalized landmark set differs from the recent mean
        by more than `spike_thresh` (in normalized units), the current raw
        values are preserved instead of averaged.
    """

    def __init__(self, window: int = 5, spike_thresh: float = 0.25):
        """Create a smoother.

        Args:
          window: number of frames to average (3-5 recommended)
          spike_thresh: relative threshold (in normalized units) for spike
                        detection; larger means fewer preserved spikes.
        """
        assert window >= 1
        self.window = window
        self.spike_thresh = float(spike_thresh)
        self.buf = deque(maxlen=window)

    def reset(self):
        self.buf.clear()

    def smooth(self, normalized_frame):
        """
        normalized_frame: list of landmark dicts with keys `nx`,`ny`,`nz`.

        Returns a list of smoothed landmark dicts (same structure). Raw frame is
        preserved elsewhere if needed.
        """
        if not normalized_frame:
            # No face detected: clear buffer so stale data doesn't leak
            self.reset()
            return []

        # Convert to numpy arrays for vectorized operations
        # Shape: (num_landmarks, 3)
        cur = np.array([[p["nx"], p["ny"], p["nz"]] for p in normalized_frame], dtype=float)

        # If we have previous frames, compute their mean to detect spikes
        if len(self.buf) > 0:
            prev_stack = np.stack(self.buf, axis=0)  # shape (t, n, 3)
            prev_mean = prev_stack.mean(axis=0)  # shape (n,3)

            # Compute per-landmark Euclidean distance between current and prev_mean
            diffs = np.linalg.norm(cur - prev_mean, axis=1)

            # If any landmark exceeds spike threshold, treat whole frame as a spike
            # (preserve raw). This is conservative — preserves micro-expression spikes
            # that affect multiple landmarks. If you want per-landmark preservation,
            # this can be relaxed to element-wise decision.
            if np.any(diffs > self.spike_thresh):
                # append current to buffer and return raw (no smoothing)
                self.buf.append(cur)
                out = []
                for p in normalized_frame:
                    out.append(p.copy())
                return out

        # Otherwise, perform simple moving average including current frame
        self.buf.append(cur)
        stack = np.stack(self.buf, axis=0)
        mean = stack.mean(axis=0)  # shape (n,3)

        smoothed = []
        for i, p in enumerate(normalized_frame):
            nx, ny, nz = mean[i].tolist()
            out = p.copy()
            out.update({"sx": float(nx), "sy": float(ny), "sz": float(nz)})
            smoothed.append(out)

        return smoothed
