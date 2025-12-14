"""
Micro-expression detector built on top of AU activations.

Definition used:
- Micro-expression: a sudden AU activation that appears briefly and disappears.
- Duration: between `min_duration_ms` and `max_duration_ms` (default 200-500 ms).

Implementation notes:
- Per-AU state machine tracks when an AU crosses an activation threshold.
- Onset time is recorded when activation rises above `spike_threshold`.
- If activation returns below `end_threshold` within the allowed duration window,
  an event is emitted with start time, duration, and peak intensity.
- If activation persists longer than `max_duration_ms` it's considered a
  non-micro (sustained) movement and is not emitted as a micro-expression.

Why duration matters (short comment):
- Duration is the critical criterion separating micro-expressions (brief,
  involuntary muscle activations) from normal expressions or speech-related
  movements. Using a 200–500 ms window helps filter out very short noise
  spikes (<200 ms) and sustained expressions (>500 ms).
"""
import time
from typing import Dict, List


class MicroExpressionDetector:
    def __init__(self, spike_threshold: float = 0.5, min_duration_ms: int = 200, max_duration_ms: int = 500, end_threshold: float = 0.3):
        """
        Args:
          spike_threshold: score above which an AU is considered to have onset
          min_duration_ms: minimum duration for a valid micro-expression
          max_duration_ms: maximum duration for a valid micro-expression
          end_threshold: score below which the AU is considered ended
        """
        self.spike_threshold = float(spike_threshold)
        self.min_dur = float(min_duration_ms) / 1000.0
        self.max_dur = float(max_duration_ms) / 1000.0
        self.end_threshold = float(end_threshold)

        # per-AU state: {au_name: {active:bool, start:float, peak:float, last_seen:float}}
        self.states: Dict[str, Dict] = {}

    def reset(self):
        self.states.clear()

    def update(self, timestamp: float, au_scores: Dict[str, Dict]) -> List[Dict]:
        """
        Feed AU scores for the current timestamp and return a list of detected
        micro-expression events (may be empty). `au_scores` expected format:
        {"AU12": {"score": 0.7, ...}, ...}

        Each returned event has fields: `au`, `start_time`, `duration`, `peak`.
        """
        events = []
        for au_name, info in au_scores.items():
            score = float(info.get("score", 0.0))
            st = self.states.setdefault(au_name, {"active": False, "start": None, "peak": 0.0, "last_seen": None})

            # Onset
            if not st["active"] and score >= self.spike_threshold:
                st["active"] = True
                st["start"] = timestamp
                st["peak"] = score
                st["last_seen"] = timestamp
                continue

            # If active, update peak and check for end
            if st["active"]:
                st["last_seen"] = timestamp
                if score > st["peak"]:
                    st["peak"] = score

                # End if falls below end_threshold
                if score < self.end_threshold:
                    start = st["start"] or timestamp
                    duration = timestamp - start
                    peak = st["peak"]
                    # Valid micro-expression if duration in allowed window
                    if self.min_dur <= duration <= self.max_dur:
                        events.append({"au": au_name, "start_time": start, "duration": duration, "peak": float(peak)})

                    # Reset state whether emitted or not
                    st["active"] = False
                    st["start"] = None
                    st["peak"] = 0.0
                else:
                    # If active and exceeds max duration -> treat as sustained, drop
                    if (timestamp - (st["start"] or timestamp)) > self.max_dur:
                        st["active"] = False
                        st["start"] = None
                        st["peak"] = 0.0

            else:
                # not active; nothing to do
                pass

        return events
