"""
Contextual interpretation layer (rule-based, explainable).

This module produces neutral, research-oriented contextual events by
combining Action Unit activations, AU micro-expression events, and audio
intensity. Rules are intentionally simple and fully explainable; no
emotion labels (e.g. "anger", "lie") are produced — only neutral
indicators such as "stress_indicator" or "engagement_indicator".

Each event contains:
  - `type`: neutral indicator name
  - `timestamp`: event time (seconds since epoch)
  - `score`: 0..1 confidence-like numeric value (from rule combining inputs)
  - `components`: inputs that contributed to the event (AU scores, audio)

Design principles and safety:
- Rule-based only (no ML). Each rule documents which inputs it uses and
  how the score is derived.
- Audio is used only as contextual intensity/speaking indicator. This code
  does NOT analyze speech content or language.
"""
from typing import Dict, List, Optional
import time


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def detect_context(action_units: Dict[str, Dict], au_micro_events: List[Dict], audio_intensity: Optional[float] = None, speaking: Optional[bool] = None, timestamp: Optional[float] = None) -> List[Dict]:
    """
    Rule-based contextual interpretation.

    Args:
      action_units: dict of AU activations, e.g. {"AU4": {"score":0.7, ...}, ...}
      au_micro_events: list of micro-expression events (each with at least 'au','start_time','duration','peak')
      audio_intensity: optional float 0..1 representing audio loudness
      speaking: optional bool if known; if None, derived from audio_intensity
      timestamp: optional timestamp to attach to events (defaults to now)

    Returns list of events (each event is a dict with `type`,`timestamp`,`score`,`components`).
    """
    ts = float(timestamp or time.time())
    events: List[Dict] = []

    # Derive speaking flag if not provided
    if speaking is None and audio_intensity is not None:
        speaking = bool(audio_intensity >= 0.25)

    # Rule 1: Stress indicator
    # Rationale: sustained or strong brow lowering (AU4) combined with
    # elevated audio intensity (speaking loudly or agitated voice) can be a
    # contextual indicator of high arousal or stress. We do NOT label an
    # emotion; we only emit a neutral `stress_indicator` with a combined score.
    au4 = action_units.get("AU4")
    if au4 is not None and audio_intensity is not None:
        au4_score = float(au4.get("score", 0.0))
        # combine multiplicatively then clamp; this emphasizes when both signals high
        score = _clamp01(au4_score * float(audio_intensity))
        if score >= 0.3:
            events.append({
                "type": "stress_indicator",
                "timestamp": ts,
                "score": score,
                "components": {"AU4": au4_score, "audio_intensity": float(audio_intensity), "rule": "AU4 * audio_intensity"},
            })

    # Rule 2: Suppressed smile
    # Rationale: a brief AU12 spike (micro-expression smile) occurring during
    # low audio intensity (silence) may indicate a suppressed or private smile.
    # We output `suppressed_smile` when a AU12 micro-event is found while audio
    # intensity is low.
    for me in au_micro_events:
        if me.get("au") == "AU12":
            peak = float(me.get("peak", 0.0))
            # silence or very low audio suggests private/suppressed
            if audio_intensity is None or float(audio_intensity) < 0.2:
                score = _clamp01(peak)
                events.append({
                    "type": "suppressed_smile",
                    "timestamp": float(me.get("start_time", ts)),
                    "score": score,
                    "components": {"micro_event": me, "audio_intensity": audio_intensity, "rule": "AU12 micro during silence"},
                })

    # Rule 3: Engagement indicator
    # Rationale: cheek raise (AU6) together with speaking may indicate vocalized
    # engagement (e.g., laughter, active speaking with expressive face). We
    # produce a neutral `engagement_indicator` when AU6 is elevated while
    # audio intensity is moderate/high.
    au6 = action_units.get("AU6")
    if au6 is not None and audio_intensity is not None:
        au6_score = float(au6.get("score", 0.0))
        score = _clamp01((au6_score + float(audio_intensity)) / 2.0)
        if score >= 0.35:
            events.append({
                "type": "engagement_indicator",
                "timestamp": ts,
                "score": score,
                "components": {"AU6": au6_score, "audio_intensity": float(audio_intensity), "rule": "avg(AU6, audio)"},
            })

    return events
