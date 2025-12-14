"""
Simple event logger for research experiments.

Appends JSON lines to `backend/logs/events.jsonl`. Each line is a JSON
object containing at least `timestamp` and `type` fields. This logger is
intended for optional experiment logging and does not impact real-time
processing when disabled.
"""
import os
import json
from typing import Dict

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'events.jsonl')


def _ensure_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass


def log_event(event: Dict):
    """Append event (dict) as JSON line to log file."""
    _ensure_dir()
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception:
        # Logging failure should not crash the pipeline
        pass
