"""
Global processing configuration for Face Analysis pipeline.

This module exposes a simple dict and helper functions so the frontend can
enable/disable processing stages at runtime via an HTTP call. The flags are
kept intentionally simple for research experimentation.
"""
from typing import Dict

# Default configuration
_config: Dict[str, bool] = {
    "enable_aus": True,
    "enable_au_micro": True,
    "enable_contextual": True,
    "enable_smoothing": True,
    "enable_facs": True,
}


def get_config() -> Dict[str, bool]:
    return dict(_config)


def update_config(updates: Dict[str, bool]):
    for k, v in updates.items():
        if k in _config:
            _config[k] = bool(v)
