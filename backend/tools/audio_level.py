"""
Microphone audio intensity monitor.

This module captures microphone input and computes a real-time audio intensity
measure (RMS and dBFS). It only measures loudness as a contextual signal and
does NOT perform any speech recognition or content analysis. Use audio only
to provide contextual cues for facial/micro-expression analysis.

API:
  monitor = AudioLevelMonitor()
  monitor.start()
  intensity, db, speaking = monitor.get_level()
  monitor.stop()

Implementation notes:
- Uses `sounddevice` InputStream with a short block size for low-latency
- Computes RMS per block, converts to decibels (dBFS), and maps to a 0..1
  intensity value using a configurable floor (e.g. -60 dB => 0)
"""
from __future__ import annotations
import threading
import time
import math
import numpy as np
import sounddevice as sd
from typing import Optional, Tuple


class AudioLevelMonitor:
    def __init__(self, samplerate: int = 16000, block_duration: float = 0.05, min_db: float = -60.0, speak_db_threshold: float = -40.0):
        """
        Args:
          samplerate: audio sampling rate in Hz
          block_duration: block size in seconds for RMS calculation (short -> low latency)
          min_db: decibel value mapped to intensity 0 (e.g. -60 dB)
          speak_db_threshold: dB threshold above which speaking=True
        """
        self.sr = int(samplerate)
        self.block_duration = float(block_duration)
        self.blocksize = max(256, int(self.sr * self.block_duration))
        self.min_db = float(min_db)
        self.speak_db_threshold = float(speak_db_threshold)

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()
        self._last_rms = 0.0
        self._last_db = self.min_db
        self._last_intensity = 0.0
        self._last_speaking = False

    def _rms_to_db(self, rms: float) -> float:
        # dBFS: 20 * log10(rms); clamp to min_db floor
        db = 20.0 * math.log10(rms + 1e-12)
        return max(self.min_db, db)

    def _db_to_intensity(self, db: float) -> float:
        # Map [min_db .. 0] -> [0 .. 1]
        v = (db - self.min_db) / (0.0 - self.min_db)
        return max(0.0, min(1.0, v))

    def _callback(self, indata, frames, time_info, status):
        # indata is shape (frames, channels). Convert to mono by averaging.
        if status:
            # For research/debugging we keep status visible in logs if needed
            pass
        audio = np.asarray(indata)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # compute RMS
        rms = float(np.sqrt(np.mean(np.square(audio))))
        db = self._rms_to_db(rms)
        intensity = self._db_to_intensity(db)
        speaking = db >= self.speak_db_threshold

        with self._lock:
            self._last_rms = rms
            self._last_db = db
            self._last_intensity = intensity
            self._last_speaking = bool(speaking)

    def start(self):
        if self._running:
            return
        self._stream = sd.InputStream(channels=1, samplerate=self.sr, blocksize=self.blocksize, callback=self._callback)
        self._stream.start()
        self._running = True

    def stop(self):
        if not self._running:
            return
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            self._running = False

    def get_level(self) -> Tuple[float, float, bool]:
        """Return (intensity, db, speaking)

        - intensity: 0..1 mapped from dB (min_db..0)
        - db: decibel value (dBFS)
        - speaking: bool
        """
        with self._lock:
            return float(self._last_intensity), float(self._last_db), bool(self._last_speaking)


if __name__ == "__main__":
    # simple CLI for testing
    print("Starting AudioLevelMonitor (press Ctrl-C to stop). This is NOT speech analysis.")
    m = AudioLevelMonitor()
    try:
        m.start()
        while True:
            intensity, db, speaking = m.get_level()
            print(f"intensity={intensity:.3f} db={db:.1f} speaking={speaking}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        m.stop()
