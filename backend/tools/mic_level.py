"""Small wrapper CLI that sends audio intensity samples to stdout as JSON lines.

Example:
  python -m backend.tools.mic_level

This prints JSON objects: {"timestamp": <float>, "intensity": <0..1>, "db": <dB>, "speaking": true/false}
"""
import time
import json
from backend.tools.audio_level import AudioLevelMonitor


def main():
    m = AudioLevelMonitor()
    try:
        m.start()
        while True:
            intensity, db, speaking = m.get_level()
            out = {"timestamp": time.time(), "intensity": intensity, "db": db, "speaking": speaking}
            print(json.dumps(out))
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        m.stop()


if __name__ == "__main__":
    main()
