"""Simple CLI to run the webcam landmark streamer.

Usage: python -m backend.tools.webcam_landmarks
Press 'q' in the display window to quit.
"""
from backend.face_engine.streamer import WebcamLandmarkStreamer


def main():
    streamer = WebcamLandmarkStreamer(src=0, width=640, height=480, fps=30)
    # show_fps=True, print_json=False by default
    streamer.run_display(show_fps=True, print_json=False)


if __name__ == "__main__":
    main()
