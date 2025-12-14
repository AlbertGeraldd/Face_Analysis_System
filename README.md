# Face Analysis System

Real-time facial interaction analysis core for a smart-glasses prototype. This repository provides a production-ready prototype with a Python FastAPI backend and a plain HTML/JS frontend that streams camera frames via WebSocket and extracts facial features using MediaPipe FaceMesh.

**Highlights**
- FastAPI backend with WebSocket endpoint for low-latency streaming
- MediaPipe FaceMesh facial landmarks for robust detection
- Modular `face_engine` for detection, landmarks, and feature extraction
- Frontend using getUserMedia and WebSocket, with FPS throttling

## How it works (overview)
- Frontend captures frames from the user's webcam and sends compressed JPEG frames via WebSocket to the backend.
- Backend decodes frames, runs MediaPipe FaceMesh to extract landmarks.
- Landmarks are processed to compute features: mouth openness, eye openness, and eyebrow intensity.
- Backend responds with a small JSON object containing face status, features, and selected landmark coordinates for visualization.

## Folder Structure
- `/backend` — Python backend
  - `main.py` — FastAPI application entry
  - `websocket_handler.py` — WebSocket message handling
  - `face_engine/` — Consolidated face processing modules
    - `detector.py` — FaceMesh detector wrapper
    - `landmarks.py` — Converts landmarks to pixel coordinates and selects key points
    - `features.py` — Executes feature computations derived from landmarks
  - `requirements.txt` — Python dependencies
- `/frontend` — Minimal web UI
  - `index.html`, `script.js`, `style.css` — Frontend for live webcam, overlay, and controls

## Quick Start (Codespaces / Local)
1. Create and activate a Python virtual environment.
2. Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```
3. Run the backend:
```bash
cd backend
python -m backend.main
# or run: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
4. Serve the frontend static files. Easiest is to open `frontend/index.html` in the browser and allow camera permissions, or run a lightweight static server (example below):
```bash
python -m http.server 5500 --directory frontend
# open http://localhost:5500 in the browser
```
5. Start the UI, click `Start`, and views will be sent at the requested FPS to the backend, with analysis displayed on screen.

## WebSocket Message Protocol
- Client -> Server: JSON object with `type: "frame"` and `data` set to a DataURL string: `{"type":"frame","data":"data:image/jpeg;base64,..."}`.
- Server -> Client: JSON object with `type: "analysis"` and fields: `face` (bool), `features` (object), `landmarks` (object)

## Important Implementation Notes
- MediaPipe FaceMesh is used for landmark detection. We select a subset of landmarks to compute primary features required for micro-expression analysis.
- The frontend throttles at 8–12 fps by default to reduce bandwidth and processing load.
- The `face_engine` is intentionally modular to allow future additions such as smoothing, tracking, or deep-learning-based classification.

## Extending the System
- Add emotion or expression classification on top of `features.py` (e.g., a lightweight model that consumes time-series features).
- Add client-side smoothing and visualization for smoother overlays and historical trends.
- Add an API for replaying saved streams and computing offline analytics.

## Troubleshooting
- If MediaPipe fails to install on your platform, ensure your environment supports it and consider using a Debian-based image that provides the necessary build dependencies; alternatively, use a native pip wheel that matches your OS and Python.
- If the video overlay landmarks do not align exactly, check the video capture resolution and ensure the server scales frames consistently (backend resizes to max width 640 for predictability).

## License
This project is scaffolded for prototyping and demos. Update license to match your organization's policies.
# Face_Analysis_System