from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import base64
import cv2
import numpy as np
from backend.face_engine.engine import FaceEngine
import json

websocket_router = APIRouter()


@websocket_router.get("/health")
async def get_health():
    # simple health endpoint (do not shadow frontend root "/")
    return HTMLResponse("<h1>Face Analysis Backend Running</h1>")


@websocket_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = FaceEngine()
    try:
        last_audio_intensity = None
        while True:
            message = await websocket.receive_text()
            # Expecting JSON messages with type and data
            # Client should send frames as data URLs. Example:
            # {"type": "frame", "data": "data:image/jpeg;base64,/..."}
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                continue

            # Support two message types: 'audio' (context only) and 'frame' (image)
            if payload.get("type") == "audio":
                # Expect payload: {"type":"audio","intensity": <0..1 float>}
                intensity = payload.get("intensity")
                try:
                    if intensity is not None:
                        last_audio_intensity = float(intensity)
                except Exception:
                    await websocket.send_text(json.dumps({"error": "invalid audio intensity"}))
                    continue
                await websocket.send_text(json.dumps({"type": "audio_ack", "intensity": last_audio_intensity}))
                continue

            if payload.get("type") != "frame" or "data" not in payload:
                await websocket.send_text(json.dumps({"error": "bad message format"}))
                continue

            b64 = payload["data"]
            # handle data URI prefix
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_text(json.dumps({"error": "image decode failed"}))
                continue

            # Process frame: FaceEngine returns a dict with keys 'face_detected',
            # 'landmarks', 'features', and 'micro_expressions'. Landmarks are
            # pixel coordinates matching the processed frame size
            # (backend resizes to width 640).
            result = engine.process_frame(img, audio_intensity=last_audio_intensity)

            # Send result back as JSON
            await websocket.send_text(json.dumps({
                "type": "analysis",
                "face": result["face_detected"],
                "features": result.get("features", {}),
                "landmarks": result.get("landmarks", {}),
                "micro_expressions": result.get("micro_expressions", {}),
                "audio_intensity": last_audio_intensity,
            }))

    except WebSocketDisconnect:
        # Clean up
        engine.close()
        return
