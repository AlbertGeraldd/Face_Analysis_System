from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from backend.websocket_handler import websocket_router
from pathlib import Path

app = FastAPI(title="Face Analysis System - Backend")

# Include the WebSocket/router first so websocket scopes are handled by the router
# before the StaticFiles mount. If StaticFiles sees a websocket scope it will assert
# because it only handles HTTP scopes.
app.include_router(websocket_router)

# Mount the frontend static files so the UI is served from the same origin (port 8000).
# This avoids cross-origin wss/ws connection issues when accessing the Codespaces
# forwarded backend URL. The frontend files live one level above backend in /frontend.
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    # Run with: python -m backend.main
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
