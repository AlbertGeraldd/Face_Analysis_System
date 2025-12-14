import asyncio
import base64
import json
from io import BytesIO
import websockets
import cv2
import numpy as np


async def send_test_frame():
    uri = "ws://localhost:8000/ws"
    # Create a simple blank image (black) matching backend expected size
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Encode to JPEG
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        print('Failed to encode image')
        return
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    data_url = f'data:image/jpeg;base64,{b64}'

    async with websockets.connect(uri) as ws:
        msg = json.dumps({"type": "frame", "data": data_url})
        await ws.send(msg)
        resp = await ws.recv()
        print('Server response:', resp)


if __name__ == '__main__':
    asyncio.run(send_test_frame())
