from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json

from core.video_manager import global_frame_store

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

@router.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket)
    try:
        # Loop to push frames down to the client
        while True:
            if camera_id in global_frame_store:
                data = global_frame_store[camera_id]
                await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.05) # ~20 FPS limit
    except WebSocketDisconnect:
        manager.disconnect(websocket)
