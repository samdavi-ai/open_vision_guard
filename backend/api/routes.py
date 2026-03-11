from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class CameraConfig(BaseModel):
    id: str
    url: str
    name: str
    modules: List[str]

    # Reference to the global video manager and db
video_manager = None
streams_db = []

def init_routes(vm):
    global video_manager
    video_manager = vm

@router.get("/cameras", response_model=List[CameraConfig])
def get_cameras():
    return streams_db

@router.post("/cameras", response_model=CameraConfig)
def add_camera(config: CameraConfig):
    streams_db.append(config)
    if video_manager:
        video_manager.start_stream(config.id, config.url, config.modules)
    return config

@router.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str):
    global streams_db
    streams_db = [c for c in streams_db if c.id != camera_id]
    if video_manager:
        video_manager.stop_stream(camera_id)
    return {"status": "success"}

@router.get("/alerts")
def get_alerts(limit: int = 50):
    return []
