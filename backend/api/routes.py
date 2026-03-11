from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import os
import shutil
import time

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

@router.post("/cameras/upload", response_model=CameraConfig)
async def upload_camera(
    file: UploadFile = File(...),
    name: str = Form(...),
    modules: str = Form(...) 
):
    """
    Accepts a video file, saves it, and starts a stream using that file path.
    Modules should be a comma-separated string, e.g., 'object_detector,pose'
    """
    os.makedirs("uploads", exist_ok=True)
    
    # Generate unique ID and save path
    cam_id = f"cam_upload_{int(time.time())}"
    file_ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{cam_id}{file_ext}"
    file_path = os.path.abspath(os.path.join("uploads", safe_filename))
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    module_list = [m.strip() for m in modules.split(",") if m.strip()]
    
    config = CameraConfig(
        id=cam_id,
        url=file_path,
        name=name,
        modules=module_list
    )
    
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
