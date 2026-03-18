import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form

from config import config
from models.schemas import ConfigUpdateRequest, FaceRegisterResponse

router = APIRouter(tags=["Config"])


@router.get("/config")
async def get_config():
    """Get current system configuration."""
    return {
        "similarity_threshold": config.similarity_threshold,
        "embedding_model": config.embedding_model,
        "face_recognition_enabled": config.face_recognition_enabled,
        "face_tolerance": config.face_tolerance,
        "min_face_height_px": config.min_face_height_px,
        "pose_enabled": config.pose_enabled,
        "fall_confidence_threshold": config.fall_confidence_threshold,
        "weapon_detection_enabled": config.weapon_detection_enabled,
        "weapon_confidence_threshold": config.weapon_confidence_threshold,
        "loitering_threshold_seconds": config.loitering_threshold_seconds,
        "alert_dedup_window_seconds": config.alert_dedup_window_seconds,
        "db_path": config.db_path,
        "thumbnails_dir": config.thumbnails_dir,
        "known_faces_dir": config.known_faces_dir,
        "frame_jpeg_quality": config.frame_jpeg_quality,
        "max_concurrent_cameras": config.max_concurrent_cameras,
    }


@router.put("/config")
async def update_config(req: ConfigUpdateRequest):
    """Update system thresholds and toggles."""
    updates = req.dict(exclude_none=True)
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return {"message": "Config updated", "updated_fields": list(updates.keys())}


@router.post("/faces/register", response_model=FaceRegisterResponse)
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    """Register a new known face (name + image upload)."""
    person_dir = os.path.join(config.known_faces_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    file_path = os.path.join(person_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return FaceRegisterResponse(
        name=name,
        status="registered",
        message=f"Face image saved to {file_path}. Restart or reload face module to activate."
    )
