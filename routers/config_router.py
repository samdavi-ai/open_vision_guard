import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form

from config import config
from models.schemas import ConfigUpdateRequest, FaceRegisterResponse

router = APIRouter(tags=["Config"])


@router.get("/config")
async def get_config():
    """Get current system configuration — all toggles and thresholds."""
    return {
        # ── Detection Features ──
        "multiscale_enabled":           config.multiscale_enabled,
        "small_object_reinference":     config.small_object_reinference,
        "redetection_on_loss_enabled":  config.redetection_on_loss_enabled,
        "zero_motion_gate_enabled":     getattr(config, "zero_motion_gate_enabled", False),
        # ── Alert Features ──
        "weapon_detection_enabled":     config.weapon_detection_enabled,
        "pose_enabled":                 config.pose_enabled,
        "face_recognition_enabled":     config.face_recognition_enabled,
        # ── Baggage / Session ──
        "session_exit_timeout_s":       getattr(config, "session_exit_timeout_s", 20.0),
        # ── Thresholds ──
        "person_conf_threshold":        config.person_conf_threshold,
        "object_conf_threshold":        config.object_conf_threshold,
        "luggage_conf_threshold":       config.luggage_conf_threshold,
        "similarity_threshold":         config.similarity_threshold,
        "loitering_threshold_seconds":  config.loitering_threshold_seconds,
        "alert_dedup_window_seconds":   config.alert_dedup_window_seconds,
        # ── Display / Performance ──
        "temporal_hold_frames":         config.temporal_hold_frames,
        "frame_jpeg_quality":           config.frame_jpeg_quality,
        "max_concurrent_cameras":       config.max_concurrent_cameras,
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
