import os

class OpenVisionConfig:
    # Embedding
    similarity_threshold: float = 0.82
    embedding_model: str = "mobilenet_v2"  # or "resnet50"
    
    # Face Recognition
    face_recognition_enabled: bool = True
    face_tolerance: float = 0.5
    min_face_height_px: int = 80
    
    # Pose
    pose_enabled: bool = True
    fall_confidence_threshold: float = 0.6
    
    # Weapon Detection
    weapon_detection_enabled: bool = True
    weapon_confidence_threshold: float = 0.5
    
    # Motion / Loitering
    loitering_threshold_seconds: int = 30
    
    # Alerts
    alert_dedup_window_seconds: int = 60
    
    # Storage
    db_path: str = "data/openvisionguard.db"
    thumbnails_dir: str = "data/thumbnails"
    known_faces_dir: str = "data/known_faces"
    
    # Streaming
    frame_jpeg_quality: int = 75  # WebSocket stream compression
    max_concurrent_cameras: int = 8

# Global config instance
config = OpenVisionConfig()
