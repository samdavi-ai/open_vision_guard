from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# ── Stream ──
class StreamStartRequest(BaseModel):
    source: str  # RTSP URL, webcam index ("0"), or video file path
    camera_id: Optional[str] = None


class StreamInfo(BaseModel):
    camera_id: str
    source: str
    status: str  # "running" | "stopped"


# ── Identity ──
class IdentityMetadata(BaseModel):
    face_name: Optional[str] = None
    activity: Optional[str] = None
    risk_level: Optional[str] = "low"
    clothing_color: Optional[str] = None
    last_seen_camera: Optional[str] = None
    last_seen_time: Optional[str] = None
    # Enhanced fields
    movement_direction: Optional[str] = None   # "left", "right", "towards", "away", "stationary"
    speed: Optional[float] = 0.0               # px/sec movement speed
    pose_detail: Optional[str] = None          # "standing", "sitting", "crouching", "falling"
    entry_time: Optional[str] = None           # ISO timestamp of first detection
    exit_time: Optional[str] = None            # ISO timestamp of last detection (updated each frame)
    carried_objects: Optional[List[str]] = []   # ["backpack", "suitcase"]
    object_log: Optional[List[Dict[str, Any]]] = []  # [{object, action, timestamp}]
    zone_history: Optional[List[str]] = []     # camera zones visited
    latitude: Optional[float] = None           # GPS tracking
    longitude: Optional[float] = None          # GPS tracking
    total_appearances: Optional[int] = 0

    # Behaviour Analysis
    behaviour_label: Optional[str] = "normal"      # "normal", "pacing", "erratic", etc.
    behaviour_score: Optional[float] = 0.0          # 0-100 score

    # Risk Engine (Composite)
    risk_score: Optional[float] = 0.0              # composite 0-100 score
    risk_factors: Optional[List[str]] = []          # ["weapon_proximity", "loitering", ...]

    # Luggage Status
    luggage_status: Optional[Dict[str, Any]] = {}   # {luggage_id: {status, owner}}

    # Presence & Frequency
    dwell_time_seconds: Optional[float] = 0.0
    visit_count: Optional[int] = 0
    is_present: Optional[bool] = True
    frequency_label: Optional[str] = "new"          # "new", "rare", "occasional", "frequent", "regular"

    # Camera Avoidance
    avoidance_score: Optional[float] = 0.0
    avoidance_flags: Optional[List[str]] = []       # ["face_hidden", "edge_hugging", ...]



class IdentityResponse(BaseModel):
    global_id: str
    metadata: IdentityMetadata
    history: List[Dict[str, Any]] = []


class AssignNameRequest(BaseModel):
    name: str


# ── Alert ──
class AlertResponse(BaseModel):
    alert_id: str
    severity: str
    type: str
    message: str
    global_id: str
    camera_id: str
    timestamp: str
    thumbnail_path: Optional[str] = None
    acknowledged: bool = False


class AlertStatsResponse(BaseModel):
    total: int
    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}


# ── Search ──
class EventFilterParams(BaseModel):
    camera_id: Optional[str] = None
    activity: Optional[str] = None
    risk_level: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


# ── Zones ──
class ZoneConfig(BaseModel):
    name: str
    x1: int
    y1: int
    x2: int
    y2: int


class ZonesUpdateRequest(BaseModel):
    zones: List[ZoneConfig]


# ── Config ──
class ConfigUpdateRequest(BaseModel):
    similarity_threshold: Optional[float] = None
    face_recognition_enabled: Optional[bool] = None
    face_tolerance: Optional[float] = None
    min_face_height_px: Optional[int] = None
    pose_enabled: Optional[bool] = None
    fall_confidence_threshold: Optional[float] = None
    weapon_detection_enabled: Optional[bool] = None
    weapon_confidence_threshold: Optional[float] = None
    loitering_threshold_seconds: Optional[int] = None
    alert_dedup_window_seconds: Optional[int] = None
    frame_jpeg_quality: Optional[int] = None
    max_concurrent_cameras: Optional[int] = None


# ── Face Registration ──
class FaceRegisterResponse(BaseModel):
    name: str
    status: str
    message: str
