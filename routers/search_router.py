import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query
from typing import Optional, List

from modules.embedding_engine import embedding_engine
from modules import database
from models.schemas import ZoneConfig, ZonesUpdateRequest

router = APIRouter(tags=["Search & Investigation"])

# In-memory zone storage per camera
camera_zones: dict = {}


@router.post("/search/person")
async def search_person(file: UploadFile = File(...)):
    """Upload image → find matching global_id."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    embedding = embedding_engine.generate_embedding(img)
    matches = embedding_engine.search_similar(embedding, top_k=5)

    results = []
    for gid in matches:
        ident = embedding_engine.get_identity(gid)
        if ident:
            results.append({
                "global_id": gid,
                "metadata": ident.get("metadata", {})
            })

    return {"matches": results}


@router.get("/search/events")
async def search_events(
    camera_id: Optional[str] = None,
    activity: Optional[str] = None,
    risk_level: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100
):
    """Filter events by time, camera, activity, risk."""
    # Get all identities and filter
    identities = embedding_engine.get_all_identities()
    results = []

    for ident in identities:
        meta = ident.get("metadata", {})

        if camera_id and meta.get("last_seen_camera") != camera_id:
            continue
        if activity and meta.get("activity") != activity:
            continue
        if risk_level and meta.get("risk_level") != risk_level:
            continue

        results.append({
            "global_id": ident["global_id"],
            "metadata": meta,
            "history": ident.get("history", [])
        })

    return {"events": results[:limit]}


@router.get("/cameras/{camera_id}/zones")
async def get_zones(camera_id: str):
    """Get configured zones for a camera."""
    zones = camera_zones.get(camera_id, [])
    return {"camera_id": camera_id, "zones": zones}


@router.put("/cameras/{camera_id}/zones")
async def update_zones(camera_id: str, req: ZonesUpdateRequest):
    """Update restricted zones for a camera."""
    camera_zones[camera_id] = [z.dict() for z in req.zones]

    # Also update the pipeline's zone config
    from routers.stream_router import get_pipeline
    pipeline = get_pipeline()
    pipeline.camera_zones[camera_id] = camera_zones[camera_id]

    return {"message": f"Zones updated for {camera_id}", "zones": camera_zones[camera_id]}
