import cv2
import numpy as np
import io
from fastapi import APIRouter, UploadFile, File
from typing import List

from modules.embedding_engine import embedding_engine
from modules import database
from models.schemas import IdentityResponse, IdentityMetadata, AssignNameRequest

router = APIRouter(tags=["Identities"])


@router.get("/identities", response_model=List[IdentityResponse])
async def list_identities():
    """List all tracked identities."""
    identities = embedding_engine.get_all_identities()
    results = []
    for ident in identities:
        results.append(IdentityResponse(
            global_id=ident["global_id"],
            metadata=IdentityMetadata(**ident.get("metadata", {})),
            history=ident.get("history", [])
        ))
    return results


@router.get("/identities/{global_id}")
async def get_identity(global_id: str):
    """Full profile: metadata + history."""
    from fastapi import HTTPException
    ident = embedding_engine.get_identity(global_id)
    if not ident:
        raise HTTPException(status_code=404, detail="Identity not found")

    history = database.get_identity_history(global_id)
    return IdentityResponse(
        global_id=ident["global_id"],
        metadata=IdentityMetadata(**ident.get("metadata", {})),
        history=history
    )


@router.post("/identities/{global_id}/name")
async def assign_name(global_id: str, req: AssignNameRequest):
    """Manually assign a name to an identity."""
    embedding_engine.update_identity_metadata(global_id, {"face_name": req.name})
    return {"message": f"Name '{req.name}' assigned to {global_id}"}


@router.post("/identities/search/image")
async def search_by_image(file: UploadFile = File(...)):
    """Upload image → find similar person (embedding search)."""
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


@router.get("/identities/{global_id}/timeline")
async def get_timeline(global_id: str):
    """All events for this person across cameras."""
    history = database.get_identity_history(global_id)
    return {"global_id": global_id, "events": history}


@router.get("/identities/{global_id}/alerts")
async def get_person_alerts(global_id: str):
    """All alerts involving this person."""
    all_alerts = database.get_alerts()
    person_alerts = [a for a in all_alerts if a.get("global_id") == global_id]
    return {"global_id": global_id, "alerts": person_alerts}

