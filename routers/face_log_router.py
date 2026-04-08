"""
Face Log Router
API endpoints for querying face recognition event history.
"""
from fastapi import APIRouter, Query
from typing import Optional
from modules import database

router = APIRouter(prefix="/face-logs", tags=["Face Logs"])


@router.get("")
async def get_all_face_logs(limit: int = Query(100, ge=1, le=1000)):
    """Get all face recognition events, most recent first."""
    logs = database.get_face_logs(limit=limit)
    return {"face_logs": logs, "count": len(logs)}


@router.get("/{person_id}")
async def get_face_logs_by_person(person_id: str):
    """Get face recognition events for a specific person."""
    logs = database.get_face_logs_by_person(person_id)
    return {"person_id": person_id, "face_logs": logs, "count": len(logs)}
