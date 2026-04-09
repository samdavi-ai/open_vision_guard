from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from modules import database

router = APIRouter(prefix="/face-logs", tags=["Face Logs"])

@router.get("/")
async def get_face_logs(limit: int = 100):
    """
    Retrieves all face recognition events.
    """
    try:
        return database.get_face_logs(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{person_id}")
async def get_face_logs_by_person(person_id: str):
    """
    Retrieves face recognition events for a specific person.
    """
    try:
        return database.get_face_logs_by_person(person_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
