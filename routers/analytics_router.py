from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from modules import database

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/frequency/{person_id}")
async def get_frequency_analytics(person_id: str):
    """
    Retrieves visit frequency data and history for a person.
    """
    try:
        history = database.get_visit_history(person_id)
        return {
            "person_id": person_id,
            "visit_count": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/presence/{person_id}")
async def get_presence_logs(person_id: str, limit: int = 100):
    """
    Retrieves all presence logs (entry/exit) for a person.
    """
    try:
        return database.get_presence_logs(person_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/movement/{person_id}")
async def get_movement_logs(person_id: str, limit: int = 200):
    """
    Retrieves movement logs for trajectory analysis.
    """
    try:
        return {"movement_logs": database.get_movement_logs(person_id, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
