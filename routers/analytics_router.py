"""
Analytics Router
API endpoints for frequency analysis, presence data, and system-wide analytics.
"""
from fastapi import APIRouter, Query
from typing import Optional
from modules import database
from modules.frequency_analyzer import frequency_analyzer
from modules.presence_tracker import presence_tracker

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/frequency/{person_id}")
async def get_frequency_data(person_id: str):
    """Get visit frequency data for a specific person."""
    data = frequency_analyzer.get_frequency_data(person_id)
    return data


@router.get("/overview")
async def get_analytics_overview():
    """Get system-wide frequency and statistics overview."""
    overview = frequency_analyzer.get_overview()
    return overview


@router.get("/presence/{person_id}")
async def get_presence_data(person_id: str):
    """Get presence/dwell data for a specific person."""
    data = presence_tracker.get_presence_data(person_id)
    if not data:
        return {"person_id": person_id, "message": "No presence data available"}
    return data


@router.get("/presence-logs")
async def get_presence_logs(
    person_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get presence event logs (entry/exit events)."""
    logs = database.get_presence_logs(person_id=person_id, limit=limit)
    return {"presence_logs": logs, "count": len(logs)}


@router.get("/movement/{person_id}")
async def get_movement_analytics(person_id: str, limit: int = Query(200, ge=1, le=1000)):
    """Get time-series movement trajectory metrics for dynamic UI charting."""
    logs = database.get_movement_logs(person_id, limit=limit)
    if not logs:
        return {"person_id": person_id, "movement_logs": []}
    return {"person_id": person_id, "movement_logs": logs}
