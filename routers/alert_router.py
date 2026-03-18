from fastapi import APIRouter
from typing import Optional
from collections import Counter

from modules.alert_engine import alert_engine
from modules import database
from models.schemas import AlertResponse, AlertStatsResponse

router = APIRouter(tags=["Alerts"])


@router.get("/alerts")
async def list_alerts(severity: Optional[str] = None, type: Optional[str] = None,
                      camera: Optional[str] = None, limit: int = 50):
    """List recent alerts with optional filters."""
    alerts = database.get_alerts()

    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    if type:
        alerts = [a for a in alerts if a.get("type") == type]
    if camera:
        alerts = [a for a in alerts if a.get("camera_id") == camera]

    return alerts[:limit]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Mark alert as handled."""
    alert_engine.acknowledge_alert(alert_id)
    return {"message": f"Alert {alert_id} acknowledged"}


@router.get("/alerts/stats", response_model=AlertStatsResponse)
async def alert_stats():
    """Alert count by type/severity."""
    alerts = database.get_alerts()

    by_type = dict(Counter(a.get("type", "unknown") for a in alerts))
    by_severity = dict(Counter(a.get("severity", "unknown") for a in alerts))

    return AlertStatsResponse(
        total=len(alerts),
        by_type=by_type,
        by_severity=by_severity
    )
