"""LLM Router — FastAPI endpoints for all LLM-powered features."""

import datetime
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from modules.llm_engine import llm_engine
from modules import database

router = APIRouter(prefix="/llm", tags=["LLM"])


class QueryRequest(BaseModel):
    question: str


class CorrelateRequest(BaseModel):
    hours: Optional[int] = 1   # look back N hours


class AlertBody(BaseModel):
    """Optional alert payload sent from the frontend to avoid a DB round-trip."""
    alert_id:    Optional[str] = None
    type:        Optional[str] = None
    global_id:   Optional[str] = None
    camera_id:   Optional[str] = None
    message:     Optional[str] = None
    severity:    Optional[str] = None
    timestamp:   Optional[str] = None
    items_added:   Optional[list] = None
    items_removed: Optional[list] = None
    entry_time:  Optional[str] = None
    exit_time:   Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_get_alerts(limit: int = 100):
    """Return alerts or [] when the database is offline."""
    try:
        if database.db_pool is None:
            return []
        return [dict(a) for a in database.get_alerts()[:limit]]
    except Exception as e:
        print(f"[LLMRouter] DB error: {e}")
        return []


def _filter_by_hours(alerts, hours: int):
    """Filter a list of alert dicts to only those within the last N hours."""
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours)
    filtered = []
    for a in alerts:
        try:
            ts = a.get("timestamp")
            if ts:
                if isinstance(ts, str):
                    alert_dt = datetime.datetime.fromisoformat(ts)
                else:
                    alert_dt = ts
                if alert_dt.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
                    filtered.append(a)
        except Exception:
            filtered.append(a)   # include on parse error
    return filtered


# ── Status ──────────────────────────────────────────────────────────────────

@router.get("/status")
async def llm_status():
    """Check if LLM is available."""
    return {
        "available":          llm_engine.available,
        "model_fast":         llm_engine.MODEL_FAST,
        "model_smart":        llm_engine.MODEL_SMART,
        "cached_narrations":  len(llm_engine._narration_cache),
        "db_online":          database.db_pool is not None,
    }


@router.post("/reinit")
async def reinit_llm():
    """Hot-reload GROQ_API_KEY from .env and reconnect."""
    ok = llm_engine.reinit()
    return {
        "success":   ok,
        "available": llm_engine.available,
        "model_fast":  llm_engine.MODEL_FAST,
        "model_smart": llm_engine.MODEL_SMART,
    }


# ── Alert narration ──────────────────────────────────────────────────────────

@router.post("/narrate/{alert_id}")
async def narrate_alert(alert_id: str, body: Optional[AlertBody] = None):
    """
    Generate an AI narrative for an alert.
    Accepts an optional JSON body containing the alert so we can narrate it
    even before it is persisted to the database (eliminates the race condition
    where a brand-new alert isn't in the DB yet when the modal opens).
    """
    # Check cache first
    cached = llm_engine.get_cached_narration(alert_id)
    if cached:
        return {"alert_id": alert_id, "narrative": cached, "cached": True}

    if not llm_engine.available:
        return {"alert_id": alert_id, "narrative": "LLM not available. Add GROQ_API_KEY to .env file.", "cached": False}

    # Use the body sent from the frontend if available (no DB round-trip needed)
    if body and body.alert_id:
        alert = body.dict(exclude_none=True)
    else:
        # Fallback: look up from DB
        alerts = _safe_get_alerts(limit=500)
        alert  = next((a for a in alerts if str(a.get("alert_id")) == alert_id), None)
        if not alert:
            return {"alert_id": alert_id, "narrative": None, "cached": False}

    try:
        narrative = llm_engine.narrate_alert(alert)
        return {"alert_id": alert_id, "narrative": narrative or None, "cached": False}
    except Exception as e:
        return {"alert_id": alert_id, "narrative": None, "error": str(e), "cached": False}


# ── Natural Language Query ────────────────────────────────────────────────────

@router.post("/query")
async def nl_query(req: QueryRequest):
    """
    Answer a natural language question about surveillance data.
    Example: 'Who entered with a bag and left without one today?'
    """
    if not llm_engine.available:
        return {"answer": "LLM not available. Add GROQ_API_KEY to .env file."}

    try:
        recent_alerts = _safe_get_alerts(limit=60)

        # Pull in-memory person metadata (no DB call needed)
        try:
            from modules.embedding_engine import embedding_engine
            recent_persons = [
                {**m, "global_id": gid}
                for gid, m in list(embedding_engine.global_to_metadata.items())[:40]
            ]
        except Exception:
            recent_persons = []

        answer = llm_engine.query(req.question, recent_alerts, recent_persons)
        return {"question": req.question, "answer": answer or "I couldn't find relevant data."}
    except Exception as e:
        return {"question": req.question, "answer": f"Query failed: {e}"}


# ── Multi-alert correlation ───────────────────────────────────────────────────

@router.post("/correlate")
async def correlate(req: CorrelateRequest):
    """
    Analyze recent alerts for coordinated suspicious activity patterns.
    """
    if not llm_engine.available:
        return {"analysis": "LLM not available."}

    try:
        all_alerts = _safe_get_alerts(limit=100)
        filtered   = _filter_by_hours(all_alerts, req.hours)

        analysis = llm_engine.correlate_alerts(filtered or all_alerts[:50])
        return {
            "hours_analyzed":   req.hours,
            "alerts_analyzed":  len(filtered),
            "analysis":         analysis or "No patterns detected.",
        }
    except Exception as e:
        return {"analysis": f"Correlation failed: {e}"}


# ── Person profile ────────────────────────────────────────────────────────────

@router.get("/profile/{person_id}")
async def person_profile(person_id: str):
    """
    Generate an LLM narrative intelligence profile for a specific person.
    """
    if not llm_engine.available:
        return {"profile": "LLM not available."}

    try:
        from modules.embedding_engine import embedding_engine
        identity  = embedding_engine.get_identity(person_id) or {}

        # Alerts for this person
        person_alerts = [
            a for a in _safe_get_alerts(limit=500)
            if a.get("global_id") == person_id
        ][:10]

        # Movement logs (graceful fallback)
        try:
            if database.db_pool is not None:
                movements = [dict(m) for m in database.get_movement_logs(person_id, limit=20)]
            else:
                movements = []
        except Exception:
            movements = []

        profile = llm_engine.person_profile(person_id, identity, person_alerts, movements)
        return {"person_id": person_id, "profile": profile or "Insufficient data to generate profile."}
    except Exception as e:
        return {"person_id": person_id, "profile": f"Profile generation failed: {e}"}


# ── Shift report ──────────────────────────────────────────────────────────────

@router.get("/shift-report")
async def shift_report(hours: int = 8):
    """
    Generate an end-of-shift security report for the last N hours.
    """
    if not llm_engine.available:
        return {"report": "LLM not available. Add GROQ_API_KEY to .env file."}

    try:
        all_alerts = _safe_get_alerts(limit=500)
        filtered   = _filter_by_hours(all_alerts, hours)
        now        = datetime.datetime.now()

        alert_types: dict = {}
        critical_count = 0
        for a in filtered:
            t = a.get("type", "unknown")
            alert_types[t] = alert_types.get(t, 0) + 1
            if a.get("severity") in ("critical", "high"):
                critical_count += 1

        incidents = [
            f"{a.get('type','').upper()} — {a.get('global_id','')} at {a.get('camera_id','')} "
            f"({str(a.get('timestamp',''))[:16]}): {a.get('message','')}"
            for a in filtered
            if a.get("severity") in ("critical", "high")
        ][:15]

        summary = {
            "start_time":    (now - datetime.timedelta(hours=hours)).strftime("%I:%M %p"),
            "end_time":      now.strftime("%I:%M %p"),
            "total_alerts":  len(filtered),
            "critical_count": critical_count,
            "alert_types":   alert_types,
            "incidents":     incidents,
            "unresolved": [
                f"{a.get('type','').upper()} at {a.get('camera_id','')} — {a.get('message','')}"
                for a in filtered if not a.get("acknowledged", False)
            ][:5],
            "total_persons": 0,
        }

        report = llm_engine.shift_report(summary)
        return {"hours": hours, "alerts_analyzed": len(filtered), "report": report}
    except Exception as e:
        return {"report": f"Report generation failed: {e}"}
