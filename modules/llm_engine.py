"""
LLM Engine — Groq API integration for OpenVisionGuard.

Features:
  1. narrate_alert()     — human-readable incident story from raw alert data
  2. query()             — natural language → DB query answer
  3. correlate_alerts()  — multi-alert pattern detection (coordinated behavior)
  4. person_profile()    — natural language summary of a specific person
  5. shift_report()      — end-of-shift incident summary
  6. reinit()            — hot-reload API key from environment

All calls are non-blocking (run in background thread) so they never
slow down the real-time inference pipeline.

Config: set GROQ_API_KEY in .env file.
"""

import os
import json
import threading
import datetime
from typing import Any, Dict, List, Optional
from collections import deque

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMEngine:
    """Groq-backed LLM engine. Gracefully disabled if API key is missing."""

    # Llama 4 Scout — multimodal, fast, context-aware (best available on free tier)
    MODEL_FAST  = "meta-llama/llama-4-scout-17b-16e-instruct"
    # Llama 3.3 70B — deep reasoning for complex correlation / reports
    MODEL_SMART = "llama-3.3-70b-versatile"

    def __init__(self) -> None:
        self.available = False
        self._client   = None
        self._lock     = threading.Lock()

        # In-memory narration cache: alert_id -> narrative string
        self._narration_cache: Dict[str, str] = {}
        # Rolling buffer of recent alerts for correlation (last 200)
        self._alert_buffer: deque = deque(maxlen=200)

        # Rate-limit backoff: when Groq returns 429, silence all calls for
        # _rate_backoff_s seconds (doubles on repeated 429s, max 60s).
        self._rate_limited_until: float = 0.0
        self._rate_backoff_s:     float = 5.0   # starts at 5s, doubles on repeat

        self._init()

    def _init(self) -> None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key or api_key in ("your_groq_api_key_here", ""):
            print("[LLMEngine] GROQ_API_KEY not set — LLM features disabled.")
            return
        try:
            from groq import Groq
            self._client  = Groq(api_key=api_key)
            # Validate key with a lightweight ping
            self._client.models.list()
            self.available = True
            print(f"[LLMEngine] Ready — fast={self.MODEL_FAST}  smart={self.MODEL_SMART}")
        except Exception as e:
            print(f"[LLMEngine] Init failed: {e}")
            self.available = False
            self._client   = None

    def reinit(self) -> bool:
        """Hot-reload the API key and reconnect. Returns True on success."""
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        self.available = False
        self._client   = None
        self._init()
        return self.available

    # ── Internal helper ───────────────────────────────────────────────────────

    def _chat(self, system: str, user: str, model: str = None, max_tokens: int = 512) -> str:
        """Send a chat completion request. Returns empty string on failure.
        Implements exponential backoff on Groq 429 rate-limit errors so the
        terminal isn't spammed and the pipeline doesn't stall."""
        if not self.available or self._client is None:
            return ""

        # Check rate-limit backoff window
        import time as _time
        now = _time.time()
        if now < self._rate_limited_until:
            return ""   # silently skip — still in backoff window

        _model = model or self.MODEL_FAST
        try:
            resp = self._client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            # Successful call — reset backoff
            self._rate_backoff_s = 5.0
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                # Exponential backoff: 5s → 10s → 20s → 40s → 60s (cap)
                self._rate_backoff_s = min(self._rate_backoff_s * 2, 60.0)
                self._rate_limited_until = _time.time() + self._rate_backoff_s
                print(f"[LLMEngine] Rate limited — backing off {self._rate_backoff_s:.0f}s")
            else:
                print(f"[LLMEngine] API error ({_model}): {err}")
                # Fallback: if smart model fails, retry with fast model
                if model == self.MODEL_SMART and self.MODEL_SMART != self.MODEL_FAST:
                    try:
                        resp = self._client.chat.completions.create(
                            model=self.MODEL_FAST,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user",   "content": user},
                            ],
                            max_tokens=max_tokens,
                            temperature=0.4,
                        )
                        self._rate_backoff_s = 5.0
                        return resp.choices[0].message.content.strip()
                    except Exception as e2:
                        print(f"[LLMEngine] Fallback also failed: {e2}")
            return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def narrate_alert(self, alert: Dict[str, Any]) -> str:
        """
        Generate a human-readable incident narrative from an alert dict.
        Result is cached by alert_id. Called async — safe from pipeline.
        """
        alert_id = str(alert.get("alert_id", ""))
        if alert_id and alert_id in self._narration_cache:
            return self._narration_cache[alert_id]

        # Feed this alert into the correlation buffer
        self._alert_buffer.append(alert)

        alert_type = alert.get("type", "unknown")
        person     = alert.get("global_id", "Unknown person")
        camera     = alert.get("camera_id", "Unknown camera")
        timestamp  = str(alert.get("timestamp", ""))
        message    = alert.get("message", "")
        removed    = alert.get("items_removed", []) or []
        added      = alert.get("items_added",   []) or []
        entry_time = alert.get("entry_time", "")
        exit_time  = alert.get("exit_time",  "")

        try:
            ts = datetime.datetime.fromisoformat(timestamp).strftime("%I:%M %p") if timestamp else ""
        except Exception:
            ts = timestamp
        try:
            et = datetime.datetime.fromisoformat(entry_time).strftime("%I:%M %p") if entry_time else ""
        except Exception:
            et = entry_time
        try:
            xt = datetime.datetime.fromisoformat(exit_time).strftime("%I:%M %p") if exit_time else ""
        except Exception:
            xt = exit_time

        removed_str = ", ".join(removed) if removed else "none"
        added_str   = ", ".join(added)   if added   else "none"

        system = (
            "You are a professional retail security analyst writing incident bulletins. "
            "Write exactly 2-3 plain-English sentences. Be specific: name the person ID, "
            "camera(s), items, and times. Do NOT use bullet points, JSON, or technical jargon. "
            "Write as if handing a printed note to a floor security guard."
        )

        # ── Per-alert-type instruction ───────────────────────────────────────
        if alert_type in ("baggage_taken", "item_taken"):
            instruction = (
                f"{person} entered the monitored area at {et or ts} on {camera} carrying: {removed_str}. "
                f"They were later observed at {xt or ts} WITHOUT those items. "
                f"Describe this as a potential theft/removal incident and state which items are missing."
            )
        elif alert_type in ("baggage_left_behind", "item_left_behind", "luggage_abandoned"):
            instruction = (
                f"{person} was seen on {camera} at {et or ts}. "
                f"They left the area at {xt or ts} leaving behind: {removed_str}. "
                f"Describe this as an abandoned-item incident and note the security concern."
            )
        elif alert_type in ("baggage_swap", "luggage_transferred"):
            instruction = (
                f"{person} on {camera} at {ts} was involved in a possible item exchange. "
                f"Items that disappeared: {removed_str}. Items that appeared: {added_str}. "
                f"Describe this as a suspicious item swap and flag it for review."
            )
        elif alert_type == "loitering":
            instruction = (
                f"{person} has been stationary or pacing on {camera} since {et or ts} (current time {ts}). "
                f"Describe this as a loitering alert and suggest what action a guard should take."
            )
        elif alert_type == "sudden_movement":
            instruction = (
                f"{person} on {camera} at {ts} exhibited a sudden change in movement — "
                f"either accelerating rapidly (possible panic/fleeing) or stopping abruptly. "
                f"Describe this as a sudden movement alert and advise the guard to visually confirm."
            )
        elif alert_type == "weapon":
            instruction = (
                f"URGENT: {person} on {camera} at {ts} was detected near a possible weapon or sharp object. "
                f"Write a high-priority security bulletin advising immediate guard response."
            )
        elif alert_type in ("fall", "falling"):
            instruction = (
                f"{person} on {camera} at {ts} appears to have fallen or collapsed. "
                f"Write a welfare alert advising staff to check on this individual immediately."
            )
        else:
            # Generic fallback
            instruction = (
                f"Alert type: {alert_type.replace('_', ' ').title()}. "
                f"Person: {person}. Camera: {camera}. Time: {ts}. "
                f"System message: {message}. "
                f"Items removed: {removed_str}. Items added: {added_str}. "
                f"Write a concise factual security incident narrative."
            )

        narrative = self._chat(system, instruction, max_tokens=220)
        if narrative and alert_id:
            self._narration_cache[alert_id] = narrative
        return narrative

    def narrate_alert_async(self, alert: Dict[str, Any], callback=None) -> None:
        """Non-blocking version — runs narration in background thread."""
        def _run():
            result = self.narrate_alert(alert)
            if callback:
                callback(alert.get("alert_id", ""), result)
        threading.Thread(target=_run, daemon=True).start()

    def query(self, question: str, recent_alerts: List[Dict], recent_persons: List[Dict]) -> str:
        """
        Answer a natural-language question about the surveillance data.
        Receives pre-fetched recent_alerts and recent_persons as context.
        """
        if not self.available:
            return "LLM not available. Please set GROQ_API_KEY in .env"

        system = (
            "You are an AI assistant for a retail security surveillance system. "
            "Answer questions about detected persons, alerts, and suspicious activity "
            "based ONLY on the provided context data. Be concise and factual. "
            "If the data doesn't contain the answer, say so clearly. "
            "Format times as 12-hour clock. Use plain English, no JSON in response."
        )

        # Summarize alerts compactly
        alerts_text = "\n".join([
            f"- [{a.get('timestamp','')}] {a.get('type','').upper()}: {a.get('global_id','')} at {a.get('camera_id','')} — {a.get('message','')}"
            for a in recent_alerts[:50]
        ])

        persons_text = "\n".join([
            f"- {p.get('global_id','')} | Risk: {p.get('risk_level','low')} | Camera: {p.get('last_seen_camera','')} | Activity: {p.get('activity','')}"
            for p in recent_persons[:30]
        ])

        user = f"""RECENT ALERTS (last 1 hour):
{alerts_text or 'No recent alerts'}

ACTIVE PERSONS:
{persons_text or 'No persons detected'}

QUESTION: {question}"""

        return self._chat(system, user, model=self.MODEL_SMART, max_tokens=400)

    def correlate_alerts(self, alerts: List[Dict]) -> str:
        """
        Analyze a batch of alerts to detect coordinated/suspicious patterns.
        Returns a narrative summary of patterns found.
        """
        if not self.available or len(alerts) < 2:
            return ""

        system = (
            "You are a senior security analyst. Analyze these surveillance alerts for patterns "
            "suggesting coordinated criminal activity, shoplifting teams, or suspicious behavior. "
            "Look for: multiple persons acting together, sequential alerts near the same location, "
            "similar behavior patterns across persons. "
            "If no significant pattern is found, say 'No coordinated activity detected.' "
            "Be specific: name the persons, times, and cameras involved."
        )

        alerts_text = "\n".join([
            f"[{a.get('timestamp','')}] {a.get('type','').upper()} | "
            f"Person: {a.get('global_id','')} | Camera: {a.get('camera_id','')} | "
            f"Items: removed={a.get('items_removed',[])} taken={a.get('items_added',[])} | "
            f"{a.get('message','')}"
            for a in alerts[-100:]  # last 100 alerts
        ])

        user = f"ALERTS TO ANALYZE:\n{alerts_text}"
        return self._chat(system, user, model=self.MODEL_SMART, max_tokens=500)

    def person_profile(self, person_id: str, identity: Dict, alerts: List[Dict], movements: List[Dict]) -> str:
        """
        Generate a narrative intelligence profile for a specific person.
        """
        if not self.available:
            return ""

        system = (
            "You are a security intelligence analyst. Write a 3-5 sentence behavioral profile "
            "of this surveillance subject based on their detected activity. "
            "Include: when they arrived, what they did, any suspicious behavior, current risk level. "
            "Write in third person, professional tone."
        )

        meta = identity.get("metadata", identity)
        activity_log = "\n".join([
            f"- {m.get('timestamp','')} | Speed: {float(m.get('speed') or 0):.1f}px/s | Activity: {m.get('event_type','')}"
            for m in movements[:20]
        ])
        alerts_text = "\n".join([
            f"- {a.get('type','').upper()} at {a.get('timestamp','')}: {a.get('message','')}"
            for a in alerts[:10]
        ])

        dwell = meta.get("dwell_time_seconds", 0)
        try:
            dwell = float(dwell or 0)
        except (TypeError, ValueError):
            dwell = 0.0

        user = f"""
Person ID: {person_id}
Face Name: {meta.get('face_name') or 'Unknown'}
Risk Level: {meta.get('risk_level', 'low')}
Activity: {meta.get('activity', 'unknown')}
Entry Time: {meta.get('entry_time', 'unknown')}
Dwell Time: {dwell:.0f} seconds
Carried Objects: {', '.join(meta.get('carried_objects', [])) or 'none'}
Camera: {meta.get('last_seen_camera', 'unknown')}

MOVEMENT LOG:
{activity_log or 'No movement data'}

ALERTS TRIGGERED:
{alerts_text or 'No alerts'}
"""
        return self._chat(system, user, model=self.MODEL_FAST, max_tokens=300)

    def shift_report(self, summary: Dict) -> str:
        """
        Generate a natural language end-of-shift report from a pre-aggregated summary.
        """
        if not self.available:
            return ""

        system = (
            "You are a security operations manager. Write a professional end-of-shift security report "
            "based on the surveillance data provided. Structure: Overview, Key Incidents, Persons of Interest, "
            "Recommendations. Keep it under 400 words. Professional tone."
        )

        user = f"""
Shift Period: {summary.get('start_time', '')} to {summary.get('end_time', '')}
Total Persons Detected: {summary.get('total_persons', 0)}
Total Alerts: {summary.get('total_alerts', 0)}
High/Critical Alerts: {summary.get('critical_count', 0)}

Alert Breakdown:
{json.dumps(summary.get('alert_types', {}), indent=2)}

Key Incidents:
{chr(10).join(['- ' + i for i in summary.get('incidents', [])])}

Unresolved Items:
{chr(10).join(['- ' + i for i in summary.get('unresolved', [])])}

Peak Activity: {summary.get('peak_time', 'N/A')} ({summary.get('peak_count', 0)} persons)
"""
        return self._chat(system, user, model=self.MODEL_SMART, max_tokens=600)

    def get_cached_narration(self, alert_id: str) -> Optional[str]:
        return self._narration_cache.get(str(alert_id))

    def get_recent_buffer(self) -> List[Dict]:
        return list(self._alert_buffer)


# Singleton
llm_engine = LLMEngine()
