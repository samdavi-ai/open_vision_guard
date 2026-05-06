"""
Session Tracker — entry/exit visit tracking with baggage-change detection.

Each "session" is one visit:  entry_time → exit_time  with a snapshot of
items carried at entry and at exit.  When the same person (matched via
OSNet-AIN Re-ID) re-enters, a new session is linked to the same global_id.

The key methods are:
  • on_person_seen(global_id, carried_items, camera_id, now)
      Call every frame a person is visible. Internally decides whether this
      is an entry, an update, or a re-entry after absence.

  • check_exits(active_global_ids, camera_id, now) → List[BaggageEvent]
      Call every frame with the current set of visible persons. Returns
      baggage-change alerts for identities that have just left the scene.

Baggage events emitted:
  • "baggage_left_behind"  — entered WITH items, exited WITHOUT
  • "baggage_taken"        — entered WITHOUT items, exited WITH items
  • "baggage_swap"         — exited with a DIFFERENT set of items
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from config import config


@dataclass
class VisitSession:
    global_id: str
    session_index: int                          # 1 = first visit, 2 = second, etc.
    camera_id: str
    entry_time: float
    exit_time: Optional[float] = None
    items_at_entry: Set[str] = field(default_factory=set)
    items_at_exit: Set[str] = field(default_factory=set)
    last_seen: float = 0.0
    alert_fired: bool = False                   # prevent duplicate baggage alerts


@dataclass
class BaggageEvent:
    event_type: str                             # baggage_left_behind | baggage_taken | baggage_swap
    global_id: str
    camera_id: str
    session_index: int
    items_removed: List[str]
    items_added: List[str]
    entry_time_iso: str
    exit_time_iso: str
    severity: str                               # high | critical
    message: str


class SessionTracker:
    """
    Tracks per-identity visit sessions and detects baggage changes on exit.
    Thread-safe for single-writer (pipeline thread) + single-reader (alert thread).
    """

    def __init__(self) -> None:
        # global_id → current open session
        self._active: Dict[str, VisitSession] = {}
        # global_id → count of completed sessions (for session_index)
        self._visit_counts: Dict[str, int] = {}
        # Completed sessions awaiting baggage-check (filled by check_exits)
        self._completed: List[VisitSession] = []

        # Seconds of absence before we consider the person to have "exited"
        self._exit_timeout_s: float = getattr(config, "session_exit_timeout_s", 8.0)

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def on_person_seen(
        self,
        global_id: str,
        carried_items: Set[str],
        camera_id: str,
        now: float,
    ) -> None:
        """
        Call every frame for each visible person with their current carried items.
        Handles entry, ongoing visit, and re-entry after absence.
        """
        session = self._active.get(global_id)

        if session is None:
            # ── New entry ────────────────────────────────────────────────────
            count = self._visit_counts.get(global_id, 0) + 1
            self._visit_counts[global_id] = count
            session = VisitSession(
                global_id=global_id,
                session_index=count,
                camera_id=camera_id,
                entry_time=now,
                last_seen=now,
                items_at_entry=set(carried_items),
                items_at_exit=set(carried_items),
            )
            self._active[global_id] = session
            print(
                f"[SessionTracker] Entry: {global_id} (visit #{count}) "
                f"items={sorted(carried_items) or 'none'}"
            )
        else:
            # ── Ongoing visit — update last seen and current items ────────────
            session.last_seen = now
            session.items_at_exit = set(carried_items)   # rolling update; final value on exit

    def check_exits(
        self,
        active_global_ids: Set[str],
        camera_id: str,
        now: float,
    ) -> List[BaggageEvent]:
        """
        Call every frame with the current set of visible persons.
        Returns a list of BaggageEvent for persons that just exited with a
        different set of items than they entered with.
        """
        events: List[BaggageEvent] = []
        exited_ids: List[str] = []

        for gid, session in self._active.items():
            if gid in active_global_ids:
                continue                         # Still visible — not exited

            absent_for = now - session.last_seen
            if absent_for < self._exit_timeout_s:
                continue                         # Give them a grace period

            # ── Confirmed exit ────────────────────────────────────────────────
            session.exit_time = session.last_seen
            exited_ids.append(gid)
            print(
                f"[SessionTracker] Exit: {gid} (visit #{session.session_index}) "
                f"entry_items={sorted(session.items_at_entry) or 'none'} "
                f"exit_items={sorted(session.items_at_exit) or 'none'}"
            )

            if not session.alert_fired:
                event = self._evaluate_baggage_change(session, camera_id)
                if event:
                    events.append(event)
                    session.alert_fired = True

            self._completed.append(session)

        for gid in exited_ids:
            del self._active[gid]

        return events

    def get_active_session(self, global_id: str) -> Optional[VisitSession]:
        return self._active.get(global_id)

    def get_visit_count(self, global_id: str) -> int:
        return self._visit_counts.get(global_id, 0)

    # ──────────────────────────────────────────────────────────────────────────
    #  Baggage comparison
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_baggage_change(
        self, session: VisitSession, camera_id: str
    ) -> Optional[BaggageEvent]:
        """
        Compare items at entry vs exit and return the appropriate BaggageEvent,
        or None if no significant change occurred.
        """
        entry = session.items_at_entry
        exit_ = session.items_at_exit

        removed = sorted(entry - exit_)         # items the person entered WITH but left WITHOUT
        added   = sorted(exit_ - entry)         # items the person is leaving WITH that they didn't enter with

        # No change — no alert
        if not removed and not added:
            return None

        entry_iso = datetime.datetime.fromtimestamp(session.entry_time).astimezone().isoformat()
        exit_iso  = datetime.datetime.fromtimestamp(session.exit_time or session.last_seen).astimezone().isoformat()
        gid       = session.global_id

        if removed and not added:
            # Entered WITH items → exited WITHOUT → possible item left behind / theft
            msg = (
                f"🚨 ITEM LEFT BEHIND: {gid} entered with {', '.join(removed)} "
                f"but exited without it. Entry {_fmt_time(session.entry_time)} → "
                f"Exit {_fmt_time(session.exit_time or session.last_seen)}."
            )
            return BaggageEvent(
                event_type="baggage_left_behind",
                global_id=gid,
                camera_id=camera_id,
                session_index=session.session_index,
                items_removed=removed,
                items_added=[],
                entry_time_iso=entry_iso,
                exit_time_iso=exit_iso,
                severity="high",
                message=msg,
            )

        elif added and not removed:
            # Entered WITHOUT items → exited WITH → possible theft
            msg = (
                f"🚨 ITEM TAKEN: {gid} entered without luggage but exited "
                f"with {', '.join(added)}. "
                f"Entry {_fmt_time(session.entry_time)} → "
                f"Exit {_fmt_time(session.exit_time or session.last_seen)}."
            )
            return BaggageEvent(
                event_type="baggage_taken",
                global_id=gid,
                camera_id=camera_id,
                session_index=session.session_index,
                items_removed=[],
                items_added=added,
                entry_time_iso=entry_iso,
                exit_time_iso=exit_iso,
                severity="critical",
                message=msg,
            )

        else:
            # Items were SWAPPED — most suspicious
            msg = (
                f"🔴 BAGGAGE SWAP: {gid} entered with {', '.join(removed)} "
                f"and exited with {', '.join(added)} — different items. "
                f"Entry {_fmt_time(session.entry_time)} → "
                f"Exit {_fmt_time(session.exit_time or session.last_seen)}."
            )
            return BaggageEvent(
                event_type="baggage_swap",
                global_id=gid,
                camera_id=camera_id,
                session_index=session.session_index,
                items_removed=removed,
                items_added=added,
                entry_time_iso=entry_iso,
                exit_time_iso=exit_iso,
                severity="critical",
                message=msg,
            )


def _fmt_time(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# Singleton — imported by pipeline.py
session_tracker = SessionTracker()
