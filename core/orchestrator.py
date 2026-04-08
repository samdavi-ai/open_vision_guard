import threading
import time
import datetime
from typing import Dict, Any

from config import config
from modules.database import get_connection
from routers import stream_router

class SystemWatchdog:
    def __init__(self):
        self.is_running = False
        self._thread = None
        self.health_status = {
            "status": "starting",
            "last_check": None,
            "streams": {},
            "database": {"status": "unknown", "latency_ms": 0},
            "alerts": []
        }

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="SystemWatchdog")
        self._thread.start()
        print("[Orchestrator] SystemWatchdog started.")

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            print("[Orchestrator] SystemWatchdog stopped.")

    def _monitor_loop(self):
        while self.is_running:
            try:
                self._run_diagnostics()
            except Exception as e:
                print(f"[Orchestrator Error] {e}")
            time.sleep(2.0)

    def _run_diagnostics(self):
        now_str = datetime.datetime.now().astimezone().isoformat()
        alerts = []
        overall_status = "healthy"

        # 1. Check Streams
        stream_health = {}
        for cam_id, stream_info in stream_router.active_streams.items():
            cam_status = stream_info.get("status", "unknown")
            # If it's running, check if we have a recent payload to verify it hasn't hung
            fps = 0
            is_hung = False
            
            if cam_status == "running":
                payload = stream_info.get("latest_ws_payload")
                if payload:
                    import json
                    try:
                        data = json.loads(payload)
                        fps = data.get("fps", 0)
                        if fps < 2:
                            is_hung = True
                            alerts.append(f"Stream {cam_id} is running at critically low FPS ({fps})")
                    except Exception:
                        pass
                else:
                    # No payload yet
                    fps = 0

            stream_health[cam_id] = {
                "state": cam_status,
                "fps": fps,
                "is_hung": is_hung
            }
            if is_hung:
                overall_status = "degraded"

        # 2. Check Database latency
        t0 = time.time()
        db_state = "healthy"
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception as e:
            db_state = f"error: {str(e)}"
            overall_status = "critical"
            alerts.append(f"Database unavailable: {e}")
            
        latency_ms = round((time.time() - t0) * 1000, 2)
        if latency_ms > 500 and db_state == "healthy":
            db_state = "slow"
            overall_status = "degraded"
            alerts.append(f"Database latency is high ({latency_ms}ms)")

        # Update absolute state
        if len(stream_router.active_streams) == 0 and overall_status == "healthy":
            overall_status = "idle"

        self.health_status = {
            "status": overall_status,
            "last_check": now_str,
            "streams": stream_health,
            "database": {
                "status": db_state,
                "latency_ms": latency_ms
            },
            "alerts": alerts
        }

    def get_health(self) -> Dict[str, Any]:
        return self.health_status

# Global singleton
orchestrator = SystemWatchdog()
