import asyncio
import cv2
import json
import os
import threading
import time
import datetime
import base64
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import Response
from typing import Dict, Any

from config import config
from core.pipeline import Pipeline
from models.schemas import StreamStartRequest, StreamInfo
from modules.geolocation import geolocation_engine

router = APIRouter(tags=["Stream"])

active_streams: Dict[str, Dict[str, Any]] = {}
_pipeline: Pipeline = None
_pipeline_lock = threading.Lock()

# camera_id → {frame, timestamp}
_latest_raw_frames: Dict[str, Any] = {}

# camera_id → {global_id → crop_jpeg_bytes}
_latest_crops: Dict[str, Dict[str, bytes]] = {}

RISK_COLORS_BGR = {
    'low':      (255, 165, 0),
    'medium':   (0, 200, 220),
    'high':     (0, 120, 255),
    'critical': (0, 0, 230),
}

# Colors for non-person object categories (BGR)
CATEGORY_COLORS_BGR = {
    'vehicle':    (0, 200, 0),       # green
    'animal':     (0, 220, 220),     # yellow
    'accessory':  (200, 130, 50),    # teal
    'sports':     (50, 200, 255),    # orange
    'food':       (100, 180, 255),   # peach
    'furniture':  (180, 180, 100),   # slate blue
    'electronic': (255, 150, 50),    # light blue
    'kitchen':    (150, 100, 200),   # mauve
    'other':      (180, 180, 180),   # gray
}


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = Pipeline()
    return _pipeline


def _draw_detections(frame: 'np.ndarray', detections: list) -> 'np.ndarray':
    """Draw detections onto the frame. Returns a new annotated copy."""
    import cv2 as _cv2
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        is_object = det.get("is_object", False)

        if is_object:
            # Non-person object: use category color
            category = det.get("object_category", "other")
            color = CATEGORY_COLORS_BGR.get(category, (180, 180, 180))
            conf = det.get("confidence", 0)
            label = f"{det.get('display_name', '?')} {conf:.0%}" if conf else det.get("display_name", "?")

            # Thinner box for objects
            _cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

            # Label background + text
            (tw, th), _ = _cv2.getTextSize(label, _cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            _cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            _cv2.putText(out, label, (x1 + 2, max(th, y1) - 3),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, _cv2.LINE_AA)
        else:
            # Person: use risk-level color
            risk = det.get("risk_level", "low")
            color = RISK_COLORS_BGR.get(risk, (255, 165, 0))

            # Box
            _cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = det.get("display_name", det.get("global_id", "?"))
            (tw, th), _ = _cv2.getTextSize(label, _cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            _cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            _cv2.putText(out, label, (x1 + 3, max(th, y1) - 4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, _cv2.LINE_AA)

            # Objects tag
            objects = det.get("carried_objects", [])
            if objects:
                tag = ", ".join(objects)[:18]
                (ow, oh), _ = _cv2.getTextSize(tag, _cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                _cv2.rectangle(out, (x1, y2), (x1 + ow + 6, y2 + oh + 6), (130, 60, 200), -1)
                _cv2.putText(out, tag, (x1 + 3, y2 + oh + 2),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, _cv2.LINE_AA)
    return out


def _update_crops(frame: 'np.ndarray', detections: list, camera_id: str):
    """Extract and cache a crop for every detected person."""
    import cv2 as _cv2
    h, w = frame.shape[:2]
    crops = {}
    for det in detections:
        gid = det["global_id"]
        x1, y1, x2, y2 = det["bbox"]
        pad = 30
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            _, jpeg = _cv2.imencode('.jpg', crop, [_cv2.IMWRITE_JPEG_QUALITY, 88])
            crops[gid] = jpeg.tobytes()
    _latest_crops[camera_id] = crops


def _ai_inference_thread(camera_id: str, pipeline, shared_state: dict):
    """
    Background AI thread: picks up frames from shared_state, runs inference,
    writes results back. Uses adaptive_controller.should_run_inference() which
    incorporates both the interval gate and the zero-motion gate (with warmup).
    """
    from modules.adaptive_inference import adaptive_controller
    from modules.production_guard import (
        latency_guard, scene_profiler, threshold_calibrator,
        edge_detector,
    )

    while shared_state["running"]:
        frame = shared_state.get("ai_frame")
        if frame is None:
            time.sleep(0.005)
            continue

        # Check interval + motion gate. If not ready, sleep briefly and retry.
        if not adaptive_controller.should_run_inference():
            time.sleep(0.005)
            continue

        # Consume the frame — take a copy so the display thread can overwrite safely
        shared_state["ai_frame"] = None
        work_frame = frame.copy()

        try:
            t0 = time.monotonic()
            result = pipeline.process_frame(work_frame, camera_id)
            ai_latency = (time.monotonic() - t0) * 1000

            shared_state["detections"] = result.current_detections
            shared_state["alerts"] = result.alerts
            shared_state["ai_latency_ms"] = ai_latency

            adaptive_controller.feed(result.current_detections, work_frame)
            threshold_calibrator.feed(camera_id, result.current_detections)
            scene_profiler.update(camera_id, work_frame, result.current_detections)
            edge_events = edge_detector.check(camera_id, work_frame, len(result.current_detections))
            if edge_events.get("lighting_change"):
                print(f"[EdgeCase] {camera_id}: lighting change")
            if edge_events.get("camera_shake"):
                print(f"[EdgeCase] {camera_id}: camera shake")
            latency_guard.record(ai_latency)

        except Exception as e:
            print(f"[AI Thread error] {e}")

    shared_state["ai_done"] = True


def _stream_worker(camera_id: str, source: str):
    """
    Display-thread worker: reads video at native FPS, draws cached detections,
    encodes JPEG, and pushes over WebSocket.  AI runs in a parallel thread.

    Optimizations vs previous version:
      - Adaptive display width and JPEG quality based on activity score.
      - Crop rebuild skipped when detection IDs haven't changed.
      - Hardcoded display constants replaced with config values.
    """
    import cv2 as _cv2
    from modules.detection_memory import DetectionMemory
    from modules.adaptive_inference import adaptive_controller
    from modules.production_guard import (
        latency_guard, scene_profiler, threshold_calibrator,
    )

    pipeline = get_pipeline()

    is_webcam = str(source) == "0"
    cap = _cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        active_streams[camera_id]["status"] = "error"
        return

    native_fps = cap.get(_cv2.CAP_PROP_FPS)
    if native_fps <= 0 or native_fps > 120:
        native_fps = 25.0
    frame_delay = 1.0 / native_fps

    active_streams[camera_id]["status"] = "running"

    memory = DetectionMemory(ttl_frames=12, interpolate=True, video_fps=native_fps)

    shared_state = {
        "running": True,
        "ai_frame": None,
        "detections": [],
        "alerts": [],
        "ai_latency_ms": 0.0,
        "ai_done": False,
    }

    ai_thread = threading.Thread(
        target=_ai_inference_thread,
        args=(camera_id, pipeline, shared_state),
        daemon=True,
        name=f"AI-{camera_id}",
    )
    ai_thread.start()

    detections = []
    frame_count = 0
    _last_crop_ids: frozenset = frozenset()   # track IDs in last crop update
    _cached_location = geolocation_engine.get_current_location()  # cache once
    _loc_refresh_counter = 0

    while active_streams.get(camera_id, {}).get("status") == "running":
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            continue

        frame_count += 1
        _latest_raw_frames[camera_id] = {"frame": frame, "timestamp": time.time()}

        # Feed AI thread the latest frame (overwrite — we only care about newest)
        shared_state["ai_frame"] = frame  # no copy; AI thread reads only

        # Consume new AI detections if available
        ai_detections = shared_state["detections"]
        if ai_detections:
            memory.update(ai_detections)
            detections = ai_detections
            active_streams[camera_id]["latest_alerts"] = shared_state["alerts"]
            shared_state["detections"] = []
            shared_state["alerts"] = []
        else:
            memory.tick()
            detections = memory.get_active()

        active_streams[camera_id]["latest_detections"] = detections

        # Smart crop update: only rebuild when the tracked identity set changes
        if frame_count % config.crop_update_interval == 0:
            current_ids = frozenset(
                d["global_id"] for d in detections if not d.get("is_object", False)
            )
            if current_ids != _last_crop_ids:
                _update_crops(frame, detections, camera_id)
                _last_crop_ids = current_ids

        # --- Adaptive display pipeline: downscale → draw → encode -------------
        h_orig, w_orig = frame.shape[:2]
        activity = adaptive_controller.activity_score
        high_activity = activity >= config.activity_high_threshold
        display_w = (
            config.display_width_high_activity if high_activity
            else config.display_width_low_activity
        )
        jpeg_q = (
            config.jpeg_quality_high_activity if high_activity
            else config.jpeg_quality_low_activity
        )

        if w_orig > display_w:
            scale = display_w / w_orig
            display_frame = _cv2.resize(
                frame, (display_w, int(h_orig * scale)),
                interpolation=_cv2.INTER_NEAREST,
            )
            scaled_dets = []
            for det in detections:
                sd = dict(det)
                x1, y1, x2, y2 = det["bbox"]
                sd["bbox"] = [int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)]
                scaled_dets.append(sd)
            annotated = _draw_detections(display_frame, scaled_dets)
        else:
            annotated = _draw_detections(frame, detections)

        _, jpeg_buf = _cv2.imencode('.jpg', annotated, [_cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        b64 = base64.b64encode(jpeg_buf.tobytes()).decode('utf-8')

        # Refresh geolocation only every 150 frames (~6 seconds at 25 FPS)
        _loc_refresh_counter += 1
        if _loc_refresh_counter >= 150:
            _cached_location = geolocation_engine.get_current_location()
            _loc_refresh_counter = 0

        display_latency_ms = (time.monotonic() - t0) * 1000
        scene = scene_profiler.get_profile(camera_id)
        active_streams[camera_id]["latest_ws_payload"] = json.dumps({
            "frame": b64,
            "width": annotated.shape[1],
            "height": annotated.shape[0],
            "detections": detections,
            "fps": int(1.0 / max(0.001, time.monotonic() - t0)),
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "latitude": _cached_location["latitude"],
            "longitude": _cached_location["longitude"],
            "latency_ms": round(display_latency_ms, 1),
            "ai_latency_ms": round(shared_state["ai_latency_ms"], 1),
            "latency_status": latency_guard.status,
            "scene_brightness": round(scene.avg_brightness, 1),
            "scene_density": round(scene.avg_density, 1),
            "adaptive_conf": round(threshold_calibrator.get_threshold(camera_id), 3),
            "activity_score": round(activity, 3),
        })

        elapsed = time.monotonic() - t0
        remaining = frame_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    shared_state["running"] = False
    ai_thread.join(timeout=3.0)
    cap.release()
    active_streams[camera_id]["status"] = "stopped"


# -- REST Endpoints ----------------------------------------------------------

@router.post("/stream/upload")
async def upload_video(file: UploadFile = File(...)):
    print(f"[Upload] Received file: {file.filename}, type: {file.content_type}")
    try:
        upload_dir = os.path.join("data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        # Use only the basename to prevent directory traversal or absolute path issues
        filename = os.path.basename(file.filename)
        path = os.path.join(upload_dir, filename)
        
        # Stream the file to disk in chunks to handle large files efficiently
        with open(path, "wb") as f:
            while chunk := await file.read(1024 * 1024): # 1MB chunks
                f.write(chunk)
                
        print(f"[Upload] Saved to: {path}")
        return {"message": "Uploaded", "path": path}
    except Exception as e:
        print(f"[Upload error] {e}")
        return {"error": str(e)}


@router.post("/stream/start")
async def start_stream(req: StreamStartRequest):
    threading.Thread(target=get_pipeline, daemon=True).start()
    camera_id = req.camera_id or f"CAM_{len(active_streams) + 1:02d}"
    if camera_id in active_streams and active_streams[camera_id]["status"] == "running":
        return {"message": "Already running", "camera_id": camera_id}
    active_streams[camera_id] = {
        "source": req.source, "status": "starting",
        "latest_ws_payload": None, "latest_alerts": [],
        "latest_detections": [],
    }
    threading.Thread(target=_stream_worker, args=(camera_id, req.source), daemon=True).start()
    return {"message": f"Stream {camera_id} started", "camera_id": camera_id}


@router.post("/stream/stop/{camera_id}")
async def stop_stream(camera_id: str):
    if camera_id not in active_streams:
        return {"error": "Not found"}
    active_streams[camera_id]["status"] = "stopping"
    return {"message": "Stopping"}


@router.get("/stream/list")
async def list_streams():
    return [StreamInfo(camera_id=c, source=i["source"], status=i["status"])
            for c, i in active_streams.items()]


@router.get("/stream/person_crop/{camera_id}/{global_id}")
async def get_person_crop(camera_id: str, global_id: str):
    crops = _latest_crops.get(camera_id, {})
    jpeg = crops.get(global_id)
    if jpeg:
        return Response(content=jpeg, media_type="image/jpeg")
    return Response(content=b"", media_type="image/jpeg", status_code=404)


# -- WebSocket Endpoints -----------------------------------------------------

@router.websocket("/ws/stream/{camera_id}")
async def ws_stream(websocket: WebSocket, camera_id: str):
    """Main video stream WebSocket."""
    await websocket.accept()
    try:
        while True:
            payload = active_streams.get(camera_id, {}).get("latest_ws_payload")
            if payload:
                await websocket.send_text(payload)
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/person_crop/{camera_id}/{global_id}")
async def ws_person_crop(websocket: WebSocket, camera_id: str, global_id: str):
    """
    Live crop WebSocket for a specific person.
    Streams JPEG crops at ~20fps as base64, always in sync with video.
    """
    await websocket.accept()
    try:
        while True:
            crops = _latest_crops.get(camera_id, {})
            jpeg = crops.get(global_id)
            if jpeg:
                b64 = base64.b64encode(jpeg).decode('utf-8')
                await websocket.send_text(json.dumps({
                    "crop": b64,
                    "found": True,
                    "global_id": global_id
                }))
            else:
                await websocket.send_text(json.dumps({"found": False, "global_id": global_id}))
            await asyncio.sleep(0.05)  # ~20fps crop stream
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            for info in active_streams.values():
                for alert in info.get("latest_alerts", []):
                    await websocket.send_text(json.dumps(alert))
                info["latest_alerts"] = []
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
