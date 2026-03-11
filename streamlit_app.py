import streamlit as st
import requests
import json
import base64
import cv2
import numpy as np
import websocket
import threading
import time

# --- Configuration ---
API_URL = "http://localhost:8000/api"
WS_URL = "ws://localhost:8000/ws"

st.set_page_config(
    page_title="OpenVisionGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .alert-box {
        padding: 5px 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        background-color: #2c0b0e;
        border: 1px solid #842029;
        color: #ea868f;
        font-family: monospace;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'cameras' not in st.session_state:
    st.session_state.cameras = []

if 'images' not in st.session_state:
    st.session_state.images = {}
    
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

def fetch_cameras():
    try:
        response = requests.get(f"{API_URL}/cameras")
        if response.status_code == 200:
            st.session_state.cameras = response.json()
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")

# --- API Functions ---
def add_camera_url(name, url, modules):
    cam_id = f"cam_{int(time.time())}"
    data = {
        "id": cam_id,
        "name": name,
        "url": url,
        "modules": modules
    }
    try:
        res = requests.post(f"{API_URL}/cameras", json=data)
        if res.status_code == 200:
            st.success(f"Added source: {name}")
            fetch_cameras()
        else:
            st.error(f"Failed to add source: {res.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

def add_camera_upload(name, uploaded_file, modules):
    try:
        files = {"file": (uploaded_file.name, uploaded_file, "video/mp4")}
        data = {
            "name": name,
            "modules": ",".join(modules)
        }
        res = requests.post(f"{API_URL}/cameras/upload", files=files, data=data)
        if res.status_code == 200:
            st.success(f"Uploaded and started source: {name}")
            fetch_cameras()
        else:
            st.error(f"Upload failed: {res.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")

def delete_camera(cam_id):
    try:
        res = requests.delete(f"{API_URL}/cameras/{cam_id}")
        if res.status_code == 200:
            fetch_cameras()
    except Exception as e:
        st.error(f"Failed to delete: {e}")

# --- WebSockets Clients using threading ---
def on_video_message(ws, message):
    try:
        data = json.loads(message)
        cam_id = data.get("camera_id")
        img_data = base64.b64decode(data['frame'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.session_state.images[cam_id] = img_rgb
        
        # Handle embedded alerts
        if 'alerts' in data and data['alerts']:
            for alert_text in data['alerts']:
                st.session_state.alerts.insert(0, {
                    "module": "AI DETECT",
                    "message": alert_text,
                    "timestamp": time.strftime('%H:%M:%S')
                })
            # Keep only the last 50
            st.session_state.alerts = st.session_state.alerts[:50]
            
    except Exception as e:
        pass

def start_video_thread(cam_id):
    def run():
        ws = websocket.WebSocketApp(f"{WS_URL}/{cam_id}", on_message=on_video_message)
        ws.run_forever()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# --- Sidebar UI ---
with st.sidebar:
    st.title("🛡️ OpenVisionGuard")
    st.markdown("---")
    st.subheader("Add New Source")
    
    source_type = st.radio("Source Type", ["URL / RTSP / Webcam", "Video File Upload"])
    cam_name = st.text_input("Display Name", "Camera 1")
    
    # Available AI Modules
    module_opts = ["object_detector", "reid_tracking", "face_recognition", "pose", "motion", "weapon"]
    selected_modules = st.multiselect("AI Modules", module_opts, default=["object_detector", "reid_tracking", "motion"])
    
    if source_type == "URL / RTSP / Webcam":
        cam_url = st.text_input("Stream URL (e.g. 0 for webcam, rtsp://...)", "0")
        if st.button("Add URL Stream", type="primary", use_container_width=True):
            add_camera_url(cam_name, cam_url, selected_modules)
    else:
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if st.button("Upload & Start", type="primary", use_container_width=True, disabled=not uploaded_file):
            add_camera_upload(cam_name, uploaded_file, selected_modules)
            
    st.markdown("---")
    st.subheader("Active Sources")
    fetch_cameras()
    if not st.session_state.cameras:
        st.info("No active streams.")
    else:
        for cam in st.session_state.cameras:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📹 **{cam['name']}**")
            if col2.button("🚫", key=f"del_{cam['id']}", help="Remove Stream"):
                delete_camera(cam['id'])
                st.rerun()

# --- Main Dashboard ---
st.title("Live Monitoring Dashboard")

col_main, col_alerts = st.columns([3, 1])

with col_main:
    st.subheader("Video Feeds")
    if not st.session_state.cameras:
        st.info("👈 Add a camera stream from the sidebar to begin monitoring.")
    else:
        # Create a dynamic grid (up to 2 cols for feeds)
        feed_cols = st.columns(2)
        
        # Start websockets for active cameras if we haven't already
        for idx, cam in enumerate(st.session_state.cameras):
            cam_id = cam['id']
            if cam_id not in st.session_state.get('active_threads', set()):
                if 'active_threads' not in st.session_state:
                    st.session_state.active_threads = set()
                st.session_state.active_threads.add(cam_id)
                start_video_thread(cam_id)

        # Render loop using st.empty for each camera
        placeholders = {cam['id']: feed_cols[idx % 2].empty() for idx, cam in enumerate(st.session_state.cameras)}

with col_alerts:
    st.subheader("Real-time Alerts")
    alerts_placeholder = st.empty()

# --- Continuous Rendering Loop ---
if st.session_state.cameras:
    try:
        while True:
            rendered = False
            # Update Video Feeds
            for cam in st.session_state.cameras:
                cam_id = cam['id']
                if cam_id in st.session_state.images:
                    placeholders[cam_id].image(st.session_state.images[cam_id], channels="RGB", use_container_width=True, caption=cam['name'])
                    rendered = True
                    
            # Update Alerts Sidebar
            if st.session_state.alerts:
                alert_html = ""
                for alert in st.session_state.alerts[:10]: # Show latest 10
                    color = "#ea868f"
                    if alert.get("module") == "motion":
                        color = "#6ea8fe"
                    alert_html += f"<div class='alert-box' style='color: {color}'><b>[{alert.get('module', 'System').upper()}]</b><br/>{alert.get('message', '')}<br/><small>{alert.get('timestamp', '')}</small></div>"
                alerts_placeholder.markdown(alert_html, unsafe_allow_html=True)
            else:
                alerts_placeholder.info("Listening for alerts...")
            
            # Avoid 'magic' expression printing
            if not rendered:
                time.sleep(0.05)
            else:
                time.sleep(0.01)
    except Exception as e:
        pass