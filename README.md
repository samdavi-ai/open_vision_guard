# OpenVisionGuard

**Advanced AI Surveillance and Behavioral Analytics Operating System**

OpenVisionGuard is a state-of-the-art security intelligence platform. Moving beyond simple bounding boxes, it processes video streams in real-time to analyze deep behavioral analytics, perform cross-camera Re-Identification (Re-ID), and dynamically assess risk levels using an ensemble of 8 specialized AI modules.

## 🚀 Key Intelligent Features

1. **Persistent Identity Locking (Re-ID):** Maintains consistent subject IDs across different cameras and occlusion events using deep embedding similarity.
2. **Behavioral Analytics:** Detects pacing, circle-walking, sudden stopping, running, and prolonged stillness using sub-second trajectory calculations.
3. **Dynamic Risk Engine:** Aggregates telemetry (weapons, location, behaviors) into a live 0-100 composite threat score per individual.
4. **Luggage Tracking:** Robust distance-based tracker assigning object ownership. Detects abandoned luggage and potential theft.
5. **Presence & Frequency Tracking:** Logs precise dwell times, entry/exit events, and categorizes visit frequency (e.g., *regular*, *new*).
6. **Camera Avoidance Detection:** Flags subjects attempting to hide their faces or hug blind-spots based on head pose and trajectory angles.
7. **Sudden Movement Detection:** Alerts on sudden velocity spikes, lunging, or panicked running.
8. **Face Logging Persistence:** Saves continuous face crops of identified individuals to SQLite.

## 📂 Project Structure

- `core/`: Contains the main `pipeline.py` which streams YOLOv8 frames through the analytics modules.
- `modules/`: Contains the specialized intelligence engines (`risk_engine`, `luggage_tracker`, etc.) and database layer.
- `routers/`: FastAPI routes (`analytics_router`, `stream_router`, etc.) that expose real-time metrics.
- `openvision-ui/`: The React-based control room HUD and intelligence dashboard.

## 🛠️ How to Run

OpenVisionGuard runs using a **FastAPI** backend and a **React (Vite)** frontend.

### 1. Start the Backend API (Terminal 1)

Ensure you have Python 3.9+ installed. From the root of the repository:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate 
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### 2. Start the Control Room UI (Terminal 2)

Open a new terminal and navigate to the frontend directory:

```bash
cd openvision-ui

# Install Node dependencies (only needed the first time)
npm install

# Start the Vite development server
npm run dev
```

Your React control room will now be accessible (usually at `http://localhost:5173`). The dashboard communicates via WebSocket and REST APIs to the backend running on port `8080`.

## ⚙️ How it Works

The system utilizes an **Ultra-Fast Nano mode (YOLOv8n)** for real-time detection without sacrificing frame rate. Detections are piped into a `ByteTrack` tracker to resolve frame-to-frame association. Feature crops are then extracted and passed through a suite of lightweight analytics engines running synchronously, which emit signals to a centralized `alert_engine` and push metadata patches to an in-memory `EmbeddingEngine` singleton. This data is then streamed down via WebSocket to the React client.
