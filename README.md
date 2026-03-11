# Security-Focused Video Object Detection and Re-Identification (Re-ID) System

This project is a Security-Focused Video Object Detection and Re-Identification (Re-ID) System. It is designed to solve the common problem in surveillance where a person's ID "flickers" or changes when they are temporarily blocked or move between different camera views.

## Core Objective
To provide a reliable tracking system that assigns a globally consistent dummy name (e.g., Subject_1, Subject_2) to an individual, ensuring they are recognized as the same person even if they appear in different videos or return to a live camera feed after leaving.

## Key Technical Features
* **Stable Identity Locking**: Unlike standard trackers that restart IDs every frame, this system "locks" YOLO's local track IDs to a Global Registry. This ensures Subject_1 stays Subject_1 throughout their entire appearance in a clip.
* **Cross-Video Re-Identification**: The system maintains a "memory" (Feature Registry) of every person it sees. If Subject_1 appears in Video_A and then again in Video_B, the system uses feature matching to recognize them and keep the same name.
* **Advanced Feature Matching**:
  * **3D Color Histograms**: Uses the distribution of colors on a person's clothing/form to create a unique "style fingerprint."
  * **Cosine Similarity**: Mathematically compares new subjects against the registry to find the best match with high precision.
  * **Temporal Averaging**: Smooths out feature data over multiple frames to prevent errors caused by sudden changes in lighting or shadows.
* **Live & Batch Processing**: supports real-time monitoring via webcam and automated batch processing of multiple recorded video files.

## Security Use Cases
* **Intruder Tracking**: Following a specific individual across different security camera angles.
* **Presence Monitoring**: Identifying if the same person has visited multiple restricted zones.
* **Automated Surveillance**: Reducing the need for manual monitoring by providing clear, persistent subject labels.

## 🚀 Features

- *Live Detection*: Real-time monitoring via webcam (`--input 0`).
- *Multi-Video Processing*: Batch process all videos in a directory.
- *Security-Focused Re-ID*: Assigns consistent "Subject IDs" (e.g., Subject_1) to the same person across different video sessions.
- *Stable Tracking*: Uses session-to-global ID locking to prevent ID flickering within a single video.
- *Temporal Averaging*: Accumulates object features over time for robust identification even with changing poses or lighting.

## 🛠️ Installation

1. *Clone or Download* the project.
2. *Install Dependencies*:
```bash
pip install -r backend/requirements.txt
```

## 📂 Project Structure

- `detect.py`: Main script for detection and Re-ID.
- `backend/requirements.txt`: Python dependencies.
- `inputs/`: Place your source videos here.
- `outputs/`: Annotated videos will be saved here.
- `streamlit_app.py`: Real-time streaming dashboard for interacting with streams.

## 💻 Usage

### 1. Live Detection (Webcam)
Run the following to start real-time monitoring:
```bash
python detect.py --input 0 --device cpu
```
- Press *q* to quit the live feed.
- Add `--output outputs/` to record the session.

### 2. Process Multiple Videos
Place your videos in an `inputs/` folder and run:
```bash
python detect.py --input inputs/ --output outputs/ --device cpu
```
The annotated videos will be saved in the `outputs/` directory with the `annotated_` prefix.

### 3. Streamlit Dashboard
To run the full dashboard with real-time feedback:
```bash
# Start backend API (Terminal 1)
cd backend
python main.py

# Start Streamlit UI (Terminal 2)
streamlit run streamlit_app.py
```

## ⚙️ Configuration

- `--model`: Path to the YOLO model (default: `yolov8n.pt`).
- `--device`: Device to run on (e.g., `cpu`, `0` for GPU).
- `--similarity_threshold`: (Internal) Adjusts how strictly subjects are matched (default: 0.85).
