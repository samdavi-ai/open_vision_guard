# OpenVisionGuard (Backend Only)

A FastAPI-based AI Vision Security Platform capable of real-time video intelligence, including object detection, tracking, face recognition, pose estimation, and anomaly detection. 

## Setup

1. Create a Python Virtual Environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS
```

2. Install Dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Run the Server:
```bash
python main.py
```

## API Endpoints

- `GET /api/cameras` - List all configured camera streams
- `POST /api/cameras` - Add a new camera stream via RTSP or Webcam
- `POST /api/cameras/upload` - Upload an `.mp4` file and start processing it
- `DELETE /api/cameras/{id}` - Stop and remove a camera stream
- `WebSocket /ws/video/{camera_id}` - Stream annotated video frames
- `WebSocket /ws/alerts` - Stream global security alerts in real-time

The backend runs on `http://localhost:8000` by default. You can view the full interactive API documentation at `http://localhost:8000/docs`.
