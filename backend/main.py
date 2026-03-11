from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from api.routes import init_routes
from api.websockets import router as ws_router
from api.websockets import manager as ws_manager
from core.video_manager import VideoManager

app = FastAPI(title="OpenVisionGuard API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Video Manager
video_manager = VideoManager(ws_manager)
init_routes(video_manager)

app.include_router(api_router, prefix="/api")
app.include_router(ws_router) # The prefix /ws is handled in router paths now

@app.get("/")
def read_root():
    return {"status": "OpenVisionGuard API is running", "active_streams": len(video_manager.streams)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
