import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from modules import database
from core.orchestrator import orchestrator
from routers import stream_router, identity_router, alert_router, search_router, config_router, face_log_router, analytics_router

app = FastAPI(
    title="OpenVisionGuard",
    description="AI Surveillance Operating System — FastAPI + YOLOv8 + OpenCV + PyTorch",
    version="1.0.0"
)

# CORS (allow all for dev — restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database and start background orchestrator on startup."""
    database.init_db()
    orchestrator.start()
    print("OpenVisionGuard started.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background threads on shutdown."""
    orchestrator.stop()
    print("OpenVisionGuard stopped.")


# Register routers
app.include_router(stream_router.router)
app.include_router(identity_router.router)
app.include_router(alert_router.router)
app.include_router(search_router.router)
app.include_router(config_router.router)
app.include_router(face_log_router.router)
app.include_router(analytics_router.router)

frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
assets_path = os.path.join(frontend_path, "assets")
if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# Mount data directory for face crops and alerts
if not os.path.exists("data"):
    os.makedirs("data")
app.mount("/static/data", StaticFiles(directory="data"), name="static")

@app.get("/")
async def root():
    return {
        "name": "OpenVisionGuard",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "ui": "/ui",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Live diagnostics payload returned from the continuous background watchdog."""
    return orchestrator.get_health()

@app.get("/ui")
async def ui():
    """Serve the OpenVisionGuard frontend dashboard."""
    index_path = os.path.join(frontend_path, "index.html")
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
