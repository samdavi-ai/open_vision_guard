from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from modules import database
from routers import stream_router, identity_router, alert_router, search_router, config_router

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
    """Initialize database on startup."""
    database.init_db()
    print("OpenVisionGuard started.")


# Register routers
app.include_router(stream_router.router)
app.include_router(identity_router.router)
app.include_router(alert_router.router)
app.include_router(search_router.router)
app.include_router(config_router.router)


@app.get("/")
async def root():
    return {
        "name": "OpenVisionGuard",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "ui": "/ui"
    }

@app.get("/ui")
async def ui():
    """Serve the OpenVisionGuard frontend dashboard."""
    import os
    index_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
