from __future__ import annotations

import shutil
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .video_processor import VideoProcessor

app = FastAPI(title="ThreatSense", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOADS_DIR = BASE_DIR / "uploads"
EVENTS_DIR = BASE_DIR / "events"
DASHBOARD_DIR = BASE_DIR / "dashboard"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

restricted_zones = [[(300, 200), (500, 200), (500, 400), (300, 400)]]
processor = VideoProcessor(
    model_path=str(BASE_DIR / "models" / "yolov8n.pt"),
    restricted_zones=restricted_zones,
    events_dir=str(EVENTS_DIR),
)

app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
app.mount("/events_files", StaticFiles(directory=str(EVENTS_DIR)), name="events_files")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(DASHBOARD_DIR / "index.html"))


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    dest = UPLOADS_DIR / file.filename
    with dest.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    processor.process_source(str(dest))
    return {"status": "processed", "file": file.filename, "alerts": len(processor.alerts)}


@app.get("/alerts")
def get_alerts() -> dict:
    return {"alerts": processor.alerts}


@app.get("/events")
def get_events() -> dict:
    return {"events": processor.get_events()}


@app.get("/live_feed")
def live_feed() -> StreamingResponse:
    def generate() -> Generator[bytes, None, None]:
        while True:
            frame = processor.latest_frame_jpeg
            if frame is None:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
