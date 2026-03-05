# ThreatSense: AI-Powered Surveillance Prototype

ThreatSense is a modular, production-style prototype that upgrades CCTV feeds with real-time AI behavior analysis and explainable threat alerts.

## Features

- Human and bag detection (`person`, `backpack`, `handbag`, `suitcase`) using YOLOv8
- Multi-object tracking with persistent IDs (centroid tracker fallback)
- Restricted zone intrusion detection (polygon zones)
- Loitering detection
- Running / sudden movement detection
- Crowd density detection
- Abandoned object detection
- Threat scoring and risk level classification
- Explainable alerts with reasons + metadata
- Smart event recording (10s pre + 10s post alert clips)
- FastAPI backend + lightweight dashboard

## Project Structure

```text
threatsense/
├── app/
│   ├── main.py
│   ├── video_processor.py
│   ├── detection.py
│   ├── tracking.py
│   ├── behavior_engine.py
│   ├── zone_detection.py
│   ├── threat_engine.py
│   ├── explainability.py
│   └── recorder.py
├── dashboard/
│   └── index.html
├── events/
├── models/
│   └── yolov8n.pt (downloaded on first run by ultralytics if missing)
├── tests/
│   └── test_threat_engine.py
├── requirements.txt
└── README.md
```

## Architecture Pipeline

Video input (file/RTSP) → frame extraction → YOLO detection → tracking → behavior analysis → threat scoring → explainable alerts → event recording → dashboard.

## Installation

```bash
cd threatsense
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd threatsense
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Dashboard: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`

## API Endpoints

- `POST /upload_video` — upload and process video file
- `GET /alerts` — list explainable alerts
- `GET /events` — list generated event clips
- `GET /live_feed` — MJPEG stream of latest processed frame

## Example Usage

```bash
curl -X POST "http://localhost:8000/upload_video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_video.mp4"

curl http://localhost:8000/alerts
curl http://localhost:8000/events
```

## Configuration Notes

- Restricted zones are currently configured in `app/main.py`.
- CPU fallback is supported by default (YOLO device set to CPU).
- For 720p/15FPS demo workloads, use `yolov8n` model and lower capture FPS if needed.

## Smart Recording

When any threat event triggers:

- Keeps prior 10 seconds from rolling buffer
- Captures next 10 seconds
- Writes clip under `events/event_*.mp4`

## Testing

```bash
cd threatsense
pytest -q
```

## Demo Tips

- Use crowded hallway footage for crowd alerts.
- Use scene with person lingering in one area for loitering.
- Place bag and walk away to trigger abandoned object logic.
