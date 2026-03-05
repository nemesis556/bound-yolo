from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from .behavior_engine import BehaviorEngine
from .detection import YoloDetector
from .explainability import build_explainable_alert
from .recorder import EventRecorder
from .threat_engine import ThreatScoringEngine
from .tracking import CentroidTracker, TrackedObject


class VideoProcessor:
    def __init__(
        self,
        model_path: str,
        restricted_zones: List[List[tuple[int, int]]],
        events_dir: str = "events",
        camera_id: str = "cam-1",
    ) -> None:
        self.detector = YoloDetector(model_path=model_path)
        self.tracker = CentroidTracker()
        self.behavior = BehaviorEngine(restricted_zones=restricted_zones)
        self.threat_engine = ThreatScoringEngine()
        self.recorder = EventRecorder(events_dir=events_dir)
        self.camera_id = camera_id
        self.alerts: List[Dict[str, object]] = []
        self.event_files: List[str] = []
        self.latest_frame_jpeg: Optional[bytes] = None
        self.frame_history: Deque[np.ndarray] = deque(maxlen=150)

    def process_source(self, source: str) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open source: {source}")

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            self.process_frame(frame, frame_index)
        cap.release()

    def process_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, object]:
        now = datetime.utcnow()
        detections = self.detector.detect(frame)
        tracked = self.tracker.update(detections)
        behavior_alerts = self.behavior.analyze(tracked, now)

        payload = {
            "frame_index": frame_index,
            "timestamp": now.isoformat(),
            "detections": [self._track_to_json(t) for t in tracked],
            "alerts": [],
        }

        event_types = []
        for alert in behavior_alerts:
            event_types.append(alert.event_type)
            payload["alerts"].append(
                {"event_type": alert.event_type, "entity_id": alert.entity_id, "message": alert.message}
            )

        if event_types:
            assessment = self.threat_engine.score(event_types)
            base = self.threat_engine.explain(assessment)
            enriched = build_explainable_alert(
                base,
                {"camera_id": self.camera_id, "frame_index": frame_index, "timestamp": now.isoformat()},
            )
            enriched["events"] = payload["alerts"]
            self.alerts.append(enriched)
            clip = self.recorder.trigger(reason=", ".join(event_types))
            self.event_files.append(clip)
            payload["threat"] = enriched

        self._annotate_frame(frame, tracked, payload.get("alerts", []))
        self.recorder.push_frame(frame)
        self.frame_history.append(frame.copy())
        _, encoded = cv2.imencode(".jpg", frame)
        self.latest_frame_jpeg = encoded.tobytes()
        return payload

    def _track_to_json(self, item: TrackedObject) -> Dict[str, object]:
        return {
            "track_id": item.track_id,
            "label": item.label,
            "confidence": item.confidence,
            "bbox": item.bbox,
            "center": item.center,
        }

    def _annotate_frame(self, frame: np.ndarray, tracked: List[TrackedObject], alerts: List[Dict[str, object]]) -> None:
        for item in tracked:
            x1, y1, x2, y2 = item.bbox
            color = (0, 255, 0) if item.label == "person" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{item.label} #{item.track_id}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        y = 20
        for alert in alerts[:3]:
            cv2.putText(frame, alert["event_type"], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 24

    def get_events(self) -> List[str]:
        return [str(Path(p)) for p in self.event_files]
