from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional import runtime fallback
    YOLO = None


TARGET_CLASSES = {"person", "backpack", "handbag", "suitcase"}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]


class YoloDetector:
    """YOLOv8 detector wrapper with graceful fallback when model is unavailable."""

    def __init__(self, model_path: str = "models/yolov8n.pt", device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None
        if YOLO is not None:
            self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(frame, device=self.device, verbose=False)
        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                if label not in TARGET_CLASSES:
                    continue

                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                    )
                )
        return detections
