from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .detection import Detection


@dataclass
class TrackedObject:
    track_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]


class CentroidTracker:
    """Simple nearest-centroid tracker as a ByteTrack/DeepSORT fallback."""

    def __init__(self, max_distance: float = 60.0, max_missing: int = 20) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        assigned = set()
        outputs: List[TrackedObject] = []

        # Increment missing counters before matching
        for data in self.tracks.values():
            data["missing"] = int(data.get("missing", 0)) + 1

        for det in detections:
            best_id = None
            best_dist = float("inf")
            for track_id, data in self.tracks.items():
                if track_id in assigned:
                    continue
                if data["label"] != det.label:
                    continue
                dist = self._distance(data["center"], det.center)
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {}

            self.tracks[best_id].update(
                {
                    "center": det.center,
                    "bbox": det.bbox,
                    "label": det.label,
                    "confidence": det.confidence,
                    "missing": 0,
                }
            )
            assigned.add(best_id)

            outputs.append(
                TrackedObject(
                    track_id=best_id,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    center=det.center,
                )
            )

        stale_ids = [tid for tid, d in self.tracks.items() if int(d.get("missing", 0)) > self.max_missing]
        for tid in stale_ids:
            self.tracks.pop(tid, None)

        return outputs
