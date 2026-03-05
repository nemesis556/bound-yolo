from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List

import cv2
import numpy as np


class EventRecorder:
    def __init__(self, events_dir: str = "events", fps: int = 15, pre_seconds: int = 10, post_seconds: int = 10) -> None:
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pre_frames = fps * pre_seconds
        self.post_frames = fps * post_seconds
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.pre_frames)
        self.active_recordings: List[Dict[str, object]] = []

    def push_frame(self, frame: np.ndarray) -> None:
        self.buffer.append(frame.copy())
        finished = []
        for recording in self.active_recordings:
            recording["frames"].append(frame.copy())
            recording["remaining"] -= 1
            if recording["remaining"] <= 0:
                finished.append(recording)

        for recording in finished:
            self._flush_recording(recording)
            self.active_recordings.remove(recording)

    def trigger(self, reason: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"event_{ts}.mp4"
        clip_path = self.events_dir / filename
        initial_frames = list(self.buffer)
        self.active_recordings.append(
            {
                "reason": reason,
                "path": clip_path,
                "frames": initial_frames,
                "remaining": self.post_frames,
            }
        )
        return str(clip_path)

    def _flush_recording(self, recording: Dict[str, object]) -> None:
        frames = recording["frames"]
        if not frames:
            return
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(recording["path"]),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
