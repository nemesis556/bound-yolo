from __future__ import annotations

from typing import Dict, List


def build_explainable_alert(base_alert: Dict[str, object], metadata: Dict[str, object]) -> Dict[str, object]:
    reasons: List[str] = list(base_alert.get("reasons", []))
    details = {
        "camera_id": metadata.get("camera_id", "default-camera"),
        "frame_index": metadata.get("frame_index"),
        "timestamp": metadata.get("timestamp"),
    }
    return {
        "title": base_alert.get("title"),
        "score": base_alert.get("score"),
        "reasons": reasons,
        "details": details,
        "message": f"{base_alert.get('title')} | Score: {base_alert.get('score')} | Reasons: {', '.join(reasons)}",
    }
