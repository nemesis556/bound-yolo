from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


EVENT_SCORES = {
    "loitering": 30,
    "restricted_zone_entry": 40,
    "running": 20,
    "crowd_alert": 15,
    "abandoned_object": 50,
    "night_activity": 10,
}


@dataclass
class ThreatAssessment:
    score: int
    level: str
    reasons: List[str]


class ThreatScoringEngine:
    def score(self, triggered_events: List[str]) -> ThreatAssessment:
        total = sum(EVENT_SCORES.get(event, 0) for event in triggered_events)
        total = max(0, min(100, total))

        if total < 30:
            level = "LOW"
        elif total < 70:
            level = "MEDIUM"
        else:
            level = "CRITICAL"

        reasons = [event.replace("_", " ").title() for event in triggered_events]
        return ThreatAssessment(score=total, level=level, reasons=reasons)

    def explain(self, assessment: ThreatAssessment) -> Dict[str, object]:
        return {
            "title": f"ALERT: {assessment.level} THREAT",
            "score": assessment.score,
            "reasons": assessment.reasons,
        }
