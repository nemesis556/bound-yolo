from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from .tracking import TrackedObject
from .zone_detection import check_restricted_zones


@dataclass
class BehaviorAlert:
    event_type: str
    entity_id: int | str
    message: str


class BehaviorEngine:
    def __init__(
        self,
        restricted_zones: Sequence[Sequence[Tuple[int, int]]],
        zone_threshold_s: float = 5.0,
        loitering_seconds: float = 60.0,
        loitering_movement_px: float = 50.0,
        running_px_per_s: float = 220.0,
        crowd_threshold: int = 8,
        abandoned_seconds: float = 20.0,
    ) -> None:
        self.restricted_zones = restricted_zones
        self.zone_threshold_s = zone_threshold_s
        self.loitering_seconds = loitering_seconds
        self.loitering_movement_px = loitering_movement_px
        self.running_px_per_s = running_px_per_s
        self.crowd_threshold = crowd_threshold
        self.abandoned_seconds = abandoned_seconds
        self.person_history: Dict[int, Dict[str, object]] = {}
        self.object_history: Dict[int, Dict[str, object]] = {}

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.dist(a, b)

    def analyze(self, tracked: List[TrackedObject], now: datetime) -> List[BehaviorAlert]:
        alerts: List[BehaviorAlert] = []
        people = [t for t in tracked if t.label == "person"]
        bags = [t for t in tracked if t.label in {"backpack", "handbag", "suitcase"}]

        if len(people) > self.crowd_threshold:
            alerts.append(BehaviorAlert("crowd_alert", "scene", f"Crowd density high: {len(people)} people detected"))

        for person in people:
            hist = self.person_history.setdefault(
                person.track_id,
                {
                    "positions": [],
                    "first_seen": now,
                    "last_seen": now,
                    "speed": 0.0,
                    "time_in_frame": 0.0,
                    "distance_traveled": 0.0,
                    "zone_entry_state": {},
                },
            )

            positions = hist["positions"]
            positions.append((now, person.center))
            hist["last_seen"] = now
            hist["time_in_frame"] = (now - hist["first_seen"]).total_seconds()

            if len(positions) >= 2:
                t0, p0 = positions[-2]
                dt = max((now - t0).total_seconds(), 1e-6)
                dist = self._distance(p0, person.center)
                hist["speed"] = dist / dt
                hist["distance_traveled"] += dist
                if hist["speed"] >= self.running_px_per_s:
                    alerts.append(
                        BehaviorAlert("running", person.track_id, f"Person {person.track_id} moving at high speed ({hist['speed']:.1f} px/s)")
                    )

            inside = check_restricted_zones(person.center, self.restricted_zones)
            zone_state = hist["zone_entry_state"]
            for zone_id in inside:
                if zone_id not in zone_state:
                    zone_state[zone_id] = now
                else:
                    zone_time = (now - zone_state[zone_id]).total_seconds()
                    if zone_time >= self.zone_threshold_s:
                        alerts.append(
                            BehaviorAlert(
                                "restricted_zone_entry",
                                person.track_id,
                                f"Person {person.track_id} in restricted zone {zone_id} for {zone_time:.1f}s",
                            )
                        )
            for zone_id in list(zone_state.keys()):
                if zone_id not in inside:
                    zone_state.pop(zone_id, None)

            total_time = hist["time_in_frame"]
            if total_time >= self.loitering_seconds and hist["distance_traveled"] <= self.loitering_movement_px:
                alerts.append(
                    BehaviorAlert(
                        "loitering",
                        person.track_id,
                        f"Person {person.track_id} loitering ({total_time:.1f}s, movement {hist['distance_traveled']:.1f}px)",
                    )
                )

        # Abandoned object heuristic: bag is stationary and no nearby person for threshold
        for bag in bags:
            hist = self.object_history.setdefault(
                bag.track_id,
                {
                    "first_seen": now,
                    "last_seen": now,
                    "last_pos": bag.center,
                    "stationary_since": now,
                    "last_near_person": now,
                },
            )
            moved = self._distance(hist["last_pos"], bag.center)
            hist["last_seen"] = now
            hist["last_pos"] = bag.center
            if moved > 10:
                hist["stationary_since"] = now

            near_person = any(self._distance(bag.center, p.center) <= 100 for p in people)
            if near_person:
                hist["last_near_person"] = now

            stationary_time = (now - hist["stationary_since"]).total_seconds()
            unattended_time = (now - hist["last_near_person"]).total_seconds()
            if stationary_time >= self.abandoned_seconds and unattended_time >= self.abandoned_seconds:
                alerts.append(
                    BehaviorAlert(
                        "abandoned_object",
                        bag.track_id,
                        f"Bag {bag.track_id} appears abandoned ({unattended_time:.1f}s unattended)",
                    )
                )

        return alerts
