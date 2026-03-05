from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def point_in_polygon(point: Tuple[int, int], polygon: Sequence[Tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersect = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1
        )
        if intersect:
            inside = not inside
    return inside


def check_restricted_zones(
    center: Tuple[int, int], restricted_zones: Iterable[Sequence[Tuple[int, int]]]
) -> List[int]:
    inside_zone_ids: List[int] = []
    for idx, polygon in enumerate(restricted_zones):
        if point_in_polygon(center, polygon):
            inside_zone_ids.append(idx)
    return inside_zone_ids
