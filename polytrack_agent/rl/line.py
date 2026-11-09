"""The fastest archived lap, as a polyline the reward can follow.

Position in a TrackData row is scaled by 200. The cursor only looks a short
distance ahead, so a shortcut across the map cannot collect the rest of the lap.
"""

from __future__ import annotations

import math

import numpy as np

from polytrack_agent.rl.state import RECORDED, RECORDED_FEATURES, _POS

_X = RECORDED.index("x")
_Z = RECORDED.index("z")


class ReferenceLine:
    """World-space (x, z) of one clean lap, with arc length in metres."""

    def __init__(self, points: list[tuple[float, float]], *, seconds: float, track: str):
        cleaned: list[tuple[float, float]] = []
        for x, z in points:
            point = (float(x), float(z))
            if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) < 0.25:
                continue
            cleaned.append(point)
        if len(cleaned) < 2:
            raise ValueError("reference lap needs at least two positions")
        self.points = np.asarray(cleaned, dtype=np.float64)
        segments = np.hypot(np.diff(self.points[:, 0]), np.diff(self.points[:, 1]))
        self.arc = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(segments)])
        self.seconds = float(seconds)
        self.track = str(track)

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    @property
    def count(self) -> int:
        return int(self.points.shape[0])

    def project(
        self,
        x: float,
        z: float,
        cursor: int,
        back_m: float,
        forward_m: float,
    ) -> tuple[float, float, int]:
        """Closest point inside a window around the cursor.

        Returns metres along the lap, metres off to the side, and the new cursor.
        The window is what stops a jump from scoring the whole lap.
        """
        cursor = int(min(max(int(cursor), 0), self.count - 1))
        here = float(self.arc[cursor])
        low = here - max(0.0, float(back_m))
        high = here + max(0.0, float(forward_m))
        start = max(0, int(np.searchsorted(self.arc, low, side="right")) - 1)
        stop = int(np.searchsorted(self.arc, high, side="left"))
        stop = min(self.count - 1, max(stop, start + 1))
        best_d2 = math.inf
        best_along = here
        best_lateral = 0.0
        best_cursor = cursor
        for index in range(start, stop):
            ax, az = self.points[index]
            bx, bz = self.points[index + 1]
            dx = float(bx - ax)
            dz = float(bz - az)
            seg_lo = float(self.arc[index])
            seg_hi = float(self.arc[index + 1])
            clip_lo = max(seg_lo, low)
            clip_hi = min(seg_hi, high)
            if clip_hi < clip_lo:
                continue
            length2 = dx * dx + dz * dz
            length = math.sqrt(length2)
            if length2 < 1e-8:
                fraction = 0.0
                along = seg_lo
            else:
                fraction = ((float(x) - ax) * dx + (float(z) - az) * dz) / length2
                fraction = min(1.0, max(0.0, fraction))
                along = min(clip_hi, max(clip_lo, seg_lo + fraction * length))
                fraction = (along - seg_lo) / length
            px = ax + fraction * dx
            pz = az + fraction * dz
            lateral2 = (float(x) - px) ** 2 + (float(z) - pz) ** 2
            if lateral2 < best_d2:
                best_d2 = lateral2
                best_along = along
                best_lateral = math.sqrt(lateral2)
                best_cursor = index if fraction < 1.0 else min(index + 1, self.count - 1)
        return best_along, best_lateral, best_cursor


def reference_line(rows: list, *, seconds: float, track: str) -> ReferenceLine:
    """Build the lap from recorded rows. `n[0]` and `n[2]` are x and z over 200."""
    points: list[tuple[float, float]] = []
    for row in rows:
        recorded = row.get("n") if isinstance(row, dict) else None
        if not isinstance(recorded, list) or len(recorded) != RECORDED_FEATURES:
            continue
        points.append((float(recorded[_X]) * _POS, float(recorded[_Z]) * _POS))
    return ReferenceLine(points, seconds=float(seconds), track=track)


def ready_for_time(stage: str, finished: bool, line_progress: float, close: float) -> bool:
    """True once the reference stage has a finish or nearly the whole lap."""
    if stage != "line":
        return False
    if finished:
        return True
    return float(line_progress) >= float(close)
