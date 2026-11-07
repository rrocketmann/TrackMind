"""Upcoming road shape, in the same numbers on every track.

The editor builds a track from straights, curves, and slopes. This module
walks those pieces in driving order and turns the next stretch into a fixed
list: how long it is, which way it bends, how steep it is, and how far the
road extends to the left and to the right of the racing line. Absolute map
coordinates stay out of this list, so a different track uses the same inputs.

A sharp bend is its own short piece. The turn sits on that piece, and the
straight after it does not keep the turn. Left is positive, the same side as
the car's +X. Uphill is positive.
"""

from __future__ import annotations

import heapq
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

# Eight pieces is enough to see a straight, the next corner, and the slope
# after it. Each piece is length, turn, slope, left side, and right side.
# The live run leaves this at 0.
ROAD_AHEAD = 8
ROAD_STRIDE = 5
_LEN_SCALE = 50.0
_SIDE_SCALE = 10.0
_SHARP = 0.35  # radians. Steeper than this is a corner, not a gentle curve.
_SLOPE_SPLIT = 0.12
_CORNER_M = 1.0
_PILLAR = 7
_SIGN = 8
_YAW180 = (0.0, 1.0, 0.0, 0.0)


def road_width(window: int) -> int:
    count = int(window)
    if count < 0:
        raise ValueError("road_window must be >= 0")
    return count * ROAD_STRIDE


def road_window_of(cfg: dict) -> int:
    """How many upcoming pieces the policy reads. 0 is the current observation."""
    window = int((cfg.get("rl") or {}).get("road_window") or 0)
    if window < 0:
        raise ValueError("rl.road_window must be >= 0")
    return window


class RoadSegment:
    """One straight, curve, or slope along the racing line."""

    __slots__ = ("s0", "s1", "turn", "slope", "left", "right")

    def __init__(self, s0: float, s1: float, turn: float, slope: float, left: float = 0.0, right: float = 0.0):
        self.s0 = float(s0)
        self.s1 = float(s1)
        self.turn = float(turn)
        self.slope = float(slope)
        self.left = float(left)
        self.right = float(right)

    @property
    def length(self) -> float:
        return self.s1 - self.s0


class RoadCourse:
    """A racing line reduced to straights, curves, and slopes."""

    def __init__(
        self,
        points: list[tuple[float, float, float]],
        gates: list[tuple[float, float, float]] | None = None,
        finish: tuple[float, float, float] | None = None,
        surface: dict | None = None,
    ):
        self.points = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
        # (column, row) of a 5 m cell -> road height in metres. Empty on a
        # point-only line; height then falls back to the line itself.
        self.surface: dict[tuple[int, int], float] = {}
        for key, height in (surface or {}).items():
            self.surface[(int(key[0]), int(key[1]))] = float(height)
        self._s = _point_arc(self.points)
        self.segments = _compile(self.points, self._s)
        self.gate_s = [self._at(*gate) for gate in (gates or [])]
        self.finish_s = None if finish is None else self._at(*finish)
        self.gate_count = len(self.gate_s)

    @property
    def length(self) -> float:
        if not self._s:
            return 0.0
        return float(self._s[-1])

    def ahead(
        self,
        x: float,
        y: float,
        z: float,
        count: int,
        checkpoint: int | None = None,
    ) -> np.ndarray:
        """The next `count` pieces from where the car is. Missing pieces are 0."""
        width = road_width(count)
        out = np.zeros(width, dtype=np.float32)
        if count < 1 or not self.segments:
            return out
        place = self._at(float(x), float(y), float(z), checkpoint)
        if place is None:
            return out
        cursor = 0
        for segment in self.segments:
            if segment.s1 <= place + 1e-3:
                continue
            remaining = segment.s1 - max(place, segment.s0)
            if remaining < 1e-3:
                continue
            fraction = min(1.0, remaining / max(segment.length, 1e-3))
            out[cursor] = np.float32(remaining / _LEN_SCALE)
            out[cursor + 1] = np.float32(segment.turn * fraction / math.pi)
            out[cursor + 2] = np.float32(max(-2.0, min(2.0, segment.slope)))
            out[cursor + 3] = np.float32(segment.left / _SIDE_SCALE)
            out[cursor + 4] = np.float32(segment.right / _SIDE_SCALE)
            cursor += ROAD_STRIDE
            if cursor >= width:
                break
        return out

    def _at(
        self,
        x: float,
        y: float,
        z: float,
        checkpoint: int | None = None,
    ) -> float | None:
        if len(self.points) < 2:
            return None
        lo, hi = self._span(checkpoint)
        best_s = None
        best_d = 1e18
        for index in range(len(self.points) - 1):
            s0 = self._s[index]
            s1 = self._s[index + 1]
            if s1 < lo - 1.0 or s0 > hi + 1.0:
                continue
            t, distance = _project(x, y, z, self.points[index], self.points[index + 1])
            if distance < best_d:
                best_d = distance
                best_s = s0 + t * (s1 - s0)
        if best_s is None and checkpoint is not None:
            return self._at(x, y, z, None)
        if best_s is None:
            return None
        return max(0.0, min(self.length, best_s))

    def _span(self, checkpoint: int | None) -> tuple[float, float]:
        if checkpoint is None or (not self.gate_s and self.finish_s is None):
            return 0.0, self.length
        index = max(0, int(checkpoint))
        if index <= 0:
            previous = 0.0
        elif index - 1 < len(self.gate_s):
            previous = self.gate_s[index - 1]
        else:
            previous = self.gate_s[-1] if self.gate_s else 0.0
        if index < len(self.gate_s):
            nxt = self.gate_s[index]
        elif self.finish_s is not None:
            nxt = self.finish_s
        else:
            nxt = self.length
        return max(0.0, previous - 20.0), min(self.length, max(nxt, previous) + 80.0)

    def car_offsets(
        self,
        x: float,
        y: float,
        z: float,
        checkpoint: int | None = None,
    ) -> tuple[float, float]:
        """Metres the chassis is above the road, and metres to the line's left.

        Left is the road's left, the same side as the car's +X when the car
        faces along the line. Facing +z, left is +x, so a car on the +x side
        is positive. The cell under the car supplies the height. A missing
        cell uses the racing line's height. No line leaves both at 0.
        """
        x, y, z = float(x), float(y), float(z)
        nearest = self._nearest_pose(x, y, z, checkpoint)
        road_y = self.surface.get((math.floor(x / 5.0), math.floor(z / 5.0)))
        if road_y is None and nearest is not None:
            road_y = nearest[1]
        above = 0.0 if road_y is None else y - road_y
        if nearest is None:
            return above, 0.0
        px, _py, pz, fx, fz = nearest
        if fx == 0.0 and fz == 0.0:
            return above, 0.0
        # up × forward. Forward +z gives left +x.
        return above, (x - px) * fz - (z - pz) * fx

    def _nearest_pose(
        self,
        x: float,
        y: float,
        z: float,
        checkpoint: int | None,
    ) -> tuple[float, float, float, float, float] | None:
        """Closest point on the line, plus its flat forward (fx, fz)."""
        if len(self.points) < 2:
            return None
        lo, hi = self._span(checkpoint)
        best = None
        best_d = 1e18
        for index in range(len(self.points) - 1):
            s0 = self._s[index]
            s1 = self._s[index + 1]
            if s1 < lo - 1.0 or s0 > hi + 1.0:
                continue
            a = self.points[index]
            b = self.points[index + 1]
            t, distance = _project(x, y, z, a, b)
            if distance >= best_d:
                continue
            best_d = distance
            px = a[0] + t * (b[0] - a[0])
            py = a[1] + t * (b[1] - a[1])
            pz = a[2] + t * (b[2] - a[2])
            dx, dz = b[0] - a[0], b[2] - a[2]
            flat = math.hypot(dx, dz)
            if flat < 1e-6:
                fx, fz = 0.0, 0.0
            else:
                fx, fz = dx / flat, dz / flat
            best = (px, py, pz, fx, fz)
        if best is None and checkpoint is not None:
            return self._nearest_pose(x, y, z, None)
        return best


def side_clearance(
    surface: dict,
    x: float,
    y: float,
    z: float,
    dx: float,
    dz: float,
) -> tuple[float, float]:
    """Metres of road to the driver's left and right of one point.

    `dx, dz` is the way the road points. Facing +z, left is +x, the same side
    as the car's body +X. The walk stops at the first point that is off the
    surface, so the number is the distance to that edge.
    """
    flat = math.hypot(dx, dz)
    if flat < 1e-6:
        return 0.0, 0.0
    fx, fz = dx / flat, dz / flat
    # up × forward. Forward +z gives left +x.
    left_x, left_z = fz, -fx
    return (
        _reach(surface, x, y, z, left_x, left_z, fx, fz),
        _reach(surface, x, y, z, -left_x, -left_z, fx, fz),
    )


def _reach(
    surface: dict,
    x: float,
    y: float,
    z: float,
    sx: float,
    sz: float,
    fx: float,
    fz: float,
) -> float:
    step = 0.5
    last = 0.0
    travelled = 0.0
    while travelled < 25.0:
        travelled += step
        px = x + sx * travelled
        pz = z + sz * travelled
        if abs((px - x) * fx + (pz - z) * fz) > 5.0:
            break
        cell = surface.get((math.floor(px / 5.0), math.floor(pz / 5.0)))
        if cell is None or abs(cell[2] - y) > 8.0:
            break
        last = travelled
    if last <= 0.0:
        return 0.0
    return last + step * 0.5


def _measure_sides(course: RoadCourse, surface: dict) -> None:
    """Tightest left and right clearance along each piece, in metres."""
    points = course.points
    arc = course._s
    for segment in course.segments:
        lefts: list[float] = []
        rights: list[float] = []
        for index in range(len(points) - 1):
            s0 = arc[index]
            s1 = arc[index + 1]
            if s1 < segment.s0 + 1e-3 or s0 > segment.s1 - 1e-3:
                continue
            x0, y0, z0 = points[index]
            x1, y1, z1 = points[index + 1]
            left, right = side_clearance(surface, (x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5, x1 - x0, z1 - z0)
            if left > 0.0 or right > 0.0:
                lefts.append(left)
                rights.append(right)
        if lefts:
            segment.left = min(lefts)
            segment.right = min(rights)


def course_from_parts(parts: list) -> RoadCourse:
    """Racing line through one placed track. Unknown piece ids are skipped."""
    catalog = _catalog()
    surface: dict[tuple[int, int], tuple[int, float, float, float]] = {}
    labels: dict[tuple[int, int], tuple[int, str | None]] = {}
    starts: list[tuple[int, int, int, int, int, tuple[float, float, float]]] = []
    for raw in parts or []:
        parsed = _parse_part(raw)
        if parsed is None:
            continue
        gx, gy, gz, kind, rotation, axis, order = parsed
        shape = catalog.get(kind)
        if shape is None or shape["category"] in (_PILLAR, _SIGN):
            continue
        if shape["start"] is not None:
            starts.append((gx, gy, gz, rotation, axis, tuple(shape["start"])))
        for tx, ty, tz in _expand(shape["boxes"], rotation, axis):
            key = (gx + tx, gz + tz)
            height = gy + ty
            current = surface.get(key)
            if current is not None and current[0] > height:
                continue
            surface[key] = (height, (key[0] + 0.5) * 5.0, height * 5.0, (key[1] + 0.5) * 5.0)
            marker = shape["detector"]
            previous = labels.get(key)
            if previous is None or marker is not None:
                labels[key] = (order, marker)
    if not surface or not starts:
        return RoadCourse([])
    spawn, forward = _spawn(starts[-1])
    origin = min(surface, key=lambda key: _dist2(surface[key][1:], spawn))
    goals = _goals(surface, labels)
    path = _racing_line(origin, goals, surface, forward)
    if len(path) < 2:
        return RoadCourse([])
    points = _simplify(_recenter(path, surface))
    gates = [_world(cell, surface) for cell, kind in goals if kind == "checkpoint"]
    finishes = [_world(cell, surface) for cell, kind in goals if kind == "finish"]
    finish = finishes[-1] if finishes else None
    heights = {key: value[2] for key, value in surface.items()}
    course = RoadCourse(points, gates=gates, finish=finish, surface=heights)
    _measure_sides(course, surface)
    return course


def rotate_tile(x: int, y: int, z: int, rotation: int, axis: int) -> tuple[int, int, int]:
    """The worker's tile rotation. `rotated` in simulation_worker.bundle.js."""
    negative = axis in (1, 3, 5)
    if negative:
        if rotation == 1:
            x, z = -z - 1, x
        elif rotation == 2:
            x, z = -x - 1, -z - 1
        elif rotation == 3:
            x, z = z, -x - 1
        elif rotation != 0:
            raise ValueError(f"rotation must be 0..3, got {rotation}")
    else:
        if rotation == 1:
            x, z = z, -x - 1
        elif rotation == 2:
            x, z = -x - 1, -z - 1
        elif rotation == 3:
            x, z = -z - 1, x
        elif rotation != 0:
            raise ValueError(f"rotation must be 0..3, got {rotation}")
    if axis == 1:
        x, y = -x - 1, -y - 1
    elif axis == 2:
        x, y = y, -x - 1
    elif axis == 3:
        x, y = -y - 1, x
    elif axis == 4:
        y, z = -z - 1, y
    elif axis == 5:
        y, z = z, -y - 1
    elif axis != 0:
        raise ValueError(f"axis must be 0..5, got {axis}")
    return x, y, z


def _parse_part(raw) -> tuple[int, int, int, int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 7:
        return None
    try:
        gx, gy, gz = int(raw[0]), int(raw[1]), int(raw[2])
        kind, rotation, axis, order = int(raw[3]), int(raw[4]), int(raw[5]), int(raw[6])
    except (TypeError, ValueError):
        return None
    if rotation not in (0, 1, 2, 3) or axis not in range(6):
        return None
    return gx, gy, gz, kind, rotation, axis, order


def _point_arc(points: list[tuple[float, float, float]]) -> list[float]:
    arc = [0.0]
    for index in range(len(points) - 1):
        x0, y0, z0 = points[index]
        x1, y1, z1 = points[index + 1]
        arc.append(arc[-1] + _flat(x1 - x0, y1 - y0, z1 - z0))
    return arc


def _flat(dx: float, dy: float, dz: float) -> float:
    flat = math.hypot(dx, dz)
    if flat < 1e-3 and abs(dy) < 1e-3:
        return 0.0
    return flat if flat >= 1e-3 else abs(dy)


def _compile(points: list[tuple[float, float, float]], arc: list[float]) -> list[RoadSegment]:
    """Steps between points, with a sharp bend pulled out as its own metre."""
    pieces: list[tuple[float, float, float, float]] = []
    heading = None
    for index in range(len(points) - 1):
        x0, y0, z0 = points[index]
        x1, y1, z1 = points[index + 1]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        length = arc[index + 1] - arc[index]
        if length < 1e-3:
            continue
        flat = math.hypot(dx, dz)
        new_heading = math.atan2(dx, dz) if flat >= 1e-3 else heading
        if new_heading is None:
            new_heading = 0.0
        turn = 0.0 if heading is None else _wrap(new_heading - heading)
        heading = new_heading
        start = arc[index]
        if abs(turn) > _SHARP:
            corner = min(_CORNER_M, length)
            pieces.append((start, corner, turn, dy * corner / length))
            rest = length - corner
            if rest > 0.25:
                pieces.append((start + corner, rest, 0.0, dy * rest / length))
        else:
            pieces.append((start, length, turn, dy))
    return _merge(pieces)


def _merge(pieces: list[tuple[float, float, float, float]]) -> list[RoadSegment]:
    segments: list[RoadSegment] = []
    buf: list[tuple[float, float, float, float]] = []

    def flush() -> None:
        if not buf:
            return
        length = sum(item[1] for item in buf)
        turn = sum(item[2] for item in buf)
        dy = sum(item[3] for item in buf)
        slope = 0.0 if length < 1e-3 else dy / length
        segments.append(RoadSegment(buf[0][0], buf[0][0] + length, turn, slope))
        buf.clear()

    for piece in pieces:
        _start, length, turn, dy = piece
        if length < 1e-3:
            continue
        if not buf:
            buf.append(piece)
            continue
        if abs(turn) > _SHARP or abs(buf[0][2]) > _SHARP:
            flush()
            buf.append(piece)
            continue
        buf_len = sum(item[1] for item in buf)
        buf_dy = sum(item[3] for item in buf)
        buf_turn = sum(item[2] for item in buf)
        step_slope = dy / length
        buf_slope = 0.0 if buf_len < 1e-3 else buf_dy / buf_len
        if abs(step_slope - buf_slope) > _SLOPE_SPLIT:
            flush()
            buf.append(piece)
            continue
        if abs(buf_turn + turn) > math.pi and abs(buf_turn) > 0.2:
            flush()
            buf.append(piece)
            continue
        # A step under about 3° keeps a straight straight. A larger step of the
        # same sign keeps a curve one piece, so a bend is not a pile of corners.
        same_curve = buf_turn * turn > 0.0 and abs(turn) >= 0.05 and abs(buf_turn) >= 0.05
        both_straight = abs(buf_turn) < 0.2 and abs(turn) < 0.05
        if same_curve or both_straight:
            buf.append(piece)
            continue
        flush()
        buf.append(piece)
    flush()
    return segments


def _expand(boxes: list, rotation: int, axis: int):
    for box in boxes:
        (x0, y0, z0), (x1, y1, z1) = box
        for x in range(int(x0), int(x1) + 1):
            for y in range(int(y0), int(y1) + 1):
                for z in range(int(z0), int(z1) + 1):
                    yield rotate_tile(x, y, z, rotation, axis)


def _spawn(
    start: tuple[int, int, int, int, int, tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """World position and forward on a start piece. The body's +Z is forward."""
    from polytrack_agent.rl.track import _PART_QUAT, _rotate

    gx, gy, gz, rotation, axis, offset = start
    quat = _mul(_PART_QUAT[axis][rotation], _YAW180)
    ox, oy, oz = _rotate(quat, [offset[0], offset[1], offset[2]])
    forward = _rotate(quat, [0.0, 0.0, 1.0])
    return (gx * 5.0 + ox, gy * 5.0 + oy, gz * 5.0 + oz), forward


def _mul(
    q: tuple[float, float, float, float],
    p: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    px, py, pz, pw = p
    return (
        qw * px + qx * pw + qy * pz - qz * py,
        qw * py - qx * pz + qy * pw + qz * px,
        qw * pz + qx * py - qy * px + qz * pw,
        qw * pw - qx * px - qy * py - qz * pz,
    )


def _goals(surface: dict, labels: dict) -> list[tuple[tuple[int, int], str]]:
    buckets: dict[int, list[tuple[int, int]]] = {}
    finishes: list[tuple[int, int]] = []
    for key, (order, marker) in labels.items():
        if key not in surface:
            continue
        if marker == "finish":
            finishes.append(key)
        elif marker == "checkpoint" and order >= 0:
            buckets.setdefault(order, []).append(key)
    goals: list[tuple[tuple[int, int], str]] = []
    for order in sorted(buckets):
        goals.append((_centroid_cell(buckets[order], surface), "checkpoint"))
    if finishes:
        goals.append((_centroid_cell(finishes, surface), "finish"))
    return goals


def _centroid_cell(cells: list[tuple[int, int]], surface: dict) -> tuple[int, int]:
    sx = sum(surface[key][1] for key in cells) / len(cells)
    sz = sum(surface[key][3] for key in cells) / len(cells)
    return min(cells, key=lambda key: (surface[key][1] - sx) ** 2 + (surface[key][3] - sz) ** 2)


def _world(cell: tuple[int, int], surface: dict) -> tuple[float, float, float]:
    _height, x, y, z = surface[cell]
    return (x, y, z)


def _racing_line(origin, goals, surface, forward) -> list[tuple[int, int]]:
    graph = _graph(surface)
    if not goals:
        return _walk_forward(origin, graph, surface, forward)
    path = [origin]
    for cell, _kind in goals:
        leg = _shortest(path[-1], cell, graph)
        if leg is None:
            # A jump, or a ramp whose tiles do not touch, leaves a short gap.
            # Bridge it only when the next gate is on the far side.
            leg = _bridge(path[-1], cell, graph, surface)
        if not leg:
            break
        path.extend(leg[1:])
    # Keep the walk back from a gate. On Summer 1 the third gate is the end of
    # a branch, and the way on is the same road in the other direction. Cutting
    # that branch drops the gate off the line.
    return path


def _graph(surface: dict) -> dict[tuple[int, int], list[tuple[tuple[int, int], float]]]:
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {key: [] for key in surface}
    for (ix, iz), (iy, _x, _y, _z) in surface.items():
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                other = (ix + dx, iz + dz)
                cell = surface.get(other)
                if cell is None or abs(cell[0] - iy) > 2:
                    continue
                rise = (cell[0] - iy) * 5.0
                graph[(ix, iz)].append((other, math.hypot(math.hypot(dx * 5.0, dz * 5.0), rise)))
    return graph


def _shortest(start, goal, graph) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    dist = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost != dist.get(node):
            continue
        if node == goal:
            break
        for nxt, step in graph.get(node, []):
            updated = cost + step
            if updated < dist.get(nxt, 1e18):
                dist[nxt] = updated
                prev[nxt] = node
                heapq.heappush(heap, (updated, nxt))
    if goal not in dist:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _bridge(start, goal, graph, surface) -> list[tuple[int, int]] | None:
    """Cross a gap of at most 40 m when the next gate is not on this road piece."""
    reached = _reachable(start, graph)
    if goal in reached:
        return _shortest(start, goal, graph)
    goal_side = _reachable(goal, graph)
    best = None
    for here in reached:
        ix, iz = here
        hy = surface[here][0]
        for dx in range(-8, 9):
            for dz in range(-8, 9):
                if dx == 0 and dz == 0:
                    continue
                other = (ix + dx, iz + dz)
                if other not in goal_side:
                    continue
                cell = surface[other]
                rise = (cell[0] - hy) * 5.0
                gap = math.hypot(dx * 5.0, dz * 5.0, rise)
                if gap > 40.0:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, here, other)
    if best is None:
        return None
    _gap, here, other = best
    head = _shortest(start, here, graph) or [start]
    tail = _shortest(other, goal, graph) or [other]
    return head + tail


def _reachable(start, graph) -> set[tuple[int, int]]:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt, _cost in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _simplify(points: list[tuple[float, float, float]], epsilon: float = 3.0) -> list[tuple[float, float, float]]:
    """Drop tile-grid wobble. A real bend sticks out farther than `epsilon` metres."""
    if len(points) < 3:
        return points
    ax, ay, az = points[0]
    bx, by, bz = points[-1]
    abx, aby, abz = bx - ax, by - ay, bz - az
    ab2 = abx * abx + aby * aby + abz * abz
    worst = 0.0
    index = 0
    for cursor in range(1, len(points) - 1):
        px, py, pz = points[cursor]
        if ab2 < 1e-8:
            distance = math.dist(points[cursor], points[0])
        else:
            t = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / ab2
            t = max(0.0, min(1.0, t))
            distance = math.dist(points[cursor], (ax + t * abx, ay + t * aby, az + t * abz))
        if distance > worst:
            worst = distance
            index = cursor
    if worst <= epsilon:
        return [points[0], points[-1]]
    left = _simplify(points[: index + 1], epsilon)
    right = _simplify(points[index:], epsilon)
    return left[:-1] + right


def _walk_forward(origin, graph, surface, forward) -> list[tuple[int, int]]:
    """Follow the road in the direction the car faces when the track has no gates."""
    path = [origin]
    seen = {origin}
    fx, fz = forward[0], forward[2]
    scale = math.hypot(fx, fz) or 1.0
    fx /= scale
    fz /= scale
    while True:
        best = None
        best_dot = 0.15
        ix, iz = path[-1]
        for nxt, _cost in graph.get(path[-1], []):
            if nxt in seen:
                continue
            dx = surface[nxt][1] - surface[(ix, iz)][1]
            dz = surface[nxt][3] - surface[(ix, iz)][3]
            dot = (dx * fx + dz * fz) / (math.hypot(dx, dz) or 1.0)
            if dot > best_dot:
                best_dot = dot
                best = nxt
        if best is None:
            return path
        seen.add(best)
        dx = surface[best][1] - surface[(ix, iz)][1]
        dz = surface[best][3] - surface[(ix, iz)][3]
        scale = math.hypot(dx, dz) or 1.0
        path.append(best)
        fx, fz = dx / scale, dz / scale


def _recenter(path: list[tuple[int, int]], surface: dict) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for index, key in enumerate(path):
        prev = path[max(0, index - 1)]
        nxt = path[min(len(path) - 1, index + 1)]
        tx = nxt[0] - prev[0]
        tz = nxt[1] - prev[1]
        norm = math.hypot(tx, tz) or 1.0
        tx /= norm
        tz /= norm
        px, pz = -tz, tx
        height = surface[key][0]
        ix, iz = key
        chosen: list[tuple[float, float, float]] = []
        for dx in range(-5, 6):
            for dz in range(-5, 6):
                cell = surface.get((ix + dx, iz + dz))
                if cell is None or abs(cell[0] - height) > 1:
                    continue
                along = dx * tx + dz * tz
                side = dx * px + dz * pz
                if abs(along) <= 0.75 and abs(side) <= 4.5:
                    chosen.append((cell[1], cell[2], cell[3]))
        if not chosen:
            cell = surface[key]
            chosen = [(cell[1], cell[2], cell[3])]
        count = float(len(chosen))
        points.append((
            sum(item[0] for item in chosen) / count,
            sum(item[1] for item in chosen) / count,
            sum(item[2] for item in chosen) / count,
        ))
    kept = [points[0]]
    for point in points[1:]:
        if math.hypot(point[0] - kept[-1][0], point[2] - kept[-1][2]) >= 2.0:
            kept.append(point)
    return kept


def _project(
    x: float,
    y: float,
    z: float,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float]:
    ax, ay, az = a
    bx, by, bz = b
    abx, aby, abz = bx - ax, by - ay, bz - az
    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 < 1e-8:
        return 0.0, math.dist((x, y, z), a)
    t = ((x - ax) * abx + (y - ay) * aby + (z - az) * abz) / ab2
    t = max(0.0, min(1.0, t))
    cx, cy, cz = ax + t * abx, ay + t * aby, az + t * abz
    return t, math.dist((x, y, z), (cx, cy, cz))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _dist2(point, other) -> float:
    return (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 + (point[2] - other[2]) ** 2


@lru_cache(maxsize=1)
def _catalog() -> dict[int, dict]:
    path = Path(__file__).with_name("part_shapes.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["id"]): row for row in rows}
