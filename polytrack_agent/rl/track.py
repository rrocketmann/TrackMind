"""One car in the PolyTrack 0.6.3 fork simulation.

The Node host steps the game's own worker. This class turns a chord into
up/down/left/right and scores progress from the car state the worker returns.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from polytrack_agent.rl.actions import action_masks, mask_for
from polytrack_agent.sim.client import SimClient

# Part orientation table from the 0.5.2 worker (`Yh[axis][rotation]`), xyzw.
_PART_QUAT = (
    ((0.0, 0.0, 0.0, 1.0), (0.0, 0.7071067811865475, 0.0, 0.7071067811865476), (0.0, 1.0, 0.0, 0.0), (0.0, 0.7071067811865476, 0.0, -0.7071067811865475)),
    ((0.0, 0.0, 1.0, 0.0), (0.7071067811865475, 0.0, 0.7071067811865476, 0.0), (1.0, 0.0, 0.0, 0.0), (0.7071067811865476, 0.0, -0.7071067811865475, 0.0)),
    ((0.0, 0.0, -0.7071067811865477, 0.7071067811865475), (0.5, 0.5, -0.5, 0.5), (0.7071067811865475, 0.7071067811865477, 0.0, 0.0), (0.5, 0.5, 0.5, -0.5)),
    ((0.0, 0.0, 0.7071067811865475, 0.7071067811865476), (0.5, -0.5, 0.5, 0.5), (0.7071067811865476, -0.7071067811865475, 0.0, 0.0), (0.5, -0.5, -0.5, -0.5)),
    ((0.7071067811865475, 0.0, 0.0, 0.7071067811865476), (0.5, 0.5, 0.5, 0.5), (0.0, 0.7071067811865476, 0.7071067811865475, 0.0), (-0.5, 0.5, 0.5, -0.5)),
    ((-0.7071067811865477, 0.0, 0.0, 0.7071067811865475), (-0.5, -0.5, 0.5, 0.5), (0.0, -0.7071067811865475, 0.7071067811865477, 0.0), (0.5, -0.5, 0.5, -0.5)),
)
_PART_SIZE = 5.0
_ROAD_HALF = 2.5  # one grid cell. A wide road is several of these side by side.

# Pillars, walls, signs, and loose blocks share the track grid but are not road.
_NOT_ROAD = frozenset(
    {
        19, 20, 21, 22, 23, 24,
        26, 27, 28, 29,
        30, 31, 32,
        40,
        45, 46, 47, 48,
        49, 50, 51,
        53, 54,
        58, 59, 61, 62, 63, 64,
        66,
        79, 80, 81,
        84,
        99, 100,
        *range(101, 117),
        119, 120, 121, 122, 123, 124, 125, 126,
        134, 135,
    }
)


def _rotate(quat: tuple[float, float, float, float], vector: list[float]) -> tuple[float, float, float]:
    x, y, z, w = quat
    vx, vy, vz = (float(vector[0]), float(vector[1]), float(vector[2]))
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _mean(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _detectors(cfg: dict) -> dict[int, dict]:
    game = cfg.get("game") or {}
    parts = Path(game.get("parts") or Path(__file__).resolve().parents[1] / "sim" / "cache" / "track_parts.json")
    catalog = json.loads(parts.read_text(encoding="utf-8"))
    items = catalog.get("parts") if isinstance(catalog, dict) else catalog
    if not isinstance(items, list):
        raise ValueError(f"{parts} has no parts list")
    return {int(item["id"]): item.get("detector") for item in items}


def gate_points(parts: list, detectors: dict[int, dict]) -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    """World positions of each checkpoint, in order, and the finish line."""
    buckets: dict[int, list[tuple[float, float, float]]] = {}
    finishes: list[tuple[float, float, float]] = []
    for raw in parts:
        gx, gy, gz, kind, rotation, axis, order = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4]), int(raw[5]), int(raw[6]))
        quat = _PART_QUAT[axis][rotation]
        detector = detectors.get(kind) or {}
        center = detector.get("center") or [0.0, 0.0, 0.0]
        ox, oy, oz = _rotate(quat, center)
        point = (gx * _PART_SIZE + ox, gy * _PART_SIZE + oy, gz * _PART_SIZE + oz)
        if order >= 0:
            buckets.setdefault(order, []).append(point)
        elif detector.get("type") == 1:
            finishes.append(point)
    gates = [_mean(buckets[order]) for order in sorted(buckets)]
    finish = _mean(finishes) if finishes else (gates[-1] if gates else (0.0, 0.0, 0.0))
    return gates, finish


def road_cells(parts: list) -> list[tuple[float, float]]:
    """Unique drivable-tile centers on the ground plane.

    Pillars, walls, and signs are not road. A pillar under a road piece shares
    that piece's cell and is dropped when the road cell is already recorded.
    """
    seen: set[tuple[float, float]] = set()
    cells: list[tuple[float, float]] = []
    for raw in parts:
        if int(raw[3]) in _NOT_ROAD:
            continue
        cell = (int(raw[0]) * _PART_SIZE, int(raw[2]) * _PART_SIZE)
        if cell in seen:
            continue
        seen.add(cell)
        cells.append(cell)
    return cells


class _Road:
    """Grid index of road cells. Border queries stay off the per-step scan."""

    def __init__(self, cells: list[tuple[float, float]]):
        self._at: dict[tuple[int, int], tuple[float, float]] = {}
        for x, z in cells:
            self._at[(int(round(x / _PART_SIZE)), int(round(z / _PART_SIZE)))] = (float(x), float(z))
        self._along: dict[tuple[int, int], tuple[float, float]] = {
            key: self._along_of(key) for key in self._at
        }

    def _neighbors(self, key: tuple[int, int]) -> list[tuple[int, int]]:
        ix, iz = key
        return [
            nxt
            for nxt in ((ix + 1, iz), (ix - 1, iz), (ix, iz + 1), (ix, iz - 1))
            if nxt in self._at
        ]

    def _along_of(self, key: tuple[int, int]) -> tuple[float, float]:
        """Track direction: the pair of neighbors that most nearly face each other."""
        neighbors = self._neighbors(key)
        if not neighbors:
            return (1.0, 0.0)
        x, z = self._at[key]
        dirs: list[tuple[float, float]] = []
        for nxt in neighbors:
            nx, nz = self._at[nxt]
            dx, dz = nx - x, nz - z
            length = math.hypot(dx, dz) or 1.0
            dirs.append((dx / length, dz / length))
        best = 0
        best_dot = 1.0
        for i, left in enumerate(dirs):
            for right in dirs[i + 1 :]:
                dot = left[0] * right[0] + left[1] * right[1]
                if dot < best_dot:
                    best_dot = dot
                    best = i
        return dirs[best]

    def border(self, x: float, z: float) -> tuple[float, float]:
        """Meters of road left to the nearest edge, and meters to the right of the tile center.

        The along-track axis is the road chain, so the joint between two tiles
        still counts as the middle. A second tile beside the first extends the
        clearance instead of being treated as the track direction. Off the road,
        clearance is negative.
        """
        if not self._at:
            return 0.0, 0.0
        ix = int(round(x / _PART_SIZE))
        iz = int(round(z / _PART_SIZE))
        nearest = None
        best = 1e18
        for dx in range(-4, 5):
            for dz in range(-4, 5):
                key = (ix + dx, iz + dz)
                pos = self._at.get(key)
                if pos is None:
                    continue
                dist = (pos[0] - x) ** 2 + (pos[1] - z) ** 2
                if dist < best:
                    best = dist
                    nearest = key
        if nearest is None:
            for key, pos in self._at.items():
                dist = (pos[0] - x) ** 2 + (pos[1] - z) ** 2
                if dist < best:
                    best = dist
                    nearest = key
        if nearest is None:
            return 0.0, 0.0
        nx, nz = self._at[nearest]
        along_x, along_z = self._along[nearest]
        side = (x - nx) * -along_z + (z - nz) * along_x
        extra = 0.0
        if abs(side) > 1e-6:
            sign = 1.0 if side > 0.0 else -1.0
            step_x = -along_z * sign
            step_z = along_x * sign
            cx, cz = nx, nz
            for _ in range(4):
                cx += step_x * _PART_SIZE
                cz += step_z * _PART_SIZE
                if (int(round(cx / _PART_SIZE)), int(round(cz / _PART_SIZE))) not in self._at:
                    break
                extra += _PART_SIZE
        return _ROAD_HALF - abs(side) + extra, side


def road_border(cells: list[tuple[float, float]], x: float, z: float) -> tuple[float, float]:
    """Meters of road left beside the car, and meters to the right of the tile center."""
    return _Road(cells).border(x, z)


def _number(mapping: dict, key: str, default: float) -> float:
    """Config float. An explicit 0 stays 0."""
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if value is None:
        return float(default)
    return float(value)


def game_root(cfg: dict) -> Path:
    """Directory that contains the polytrack fork worker."""
    raw = os.environ.get("POLYTRACK_ROOT") or (cfg.get("game") or {}).get("root")
    if not raw:
        raise ValueError(
            "Set POLYTRACK_ROOT to the polytrack fork directory, or set game.root in --config."
        )
    root = Path(raw).expanduser().resolve()
    worker = root / "simulation_worker.bundle.js"
    if not worker.is_file():
        raise FileNotFoundError(
            f"{root} is not the PolyTrack 0.6.3 web directory. "
            "It needs simulation_worker.bundle.js."
        )
    return root


def _controls(action_id: int, keys: list[str], reset_action: bool = False) -> dict[str, bool]:
    mask = mask_for(action_id, keys, reset_action=reset_action)
    return {
        "up": bool(mask.get("w") or mask.get("up")),
        "left": bool(mask.get("a") or mask.get("left")),
        "down": bool(mask.get("s") or mask.get("down")),
        "right": bool(mask.get("d") or mask.get("right")),
        "reset": bool(mask.get("reset")),
    }


class SchoolagyTrack:
    """Gymnasium-style env. `step` holds one chord for `rl.action_repeat` physics frames."""

    def __init__(self, cfg: dict, client: SimClient | None = None):
        game = cfg["game"]
        root = game_root(cfg)
        track = Path(game["track"])
        if not track.is_absolute():
            track = root / track
        if not track.is_file():
            raise FileNotFoundError(f"Track file not found: {track}")
        parts = Path(game.get("parts") or Path(__file__).resolve().parents[1] / "sim" / "cache" / "track_parts.json")
        if not parts.is_file():
            raise FileNotFoundError(
                f"Missing physics meshes at {parts}. Run `node scripts/dump_parts.mjs` once on this machine."
            )
        self.cfg = cfg
        self.keys = list(cfg["keys"])
        self.client = client if client is not None else SimClient(root, parts, track)
        self.repeat = max(1, int(cfg["rl"].get("action_repeat") or 50))
        self.timeout_s = float(cfg["rl"].get("timeout_s") or 90)
        self.stuck_s = float(cfg["rl"].get("stuck_s") or 8)
        self.reset_action = bool(cfg["rl"].get("reset_action", False))
        self.auto_respawn = bool(cfg["rl"].get("auto_respawn", False))
        self.airborne_grace_s = float(cfg["rl"].get("airborne_grace_s") or 0.5)
        reward = cfg["reward"]
        self.progress_scale = _number(reward, "progress_scale", 1.0)
        self.time_penalty = _number(reward, "time_penalty", 0.0)
        self.finish_bonus = _number(reward, "finish_bonus", 0.0)
        self.respawn_penalty = _number(reward, "respawn_penalty", 0.0)
        self.contact_reward = _number(reward, "contact_reward", 0.0)
        self.beach_penalty = _number(reward, "beach_penalty", 0.0)
        self.stall_penalty = _number(reward, "stall_penalty", 0.0)
        self.speed_reward = _number(reward, "speed_reward", 0.0)
        self.speed_cap_kmh = _number(reward, "speed_cap_kmh", 120.0)
        self.speed_drop_penalty = _number(reward, "speed_drop_penalty", 0.0)
        self.speed_drop_margin = _number(reward, "speed_drop_margin", 0.0)
        self.fail_penalty = _number(reward, "fail_penalty", 0.0)
        self.offtrack_penalty = _number(reward, "offtrack_penalty", 0.0)
        self._state: dict = {}
        self._origin_y = 0.0
        self._origin_xz = (0.0, 0.0)
        self._progress = 0.0
        self._mark = 0.0
        self._flat = 0.0
        self._checkpoint = 0
        self._slow = 0.0
        self._airborne_s = 0.0
        self._last_xy = (0.0, 0.0)
        self._pace = 0.0
        self._beached = False
        self._respawns = 0
        self.gates, self.finish = gate_points(self.client.parts, _detectors(cfg))
        self._road = _Road(road_cells(self.client.parts))

    def _point(self, index: int) -> tuple[float, float, float]:
        if 0 <= index < len(self.gates):
            return self.gates[index]
        return self.finish

    def _view(self, state: dict) -> dict:
        viewed = dict(state)
        viewed["checkpointCount"] = self.client.checkpoint_count
        point = self._point(int(viewed["nextCheckpointIndex"]))
        viewed["target"] = {"x": point[0], "y": point[1], "z": point[2]}
        position = viewed.get("position") or {}
        clearance, side = self._road.border(
            float(position.get("x") or 0.0),
            float(position.get("z") or 0.0),
        )
        viewed["border"] = {"clearance": clearance, "side": side}
        return viewed

    def _distance(self, state: dict, index: int) -> float:
        point = self._point(index)
        position = state["position"]
        return math.hypot(point[0] - float(position["x"]), point[2] - float(position["z"]))

    def _wheels_down(self, state: dict) -> int:
        """How many wheels report contact with the track surface."""
        wheels = state.get("wheels") or []
        return sum(1 for wheel in wheels if wheel)

    def _moved(self, state: dict) -> float:
        """Horizontal distance the car actually travelled since the last step."""
        position = state["position"]
        x, z = float(position["x"]), float(position["z"])
        dx, dz = x - self._last_xy[0], z - self._last_xy[1]
        self._last_xy = (x, z)
        return math.hypot(dx, dz)

    def _ground_speed(self, state: dict) -> float:
        """Chassis speed in km/h. Falls back to the wheel figure if absent.

        `speedKmh` is derived from wheel rotation, so a car spinning its wheels
        while pinned against a wall still reads as moving. The velocity vector is
        the honest one when the worker supplies it.
        """
        velocity = state.get("velocity")
        if isinstance(velocity, dict):
            # A stopped chassis is (0, 0). Falling through to speedKmh here made
            # a car spinning its wheels against a wall look like it was moving.
            return math.hypot(float(velocity.get("x") or 0.0), float(velocity.get("z") or 0.0)) * 3.6
        return float(state["speedKmh"])

    def _motion_kmh(self, state: dict) -> tuple[float, float]:
        """Chassis speed, and speed toward the next gate, both in km/h.

        Both come from the velocity vector. Wheel spin does not count. Toward
        the gate is positive. With no chord to follow, the chassis speed is used
        for both.
        """
        velocity = state.get("velocity")
        if not isinstance(velocity, dict):
            return 0.0, 0.0
        vx = float(velocity.get("x") or 0.0)
        vz = float(velocity.get("z") or 0.0)
        ground = math.hypot(vx, vz) * 3.6
        index = int(state["nextCheckpointIndex"])
        prev, nxt = self._chord(index)
        dx = nxt[0] - prev[0]
        dz = nxt[1] - prev[1]
        length2 = dx * dx + dz * dz
        if length2 < 1.0:
            return ground, ground
        along = (vx * dx + vz * dz) / math.sqrt(length2) * 3.6
        return ground, along

    def _chord(self, index: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """The ground segment from the previous gate (or the spawn) to this gate."""
        if index <= 0:
            prev = self._origin_xz
        else:
            point = self._point(index - 1)
            prev = (point[0], point[2])
        point = self._point(index)
        return prev, (point[0], point[2])

    def _progress_of(self, state: dict, finished: bool) -> float:
        """Checkpoint index plus the fraction of the chord to that gate.

        The fraction is the projection onto the segment, clamped to the segment.
        Reaching the gate is 1. Crossing it advances the index and starts the
        next segment at 0, so the crossing pays whatever was still unpaid and
        nothing twice. Driving back along the chord lowers it.
        """
        if finished:
            return float(self.client.checkpoint_count) + 1.0
        index = int(state["nextCheckpointIndex"])
        position = state["position"]
        x = float(position["x"])
        z = float(position["z"])
        prev, nxt = self._chord(index)
        dx = nxt[0] - prev[0]
        dz = nxt[1] - prev[1]
        length2 = dx * dx + dz * dz
        if length2 < 1.0:
            covered = 1.0
        else:
            covered = ((x - prev[0]) * dx + (z - prev[1]) * dz) / length2
            covered = min(1.0, max(0.0, covered))
        self._checkpoint = index
        return float(index) + covered

    def reset(self, restart: bool = True) -> dict:
        del restart
        state = self._view(self.client.reset())
        self._state = state
        self._origin_y = float(state["position"]["y"])
        self._origin_xz = (float(state["position"]["x"]), float(state["position"]["z"]))
        self._progress = 0.0
        self._mark = 0.0
        self._flat = 0.0
        self._checkpoint = int(state["nextCheckpointIndex"])
        self._slow = 0.0
        self._airborne_s = 0.0
        self._last_xy = (float(state["position"]["x"]), float(state["position"]["z"]))
        self._pace = 0.0
        self._beached = False
        return {
            "state": state,
            "progress": 0.0,
            "dt": 0.0,
            "counted": True,
            "finished": False,
            "restarted": True,
        }

    def step(self, action_id: int):
        before = self._state
        target = int(before["frames"]) + self.repeat
        state = self._view(self.client.step(_controls(action_id, self.keys, self.reset_action), target))
        dt = max(0.0, (int(state["frames"]) - int(before["frames"])) * 0.001)
        finished = state["finishFrames"] is not None
        progress = self._progress_of(state, finished)
        # Wheel contact plus the road edge is "on the track". A car in the air
        # or past the edge can still project onto the chord, and that projection
        # must not pay. The baseline still advances, so landing does not collect
        # the jump afterwards.
        down = self._wheels_down(state)
        clearance = (state.get("border") or {}).get("clearance")
        on_track = down > 0 and (clearance is None or float(clearance) >= 0.0)
        reward = -self.time_penalty * dt
        if on_track:
            reward += (progress - self._progress) * self.progress_scale
        self._progress = progress
        # Speed toward the next gate pays on top of the chord, and only on the
        # track. A sudden loss of chassis speed is a hit either way.
        ground_kmh, along_kmh = self._motion_kmh(state)
        cap = max(1.0, self.speed_cap_kmh)
        if on_track:
            reward += self.speed_reward * min(cap, max(-cap, along_kmh)) / cap
        allowed = self.speed_drop_margin * dt
        lost = self._pace - ground_kmh
        if lost > allowed:
            reward -= self.speed_drop_penalty * (lost - allowed) / cap
        self._pace = ground_kmh
        # Speed with no chord progress is a stuck car. The first two seconds
        # are the launch, so that timer waits until the car has had a chance to move.
        if progress >= self._mark + 0.05:
            self._mark = progress
            self._flat = 0.0
        elif int(state["frames"]) >= 2000:
            self._flat += dt
        if finished:
            reward += self.finish_bonus
        if clearance is not None and float(clearance) < 0.0:
            reward -= self.offtrack_penalty * dt
        # Wheel contact is the authoritative "is the car on the track" signal. A
        # car that leaves the road lands on its chassis with every wheel in the
        # air and cannot be driven again, so staying on the surface has to pay.
        moved = self._moved(state)
        # A fixed distance threshold misfires at small `action_repeat`: a car at
        # 28 km/h covers only ~0.08 units in a 10-frame step, so it looked parked.
        # Scale the floor by the step's own duration instead.
        floor = max(0.02, 5.0 * dt / 3.6)  # 5 km/h over this step, in metres
        self._airborne_s = self._airborne_s + dt if down == 0 else 0.0
        if on_track and moved > 0.0 and not finished:
            # Pay only for wheels on the road *and* the car actually going
            # somewhere. Paying for contact alone let the policy park on the
            # grid and farm the reward instead of driving.
            reward += self.contact_reward * (down / 4.0) * min(moved, 4.0)
        ground = self._ground_speed(state)
        # Beached: no wheel touches anything for longer than a bump, and the car
        # is not actually moving. This is a dead end, and is scored as one.
        beached = down == 0 and self._airborne_s >= self.airborne_grace_s and moved < floor
        if beached and not self._beached:
            reward += self.beach_penalty
        self._beached = beached
        speed = float(state["speedKmh"])
        creeping = moved < floor and down == 0
        # Ground motion, not the speed field, drives the stuck timer: a beached
        # car can still report tens of km/h while it cannot move.
        self._slow = 0.0 if (not creeping and ground > 3.0) or int(state["frames"]) < 2000 else self._slow + dt
        # Sitting on the road without going anywhere must cost, or the policy
        # learns that the safest way to never leave the road is to never drive.
        if on_track and moved < floor and not finished:
            reward += self.stall_penalty
        fell = float(state["position"]["y"]) < self._origin_y - 8.0
        timed_out = int(state["frames"]) * 0.001 >= self.timeout_s
        stuck = self._slow >= self.stuck_s or self._flat >= self.stuck_s
        respawned = False
        if (stuck or beached) and self.auto_respawn and not finished:
            # The car is wedged. Hold reset for one decision so the worker
            # teleports it back to the last checkpoint, then keep the episode
            # alive. Ending here taught the policy nothing about recovering.
            first = state
            state = self._view(
                self.client.step(_controls(self._reset_id(), self.keys, True), int(first["frames"]) + self.repeat)
            )
            dt += max(0.0, (int(state["frames"]) - int(first["frames"])) * 0.001)
            progress = self._progress_of(state, finished)
            # The teleport is not driving. Keep the score at the new place
            # without adding that jump to the reward.
            self._progress = progress
            self._mark = progress
            self._flat = 0.0
            self._pace = self._motion_kmh(state)[0]
            reward += self.respawn_penalty
            self._slow = 0.0
            self._airborne_s = 0.0
            self._beached = False
            self._moved(state)
            self._respawns += 1
            respawned = True
            stuck = False
            beached = False
        done = finished or fell or timed_out or ((stuck or beached) and not self.auto_respawn)
        if done and not finished:
            reward -= self.fail_penalty
        self._state = state
        info = {
            "state": state,
            "progress": progress,
            "dt": dt,
            "counted": True,
            "finished": finished,
            "restarted": False,
            "stuck": stuck,
            "beached": beached,
            "fell": fell,
            "wheels_down": down,
            "moved": moved,
            "respawned": respawned,
            "respawns": self._respawns,
            "timeout": timed_out,
            "speed": speed,
            "ground_speed": ground,
        }
        return None, float(reward), done, info

    def _reset_id(self) -> int:
        """The action id whose mask holds `reset`, or 0 when it is disabled."""
        for index, mask in enumerate(action_masks(self.keys, reset_action=True)):
            if mask.get("reset"):
                return index
        return 0

    def close(self) -> None:
        self.client.close()


def make_track(cfg: dict, client: SimClient | None = None) -> SchoolagyTrack:
    return SchoolagyTrack(cfg, client=client)
