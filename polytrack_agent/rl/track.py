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
_ROAD_HALF = 2.5  # one grid cell. Summer 1's road is one cell wide.


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
    return {int(item["id"]): item.get("detector") for item in catalog}


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
    """Unique road-tile centers on the ground plane. Pillars share a cell with the surface."""
    seen: set[tuple[float, float]] = set()
    cells: list[tuple[float, float]] = []
    for raw in parts:
        cell = (int(raw[0]) * _PART_SIZE, int(raw[2]) * _PART_SIZE)
        if cell in seen:
            continue
        seen.add(cell)
        cells.append(cell)
    return cells


def road_border(cells: list[tuple[float, float]], x: float, z: float) -> tuple[float, float]:
    """Meters of road left beside the car, and meters to the right of the tile center.

    The road is the chain of 5 m tiles. Clearance is measured to the side of that
    chain, so the joint between two tiles still counts as the middle of the road.
    Off the road, clearance is negative.
    """
    if not cells:
        return 0.0, 0.0
    nearest = min(cells, key=lambda cell: (cell[0] - x) ** 2 + (cell[1] - z) ** 2)
    neighbor = None
    neighbor_distance = 1e9
    for cell in cells:
        distance = math.hypot(cell[0] - nearest[0], cell[1] - nearest[1])
        if 3.0 < distance < 8.0 and distance < neighbor_distance:
            neighbor = cell
            neighbor_distance = distance
    offset_x = x - nearest[0]
    offset_z = z - nearest[1]
    if neighbor is None:
        clearance = _ROAD_HALF - max(abs(offset_x), abs(offset_z))
        return clearance, 0.0
    along_x = neighbor[0] - nearest[0]
    along_z = neighbor[1] - nearest[1]
    length = math.hypot(along_x, along_z)
    along_x /= length
    along_z /= length
    side = offset_x * -along_z + offset_z * along_x
    return _ROAD_HALF - abs(side), side


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
        self.progress_scale = float(reward.get("progress_scale") or 1)
        self.time_penalty = float(reward.get("time_penalty") or 0)
        self.finish_bonus = float(reward.get("finish_bonus") or 0)
        self.respawn_penalty = float(reward.get("respawn_penalty") or 0)
        self.contact_reward = float(reward.get("contact_reward") or 0)
        self.beach_penalty = float(reward.get("beach_penalty") or 0)
        self.stall_penalty = float(reward.get("stall_penalty") or 0)
        self.speed_reward = float(reward.get("speed_reward") or 0)
        self.speed_cap_kmh = float(reward.get("speed_cap_kmh") or 120)
        self._state: dict = {}
        self._origin_y = 0.0
        self._progress = 0.0
        self._checkpoint = 0
        self._span = 1.0
        self._slow = 0.0
        self._airborne_s = 0.0
        self._last_xy = (0.0, 0.0)
        self._beached = False
        self._respawns = 0
        self.gates, self.finish = gate_points(self.client.parts, _detectors(cfg))
        self._road = road_cells(self.client.parts)

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
        clearance, side = road_border(
            self._road,
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
        velocity = state.get("velocity") or {}
        vx = float(velocity.get("x") or 0.0)
        vz = float(velocity.get("z") or 0.0)
        if vx != 0.0 or vz != 0.0:
            return math.hypot(vx, vz) * 3.6
        return float(state["speedKmh"])

    def _progress_of(self, state: dict, finished: bool) -> float:
        """Checkpoint index plus how much of the way to that gate is done.

        Closing the distance raises it. Driving away lowers it. Passing the
        gate moves the index, which the step scores with its own bonus.
        """
        if finished:
            return float(self.client.checkpoint_count) + 1.0
        index = int(state["nextCheckpointIndex"])
        distance = self._distance(state, index)
        if index != self._checkpoint or self._span <= 0.0:
            self._checkpoint = index
            self._span = max(distance, 1.0)
        covered = min(0.99, max(0.0, 1.0 - distance / self._span))
        return float(index) + covered

    def reset(self, restart: bool = True) -> dict:
        del restart
        state = self._view(self.client.reset())
        self._state = state
        self._origin_y = float(state["position"]["y"])
        self._progress = 0.0
        self._checkpoint = int(state["nextCheckpointIndex"])
        self._span = max(self._distance(state, self._checkpoint), 1.0)
        self._slow = 0.0
        self._airborne_s = 0.0
        self._last_xy = (float(state["position"]["x"]), float(state["position"]["z"]))
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
        previous = self._checkpoint
        progress = self._progress_of(state, finished)
        reward = (progress - self._progress) * self.progress_scale - self.time_penalty
        if int(state["nextCheckpointIndex"]) > previous:
            reward += 1.0
        self._progress = progress
        if finished:
            reward += self.finish_bonus
        # Wheel contact is the authoritative "is the car on the track" signal. A
        # car that leaves the road lands on its chassis with every wheel in the
        # air and cannot be driven again, so staying on the surface has to pay.
        down = self._wheels_down(state)
        moved = self._moved(state)
        # A fixed distance threshold misfires at small `action_repeat`: a car at
        # 28 km/h covers only ~0.08 units in a 10-frame step, so it looked parked.
        # Scale the floor by the step's own duration instead.
        floor = max(0.02, 5.0 * dt / 3.6)  # 5 km/h over this step, in metres
        self._airborne_s = self._airborne_s + dt if down == 0 else 0.0
        if down > 0 and moved > 0.0 and not finished:
            # Pay only for wheels on the road *and* the car actually going
            # somewhere. Paying for contact alone let the policy park on the
            # grid and farm the reward instead of driving.
            reward += self.contact_reward * (down / 4.0) * min(moved, 4.0)
        ground = self._ground_speed(state)
        if self.speed_reward and down > 0 and not finished:
            # The game's cars are steered by a slip-cancelling assist that only
            # works with the front wheels down and real speed on. A car crawling
            # at walking pace has almost no automatic steering, so paying for
            # speed while on the road pushes the policy into the regime where
            # the car steers itself. Capped so flat-out is not blindly optimal,
            # and gated on contact so a beached car cannot collect it.
            reward += self.speed_reward * (down / 4.0) * min(ground, self.speed_cap_kmh) / max(1.0, self.speed_cap_kmh)
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
        if down > 0 and moved < floor and not finished:
            reward += self.stall_penalty
        fell = float(state["position"]["y"]) < self._origin_y - 8.0
        timed_out = int(state["frames"]) * 0.001 >= self.timeout_s
        stuck = self._slow >= self.stuck_s
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
