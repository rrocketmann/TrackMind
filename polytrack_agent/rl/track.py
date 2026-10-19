"""One car in the PolyTrack 0.6.3 fork simulation.

The Node host steps the game's own worker. This class turns a chord into
up/down/left/right and scores progress from the car state the worker returns.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from polytrack_agent.rl.actions import mask_for
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


def _controls(action_id: int, keys: list[str]) -> dict[str, bool]:
    mask = mask_for(action_id, keys)
    return {
        "up": bool(mask.get("w") or mask.get("up")),
        "left": bool(mask.get("a") or mask.get("left")),
        "down": bool(mask.get("s") or mask.get("down")),
        "right": bool(mask.get("d") or mask.get("right")),
        "reset": False,
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
        reward = cfg["reward"]
        self.progress_scale = float(reward.get("progress_scale") or 1)
        self.time_penalty = float(reward.get("time_penalty") or 0)
        self.finish_bonus = float(reward.get("finish_bonus") or 0)
        self._state: dict = {}
        self._origin_y = 0.0
        self._progress = 0.0
        self._checkpoint = 0
        self._span = 1.0
        self._slow = 0.0
        self.gates, self.finish = gate_points(self.client.parts, _detectors(cfg))

    def _point(self, index: int) -> tuple[float, float, float]:
        if 0 <= index < len(self.gates):
            return self.gates[index]
        return self.finish

    def _view(self, state: dict) -> dict:
        viewed = dict(state)
        viewed["checkpointCount"] = self.client.checkpoint_count
        point = self._point(int(viewed["nextCheckpointIndex"]))
        viewed["target"] = {"x": point[0], "y": point[1], "z": point[2]}
        return viewed

    def _distance(self, state: dict, index: int) -> float:
        point = self._point(index)
        position = state["position"]
        return math.hypot(point[0] - float(position["x"]), point[2] - float(position["z"]))

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
        state = self._view(self.client.step(_controls(action_id, self.keys), target))
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
        speed = float(state["speedKmh"])
        self._slow = 0.0 if speed > 3.0 or int(state["frames"]) < 2000 else self._slow + dt
        fell = float(state["position"]["y"]) < self._origin_y - 8.0
        timed_out = int(state["frames"]) * 0.001 >= self.timeout_s
        stuck = self._slow >= self.stuck_s
        done = finished or fell or timed_out or stuck
        self._state = state
        info = {
            "state": state,
            "progress": progress,
            "dt": dt,
            "counted": True,
            "finished": finished,
            "restarted": False,
            "stuck": stuck,
            "timeout": timed_out,
            "speed": speed,
        }
        return None, float(reward), done, info

    def close(self) -> None:
        self.client.close()


def make_track(cfg: dict) -> SchoolagyTrack:
    return SchoolagyTrack(cfg)
