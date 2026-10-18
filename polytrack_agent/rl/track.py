"""One car in the schoolagy PolyTrack 0.5.2 simulation.

The Node host steps the game's own worker. This class turns a chord into
up/down/left/right and scores progress from the car state the worker returns.
"""

from __future__ import annotations

import math
from pathlib import Path

from polytrack_agent.rl.actions import mask_for
from polytrack_agent.sim.client import SimClient


class _NoKeys:
    dry_run = True

    def release_keys(self) -> None:
        return None


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
        root = Path(game["root"])
        track = Path(game["track"])
        if not track.is_absolute():
            track = root / track
        parts = Path(game.get("parts") or Path(__file__).resolve().parents[1] / "sim" / "cache" / "track_parts.json")
        self.cfg = cfg
        self.keys = list(cfg["keys"])
        self.runtime = _NoKeys()
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
        self._meters = 0.0
        self._checkpoint = 0
        self._slow = 0.0

    def _view(self, state: dict) -> dict:
        viewed = dict(state)
        viewed["checkpointCount"] = self.client.checkpoint_count
        return viewed

    def _progress_of(self, state: dict, finished: bool) -> float:
        if finished:
            return float(self.client.checkpoint_count) + 1.0
        checkpoint = int(state["nextCheckpointIndex"])
        if checkpoint != self._checkpoint:
            self._checkpoint = checkpoint
            self._meters = 0.0
        return float(checkpoint) + min(0.99, self._meters / 150.0)

    def reset(self, restart: bool = True) -> dict:
        del restart
        state = self._view(self.client.reset())
        self._state = state
        self._origin_y = float(state["position"]["y"])
        self._progress = 0.0
        self._meters = 0.0
        self._checkpoint = int(state["nextCheckpointIndex"])
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
        moved = math.hypot(
            float(state["position"]["x"]) - float(before["position"]["x"]),
            float(state["position"]["z"]) - float(before["position"]["z"]),
        )
        self._meters += moved
        finished = state["finishFrames"] is not None
        progress = self._progress_of(state, finished)
        reward = (progress - self._progress) * self.progress_scale - self.time_penalty
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
