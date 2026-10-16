"""Headless lap. No display, browser, or window tools.

The car is a point with a speed and a heading. Checkpoints are coordinates.
Getting closer to the next one is the reward. This is not the game's physics.
"""

from __future__ import annotations

import math

from polytrack_agent.rl.actions import mask_for

# A short loop in the plane. The car starts at the origin, facing +x.
GATES = ((12.0, 0.0), (36.0, 0.0), (48.0, 18.0), (28.0, 32.0), (4.0, 20.0), (0.0, 0.0))


class _NoKeys:
    dry_run = True

    def release_keys(self) -> None:
        return None


class HeadlessTrack:
    """Gymnasium-style env. `step` takes one chord id and advances simulated time."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.keys = list(cfg["record"]["keys"])
        self.runtime = _NoKeys()
        rl_cfg = cfg.get("rl") or {}
        capture = cfg.get("capture") or {}
        fps = float(capture.get("fps") or 30.0)
        repeat = max(1, int(rl_cfg.get("action_repeat") or 1))
        self.dt = repeat / fps
        self.timeout_s = float(rl_cfg.get("timeout_s") or 90.0)
        self.stuck_s = float(rl_cfg.get("stuck_s") or 8.0)
        reward_cfg = cfg.get("reward") or {}
        self.progress_scale = float(reward_cfg.get("progress_scale") or 1.0)
        self.time_penalty = float(reward_cfg.get("time_penalty") or 0.0)
        self.finish_bonus = float(reward_cfg.get("finish_bonus") or 0.0)
        self.reach = 20.0
        self.gate_radius = 2.5
        self.max_speed = 16.0
        self.reset(restart=False)

    def reset(self, restart: bool = False) -> None:
        del restart
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.gate = 0
        self.time = 0.0
        self.prev_progress = 0.0
        self.best_progress = 0.0
        self.since_progress = 0.0
        return None

    def _closest_progress(self) -> tuple[float, float, bool]:
        if self.gate >= len(GATES):
            return float(len(GATES)), 1.0, True
        gx, gy = GATES[self.gate]
        dist = math.hypot(gx - self.x, gy - self.y)
        closeness = max(0.0, 1.0 - dist / self.reach)
        passed = dist <= self.gate_radius
        return float(self.gate) + closeness, closeness, passed

    def step(self, action_id: int):
        mask = mask_for(int(action_id), self.keys)
        throttle = 1.0 if mask.get("w") or mask.get("up") else 0.0
        if mask.get("s") or mask.get("down"):
            throttle = -1.0
        steer = 0.0
        if mask.get("a") or mask.get("left"):
            steer -= 1.0
        if mask.get("d") or mask.get("right"):
            steer += 1.0
        if throttle > 0.0:
            self.speed = min(self.max_speed, self.speed + 10.0 * self.dt)
        elif throttle < 0.0:
            self.speed = max(0.0, self.speed - 14.0 * self.dt)
        else:
            self.speed = max(0.0, self.speed - 4.0 * self.dt)
        # Positive steer is `d`, which turns the nose clockwise (heading down).
        self.heading -= steer * (0.6 + 0.12 * self.speed) * self.dt
        self.x += math.cos(self.heading) * self.speed * self.dt
        self.y += math.sin(self.heading) * self.speed * self.dt
        self.time += self.dt

        progress, _closeness, passed = self._closest_progress()
        if passed and self.gate < len(GATES):
            self.gate += 1
            progress, _closeness, _passed = self._closest_progress()
        finished = self.gate >= len(GATES)
        delta = progress - self.prev_progress
        self.prev_progress = progress
        if delta > 1e-4:
            self.best_progress = max(self.best_progress, progress)
            self.since_progress = 0.0
        else:
            self.since_progress += self.dt
        reward = delta * self.progress_scale - self.dt * self.time_penalty
        if finished:
            reward += self.finish_bonus
        stuck = self.since_progress >= self.stuck_s
        timeout = self.time >= self.timeout_s
        done = bool(finished or stuck or timeout)
        info = {
            "progress": float(progress),
            "confidence": 1.0,
            "dt": float(self.dt),
            "finished": bool(finished),
            "restarted": False,
            "counted": True,
            "stuck": bool(stuck),
            "timeout": bool(timeout),
            "keys": self.keys,
        }
        return None, float(reward), done, info

    def close(self) -> None:
        return None
