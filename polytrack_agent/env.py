"""Gymnasium-style loop around the real PolyTrack window.

`step` accepts a discrete action id from `polytrack_agent.rl.actions`. The id
is held for `rl.action_repeat` capture frames. A float vector is still accepted
for one frame so older callers keep working.

`reset(restart=True)` taps `R` through the restart hook, then waits until the
reference lap's start line is visible. `R` is never part of the driving mask.
"""

from __future__ import annotations

import time

import numpy as np

from polytrack_agent.obs.preprocess import FrameStack
from polytrack_agent.obs.progress import NullProgress
from polytrack_agent.reward import RewardTracker
from polytrack_agent.rl.actions import mask_for
from polytrack_agent.runtime import VisionRuntime


class PolyTrackVisionEnv:
    def __init__(self, cfg: dict, capture=None, dry_run: bool = True, progress=None, restart_hook=None):
        self.cfg = cfg
        self.keys = list(cfg["record"]["keys"])
        self.runtime = VisionRuntime(cfg, self.keys, dry_run=dry_run, capture=capture)
        height, width = int(cfg["train"]["image_size"][0]), int(cfg["train"]["image_size"][1])
        self.stacker = FrameStack(int(cfg["train"]["stack"]), height, width)
        self.progress = progress if progress is not None else NullProgress()
        reward_cfg = cfg["reward"]
        self.reward = RewardTracker(
            progress_scale=float(reward_cfg["progress_scale"]),
            time_penalty=float(reward_cfg["time_penalty"]),
            restart_penalty=float(reward_cfg["restart_penalty"]),
            finish_bonus=float(reward_cfg["finish_bonus"]),
        )
        self.restart_hook = restart_hook
        rl_cfg = cfg.get("rl") or {}
        self.action_repeat = max(1, int(rl_cfg.get("action_repeat") or 1))
        self.stuck_s = float(rl_cfg.get("stuck_s") or 8.0)
        self.timeout_s = float(rl_cfg.get("timeout_s") or 90.0)
        self.reset_timeout_s = float(rl_cfg.get("reset_timeout_s") or 5.0)
        self.min_confidence = float(cfg["progress"].get("match_min") or 0.0)
        self.observation_shape = (int(cfg["train"]["stack"]), height, width, 3)
        self.action_size = len(self.keys)
        self._last_obs: np.ndarray | None = None
        self._best_progress = 0.0
        self._progress_mark = 0.0
        self._episode_start = 0.0

    def reset(self, restart: bool = False) -> np.ndarray:
        self.runtime.release_keys()
        self.progress.reset()
        self.reward.reset()
        self._best_progress = 0.0
        now = time.perf_counter()
        self._progress_mark = now
        self._episode_start = now
        if restart:
            if self.restart_hook is not None:
                self.restart_hook()
            else:
                self.runtime.tap_restart()
            self._wait_for_start()
            self.progress.reset()
            self.reward.reset()
        frame = self.runtime.grab()
        if frame is None:
            raise RuntimeError("Capture returned no frame")
        obs = self.stacker.reset(frame)
        self._last_obs = obs
        return obs

    def _wait_for_start(self) -> None:
        probe = getattr(self.progress, "start_similarity", None)
        if probe is None:
            raise RuntimeError("Restart needs a reference lap so the start line can be seen")
        deadline = time.perf_counter() + self.reset_timeout_s
        while time.perf_counter() < deadline:
            frame = self.runtime.grab()
            if frame is not None and float(probe(frame)) >= self.min_confidence:
                return
            time.sleep(self.runtime.period)
        raise RuntimeError("Restart did not return to the start line")

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            return self._step_id(int(action))
        return self._step_once(np.asarray(action, dtype=np.float32).reshape(-1))

    def _step_id(self, action_id: int):
        mask = mask_for(action_id, self.keys)
        total = 0.0
        done = False
        info = self._info_shell()
        obs = self._last_obs
        for _ in range(self.action_repeat):
            obs, reward, done, info = self._one_frame(mask)
            if not info["counted"]:
                return obs, 0.0, False, info
            total += reward
            if done:
                break
        info["reward"] = total
        return obs, float(total), bool(done), info

    def _step_once(self, values: np.ndarray):
        if values.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {values.shape}")
        mask = {key: bool(values[index] >= 0.5) for index, key in enumerate(self.keys)}
        return self._one_frame(mask)

    def _one_frame(self, mask: dict[str, bool]):
        started = time.perf_counter()
        focused = self.runtime.apply_mask(mask)
        remain = self.runtime.period - (time.perf_counter() - started)
        if remain > 0:
            time.sleep(remain)
        if not focused:
            self.runtime.release_keys()
            info = self._info_shell()
            info["focused"] = False
            info["counted"] = False
            obs = self._last_obs if self._last_obs is not None else self.reset()
            return obs, 0.0, False, info
        frame = self.runtime.grab()
        if frame is None:
            raise RuntimeError("Capture returned no frame")
        obs = self.stacker.push(frame)
        self._last_obs = obs
        prog = self.progress.update(frame)
        dt = time.perf_counter() - started
        reward, done = self.reward.step(prog, dt, min_confidence=self.min_confidence)
        now = time.perf_counter()
        if prog.confidence >= self.min_confidence and prog.progress > self._best_progress + 1e-4:
            self._best_progress = float(prog.progress)
            self._progress_mark = now
        stuck = (now - self._progress_mark) >= self.stuck_s
        timeout = (now - self._episode_start) >= self.timeout_s
        if stuck or timeout:
            done = True
        info = {
            "progress": float(prog.progress),
            "restarted": bool(prog.restarted),
            "finished": bool(prog.finished),
            "confidence": float(prog.confidence),
            "focused": True,
            "counted": True,
            "stuck": bool(stuck),
            "timeout": bool(timeout),
            "vision": True,
            "keys": self.keys,
        }
        return obs, float(reward), bool(done), info

    def _info_shell(self) -> dict:
        return {
            "progress": float(self._best_progress),
            "restarted": False,
            "finished": False,
            "confidence": 0.0,
            "focused": False,
            "counted": False,
            "stuck": False,
            "timeout": False,
            "vision": True,
            "keys": self.keys,
        }

    def close(self) -> None:
        self.runtime.close()
