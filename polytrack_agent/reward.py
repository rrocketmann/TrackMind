"""Reward from visual progress. PPO scores each step with this signal."""

from __future__ import annotations

from polytrack_agent.obs.progress import ProgressState


class RewardTracker:
    """Credit forward visual progress, penalize time, and penalize a detected restart.

    Small backward match jitter is ignored. Progress credit is monotonic until reset.
    """

    def __init__(
        self,
        progress_scale: float = 1.0,
        time_penalty: float = 0.002,
        restart_penalty: float = 1.0,
        finish_bonus: float = 1.0,
        monotonic: bool = True,
    ):
        self.progress_scale = float(progress_scale)
        self.time_penalty = float(time_penalty)
        self.restart_penalty = float(restart_penalty)
        self.finish_bonus = float(finish_bonus)
        self.monotonic = bool(monotonic)
        self.prev = 0.0

    def reset(self) -> None:
        self.prev = 0.0

    def step(self, state: ProgressState, dt: float, min_confidence: float = 0.0) -> tuple[float, bool]:
        if state.restarted:
            self.prev = float(state.progress)
            return -self.restart_penalty, True
        progress = float(state.progress)
        if float(state.confidence) < float(min_confidence) and not state.finished:
            progress = self.prev
        if self.monotonic:
            delta = max(0.0, progress - self.prev)
            self.prev = max(self.prev, progress)
        else:
            delta = progress - self.prev
            self.prev = progress
        reward = delta * self.progress_scale - float(dt) * self.time_penalty
        if state.finished:
            reward += self.finish_bonus
            return reward, True
        return reward, False
