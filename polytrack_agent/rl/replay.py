"""Fixed-size replay for off-policy updates."""

from __future__ import annotations

import numpy as np


class Replay:
    """Circular buffer of one-step transitions. Samples with replacement."""

    def __init__(self, capacity: int, obs_dim: int):
        slots = int(capacity)
        width = int(obs_dim)
        if slots < 1 or width < 1:
            raise ValueError("capacity and obs_dim must be positive")
        self.capacity = slots
        self.obs = np.zeros((slots, width), dtype=np.float32)
        self.next_obs = np.zeros((slots, width), dtype=np.float32)
        self.actions = np.zeros((slots,), dtype=np.int64)
        self.rewards = np.zeros((slots,), dtype=np.float32)
        self.dones = np.zeros((slots,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        count = int(batch)
        if count < 1:
            raise ValueError("batch must be positive")
        if self.size < 1:
            raise ValueError("replay is empty")
        index = rng.integers(0, self.size, size=count)
        return (
            self.obs[index],
            self.actions[index],
            self.rewards[index],
            self.next_obs[index],
            self.dones[index],
        )
