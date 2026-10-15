"""Numeric lap observation. No pixels go into the actor."""

from __future__ import annotations

import numpy as np

# progress, change since the last counted decision, match confidence,
# episode time scaled by 90s, stall time scaled by 8s, then the last chord.
SCALAR_FEATURES = 5


def obs_dim(n_actions: int) -> int:
    if int(n_actions) < 1:
        raise ValueError("n_actions must be positive")
    return SCALAR_FEATURES + int(n_actions)


class LapNumbers:
    """The numbers a decision is allowed to see."""

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)
        self._vector = np.zeros(obs_dim(self.n_actions), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._vector[:] = 0.0

    def vector(self) -> np.ndarray:
        return self._vector.copy()

    def advance(self, info: dict, action_id: int) -> None:
        """Fold one counted step into the vector. The next decision reads the result."""
        progress = float(info.get("progress") or 0.0)
        delta = progress - float(self._vector[0])
        dt = max(0.0, float(info.get("dt") or 0.0))
        stall = 0.0 if delta > 1e-4 else float(self._vector[4]) * 8.0 + dt
        self._vector[0] = progress
        self._vector[1] = float(np.clip(delta, -1.0, 1.0))
        self._vector[2] = float(info.get("confidence") or 0.0)
        self._vector[3] = min(float(self._vector[3]) * 90.0 + dt, 180.0) / 90.0
        self._vector[4] = min(stall, 16.0) / 8.0
        self._vector[SCALAR_FEATURES:] = 0.0
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError(f"action id {action_id} is outside 0..{self.n_actions - 1}")
        self._vector[SCALAR_FEATURES + int(action_id)] = 1.0
