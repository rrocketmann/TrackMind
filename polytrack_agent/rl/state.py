"""Car-state observation from the PolyTrack 0.6.3 simulation. No pixels."""

from __future__ import annotations

import numpy as np

# speed, quaternion, upright, forward xz, wheels down, checkpoint, height, airborne.
FEATURES = 12


def obs_dim(n_actions: int) -> int:
    if int(n_actions) < 1:
        raise ValueError("n_actions must be positive")
    return FEATURES + int(n_actions)


def _features(state: dict) -> list[float]:
    quat = state["quaternion"]
    x = float(quat["x"])
    y = float(quat["y"])
    z = float(quat["z"])
    w = float(quat["w"])
    upright = 1.0 - 2.0 * (x * x + z * z)
    forward_x = 2.0 * (x * z + w * y)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    wheels = state.get("wheels") or []
    contact = sum(1.0 for wheel in wheels if wheel) / 4.0
    checkpoints = float(state.get("checkpointCount") or 3)
    return [
        float(state["speedKmh"]) / 200.0,
        x,
        y,
        z,
        w,
        upright,
        forward_x,
        forward_z,
        contact,
        float(state["nextCheckpointIndex"]) / max(1.0, checkpoints),
        float(state["position"]["y"]) / 20.0,
        1.0 if contact == 0.0 else 0.0,
    ]


class LapNumbers:
    """The numbers a decision is allowed to see."""

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)
        self._vector = np.zeros(obs_dim(self.n_actions), dtype=np.float32)

    def reset(self) -> None:
        self._vector[:] = 0.0

    def vector(self) -> np.ndarray:
        return self._vector.copy()

    def advance(self, info: dict, action_id: int) -> None:
        """Store the car state that the next decision will read."""
        features = _features(info["state"])
        if len(features) != FEATURES:
            raise RuntimeError(f"expected {FEATURES} car features, got {len(features)}")
        self._vector[:FEATURES] = np.asarray(features, dtype=np.float32)
        self._vector[FEATURES:] = 0.0
        if 0 <= int(action_id) < self.n_actions:
            self._vector[FEATURES + int(action_id)] = 1.0
