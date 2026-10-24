"""Car-state observation from the PolyTrack 0.6.3 simulation. No pixels.

Each decision sees a short list of numbers: where the car is, how it is moving,
how that motion is changing, and how much road is left to the side.
"""

from __future__ import annotations

import math

import numpy as np

# Order is the observation layout. Scales below keep a Summer 1 lap near ±1.
NAMES = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "forward_x",
    "forward_z",
    "upright",
    "wheels",
    "border",
    "border_side",
    "gate_right",
    "gate_forward",
    "gate_distance",
    "checkpoint",
)
FEATURES = len(NAMES)
_POS = 200.0
_VEL = 40.0
_ACC = 40.0
_BORDER = 10.0
_GATE = 100.0


def obs_dim(n_actions: int) -> int:
    if int(n_actions) < 1:
        raise ValueError("n_actions must be positive")
    return FEATURES + int(n_actions)


def _features(state: dict, acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    quat = state.get("quaternion") or {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    qx = float(quat["x"])
    qy = float(quat["y"])
    qz = float(quat["z"])
    qw = float(quat["w"])
    upright = 1.0 - 2.0 * (qx * qx + qz * qz)
    forward_x = 2.0 * (qx * qz + qw * qy)
    forward_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    right_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    right_z = 2.0 * (qx * qz - qw * qy)
    wheels = state.get("wheels") or []
    contact = sum(1.0 for wheel in wheels if wheel) / 4.0
    checkpoints = float(state.get("checkpointCount") or 3)
    position = state.get("position") or {}
    px = float(position.get("x") or 0.0)
    py = float(position.get("y") or 0.0)
    pz = float(position.get("z") or 0.0)
    target = state.get("target") or {"x": px, "z": pz}
    dx = float(target["x"]) - px
    dz = float(target["z"]) - pz
    # The body's +Z is forward, and its +X points to the driver's left in this sim.
    local_right = -(dx * right_x + dz * right_z)
    local_forward = dx * forward_x + dz * forward_z
    velocity = state.get("velocity") or {}
    vx = float(velocity.get("x") or 0.0)
    vy = float(velocity.get("y") or 0.0)
    vz = float(velocity.get("z") or 0.0)
    ax, ay, az = acceleration
    border = state.get("border") or {}
    clearance = float(border.get("clearance") or 0.0)
    side = float(border.get("side") or 0.0)
    return [
        px / _POS,
        py / _POS,
        pz / _POS,
        vx / _VEL,
        vy / _VEL,
        vz / _VEL,
        ax / _ACC,
        ay / _ACC,
        az / _ACC,
        forward_x,
        forward_z,
        upright,
        contact,
        clearance / _BORDER,
        side / _BORDER,
        local_right / _GATE,
        local_forward / _GATE,
        math.hypot(dx, dz) / _GATE,
        float(state.get("nextCheckpointIndex") or 0.0) / max(1.0, checkpoints),
    ]


class LapNumbers:
    """The numbers a decision is allowed to see."""

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)
        self._vector = np.zeros(obs_dim(self.n_actions), dtype=np.float32)
        self._velocity = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._vector[:] = 0.0
        self._velocity = (0.0, 0.0, 0.0)

    def vector(self) -> np.ndarray:
        return self._vector.copy()

    def advance(self, info: dict, action_id: int) -> None:
        """Store the car state that the next decision will read."""
        state = info["state"]
        velocity = state.get("velocity") or {}
        current = (
            float(velocity.get("x") or 0.0),
            float(velocity.get("y") or 0.0),
            float(velocity.get("z") or 0.0),
        )
        dt = float(info.get("dt") or 0.0)
        if dt > 1e-6:
            previous = self._velocity
            acceleration = tuple((current[i] - previous[i]) / dt for i in range(3))
        else:
            acceleration = (0.0, 0.0, 0.0)
        self._velocity = current
        features = _features(state, acceleration)
        if len(features) != FEATURES:
            raise RuntimeError(f"expected {FEATURES} car features, got {len(features)}")
        self._vector[:FEATURES] = np.asarray(features, dtype=np.float32)
        self._vector[FEATURES:] = 0.0
        if 0 <= int(action_id) < self.n_actions:
            self._vector[FEATURES + int(action_id)] = 1.0
