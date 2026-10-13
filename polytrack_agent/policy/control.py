"""Turn vision-policy probabilities into key holds.

Discrete mode is the reliable one for PolyTrack: the game reads digital keys.
Continuous mode duty-cycles a key for a fraction of the tick. Short taps can
be missed by the browser, so continuous mode is off by default.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from polytrack_agent.input.keys import axis_indices


def _exclusive(state: np.ndarray, probs: np.ndarray, left: list[int], right: list[int]) -> None:
    if not left or not right:
        return
    if not any(state[i] for i in left) or not any(state[i] for i in right):
        return
    left_score = max(float(probs[i]) for i in left)
    right_score = max(float(probs[i]) for i in right)
    drop = right if left_score >= right_score else left
    for index in drop:
        state[index] = False


class Controller:
    def __init__(
        self,
        keys: Sequence[str],
        mode: str = "discrete",
        threshold_on: float = 0.62,
        threshold_off: float = 0.38,
        deadzone: float = 0.18,
        exclusive_axes: bool = True,
        min_key_freq: float = 0.0,
        key_freq: Sequence[float] | None = None,
    ):
        self.keys = [str(key) for key in keys]
        mode = mode.lower().strip()
        if mode not in {"discrete", "continuous"}:
            raise ValueError(f"control mode must be discrete or continuous, got {mode!r}")
        self.mode = mode
        self.threshold_on = float(threshold_on)
        self.threshold_off = float(threshold_off)
        if self.threshold_off > self.threshold_on:
            raise ValueError("threshold_off must be <= threshold_on")
        self.deadzone = float(deadzone)
        self.exclusive_axes = bool(exclusive_axes)
        self.axes = axis_indices(self.keys)
        self.state = np.zeros(len(self.keys), dtype=bool)
        freq = np.ones(len(self.keys), dtype=np.float64) if key_freq is None else np.asarray(key_freq, dtype=np.float64)
        if freq.shape != (len(self.keys),):
            raise ValueError("key_freq length does not match keys")
        self.gated = freq < float(min_key_freq)

    def _filter(self, probs: np.ndarray) -> np.ndarray:
        values = np.asarray(probs, dtype=np.float64).reshape(-1)
        if values.shape != (len(self.keys),):
            raise ValueError(f"expected {len(self.keys)} probabilities, got {values.shape}")
        values = values.copy()
        values[self.gated] = 0.0
        return values

    def discrete_mask(self, probs: np.ndarray) -> dict[str, bool]:
        values = self._filter(probs)
        for index, prob in enumerate(values):
            if self.state[index]:
                if prob < self.threshold_off:
                    self.state[index] = False
            elif prob > self.threshold_on:
                self.state[index] = True
        if self.exclusive_axes:
            for left, right in self.axes.values():
                _exclusive(self.state, values, left, right)
        return {key: bool(self.state[index]) for index, key in enumerate(self.keys)}

    def continuous_holds(self, probs: np.ndarray, period: float) -> dict[str, float]:
        """Seconds to hold each key during this tick. Full deflection holds the whole tick."""
        values = self._filter(probs)
        holds = {key: 0.0 for key in self.keys}
        span = max(1e-6, 1.0 - self.deadzone)
        for left, right in self.axes.values():
            pos = max((float(values[i]) for i in left), default=0.0)
            neg = max((float(values[i]) for i in right), default=0.0)
            # left/right here are the two sides. For steer, left is A and right is D.
            # The value we duty-cycle is the winning side's probability, not a difference,
            # so a confident 0.9 press becomes a long hold.
            if pos >= neg and pos > self.deadzone and left:
                frac = min(1.0, (pos - self.deadzone) / span)
                winner = max(left, key=lambda i: float(values[i]))
                holds[self.keys[winner]] = frac * float(period)
            elif neg > pos and neg > self.deadzone and right:
                frac = min(1.0, (neg - self.deadzone) / span)
                winner = max(right, key=lambda i: float(values[i]))
                holds[self.keys[winner]] = frac * float(period)
        # Keys that are not on a known axis (custom extras) fall back to a hard threshold.
        covered = {i for pair in self.axes.values() for side in pair for i in side}
        for index, key in enumerate(self.keys):
            if index in covered:
                continue
            if values[index] > self.threshold_on:
                holds[key] = float(period)
        return holds

    def reset(self) -> None:
        self.state[:] = False
