"""How close the car is to the next checkpoint gate.

PolyTrack paints that gate yellow. This looks at the forward part of the
captured frame, not at the game's memory. A bigger, more central gate means
the car is closer. Driving through it counts one checkpoint.
"""

from __future__ import annotations

import numpy as np

from polytrack_agent.obs.preprocess import as_rgb, resize_nn
from polytrack_agent.obs.progress import ProgressState


class CheckpointProgress:
    """Progress is checkpoints passed plus a 0..1 closeness to the next gate."""

    def __init__(
        self,
        min_r: int = 190,
        min_g: int = 150,
        max_b: int = 90,
        view_top: float = 0.18,
        view_bottom: float = 0.72,
        view_side: float = 0.18,
        full_fraction: float = 0.12,
        pass_high: float = 0.62,
        pass_low: float = 0.18,
        restart_drop: float = 0.85,
        restart_hold: int = 3,
        sample_hw: tuple[int, int] = (48, 80),
    ):
        self.min_r = int(min_r)
        self.min_g = int(min_g)
        self.max_b = int(max_b)
        self.view_top = float(view_top)
        self.view_bottom = float(view_bottom)
        self.view_side = float(view_side)
        self.full_fraction = max(1e-3, float(full_fraction))
        self.pass_high = float(pass_high)
        self.pass_low = float(pass_low)
        self.restart_drop = float(restart_drop)
        self.restart_hold = max(1, int(restart_hold))
        self.sample_hw = (int(sample_hw[0]), int(sample_hw[1]))
        self.min_confidence = 0.0
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict) -> "CheckpointProgress":
        raw = dict(cfg.get("checkpoint") or {})
        sample = raw.pop("sample_hw", None)
        if sample is not None:
            raw["sample_hw"] = (int(sample[0]), int(sample[1]))
        return cls(**raw)

    def reset(self) -> None:
        self.passed = 0
        self.closeness = 0.0
        self.armed = False
        self.peak = 0.0
        self.restart_count = 0

    def closeness_of(self, frame: np.ndarray) -> tuple[float, float]:
        """Return `(closeness, confidence)` for one RGB frame."""
        height, width = self.sample_hw
        small = resize_nn(as_rgb(frame), height, width)
        red = small[:, :, 0].astype(np.int16)
        green = small[:, :, 1].astype(np.int16)
        blue = small[:, :, 2].astype(np.int16)
        yellow = (red >= self.min_r) & (green >= self.min_g) & (blue <= self.max_b)
        top = int(height * self.view_top)
        bottom = max(top + 1, int(height * self.view_bottom))
        side = int(width * self.view_side)
        view = yellow[top:bottom, side : width - side]
        if view.size == 0 or not np.any(view):
            return 0.0, 0.0
        rows = np.linspace(0.45, 1.0, view.shape[0], dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, view.shape[1], dtype=np.float32)
        center = 1.0 - np.abs(cols - 0.5) * 1.4
        center = np.clip(center, 0.2, 1.0)[None, :]
        weight = rows * center
        fraction = float((view * weight).sum() / weight.sum())
        closeness = float(np.clip(fraction / self.full_fraction, 0.0, 1.0))
        confidence = float(np.clip(fraction / (self.full_fraction * 0.25), 0.0, 1.0))
        return closeness, confidence

    def update(self, frame: np.ndarray) -> ProgressState:
        closeness, confidence = self.closeness_of(frame)
        if closeness >= self.pass_high:
            self.armed = True
        passed_now = self.armed and closeness <= self.pass_low and self.closeness >= self.pass_high
        if passed_now:
            self.passed += 1
            self.armed = False
        self.closeness = closeness
        progress = float(self.passed) + closeness
        self.peak = max(self.peak, progress)
        restarted = False
        if self.passed > 0 and self.peak - progress >= self.restart_drop:
            self.restart_count += 1
        else:
            self.restart_count = 0
        if self.restart_count >= self.restart_hold:
            restarted = True
            self.passed = 0
            self.armed = False
            self.peak = closeness
            self.restart_count = 0
            progress = closeness
        return ProgressState(progress, restarted, False, confidence, self.passed)
