"""Live vision loop primitives: grab a frame, press keys, keep focus."""

from __future__ import annotations

import time

from polytrack_agent.capture.factory import FocusGate, RegionFollower, make_capture
from polytrack_agent.input.keyboard import KeyOutput


class VisionRuntime:
    """Screen in, driving keys out. `dry_run` computes the same path and presses nothing."""

    def __init__(self, cfg: dict, keys: list[str], dry_run: bool = False, capture=None):
        self.cfg = cfg
        self.keys = list(keys)
        self.dry_run = bool(dry_run)
        self.capture = capture if capture is not None else make_capture(cfg)
        self.follower = RegionFollower(cfg, self.capture)
        self.keys_out = KeyOutput(enabled=not self.dry_run)
        play = cfg["play"]
        # A dry run never presses keys, so it does not need the focus gate.
        self.focus = FocusGate(
            title=str(cfg["capture"].get("window_title") or "PolyTrack"),
            hz=float(play.get("focus_check_hz") or 20),
            required=bool(play.get("require_focus", True)) and not self.dry_run,
        )
        fps = float(cfg["capture"]["fps"])
        if fps <= 0:
            raise ValueError("capture.fps must be positive")
        self.period = 1.0 / fps

    def grab(self):
        self.follower.maybe_update()
        return self.capture.grab()

    def apply_mask(self, mask: dict[str, bool]) -> bool:
        """Apply `mask`. Returns False when the game window is not focused."""
        if not self.focus.allowed():
            self.keys_out.release_all()
            return False
        self.keys_out.apply_mask(mask)
        return True

    def apply_timed_holds(self, holds: dict[str, float], budget: float) -> None:
        """Hold keys for the given seconds, then sleep out the rest of `budget`."""
        if budget <= 0:
            self.apply_mask({key: holds.get(key, 0.0) > 0 for key in self.keys})
            return
        start = time.perf_counter()
        self.apply_mask({key: holds.get(key, 0.0) > 1e-4 for key in self.keys})
        release_at = sorted(
            {
                min(float(holds[key]), budget)
                for key in self.keys
                if 0.0 < holds.get(key, 0.0) < budget
            }
        )
        for moment in release_at:
            wait = moment - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
            self.apply_mask({key: holds.get(key, 0.0) > moment + 1e-4 for key in self.keys})
        remain = budget - (time.perf_counter() - start)
        if remain > 0:
            time.sleep(remain)
        # Keep a full-tick hold down so the key does not flicker between decisions.
        self.apply_mask({key: holds.get(key, 0.0) >= budget - 1e-4 for key in self.keys})

    def release_keys(self) -> None:
        self.keys_out.release_all()

    def tap_restart(self) -> None:
        """Tap the game restart key. Driving actions never include this key."""
        self.release_keys()
        if self.dry_run:
            return
        if not self.focus.allowed():
            raise RuntimeError("PolyTrack window is not focused, so restart was not pressed")
        self.keys_out.tap("r")

    def close(self) -> None:
        self.release_keys()
        close = getattr(self.capture, "close", None)
        if callable(close):
            close()
