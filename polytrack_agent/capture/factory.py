"""Build a capture backend and keep its rectangle on the game window."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from polytrack_agent.capture.dxcam_capture import DxcamCapture
from polytrack_agent.capture.mss_capture import MssCapture
from polytrack_agent.capture.region import Rect
from polytrack_agent.capture.window import find_window, list_monitors, origin_for_rect

log = logging.getLogger(__name__)


def resolve_region(cfg: dict[str, Any]) -> Rect:
    cap = cfg["capture"]
    raw = cap.get("region")
    if raw:
        rect = Rect(int(raw["left"]), int(raw["top"]), int(raw["width"]), int(raw["height"]))
        log.info("Using fixed capture region %s,%s %sx%s", rect.left, rect.top, rect.width, rect.height)
    else:
        title = str(cap.get("window_title") or "PolyTrack")
        found = find_window(title, str(cap.get("browser") or "") or None)
        if found is None:
            monitors = list_monitors()
            index = int(cap.get("monitor") or 1)
            if index < 0 or index >= len(monitors):
                index = 0
            rect = monitors[index]
            log.warning(
                "No window titled %r. Capturing monitor %s (%sx%s).",
                title,
                index,
                rect.width,
                rect.height,
            )
        else:
            rect = found
            log.info(
                "Window %r at %s,%s %sx%s",
                title,
                rect.left,
                rect.top,
                rect.width,
                rect.height,
            )
    return rect.cropped(cap.get("crop") or {})


def make_capture(cfg: dict[str, Any]):
    """Open mss, or dxcam on Windows. `auto` falls back to mss if dxcam is missing."""
    backend = str(cfg["capture"].get("backend") or "auto").lower()
    if backend == "auto":
        backend = "dxcam" if sys.platform == "win32" else "mss"
    region = resolve_region(cfg)
    if backend == "dxcam":
        if sys.platform != "win32":
            raise RuntimeError("dxcam only runs on Windows. Set capture.backend to mss.")
        try:
            import dxcam  # noqa: F401
        except Exception as exc:
            if str(cfg["capture"].get("backend") or "auto").lower() != "auto":
                raise RuntimeError(
                    "dxcam is not installed. On Windows: pip install -r requirements-windows.txt"
                ) from exc
            log.warning("dxcam unavailable (%s). Using mss.", exc)
            backend = "mss"
        else:
            origin = origin_for_rect(region)
            return DxcamCapture(
                region,
                origin=origin,
                device_idx=int(cfg["capture"].get("dxcam_device") or 0),
                output_idx=int(cfg["capture"].get("dxcam_output") or 0),
            )
    if backend == "mss":
        return MssCapture(region)
    raise RuntimeError(f"Unknown capture backend {backend!r}")


class RegionFollower:
    """Re-read the game window a few times a second when no fixed region is set."""

    def __init__(self, cfg: dict[str, Any], capture):
        cap = cfg["capture"]
        self.capture = capture
        self.title = str(cap.get("window_title") or "PolyTrack")
        self.browser = str(cap.get("browser") or "") or None
        self.crop = cap.get("crop") or {}
        self.enabled = bool(cap.get("follow_window", True)) and not cap.get("region")
        self.period = float(cap.get("geometry_refresh_s") or 0.5)
        self._next = 0.0

    def maybe_update(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if now < self._next:
            return
        self._next = now + self.period
        found = find_window(self.title, self.browser)
        if found is None:
            return
        try:
            self.capture.set_region(found.cropped(self.crop))
        except ValueError as exc:
            log.warning("%s", exc)


class FocusGate:
    """Press keys only while the foreground window title contains the game name."""

    def __init__(self, title: str, hz: float, required: bool):
        self.title = title.lower()
        self.required = required
        self.period = 0.0 if hz <= 0 else 1.0 / float(hz)
        self._next = 0.0
        self._ok = True
        self._warned = False

    def allowed(self) -> bool:
        if not self.required:
            return True
        now = time.perf_counter()
        if now < self._next:
            return self._ok
        self._next = now + self.period
        from polytrack_agent.capture.window import active_window_title

        current = active_window_title()
        if current is None:
            if not self._warned:
                log.warning(
                    "Cannot read the active window title (xdotool missing, or no X11 session). "
                    "The focus gate is off, so keys will be sent whenever the agent runs."
                )
                self._warned = True
            self._ok = True
        else:
            self._ok = self.title in current.lower()
        return self._ok
