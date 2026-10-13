"""Find the PolyTrack window and the monitor that contains it.

Geometry is queried from the window manager. Nothing is injected into the game.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Iterable

from polytrack_agent.capture.region import Rect

log = logging.getLogger(__name__)

_DPI_READY = False


def ensure_dpi_aware() -> None:
    """Make Win32 window rectangles match physical pixels."""
    global _DPI_READY
    if _DPI_READY or sys.platform != "win32":
        return
    _DPI_READY = True
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            log.debug("Could not set a DPI-aware process", exc_info=True)


def _class_matches(wm_class: str, browser: str) -> bool:
    klass = wm_class.lower()
    if "polytrack-official" in klass:
        return True
    if browser in {"chromium", "chrome"}:
        return "chrom" in klass
    return browser in klass


def pick_game_window(
    windows: Iterable[tuple[str, str, Rect]],
    title: str,
    browser: str | None = None,
) -> Rect | None:
    """Largest window whose title contains `title`.

    When `browser` is set, the window class has to be that browser. A file
    editor titled PolyTrack does not qualify.
    """
    needle = title.lower()
    browser_name = (browser or "").lower().strip()
    chosen: list[tuple[str, str, Rect]] = []
    for name, wm_class, rect in windows:
        if needle not in name.lower():
            continue
        if browser_name and not _class_matches(wm_class, browser_name):
            continue
        chosen.append((name, wm_class, rect))
    official = [item for item in chosen if "polytrack-official" in item[1].lower()]
    pool = official or chosen
    best: Rect | None = None
    best_area = -1
    for _name, _klass, rect in pool:
        area = rect.width * rect.height
        if area > best_area:
            best = rect
            best_area = area
    return best


def pick_largest(windows: Iterable[tuple[str, Rect]], title: str) -> Rect | None:
    needle = title.lower()
    best: Rect | None = None
    best_area = -1
    for name, rect in windows:
        if needle not in name.lower():
            continue
        area = rect.width * rect.height
        if area > best_area:
            best = rect
            best_area = area
    return best


def list_monitors() -> list[Rect]:
    """Return mss monitors. Index 0 is the virtual desktop, then one entry per display."""
    import mss

    with mss.mss() as sct:
        return [
            Rect(int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"]))
            for mon in sct.monitors
        ]


def origin_for_rect(rect: Rect) -> tuple[int, int]:
    """Top-left of the monitor that fully contains `rect`, else (0, 0)."""
    try:
        monitors = list_monitors()
    except Exception:
        log.debug("Monitor list unavailable", exc_info=True)
        return (0, 0)
    for index, mon in enumerate(monitors):
        if index == 0 and len(monitors) > 1:
            continue
        if (
            mon.left <= rect.left
            and mon.top <= rect.top
            and rect.right <= mon.right
            and rect.bottom <= mon.bottom
        ):
            return (mon.left, mon.top)
    return (0, 0)


def find_window(title: str, browser: str | None = None) -> Rect | None:
    ensure_dpi_aware()
    browser_name = (browser or "").lower().strip() or None
    if sys.platform == "win32":
        return _find_window_windows(title, browser_name)
    if browser_name:
        classed = _windows_wmctrl()
        if classed:
            return pick_game_window(classed, title, browser_name)
        classed = _windows_xdotool_class(browser_name, title)
        if classed:
            return pick_game_window(classed, title, browser_name)
        return None
    found = _find_window_xdotool(title)
    if found is not None:
        return found
    return _find_window_wmctrl(title)


def active_window_title() -> str | None:
    ensure_dpi_aware()
    if sys.platform == "win32":
        return _active_title_windows()
    try:
        out = subprocess.check_output(
            ["xdotool", "getactivewindow", "getwindowname"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    title = out.strip()
    return title or None


def _find_window_xdotool(title: str) -> Rect | None:
    try:
        raw = subprocess.check_output(
            ["xdotool", "search", "--name", title],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    windows: list[tuple[str, Rect]] = []
    for wid in raw.split():
        try:
            geo = subprocess.check_output(
                ["xdotool", "getwindowgeometry", "--shell", wid],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            name = subprocess.check_output(
                ["xdotool", "getwindowname", wid],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
        vals: dict[str, str] = {}
        for line in geo.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                vals[key] = value
        if "WIDTH" not in vals or "HEIGHT" not in vals:
            continue
        windows.append(
            (
                name,
                Rect(int(vals["X"]), int(vals["Y"]), int(vals["WIDTH"]), int(vals["HEIGHT"])),
            )
        )
    return pick_largest(windows, title)


def _windows_wmctrl() -> list[tuple[str, str, Rect]]:
    try:
        out = subprocess.check_output(["wmctrl", "-lGx"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    windows: list[tuple[str, str, Rect]] = []
    for line in out.splitlines():
        parts = line.split(None, 8)
        if len(parts) < 9:
            continue
        _wid, _desk, x, y, width, height, _host, wm_class, name = parts
        try:
            windows.append((name, wm_class, Rect(int(x), int(y), int(width), int(height))))
        except ValueError:
            continue
    return windows


def _find_window_wmctrl(title: str) -> Rect | None:
    windows = _windows_wmctrl()
    if not windows:
        return None
    return pick_largest([(name, rect) for name, _klass, rect in windows], title)


def _windows_xdotool_class(browser: str, title: str) -> list[tuple[str, str, Rect]]:
    classes = ["polytrack-official"]
    if browser in {"chromium", "chrome"}:
        classes.extend(["chromium", "google-chrome", "chrome"])
    else:
        classes.append(browser)
    windows: list[tuple[str, str, Rect]] = []
    for wm_class in classes:
        try:
            raw = subprocess.check_output(
                ["xdotool", "search", "--class", wm_class],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        for wid in raw.split():
            try:
                geo = subprocess.check_output(
                    ["xdotool", "getwindowgeometry", "--shell", wid],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                name = subprocess.check_output(
                    ["xdotool", "getwindowname", wid],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError, ValueError):
                continue
            vals: dict[str, str] = {}
            for line in geo.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    vals[key] = value
            if "WIDTH" not in vals or "HEIGHT" not in vals:
                continue
            if title.lower() not in name.lower():
                continue
            windows.append(
                (name, wm_class, Rect(int(vals["X"]), int(vals["Y"]), int(vals["WIDTH"]), int(vals["HEIGHT"])))
            )
    return windows


def _find_window_windows(title: str, browser: str | None = None) -> Rect | None:
    classed = _enum_windows()
    found = pick_game_window(classed, title, browser)
    if found is not None or browser:
        return found
    return pick_largest([(name, rect) for name, _klass, rect in classed], title)


def _enum_windows() -> list[tuple[str, Rect]]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    results: list[tuple[str, str, Rect]] = []
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width > 0 and height > 0:
            class_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buf, 256)
            results.append((buf.value, class_buf.value, Rect(int(rect.left), int(rect.top), width, height)))
        return True

    callback_ref = enum_proc(callback)
    user32.EnumWindows(callback_ref, 0)
    return results


def _active_title_windows() -> str | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    length = int(user32.GetWindowTextLengthW(hwnd))
    buf = ctypes.create_unicode_buffer(max(0, length) + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    title = buf.value.strip()
    return title or None
