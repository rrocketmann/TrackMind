"""Find the PolyTrack window and the monitor that contains it.

Geometry is queried from the window manager. Nothing is injected into the game.
"""

from __future__ import annotations

import logging
import re
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


def find_window_id(title: str, browser: str | None = None) -> int | None:
    """X11 window id of the game window, or None when it is not open."""
    if sys.platform == "win32":
        return None
    needle = title.lower()
    browser_name = (browser or "").lower().strip()
    best_id = None
    best_area = -1
    official_id = None
    official_area = -1
    for wid in _xprop_ids("_NET_CLIENT_LIST"):
        name = _xprop_name(wid)
        if needle not in name.lower():
            continue
        wm_class = _xprop_class(wid)
        if browser_name and not _class_matches(wm_class, browser_name):
            continue
        rect = _x_geometry(wid)
        if rect is None:
            continue
        area = rect.width * rect.height
        if area > best_area:
            best_id = int(wid, 16)
            best_area = area
        if "polytrack-official" in wm_class.lower() and area > official_area:
            official_id = int(wid, 16)
            official_area = area
    return official_id if official_id is not None else best_id


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
        classed = _windows_xprop()
        if classed:
            return pick_game_window(classed, title, browser_name)
        return None
    found = _find_window_xdotool(title)
    if found is not None:
        return found
    found = _find_window_wmctrl(title)
    if found is not None:
        return found
    classed = _windows_xprop()
    if not classed:
        return None
    return pick_largest([(name, rect) for name, _klass, rect in classed], title)


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
        out = ""
    title = out.strip()
    if title:
        return title
    ids = _xprop_ids("_NET_ACTIVE_WINDOW")
    if not ids:
        return None
    return _xprop_name(ids[0]) or None


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


def _xprop_ids(prop: str) -> list[str]:
    try:
        text = subprocess.check_output(["xprop", "-root", prop], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    return re.findall(r"0x[0-9a-fA-F]+", text)


def _xprop_name(wid: str) -> str:
    try:
        text = subprocess.check_output(["xprop", "-id", wid, "WM_NAME"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""
    match = re.search(r'WM_NAME\([^)]*\) = "(.*)"', text)
    if match:
        return match.group(1)
    match = re.search(r"WM_NAME\([^)]*\) = (.*)", text)
    return match.group(1).strip() if match else ""


def _xprop_class(wid: str) -> str:
    try:
        text = subprocess.check_output(["xprop", "-id", wid, "WM_CLASS"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""
    parts = re.findall(r'"([^"]*)"', text)
    return " ".join(parts)


def _x_geometry(wid: str) -> Rect | None:
    import ctypes

    lib = ctypes.CDLL("libX11.so.6")
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XGetGeometry.restype = ctypes.c_int
    lib.XGetGeometry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.XTranslateCoordinates.restype = ctypes.c_int
    lib.XTranslateCoordinates.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
    ]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = lib.XOpenDisplay(None)
    if not display:
        return None
    try:
        root = lib.XDefaultRootWindow(display)
        root_ret = ctypes.c_ulong()
        x = ctypes.c_int()
        y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        depth = ctypes.c_uint()
        window = int(wid, 16)
        ok = lib.XGetGeometry(
            display,
            ctypes.c_ulong(window),
            ctypes.byref(root_ret),
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(border),
            ctypes.byref(depth),
        )
        if not ok or int(width.value) <= 0 or int(height.value) <= 0:
            return None
        rx = ctypes.c_int()
        ry = ctypes.c_int()
        child = ctypes.c_ulong()
        lib.XTranslateCoordinates(
            display,
            ctypes.c_ulong(window),
            root,
            0,
            0,
            ctypes.byref(rx),
            ctypes.byref(ry),
            ctypes.byref(child),
        )
        return Rect(int(rx.value), int(ry.value), int(width.value), int(height.value))
    finally:
        lib.XCloseDisplay(display)


def _windows_xprop() -> list[tuple[str, str, Rect]]:
    windows: list[tuple[str, str, Rect]] = []
    for wid in _xprop_ids("_NET_CLIENT_LIST"):
        rect = _x_geometry(wid)
        if rect is None:
            continue
        windows.append((_xprop_name(wid), _xprop_class(wid), rect))
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
