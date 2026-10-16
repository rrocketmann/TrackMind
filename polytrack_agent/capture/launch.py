"""Open the official game in the user's Chromium profile.

The profile is the normal one, so a nickname and login already in Chromium
are still there. A separate profile would be logged out of the leaderboard.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

_BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")


def chromium_binary() -> str | None:
    for name in _BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    cache = Path.home() / ".cache" / "ms-playwright"
    if cache.is_dir():
        found = sorted(cache.glob("chromium-*/chrome-linux*/chrome"))
        if found:
            return str(found[-1])
    return None


def chromium_command(url: str) -> list[str] | None:
    binary = chromium_binary()
    if binary is None:
        return None
    # X11 so the window is a real tab we can find and capture. Wayland hides it.
    return [
        binary,
        f"--app={url}",
        "--class=polytrack-official",
        "--ozone-platform=x11",
        "--new-window",
        "--no-sandbox",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
    ]


def ensure_chromium_window(cfg: dict, timeout_s: float = 20.0) -> None:
    """Focus the capture code on a Chromium window at the official URL."""
    capture = cfg["capture"]
    browser = str(capture.get("browser") or "").lower()
    if browser not in {"chromium", "chrome"}:
        return
    from polytrack_agent.capture.window import find_window

    title = str(capture.get("window_title") or "PolyTrack")
    if find_window(title, browser) is not None:
        log.info("Using the open Chromium window for %s", title)
        return
    url = str(capture.get("url") or "https://www.kodub.com/apps/polytrack")
    command = chromium_command(url)
    if command is None:
        log.warning("Chromium is not installed. Open %s in Chromium yourself.", url)
        return
    log.info("Opening %s in Chromium", url)
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.perf_counter() + float(timeout_s)
    while time.perf_counter() < deadline:
        if find_window(title, browser) is not None:
            log.info("Chromium window is up. Load Summer 1 if it is not already showing.")
            return
        time.sleep(0.5)
    log.warning("Chromium did not show a %s window. Open %s yourself.", title, url)
