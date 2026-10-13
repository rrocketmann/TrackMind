"""Open the official game in the user's Chromium profile.

The profile is the normal one, so a nickname and login already in Chromium
are still there. A separate profile would be logged out of the leaderboard.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time

log = logging.getLogger(__name__)

_BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")


def chromium_command(url: str) -> list[str] | None:
    for name in _BROWSERS:
        path = shutil.which(name)
        if path:
            return [path, f"--app={url}", "--class=polytrack-official"]
    return None


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
    url = str(capture.get("url") or "https://polytrack.dev")
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
