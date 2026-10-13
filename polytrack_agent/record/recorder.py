"""Record the game window and the keys the human is holding.

This process only listens. It does not press keys, so it cannot fight the driver.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from polytrack_agent.capture.factory import RegionFollower, make_capture
from polytrack_agent.input.keyboard import KeyObserver
from polytrack_agent.obs.preprocess import as_rgb, resize_nn
from polytrack_agent.record.session import SessionWriter, new_session_dir

log = logging.getLogger(__name__)


def record_session(cfg: dict, out_root: Path | None = None) -> Path:
    keys = list(cfg["record"]["keys"])
    height, width = int(cfg["record"]["size"][0]), int(cfg["record"]["size"][1])
    fps = float(cfg["capture"]["fps"])
    root = Path(out_root) if out_root is not None else Path(cfg["record"]["out_dir"])
    session = new_session_dir(root)
    writer = SessionWriter(
        session,
        keys=keys,
        frame_shape=(height, width, 3),
        chunk_size=int(cfg["record"].get("chunk_size") or 256),
        fps_target=fps,
        window_title=str(cfg["capture"].get("window_title") or "PolyTrack"),
    )
    capture = make_capture(cfg)
    follower = RegionFollower(cfg, capture)
    observer = KeyObserver(keys)
    try:
        observer.start()
    except RuntimeError:
        capture.close()
        writer.close()
        raise
    except Exception as exc:
        capture.close()
        writer.close()
        raise RuntimeError(
            "Could not listen to the keyboard. On Linux, pynput needs an X11 session. "
            "Wayland does not deliver global key events to this recorder."
        ) from exc

    megabytes = height * width * 3 * fps * 60 / 1_000_000
    log.info("Recording vision to %s", session)
    log.info("Focus the official PolyTrack window and drive. This recorder does not press keys.")
    log.info("Release every driving key, then press them again so the listener sees the key-down.")
    log.info("F6 pause/resume    F7 new episode (do this on each restart)    F8 save and quit")
    log.info("About %.1f MB/minute at %sx%s, %.0f fps.", megabytes, width, height, fps)

    period = 1.0 / fps
    start = time.perf_counter()
    next_tick = start
    last_sample: float | None = None
    episode = 0
    max_minutes = cfg["record"].get("max_minutes")
    max_gap = float(cfg["record"]["max_gap_s"])
    try:
        while not observer.stop_event.is_set():
            if max_minutes is not None and (time.perf_counter() - start) > float(max_minutes) * 60.0:
                log.info("Reached max_minutes=%s", max_minutes)
                break
            if observer.consume_mark():
                episode += 1
                last_sample = None
                log.info("Episode %s", episode)
            if observer.paused:
                time.sleep(0.05)
                last_sample = None
                continue
            follower.maybe_update()
            frame = capture.grab()
            now = time.perf_counter()
            if frame is None:
                time.sleep(0.002)
                continue
            if last_sample is not None and (now - last_sample) > max_gap:
                episode += 1
                log.info("Time gap, episode %s", episode)
            small = resize_nn(as_rgb(frame), height, width)
            writer.add(small, observer.snapshot(keys), now - start, episode)
            last_sample = now
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        observer.stop_listening()
        writer.close()
        capture.close()
    log.info("Saved %s frames to %s", writer.count, session)
    return session
