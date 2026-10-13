"""A tiny synthetic driving video so the train/eval path can be checked without the game.

Left-half red means steer left. Right-half blue means steer right. Throttle is always held.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polytrack_agent.record.session import SessionWriter


def write_synthetic_session(
    directory: Path,
    keys: list[str],
    n: int = 200,
    height: int = 90,
    width: int = 160,
) -> Path:
    directory = Path(directory)
    writer = SessionWriter(
        directory,
        keys=list(keys),
        frame_shape=(height, width, 3),
        chunk_size=64,
        fps_target=30.0,
        window_title="synthetic",
    )
    index = {key: i for i, key in enumerate(keys)}
    episode = 0
    for t in range(n):
        if t == n // 2:
            episode = 1
        left = (t % 40) < 20
        brake = t % 47 == 0
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = 30
        if left:
            frame[:, : width // 2, 0] = 255
        else:
            frame[:, width // 2 :, 2] = 255
        if brake:
            band = max(1, height // 5)
            frame[:band, :, 0] = 220
            frame[:band, :, 1] = 220
        row = (t * 3) % height
        frame[row, :, 1] = 90
        action = np.zeros(len(keys), dtype=np.uint8)
        for name in ("w", "up"):
            if name in index:
                action[index[name]] = 1
        if left:
            for name in ("a", "left"):
                if name in index:
                    action[index[name]] = 1
        else:
            for name in ("d", "right"):
                if name in index:
                    action[index[name]] = 1
        if brake:
            for name in ("s", "down"):
                if name in index:
                    action[index[name]] = 1
        writer.add(frame, action, t / 30.0, episode)
    writer.close()
    return directory
