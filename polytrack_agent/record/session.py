"""On-disk sessions: RGB uint8 frames and a multi-hot key mask per frame."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np


def new_session_dir(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = root / f"session-{stamp}"
    suffix = 0
    while path.exists():
        suffix += 1
        path = root / f"session-{stamp}-{suffix}"
    path.mkdir()
    return path


def load_meta(session: Path) -> dict:
    path = Path(session) / "meta.json"
    with path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(f"{path} is not a session meta object")
    return meta


def find_sessions(path: Path) -> list[Path]:
    path = Path(path)
    if (path / "meta.json").is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    found = sorted(child for child in path.iterdir() if (child / "meta.json").is_file())
    if not found:
        raise FileNotFoundError(f"No recorded sessions in {path}")
    return found


def iter_chunks(session: Path) -> Iterator[dict[str, np.ndarray]]:
    session = Path(session)
    for chunk_dir in sorted(session.glob("chunk_*")):
        frames_path = chunk_dir / "frames.npy"
        if not frames_path.is_file():
            continue
        yield {
            "frames": np.load(frames_path, mmap_mode="r"),
            "actions": np.load(chunk_dir / "actions.npy", mmap_mode="r"),
            "timestamps": np.load(chunk_dir / "timestamps.npy", mmap_mode="r"),
            "episode": np.load(chunk_dir / "episode.npy", mmap_mode="r"),
        }


class SessionWriter:
    """Append frames into fixed-size numpy chunks so a crash keeps earlier chunks."""

    def __init__(
        self,
        directory: Path,
        keys: list[str],
        frame_shape: tuple[int, int, int],
        chunk_size: int = 256,
        fps_target: float = 30.0,
        window_title: str = "PolyTrack",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.glob("chunk_*")):
            raise FileExistsError(f"{self.directory} already has chunks")
        if len(frame_shape) != 3 or frame_shape[2] != 3:
            raise ValueError(f"frame_shape must be (H, W, 3), got {frame_shape}")
        self.keys = list(keys)
        self.frame_shape = tuple(int(v) for v in frame_shape)
        self.chunk_size = max(1, int(chunk_size))
        self.count = 0
        self.chunk_id = 0
        self._frames: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._episodes: list[int] = []
        self.meta = {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "keys": self.keys,
            "frame_shape": list(self.frame_shape),
            "chunk_size": self.chunk_size,
            "num_frames": 0,
            "chunks": 0,
            "fps_target": float(fps_target),
            "window_title": window_title,
            "modality": "rgb-pixels+keymask",
            "notes": "Screen frames only. No game memory, DOM, or network capture.",
        }
        self._write_meta()

    def add(self, frame: np.ndarray, action: np.ndarray, timestamp: float, episode: int) -> None:
        frame = np.ascontiguousarray(frame)
        if frame.shape != self.frame_shape or frame.dtype != np.uint8:
            raise ValueError(
                f"frame must be uint8 {self.frame_shape}, got {frame.dtype} {frame.shape}"
            )
        action = np.asarray(action, dtype=np.uint8).reshape(-1)
        if action.shape != (len(self.keys),):
            raise ValueError(f"action must have length {len(self.keys)}, got {action.shape}")
        self._frames.append(frame)
        self._actions.append(action.copy())
        self._timestamps.append(float(timestamp))
        self._episodes.append(int(episode))
        self.count += 1
        if len(self._frames) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._frames:
            return
        chunk_dir = self.directory / f"chunk_{self.chunk_id:05d}"
        chunk_dir.mkdir()
        np.save(chunk_dir / "frames.npy", np.stack(self._frames))
        np.save(chunk_dir / "actions.npy", np.stack(self._actions))
        np.save(chunk_dir / "timestamps.npy", np.asarray(self._timestamps, dtype=np.float64))
        np.save(chunk_dir / "episode.npy", np.asarray(self._episodes, dtype=np.int32))
        self.chunk_id += 1
        self._frames.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._episodes.clear()
        self._write_meta()

    def close(self) -> None:
        self.flush()
        self._write_meta()

    def _write_meta(self) -> None:
        self.meta["num_frames"] = self.count
        self.meta["chunks"] = self.chunk_id
        payload = json.dumps(self.meta, indent=2)
        tmp = self.directory / "meta.json.tmp"
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.directory / "meta.json")
