"""Visual lap progress.

A reference lap is a sequence of tiny grayscale pictures. The live frame is
matched to that sequence. A jump back toward the start is a restart. Nothing
here reads the game's timer, checkpoints, or memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polytrack_agent.obs.preprocess import rgb_to_tiny_gray


@dataclass
class ProgressState:
    progress: float
    restarted: bool
    finished: bool
    confidence: float
    index: int


class NullProgress:
    """Used when no reference lap has been built yet."""

    def reset(self) -> None:
        return None

    def update(self, frame: np.ndarray) -> ProgressState:
        del frame
        return ProgressState(0.0, False, False, 0.0, 0)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float32, copy=False)
    centered = centered - centered.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norm, 1e-6)


class ReferenceProgress:
    def __init__(
        self,
        frames: np.ndarray,
        search_window: int = 8,
        restart_backtrack: float = 0.12,
        restart_hold: int = 3,
        start_sim_margin: float = 0.05,
        match_min: float = 0.55,
        finish_progress: float = 0.97,
        finish_sim_min: float = 0.8,
        finish_hold: int = 8,
    ):
        if frames.ndim != 3 or frames.shape[0] < 1:
            raise ValueError(f"reference frames must be (N, H, W), got {frames.shape}")
        self.ref = frames.astype(np.float32, copy=False)
        self.norm = normalize_rows(self.ref.reshape(len(self.ref), -1))
        self.search_window = max(1, int(search_window))
        self.restart_backtrack = float(restart_backtrack)
        self.restart_hold = max(1, int(restart_hold))
        self.start_sim_margin = float(start_sim_margin)
        self.match_min = float(match_min)
        self.finish_progress = float(finish_progress)
        self.finish_sim_min = float(finish_sim_min)
        self.finish_hold = max(1, int(finish_hold))
        self.index = 0
        self.restart_count = 0
        self.finish_count = 0

    def reset(self) -> None:
        self.index = 0
        self.restart_count = 0
        self.finish_count = 0

    def start_similarity(self, frame: np.ndarray) -> float:
        """Cosine similarity between `frame` and the first reference picture."""
        gray = rgb_to_tiny_gray(frame, int(self.ref.shape[1]), int(self.ref.shape[2]))
        vector = normalize_rows(np.asarray(gray, dtype=np.float32).reshape(1, -1))[0]
        return float(self.norm[0] @ vector)

    def progress_of(self, index: int) -> float:
        if len(self.ref) <= 1:
            return 0.0
        return float(index) / float(len(self.ref) - 1)

    def update(self, frame: np.ndarray) -> ProgressState:
        gray = rgb_to_tiny_gray(frame, int(self.ref.shape[1]), int(self.ref.shape[2]))
        return self.update_gray(gray)

    def update_gray(self, gray: np.ndarray) -> ProgressState:
        vector = normalize_rows(np.asarray(gray, dtype=np.float32).reshape(1, -1))[0]
        scores = self.norm @ vector
        n = int(scores.shape[0])
        window = self.search_window
        i0 = max(0, self.index - window)
        i1 = min(n, self.index + window + 1)
        local_i = i0 + int(np.argmax(scores[i0:i1]))
        local_s = float(scores[local_i])
        global_i = int(np.argmax(scores))
        global_s = float(scores[global_i])

        back = max(window + 1, int(self.restart_backtrack * n))
        pending = (
            self.index - global_i >= back
            and global_s >= self.match_min
            and global_s >= local_s + self.start_sim_margin
            and self.progress_of(self.index) >= self.restart_backtrack
        )
        restarted = False
        if pending:
            self.restart_count += 1
        else:
            self.restart_count = 0

        if self.restart_count >= self.restart_hold:
            self.index = global_i
            self.restart_count = 0
            self.finish_count = 0
            restarted = True
            chosen_s = global_s
        elif not pending and local_s >= self.match_min:
            self.index = local_i
            chosen_s = local_s
        else:
            chosen_s = local_s

        progress = self.progress_of(self.index)
        end_lo = max(0, n - window)
        end_s = float(np.max(scores[end_lo:]))
        if progress >= self.finish_progress and end_s >= self.finish_sim_min:
            self.finish_count += 1
        else:
            self.finish_count = 0
        finished = self.finish_count >= self.finish_hold
        return ProgressState(progress, restarted, finished, chosen_s, int(self.index))


def save_reference(path: Path, frames: np.ndarray, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames.astype(np.float32),
        meta_json=np.asarray(json.dumps(meta)),
    )


def load_reference(path: Path, progress_cfg: dict) -> ReferenceProgress:
    data = np.load(path)
    frames = np.asarray(data["frames"], dtype=np.float32)
    return ReferenceProgress(
        frames,
        search_window=int(progress_cfg.get("search_window", 8)),
        restart_backtrack=float(progress_cfg.get("restart_backtrack", 0.12)),
        restart_hold=int(progress_cfg.get("restart_hold", 3)),
        start_sim_margin=float(progress_cfg.get("start_sim_margin", 0.05)),
        match_min=float(progress_cfg.get("match_min", 0.55)),
        finish_progress=float(progress_cfg.get("finish_progress", 0.97)),
        finish_sim_min=float(progress_cfg.get("finish_sim_min", 0.8)),
        finish_hold=int(progress_cfg.get("finish_hold", 8)),
    )


def build_reference_from_session(
    session: Path,
    downsample: tuple[int, int],
    stride: int,
    max_frames: int,
    episode: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Sample one episode into a grayscale picture strip.

    Imported lazily from the dataset reader to keep this module free of I/O layout details
    except the call below.
    """
    from polytrack_agent.record.session import iter_chunks, load_meta

    meta = load_meta(session)
    counts: dict[int, int] = {}
    chunks = list(iter_chunks(session))
    if not chunks:
        raise ValueError(f"{session} has no frame chunks")
    for chunk in chunks:
        for value in np.asarray(chunk["episode"]):
            counts[int(value)] = counts.get(int(value), 0) + 1
    if episode is None:
        episode = max(counts, key=counts.get)
    if episode not in counts:
        raise ValueError(f"Episode {episode} is not in {session}. Have {sorted(counts)}")

    total = counts[episode]
    stride = max(1, int(stride))
    if total // stride > max_frames:
        stride = max(stride, int(np.ceil(total / max_frames)))
    height, width = int(downsample[0]), int(downsample[1])
    picked: list[np.ndarray] = []
    seen = 0
    for chunk in chunks:
        episodes = np.asarray(chunk["episode"])
        frames = chunk["frames"]
        for local in range(len(episodes)):
            if int(episodes[local]) != int(episode):
                continue
            if seen % stride == 0:
                picked.append(rgb_to_tiny_gray(frames[local], height, width))
            seen += 1
    if not picked:
        raise ValueError(f"Episode {episode} produced no reference frames")
    strip = np.stack(picked).astype(np.float32)
    info = {
        "session": str(session),
        "episode": int(episode),
        "stride": int(stride),
        "frames": int(strip.shape[0]),
        "downsample": [height, width],
        "source_frames": int(total),
        "keys": meta.get("keys", []),
    }
    return strip, info
