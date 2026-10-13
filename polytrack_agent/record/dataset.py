"""Random-access dataset of stacked RGB frames and the key mask at the last frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from polytrack_agent.obs.preprocess import augment_stack, resize_nn
from polytrack_agent.record.session import find_sessions, iter_chunks, load_meta


class DrivingDataset(Dataset):
    """Samples do not cross an episode boundary or a timestamp gap.

    Each item is (frames uint8 [T,H,W,3], prev action uint8 [K], action uint8 [K]).
    """

    def __init__(
        self,
        sessions: list[Path],
        image_size: tuple[int, int],
        stack: int,
        augment: bool,
        max_gap_s: float,
        seed: int = 0,
        use_keys: list[str] | None = None,
    ):
        if stack < 1:
            raise ValueError("stack must be >= 1")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.stack = int(stack)
        self.augment = bool(augment)
        self.max_gap_s = float(max_gap_s)
        self.rng = np.random.default_rng(seed)
        self.frames: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.samples: list[tuple[int, int, str]] = []
        keys: list[str] | None = None

        for session in sessions:
            meta = load_meta(session)
            session_keys = [str(key) for key in meta["keys"]]
            if use_keys is not None and session_keys != list(use_keys):
                raise ValueError(
                    f"{session} keys {session_keys} do not match expected {list(use_keys)}"
                )
            if keys is None:
                keys = session_keys
            elif session_keys != keys:
                raise ValueError(f"{session} keys {session_keys} do not match {keys}")
            session_name = Path(session).name
            for chunk in iter_chunks(session):
                actions = np.asarray(chunk["actions"])
                episodes = np.asarray(chunk["episode"])
                timestamps = np.asarray(chunk["timestamps"], dtype=np.float64)
                frames = chunk["frames"]
                if len(frames) != len(actions) or len(frames) != len(episodes):
                    raise ValueError(f"ragged chunk in {session}")
                chunk_index = len(self.frames)
                self.frames.append(frames)
                self.actions.append(actions)
                if len(frames) < self.stack:
                    continue
                gaps = np.diff(timestamps) if len(timestamps) > 1 else np.zeros(0, dtype=np.float64)
                for t in range(self.stack - 1, len(frames)):
                    t0 = t - self.stack + 1
                    if np.any(np.asarray(episodes[t0 : t + 1]) != int(episodes[t])):
                        continue
                    if t0 < t and np.any(gaps[t0:t] > self.max_gap_s):
                        continue
                    episode_key = f"{session_name}:{int(episodes[t])}"
                    self.samples.append((chunk_index, t, episode_key))

        if keys is None:
            raise ValueError("No sessions to load")
        self.keys = keys
        self.n_actions = len(keys)
        if not self.samples:
            raise ValueError(
                "No training samples. Each episode needs at least `stack` frames "
                "with no large time gap."
            )
        self.interesting = [i for i, key in enumerate(self.keys) if key not in {"w", "up"}]

    def __len__(self) -> int:
        return len(self.samples)

    def weight_for(self, index: int) -> float:
        """Oversample frames where the human changes keys or is not just holding throttle."""
        chunk_index, t, _episode = self.samples[index]
        action = np.asarray(self.actions[chunk_index][t])
        prev = (
            np.zeros(self.n_actions, dtype=np.uint8)
            if t == 0
            else np.asarray(self.actions[chunk_index][t - 1])
        )
        weight = 1.0
        if not np.array_equal(action, prev):
            weight *= 3.0
        if self.interesting and bool(np.any(action[self.interesting] > 0)):
            weight *= 2.0
        return float(max(weight, 1e-3))

    def action_frequency(self) -> np.ndarray:
        total = 0
        acc = np.zeros(self.n_actions, dtype=np.float64)
        for actions in self.actions:
            acc += np.asarray(actions, dtype=np.float64).sum(axis=0)
            total += int(len(actions))
        if total == 0:
            return acc
        return acc / float(total)

    def __getitem__(self, index: int):
        chunk_index, t, _episode = self.samples[index]
        frames = self.frames[chunk_index]
        height, width = self.image_size
        window = [
            resize_nn(np.asarray(frames[t - self.stack + 1 + step]), height, width)
            for step in range(self.stack)
        ]
        stacked = np.stack(window, axis=0)
        if self.augment:
            stacked = augment_stack(stacked, self.rng)
        actions = self.actions[chunk_index]
        prev = (
            np.zeros(self.n_actions, dtype=np.uint8)
            if t == 0
            else np.array(actions[t - 1], dtype=np.uint8, copy=True)
        )
        action = np.array(actions[t], dtype=np.uint8, copy=True)
        return stacked, prev, action


def collate_batch(batch):
    frames, prevs, actions = zip(*batch)
    stacked = np.stack(frames, axis=0)
    frame_t = torch.from_numpy(np.ascontiguousarray(stacked))
    frame_t = frame_t.permute(0, 1, 4, 2, 3).contiguous().float().div_(255.0)
    prev_t = torch.from_numpy(np.ascontiguousarray(np.stack(prevs))).float()
    action_t = torch.from_numpy(np.ascontiguousarray(np.stack(actions))).float()
    return frame_t, prev_t, action_t


def make_splits(
    dataset: DrivingDataset,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Hold out whole episodes when more than one exists, so val frames are not neighbors."""
    if len(dataset) < 2:
        raise ValueError("Need at least 2 samples to train")
    by_episode: dict[str, list[int]] = {}
    for index, (_chunk, _t, episode) in enumerate(dataset.samples):
        by_episode.setdefault(episode, []).append(index)
    episodes = list(by_episode)
    if len(episodes) < 2:
        cut = int(len(dataset) * (1.0 - float(val_fraction)))
        cut = min(max(cut, 1), len(dataset) - 1)
        return list(range(cut)), list(range(cut, len(dataset)))
    rng = np.random.default_rng(seed)
    order = list(episodes)
    rng.shuffle(order)
    n_val = int(round(len(order) * float(val_fraction)))
    n_val = min(max(n_val, 1), len(order) - 1)
    val_eps = set(order[:n_val])
    train: list[int] = []
    val: list[int] = []
    for episode, indices in by_episode.items():
        (val if episode in val_eps else train).extend(indices)
    return train, val


def discover(path: Path) -> list[Path]:
    return find_sessions(Path(path))
