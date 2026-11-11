"""Behavioral cloning warm start from the fastest laps in TrackData.

The archive is one gzipped lap per file, indexed by laps/index.jsonl.
The specialist copies the most common track, and only laps within a short
margin of the fastest clean time on that track. A reset press is not a label.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from polytrack_agent.rl.actions import n_actions
from polytrack_agent.rl.line import ReferenceLine, reference_line
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.state import FEATURES, RECORDED_FEATURES, project_recorded

_GAME = "polytrack-0.6.3"


@dataclass(frozen=True)
class Demos:
    """Driving samples from the laps the specialist is allowed to copy."""

    track: str
    track_file: str
    observations: np.ndarray
    actions: np.ndarray
    laps: int
    best_seconds: float
    margin_s: float

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])


def archive_root(bc_cfg: dict) -> Path:
    """Directory that contains laps/index.jsonl. TRACKDATA overrides the config."""
    raw = os.environ.get("TRACKDATA") or bc_cfg.get("root") or str(Path.home() / "dev" / "trackdata")
    return Path(raw).expanduser()


def track_file(name: str) -> str:
    """`Winter 4` is `tracks/official/winter4.track`."""
    slug = "".join(name.lower().split())
    if not slug:
        raise ValueError("track name is empty")
    return f"tracks/official/{slug}.track"


def _index_rows(root: Path) -> list[dict]:
    path = root / "laps" / "index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No lap index at {path}. Set TRACKDATA to the TrackData checkout.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("track") and row.get("file"):
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no laps")
    return rows


def _popular_track(rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row["track"])
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=lambda name: (counts[name], name))


def _load_lap(root: Path, row: dict) -> dict | None:
    path = root / str(row["file"])
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lap = json.load(handle)
    if not isinstance(lap, dict):
        return None
    if lap.get("recovery"):
        return None
    if str(lap.get("determinism") or "ok") != "ok":
        return None
    if lap.get("rows_truncated") or lap.get("ticks_truncated"):
        return None
    if str(lap.get("game") or row.get("game") or _GAME) != _GAME:
        return None
    return lap


def _samples(rows: list[dict], chords: int) -> tuple[np.ndarray, np.ndarray]:
    """One decision per driving row. The previous chord is the one-hot tail."""
    observations: list[np.ndarray] = []
    actions: list[int] = []
    previous = -1
    width = FEATURES + chords
    for row in rows:
        try:
            action = int(row.get("action"))
        except (TypeError, ValueError):
            continue
        recorded = row.get("n")
        if action < 0:
            previous = -1
            continue
        if action >= chords or not isinstance(recorded, list) or len(recorded) != RECORDED_FEATURES:
            continue
        vector = np.zeros(width, dtype=np.float32)
        vector[:FEATURES] = np.asarray(project_recorded(recorded), dtype=np.float32)
        if 0 <= previous < chords:
            vector[FEATURES + previous] = 1.0
        observations.append(vector)
        actions.append(action)
        previous = action
    if not actions:
        return np.zeros((0, width), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(observations), np.asarray(actions, dtype=np.int64)


def load_demos(bc_cfg: dict, keys: list[str], *, reset_action: bool = False) -> Demos:
    """Fast clean laps on one track (or the track with the most archived finishes)."""
    root = archive_root(bc_cfg)
    margin = float(bc_cfg.get("margin_s") if bc_cfg.get("margin_s") is not None else 0.5)
    if margin < 0:
        raise ValueError(f"bc.margin_s must be >= 0, got {margin}")
    chords = n_actions(keys, reset_action=reset_action)
    indexed = _index_rows(root)
    wanted = bc_cfg.get("track")
    if wanted:
        track = str(wanted)
        if not any(str(row.get("track")) == track for row in indexed):
            raise ValueError(f"{track} has no laps in {root}")
    else:
        track = _popular_track(indexed)
    ranked = sorted(
        (row for row in indexed if str(row["track"]) == track and row.get("seconds") is not None),
        key=lambda row: float(row["seconds"]),
    )
    if not ranked:
        raise ValueError(f"{track} has no timed laps in {root}")
    best = float(ranked[0]["seconds"])
    kept_obs: list[np.ndarray] = []
    kept_act: list[np.ndarray] = []
    laps = 0
    for row in ranked:
        if float(row["seconds"]) > best + margin:
            break
        lap = _load_lap(root, row)
        if lap is None:
            continue
        observations, actions = _samples(list(lap.get("rows") or []), chords)
        if actions.shape[0] == 0:
            continue
        kept_obs.append(observations)
        kept_act.append(actions)
        laps += 1
    if laps == 0:
        raise ValueError(f"No clean {track} laps within {margin:.3f}s of {best:.3f}s in {root}")
    return Demos(
        track=track,
        track_file=track_file(track),
        observations=np.concatenate(kept_obs),
        actions=np.concatenate(kept_act),
        laps=laps,
        best_seconds=best,
        margin_s=margin,
    )


def load_best_line(bc_cfg: dict) -> ReferenceLine:
    """The single fastest clean lap on one track (or the track with the most finishes)."""
    root = archive_root(bc_cfg)
    indexed = _index_rows(root)
    wanted = bc_cfg.get("track")
    if wanted:
        track = str(wanted)
        if not any(str(row.get("track")) == track for row in indexed):
            raise ValueError(f"{track} has no laps in {root}")
    else:
        track = _popular_track(indexed)
    ranked = sorted(
        (row for row in indexed if str(row["track"]) == track and row.get("seconds") is not None),
        key=lambda row: float(row["seconds"]),
    )
    if not ranked:
        raise ValueError(f"{track} has no timed laps in {root}")
    for row in ranked:
        lap = _load_lap(root, row)
        if lap is None:
            continue
        try:
            line = reference_line(list(lap.get("rows") or []), seconds=float(row["seconds"]), track=track)
        except ValueError:
            continue
        if line.length > 1.0:
            return line
    best = float(ranked[0]["seconds"])
    raise ValueError(f"No clean {track} lap to follow near {best:.3f}s in {root}")


def clone_coef(update: int, full: float, hold: int, end: int) -> float:
    """Full clone weight through `hold` updates, then a straight taper to zero at `end`.

    `update` is how many PPO updates have already finished. Past `end` the
    clone loss is gone.
    """
    weight = float(full)
    if weight <= 0.0:
        return 0.0
    done = int(update)
    stop = int(end)
    if stop <= 0 or done >= stop:
        return 0.0
    kept = int(hold)
    if kept < 0:
        kept = 0
    if kept >= stop or done <= kept:
        return weight
    return weight * (stop - done) / (stop - kept)


def behavior_clone(
    model: StateActorCritic,
    demos: Demos,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    """Fit the actor to the archived chords. The critic is left for PPO."""
    if demos.steps < 1:
        raise ValueError("behavioral cloning needs at least one decision")
    if int(epochs) < 1:
        raise ValueError(f"bc.epochs must be >= 1, got {epochs}")
    observations = torch.as_tensor(demos.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(demos.actions, dtype=torch.int64, device=device)
    if observations.shape[-1] != model.obs_dim:
        raise ValueError(f"demo width {observations.shape[-1]} does not match obs_dim {model.obs_dim}")
    parameters = list(model.net.parameters()) + list(model.policy.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(lr))
    count = int(actions.shape[0])
    batch = max(1, int(batch_size))
    model.train()
    loss_value = 0.0
    correct = 0
    for _ in range(int(epochs)):
        order = torch.randperm(count, device=device)
        total = 0.0
        correct = 0
        for start in range(0, count, batch):
            chosen = order[start : start + batch]
            logits, _value = model(observations[chosen])
            loss = nn.functional.cross_entropy(logits, actions[chosen])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(chosen.shape[0])
            correct += int((logits.argmax(dim=-1) == actions[chosen]).sum().item())
        loss_value = total / count
    model.eval()
    return {
        "loss": loss_value,
        "accuracy": correct / count,
        "steps": float(count),
        "laps": float(demos.laps),
    }
