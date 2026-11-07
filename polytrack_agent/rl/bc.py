"""Behavioral cloning on numeric TrackData laps.

Each row is the nineteen car numbers plus the previous hold. The label is the
hold the player used. Videos are not read. The actor is the same net PPO
loads, so a saved file can be the start of a closed-loop run.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from polytrack_agent.rl.checkpoint import save_checkpoint
from polytrack_agent.rl.device import configure_torch, resolve_device
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.seed import set_seed
from polytrack_agent.rl.road import road_width
from polytrack_agent.rl.state import FEATURES, NAMES, checkpoint_from_scaled, extra_from_scaled, extra_width, obs_dim

log = logging.getLogger(__name__)

# One decision is 50 physics frames. A longer gap is a missing sample, so the
# previous-hold bit is cleared instead of pointing at a hold from far back.
_DECISION_S = 0.05
_GAP_S = _DECISION_S * 1.5


def pairs_from_rows(
    rows: list,
    n_actions: int = 9,
    road_window: int = 0,
    course=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack (features + extra car + road + previous hold) and the hold on that row.

    A row is skipped when the hold id or the nineteen numbers are missing.
    The first row, and the first row after a gap, has no previous hold.
    `road_window` 0 leaves the row as the nineteen numbers plus the hold.
    A wider window inserts the extra car numbers, then the road. Spin, skid,
    and suspension are not in a row, so those stay 0.
    """
    window = int(road_window)
    extra_n = extra_width(window)
    road_n = road_width(window)
    width = FEATURES + extra_n + road_n + int(n_actions)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    previous = -1
    for row in rows:
        if not isinstance(row, dict):
            previous = -1
            continue
        try:
            dt = float(row.get("dt") or 0.0)
        except (TypeError, ValueError):
            dt = _GAP_S + 1.0
        if dt > _GAP_S:
            previous = -1
        action = row.get("action")
        features = row.get("n")
        if isinstance(action, float) and action.is_integer():
            action = int(action)
        if isinstance(action, bool) or not isinstance(action, int):
            previous = -1
            continue
        if action < 0 or action >= int(n_actions):
            previous = -1
            continue
        if not isinstance(features, list) or len(features) != FEATURES:
            previous = -1
            continue
        vector = np.zeros(width, dtype=np.float32)
        try:
            vector[:FEATURES] = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError):
            previous = -1
            continue
        if not np.isfinite(vector[:FEATURES]).all():
            previous = -1
            continue
        if extra_n:
            vector[FEATURES : FEATURES + extra_n] = extra_from_scaled(vector[:FEATURES], course)
        if window and course is not None:
            block = _road_block(vector, course, window)
            vector[FEATURES + extra_n : FEATURES + extra_n + road_n] = block
        if 0 <= previous < int(n_actions):
            vector[FEATURES + extra_n + road_n + previous] = 1.0
        observations.append(vector)
        actions.append(int(action))
        previous = int(action)
    if not observations:
        return (
            np.zeros((0, width), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return np.stack(observations), np.asarray(actions, dtype=np.int64)


def _road_block(vector: np.ndarray, course, window: int) -> np.ndarray:
    """Road ahead at this row. x, y, z in the row are scaled by 200."""
    x = float(vector[NAMES.index("x")]) * 200.0
    y = float(vector[NAMES.index("y")]) * 200.0
    z = float(vector[NAMES.index("z")]) * 200.0
    index = checkpoint_from_scaled(
        float(vector[NAMES.index("checkpoint")]),
        int(getattr(course, "gate_count", 0) or 0),
    )
    return course.ahead(x, y, z, int(window), checkpoint=index)


def _read_run(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a lap object")
    return payload


def _dense(run: dict, frames: int) -> bool:
    rows = run.get("rows")
    if not isinstance(rows, list):
        return False
    expected = max(10, int(frames) / 50.0 * 0.5)
    return len(rows) >= expected


def load_track_pairs(
    root: Path,
    track: str,
    n_actions: int,
    seed: int,
    val_fraction: float = 0.2,
    road_window: int = 0,
    course=None,
) -> dict:
    """Load one campaign track. Whole laps are held out, not single rows."""
    index_path = Path(root) / "laps" / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"No TrackData index at {index_path}")
    listed: list[dict] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("track") == track:
            listed.append(row)
    if not listed:
        raise ValueError(f"No {track} laps in {index_path}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(listed))
    val_count = max(1, int(round(len(listed) * float(val_fraction))))
    if val_count >= len(listed):
        val_count = max(1, len(listed) // 5)
    val_ids = {int(index) for index in order[:val_count]}
    train_obs: list[np.ndarray] = []
    train_act: list[np.ndarray] = []
    val_obs: list[np.ndarray] = []
    val_act: list[np.ndarray] = []
    used = 0
    skipped = 0
    for index, row in enumerate(listed):
        path = Path(root) / str(row["file"])
        run = _read_run(path)
        if not _dense(run, int(row.get("frames") or 0)):
            skipped += 1
            continue
        observations, actions = pairs_from_rows(
            run["rows"],
            n_actions=n_actions,
            road_window=road_window,
            course=course,
        )
        if len(actions) < 10:
            skipped += 1
            continue
        used += 1
        if index in val_ids:
            val_obs.append(observations)
            val_act.append(actions)
        else:
            train_obs.append(observations)
            train_act.append(actions)
    if not train_obs or not val_obs:
        raise ValueError(f"{track} did not yield both a train set and a val set")
    return {
        "train_obs": np.concatenate(train_obs),
        "train_act": np.concatenate(train_act),
        "val_obs": np.concatenate(val_obs),
        "val_act": np.concatenate(val_act),
        "laps": used,
        "skipped": skipped,
        "val_laps": len(val_obs),
    }


def _epoch(
    model: StateActorCritic,
    optimizer: torch.optim.Optimizer | None,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    count = int(actions.shape[0])
    order = torch.randperm(count) if training else torch.arange(count)
    total_loss = 0.0
    total_correct = 0.0
    seen = 0
    for start in range(0, count, max(1, int(batch_size))):
        batch = order[start : start + batch_size]
        obs = torch.from_numpy(observations[batch.numpy()]).to(device)
        act = torch.from_numpy(actions[batch.numpy()]).to(device)
        logits, _value = model(obs)
        loss = F.cross_entropy(logits, act)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * int(act.shape[0])
        total_correct += float((logits.argmax(dim=-1) == act).sum().detach())
        seen += int(act.shape[0])
    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, total_correct / seen


def train_bc(
    root: Path,
    out_dir: Path,
    track_name: str = "Summer 1",
    track_stem: str = "summer1",
    keys: list[str] | None = None,
    hidden: int = 128,
    seed: int = 0,
    epochs: int = 40,
    patience: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    n_actions: int = 9,
    road_window: int = 0,
    course=None,
) -> dict:
    """Fit the actor on held-out laps and write best.pt when val loss improves."""
    set_seed(int(seed))
    device = resolve_device("auto")
    configure_torch(device)
    window = int(road_window)
    loaded = load_track_pairs(
        Path(root),
        track_name,
        int(n_actions),
        int(seed),
        road_window=window,
        course=course,
    )
    train_act = loaded["train_act"]
    counts = np.bincount(train_act, minlength=int(n_actions))
    majority = float(counts.max()) / float(len(train_act))
    log.info(
        "laps %s  skipped %s  train_rows %s  val_rows %s  val_laps %s  majority %.4f",
        loaded["laps"],
        loaded["skipped"],
        len(train_act),
        len(loaded["val_act"]),
        loaded["val_laps"],
        majority,
    )
    model = StateActorCritic(
        obs_dim=obs_dim(int(n_actions), window),
        n_actions=int(n_actions),
        hidden=int(hidden),
    ).to(device)
    actor = list(model.net.parameters()) + list(model.policy.parameters())
    optimizer = torch.optim.AdamW(actor, lr=float(lr), weight_decay=1e-4)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(keys or ["w", "a", "s", "d"])
    meta = {
        "obs_dim": int(model.obs_dim),
        "hidden": int(model.hidden),
        "n_actions": int(model.n_actions),
        "keys": keys,
        "track": track_stem,
        "road_window": window,
        "update": 0,
        "progress": -1.0,
    }
    log_path = out / "log.csv"
    best_loss = float("inf")
    best_epoch = 0
    best_acc = 0.0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        train_loss, train_acc = _epoch(
            model, optimizer, loaded["train_obs"], train_act, batch_size, device
        )
        with torch.no_grad():
            val_loss, val_acc = _epoch(
                model, None, loaded["val_obs"], loaded["val_act"], batch_size, device
            )
        log.info(
            "epoch %s  train_loss %.4f  train_acc %.4f  val_loss %.4f  val_acc %.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        fresh = not log_path.is_file() or epoch == 1
        with log_path.open("a" if epoch > 1 else "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if fresh:
                writer.writeheader()
            writer.writerow(row)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_acc = val_acc
            stale = 0
            payload = {**meta, "bc_epoch": epoch, "bc_val_acc": val_acc, "bc_val_loss": val_loss}
            save_checkpoint(out / "best.pt", model, payload)
            save_checkpoint(out / "last.pt", model, payload)
        else:
            stale += 1
            if stale >= max(1, int(patience)):
                break
    log.info(
        "best_epoch %s  val_loss %.4f  val_acc %.4f  wrote %s",
        best_epoch,
        best_loss,
        best_acc,
        out / "best.pt",
    )
    return {
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "val_acc": best_acc,
        "majority": majority,
        "laps": loaded["laps"],
        "train_rows": int(len(train_act)),
        "val_rows": int(len(loaded["val_act"])),
        "out_dir": str(out),
    }
