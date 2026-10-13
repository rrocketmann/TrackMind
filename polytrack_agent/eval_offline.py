"""Replay recorded frames through the vision policy and score key agreement.

This does not open the game. It answers whether the policy has learned the demos
before you let it press keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from polytrack_agent.policy.cnn import frames_to_batch, load_checkpoint, resolve_device
from polytrack_agent.policy.control import Controller
from polytrack_agent.record.dataset import DrivingDataset


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_session(
    checkpoint: Path,
    session: Path,
    device: str = "auto",
    max_frames: int = 0,
    threshold_on: float = 0.62,
    threshold_off: float = 0.38,
    max_gap_s: float = 0.5,
) -> dict:
    dev = resolve_device(device)
    model, payload = load_checkpoint(Path(checkpoint), dev)
    keys = [str(key) for key in payload["keys"]]
    dataset = DrivingDataset(
        [Path(session)],
        image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
        stack=int(payload["stack"]),
        augment=False,
        max_gap_s=max_gap_s,
        use_keys=keys,
    )
    limit = len(dataset) if max_frames <= 0 else min(len(dataset), int(max_frames))
    controller = Controller(
        keys,
        mode="discrete",
        threshold_on=threshold_on,
        threshold_off=threshold_off,
        key_freq=payload.get("key_freq"),
        min_key_freq=0.0,
    )
    k = len(keys)
    tp = np.zeros(k, dtype=np.int64)
    fp = np.zeros(k, dtype=np.int64)
    fn = np.zeros(k, dtype=np.int64)
    correct = np.zeros(k, dtype=np.int64)
    hyst_correct = np.zeros(k, dtype=np.int64)
    model.eval()
    for index in range(limit):
        frames, prev, action = dataset[index]
        batch, prev_t = frames_to_batch(frames, prev, dev)
        with torch.inference_mode():
            logits = model(batch, prev_t)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        pred = probs >= 0.5
        truth = np.asarray(action).astype(bool)
        tp += np.logical_and(pred, truth)
        fp += np.logical_and(pred, np.logical_not(truth))
        fn += np.logical_and(np.logical_not(pred), truth)
        correct += pred == truth
        mask = controller.discrete_mask(probs)
        hyst = np.array([mask[key] for key in keys], dtype=bool)
        hyst_correct += hyst == truth
    per_key = {}
    for i, key in enumerate(keys):
        per_key[key] = {
            "accuracy": float(correct[i] / max(1, limit)),
            "f1": _f1(int(tp[i]), int(fp[i]), int(fn[i])),
            "hysteresis_accuracy": float(hyst_correct[i] / max(1, limit)),
        }
    steer = [per_key[key]["f1"] for key in ("a", "d", "left", "right") if key in per_key]
    return {
        "frames": int(limit),
        "per_key": per_key,
        "steer_f1": float(np.mean(steer)) if steer else float(np.mean([m["f1"] for m in per_key.values()])),
    }


def format_report(stats: dict) -> str:
    lines = [f"frames {stats['frames']}  steer_f1 {stats['steer_f1']:.3f}"]
    for key, metrics in stats["per_key"].items():
        lines.append(
            f"{key:>8}  acc {metrics['accuracy']:.3f}  f1 {metrics['f1']:.3f}  "
            f"hyst {metrics['hysteresis_accuracy']:.3f}"
        )
    if stats["steer_f1"] < 0.6:
        lines.append(
            "Steering agreement is below 0.60. Record more laps, including recoveries, "
            "before live play."
        )
    return "\n".join(lines)
