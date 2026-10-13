"""Behavioral cloning: imitate the human key mask from a stack of RGB frames."""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from polytrack_agent.policy.cnn import DrivingPolicy, resolve_device, save_checkpoint
from polytrack_agent.record.dataset import DrivingDataset, collate_batch, make_splits

log = logging.getLogger(__name__)


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _selection_score(per_key_f1: dict[str, float], key_freq: dict[str, float]) -> float:
    steer = [key for key in per_key_f1 if key in {"a", "d", "left", "right"} and 0.01 < key_freq.get(key, 0.0) < 0.99]
    names = steer or [key for key, freq in key_freq.items() if 0.01 < freq < 0.99]
    if not names:
        names = list(per_key_f1)
    return float(np.mean([per_key_f1[key] for key in names]))


def _ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None) -> tuple[float, dict[str, dict[str, float]]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    seen = 0
    n_actions = None
    tp = fp = fn = None
    for frames, prev, target in loader:
        frames = frames.to(device)
        prev = prev.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if scaler is not None and training:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(frames, prev)
                    loss = criterion(logits, target)
            else:
                logits = model(frames, prev)
                loss = criterion(logits, target)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch = int(target.shape[0])
        total_loss += float(loss.item()) * batch
        seen += batch
        pred = (torch.sigmoid(logits.detach()) >= 0.5).to(torch.int32)
        truth = target.detach().to(torch.int32)
        if n_actions is None:
            n_actions = int(truth.shape[1])
            tp = np.zeros(n_actions, dtype=np.int64)
            fp = np.zeros(n_actions, dtype=np.int64)
            fn = np.zeros(n_actions, dtype=np.int64)
        pred_np = pred.cpu().numpy()
        truth_np = truth.cpu().numpy()
        tp += np.logical_and(pred_np == 1, truth_np == 1).sum(axis=0)
        fp += np.logical_and(pred_np == 1, truth_np == 0).sum(axis=0)
        fn += np.logical_and(pred_np == 0, truth_np == 1).sum(axis=0)
    metrics: dict[str, dict[str, float]] = {}
    if tp is None:
        return 0.0, metrics
    return total_loss / max(1, seen), {
        str(index): {"f1": _f1(int(tp[index]), int(fp[index]), int(fn[index]))}
        for index in range(len(tp))
    }


def train_behavior_cloning(cfg: dict, sessions: list[Path]) -> dict:
    train_cfg = cfg["train"]
    seed = int(train_cfg.get("seed") or 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    image_size = (int(train_cfg["image_size"][0]), int(train_cfg["image_size"][1]))
    stack = int(train_cfg["stack"])
    backbone = str(train_cfg.get("backbone") or "dinov3").lower()
    if backbone not in {"cnn", "dinov3"}:
        raise ValueError(f"Unknown backbone {backbone!r}. Use dinov3 or cnn.")
    augment = bool(train_cfg.get("augment", True))
    if backbone == "dinov3" and augment:
        log.info(
            "Pixel augment stays off for DINOv3. Each frame is encoded once, "
            "and the transformer trains on those tokens."
        )
        augment = False
    dataset = DrivingDataset(
        [Path(session) for session in sessions],
        image_size=image_size,
        stack=stack,
        augment=augment,
        max_gap_s=float(cfg["record"].get("max_gap_s") or 0.25),
        seed=seed,
    )
    train_idx, val_idx = make_splits(dataset, float(train_cfg.get("val_fraction") or 0.15), seed)

    freq = dataset.action_frequency()
    pos_weight = (1.0 - freq) / np.clip(freq, 0.02, 1.0)
    pos_weight = np.clip(pos_weight, 0.25, 20.0)
    key_freq = {key: float(freq[i]) for i, key in enumerate(dataset.keys)}
    episode_ids = {episode for _chunk, _t, episode in dataset.samples}
    if len(episode_ids) < 2:
        log.warning(
            "Only one episode is in the recordings. Validation uses a time tail, so the score is optimistic. "
            "Press F7 each time you restart while recording."
        )
    device = resolve_device(str(train_cfg.get("device") or "auto"))
    extra_meta: dict = {}
    if backbone == "cnn":
        # Validation must not be augmented. Share the sample index, use a second view with augment off.
        val_dataset = DrivingDataset(
            [Path(session) for session in sessions],
            image_size=image_size,
            stack=stack,
            augment=False,
            max_gap_s=float(cfg["record"].get("max_gap_s") or 0.25),
            seed=seed,
        )
        if len(val_dataset) != len(dataset):
            raise RuntimeError("train/val datasets indexed different samples")
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(val_dataset, val_idx)
        model = DrivingPolicy(
            stack=stack,
            n_actions=dataset.n_actions,
            use_prev_action=bool(train_cfg.get("use_prev_action", True)),
            dropout=float(train_cfg.get("dropout") or 0.0),
        ).to(device)
        collate_fn = collate_batch
    else:
        from polytrack_agent.policy.dinov3 import (
            EncodedDrivingDataset,
            build_dino_policy,
            collate_tokens,
            encode_chunks,
        )

        encoder, head = build_dino_policy(cfg, dataset.n_actions, device)
        encoded = encode_chunks(dataset, encoder, device, batch_size=8)
        encoded_ds = EncodedDrivingDataset(dataset, encoded)
        train_ds = Subset(encoded_ds, train_idx)
        val_ds = Subset(encoded_ds, val_idx)
        model = head
        collate_fn = collate_tokens
        extra_meta = {
            "arch": "dinov3-transformer-v1",
            "dino_model": encoder.model_id,
            "feature_dim": encoder.hidden_size,
            "pool": [encoder.pool_hw[0], encoder.pool_hw[1]],
            "n_tokens": encoder.n_tokens,
            "transformer_layers": head.layers,
            "transformer_heads": head.heads,
        }
        encoder.model.cpu()
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log.info(
        "Vision samples %s  train %s  val %s  keys %s",
        len(dataset),
        len(train_ds),
        len(val_ds),
        " ".join(f"{key}={key_freq[key]:.2f}" for key in dataset.keys),
    )
    ema = copy.deepcopy(model)
    ema.eval()
    for param in ema.parameters():
        param.requires_grad_(False)

    batch_size = int(train_cfg.get("batch_size") or 64)
    workers = int(train_cfg.get("num_workers") or 0)
    sampler_name = str(train_cfg.get("sampler") or "weighted").lower()
    generator = torch.Generator()
    generator.manual_seed(seed)
    if sampler_name == "shuffle":
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
            generator=generator,
        )
    elif sampler_name == "weighted":
        weights = torch.as_tensor([dataset.weight_for(i) for i in train_idx], dtype=torch.double)
        n_samples = train_cfg.get("samples_per_epoch")
        if n_samples is None or int(n_samples) <= 0:
            n_samples = len(train_ds)
        else:
            n_samples = int(n_samples)
        sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True, generator=generator)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=collate_fn,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler_name!r}")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr") or 3e-4),
        weight_decay=float(train_cfg.get("weight_decay") or 0.0),
    )
    epochs = int(train_cfg.get("epochs") or 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    ema_decay = float(train_cfg.get("ema_decay") if train_cfg.get("ema_decay") is not None else 0.99)
    patience = int(train_cfg.get("patience") or epochs)
    out_dir = Path(train_cfg.get("out_dir") or "runs/bc")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0
    best_epoch = -1
    stale = 0
    best_path = out_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        train_loss, _train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler
        )
        if ema_decay <= 0.0 or ema_decay >= 1.0:
            ema.load_state_dict(model.state_dict())
        else:
            _ema_update(ema, model, ema_decay)
        val_loss, val_raw = _run_epoch(ema, val_loader, criterion, device)
        per_key = {dataset.keys[int(index)]: values["f1"] for index, values in val_raw.items()}
        score = _selection_score(per_key, key_freq)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "score": score,
            "f1": per_key,
        }
        history.append(row)
        f1_text = " ".join(f"{key}={per_key[key]:.2f}" for key in dataset.keys)
        log.info(
            "epoch %s/%s  train %.4f  val %.4f  score %.3f  %s",
            epoch,
            epochs,
            train_loss,
            val_loss,
            score,
            f1_text,
        )
        meta = {
            "keys": dataset.keys,
            "image_size": [image_size[0], image_size[1]],
            "stack": stack,
            "use_prev_action": bool(train_cfg.get("use_prev_action", True)),
            "dropout": float(train_cfg.get("dropout") or 0.0),
            "key_freq": [float(freq[i]) for i in range(len(dataset.keys))],
            "epoch": epoch,
            "score": score,
            **extra_meta,
        }
        save_checkpoint(out_dir / "last.pt", ema, meta)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(best_path, ema, meta)
        else:
            stale += 1
            if stale >= patience:
                log.info("Early stop at epoch %s. Best score %.3f at epoch %s.", epoch, best_score, best_epoch)
                break

    report = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_path": str(best_path),
        "keys": dataset.keys,
        "key_freq": key_freq,
        "epochs": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Best vision checkpoint %s  score %.3f (epoch %s)", best_path, best_score, best_epoch)
    if best_score < 0.6:
        log.warning(
            "Steering score %.2f is below 0.60. The policy is not ready to drive. "
            "Record more laps, including recoveries, and train again.",
            best_score,
        )
    return report
