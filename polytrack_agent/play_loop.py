"""Real-time vision loop: grab, infer, press keys, track visual progress."""

from __future__ import annotations

import csv
import logging
import signal
import time
from pathlib import Path

import numpy as np
import torch

from polytrack_agent.input.keys import format_mask, zeros_action
from polytrack_agent.input.keyboard import StopHotkey
from polytrack_agent.obs.preprocess import FrameStack
from polytrack_agent.obs.progress import NullProgress, load_reference
from polytrack_agent.policy.cnn import frames_to_batch, load_checkpoint, resolve_device
from polytrack_agent.policy.control import Controller
from polytrack_agent.reward import RewardTracker
from polytrack_agent.runtime import VisionRuntime

log = logging.getLogger(__name__)


def _tracker(reference: Path | None, cfg: dict):
    if reference is None:
        log.info("No visual reference. Progress stays at 0 until you build one.")
        return NullProgress()
    path = Path(reference)
    if not path.is_file():
        log.warning("Reference %s was not found. Progress stays at 0.", path)
        return NullProgress()
    tracker = load_reference(path, cfg["progress"])
    log.info("Visual reference %s (%s frames)", path, len(tracker.ref))
    return tracker


def play_live(
    cfg: dict,
    checkpoint: Path,
    dry_run: bool = False,
    reference: Path | None = None,
    log_path: Path | None = None,
) -> None:
    device = resolve_device(str(cfg["play"].get("device") or "auto"))
    model, payload = load_checkpoint(Path(checkpoint), device)
    keys = [str(key) for key in payload["keys"]]
    image_h, image_w = int(payload["image_size"][0]), int(payload["image_size"][1])
    stacker = FrameStack(int(payload["stack"]), image_h, image_w)
    controller = Controller(
        keys,
        mode=str(cfg["play"].get("mode") or "discrete"),
        threshold_on=float(cfg["play"]["threshold_on"]),
        threshold_off=float(cfg["play"]["threshold_off"]),
        deadzone=float(cfg["play"]["deadzone"]),
        exclusive_axes=bool(cfg["play"].get("exclusive_axes", True)),
        min_key_freq=float(cfg["play"].get("min_key_freq") or 0.0),
        key_freq=payload.get("key_freq"),
    )
    runtime = VisionRuntime(cfg, keys, dry_run=dry_run)
    progress = _tracker(reference, cfg)
    reward_cfg = cfg["reward"]
    reward_tracker = RewardTracker(
        progress_scale=float(reward_cfg["progress_scale"]),
        time_penalty=float(reward_cfg["time_penalty"]),
        restart_penalty=float(reward_cfg["restart_penalty"]),
        finish_bonus=float(reward_cfg["finish_bonus"]),
    )
    hotkey = StopHotkey()
    try:
        hotkey.start()
    except Exception as exc:
        log.warning("F8 listener unavailable (%s). Ctrl+C still releases keys.", exc)

    previous = signal.getsignal(signal.SIGTERM)

    def _term(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    log_file = None
    writer = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(log_file)
        writer.writerow(
            ["t", "dt", "progress", "reward", "restarted", "finished", "confidence", "infer_ms", "keys", "probs"]
        )

    prev = zeros_action(len(keys))
    arch = str(payload.get("arch") or "vision-cnn-v1")
    log.info("Vision policy %s (%s) on %s", checkpoint, arch, device)
    if payload.get("dino_model"):
        log.info("Frozen encoder %s", payload["dino_model"])
    log.info("Keys %s  mode %s  input %sx%s x%s", " ".join(keys), controller.mode, image_w, image_h, stacker.stack)
    if dry_run:
        log.info("Dry run: predictions only, no keys will be pressed.")
    else:
        log.info("Pressing keys only while a window titled %r is focused.", cfg["capture"].get("window_title"))
        log.info("F8 or Ctrl+C releases every key and stops.")
        log.info("A real finish on polytrack.dev is submitted by the game. This agent does not POST times.")
    finish_latched = False
    last_report = time.perf_counter()
    loops = 0
    infer_acc = 0.0
    last_render = ""
    started = time.perf_counter()
    try:
        while not hotkey.event.is_set():
            t0 = time.perf_counter()
            frame = runtime.grab()
            if frame is None:
                time.sleep(0.01)
                continue
            obs = stacker.push(frame)
            infer_t = time.perf_counter()
            with torch.inference_mode():
                if arch == "dinov3-transformer-v1":
                    prev_t = torch.from_numpy(np.ascontiguousarray(prev.astype(np.float32, copy=False)))
                    prev_t = prev_t.view(1, -1).to(device=device, dtype=torch.float32)
                    logits = model.forward_frame(obs[-1], prev_t)
                else:
                    batch, prev_t = frames_to_batch(obs, prev, device)
                    logits = model(batch, prev_t)
                probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
            infer_ms = (time.perf_counter() - infer_t) * 1000.0
            if controller.mode == "continuous":
                holds = controller.continuous_holds(probs, runtime.period)
                elapsed = time.perf_counter() - t0
                budget = max(0.0, runtime.period - elapsed)
                scale = (budget / runtime.period) if runtime.period else 0.0
                scaled = {key: holds[key] * scale for key in keys}
                runtime.apply_timed_holds(scaled, budget)
                mask = {key: scaled.get(key, 0.0) > 1e-4 for key in keys}
            else:
                mask = controller.discrete_mask(probs)
                runtime.apply_mask(mask)
                remain = runtime.period - (time.perf_counter() - t0)
                if remain > 0:
                    time.sleep(remain)
            prev = np.array([1 if mask[key] else 0 for key in keys], dtype=np.uint8)
            prog = progress.update(frame)
            dt = time.perf_counter() - t0
            reward, _done = reward_tracker.step(prog, dt)
            if prog.restarted:
                log.info("Visual restart. Progress %.2f", prog.progress)
                finish_latched = False
            elif prog.finished and not finish_latched:
                log.info("Visual finish. Progress %.2f", prog.progress)
                finish_latched = True
            rendered = format_mask(mask, keys)
            if dry_run and rendered != last_render:
                pretty = " ".join(f"{value:.2f}" for value in probs)
                log.info("keys %s  probs %s", rendered, pretty)
            last_render = rendered
            if writer is not None:
                writer.writerow(
                    [
                        f"{time.perf_counter() - started:.3f}",
                        f"{dt:.4f}",
                        f"{prog.progress:.4f}",
                        f"{reward:.4f}",
                        int(prog.restarted),
                        int(prog.finished),
                        f"{prog.confidence:.3f}",
                        f"{infer_ms:.2f}",
                        rendered,
                        " ".join(f"{value:.3f}" for value in probs),
                    ]
                )
                log_file.flush()
            loops += 1
            infer_acc += infer_ms
            now = time.perf_counter()
            if now - last_report >= 2.0:
                log.info(
                    "%.1f fps  infer %.1f ms  progress %.2f  %s",
                    loops / (now - last_report),
                    infer_acc / max(1, loops),
                    prog.progress,
                    rendered,
                )
                loops = 0
                infer_acc = 0.0
                last_report = now
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        signal.signal(signal.SIGTERM, previous)
        hotkey.stop()
        runtime.close()
        if log_file is not None:
            log_file.close()
        log.info("Keys released.")
