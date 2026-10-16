"""Run a fixed number of episodes and record finish, time, and progress."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch

from polytrack_agent.policy.cnn import resolve_device
from polytrack_agent.rl.headless import HeadlessTrack
from polytrack_agent.rl.device import configure_torch
from polytrack_agent.rl.checkpoint import bind_runtime, load_actor, read_checkpoint
from polytrack_agent.rl.state import LapNumbers

log = logging.getLogger(__name__)


def run_episodes(
    cfg: dict,
    checkpoint: Path,
    episodes: int,
    dry_run: bool = False,
    reference: Path | None = None,
    log_path: Path | None = None,
) -> list[dict]:
    device = resolve_device(str(cfg["rl"].get("device") or cfg["play"].get("device") or "auto"))
    configure_torch(device)
    payload = read_checkpoint(Path(checkpoint), device)
    cfg = bind_runtime(cfg, payload)
    model = load_actor(payload, device)
    del reference, dry_run
    env = HeadlessTrack(cfg)
    numbers = LapNumbers(int(payload["n_actions"]))
    rows: list[dict] = []
    log.info("PPO eval, %s headless episodes.", episodes)
    try:
        for episode in range(1, int(episodes) + 1):
            numbers.reset()
            env.reset(restart=True)
            started = time.perf_counter()
            finished = False
            max_progress = 0.0
            info: dict = {}
            while True:
                observation = torch.from_numpy(numbers.vector()).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _logp, _value = model.act(observation, greedy=True)
                action_id = int(action.item())
                _frame, _reward, done, info = env.step(action_id)
                if not info.get("counted", False):
                    continue
                numbers.advance(info, action_id)
                max_progress = max(max_progress, float(info.get("progress") or 0.0))
                if done:
                    finished = bool(info.get("finished"))
                    break
            seconds = time.perf_counter() - started
            row = {
                "episode": episode,
                "finished": int(finished),
                "seconds": round(seconds, 3),
                "max_progress": round(max_progress, 4),
                "stuck": int(bool(info.get("stuck"))),
                "timeout": int(bool(info.get("timeout"))),
                "restarted": int(bool(info.get("restarted"))),
            }
            rows.append(row)
            log.info(
                "episode %s  finished %s  %.2fs  progress %.2f",
                episode,
                bool(finished),
                seconds,
                max_progress,
            )
    finally:
        env.runtime.release_keys()
        env.close()
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["episode"])
            writer.writeheader()
            writer.writerows(rows)
    return rows
