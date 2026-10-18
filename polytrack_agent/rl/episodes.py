"""Run a fixed number of greedy episodes in the 0.6.3 simulation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from polytrack_agent.rl.checkpoint import bind_runtime, load_actor, read_checkpoint
from polytrack_agent.rl.device import configure_torch, resolve_device
from polytrack_agent.rl.state import LapNumbers
from polytrack_agent.rl.track import make_track

log = logging.getLogger(__name__)


def run_episodes(cfg: dict, checkpoint: Path, episodes: int) -> list[dict]:
    device = resolve_device(str(cfg["rl"].get("device") or cfg["play"].get("device") or "auto"))
    configure_torch(device)
    payload = read_checkpoint(Path(checkpoint), device)
    cfg = bind_runtime(cfg, payload)
    model = load_actor(payload, device)
    env = make_track(cfg)
    numbers = LapNumbers(int(payload["n_actions"]))
    rows: list[dict] = []
    try:
        for episode in range(1, int(episodes) + 1):
            numbers.reset()
            numbers.advance(env.reset(restart=True), -1)
            finished = False
            info: dict = {}
            while True:
                observation = torch.from_numpy(numbers.vector()).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _logp, _value = model.act(observation, greedy=True)
                action_id = int(action.item())
                _frame, _reward, done, info = env.step(action_id)
                numbers.advance(info, action_id)
                if done:
                    finished = bool(info.get("finished"))
                    break
            seconds = int(info["state"]["frames"]) * 0.001
            row = {
                "episode": episode,
                "finished": finished,
                "seconds": round(seconds, 3),
                "progress": round(float(info.get("progress") or 0.0), 3),
                "checkpoints": int(info["state"]["nextCheckpointIndex"]),
            }
            rows.append(row)
            log.info(
                "episode %s  finish %s  %.2fs  progress %.2f  checkpoints %s",
                episode,
                finished,
                seconds,
                row["progress"],
                row["checkpoints"],
            )
    finally:
        env.close()
    return rows
