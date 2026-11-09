#!/usr/bin/env python3
"""Record a greedy drive of a saved checkpoint in the local PolyTrack game.

The video is the chase camera of the polytrack-0.6.3 checkout, served on
127.0.0.1. It is a fresh playback of the weights, not the stochastic episode
that produced the checkpoint. The script does not open an external page and
it does not submit a time.

Live graphs of a training run are a different window:

    python scripts/live.py --run runs/winter4

A chase-camera video:

    python scripts/record.py --checkpoint runs/winter4/best.pt --out runs/winter4/best.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.config import load_config
from polytrack_agent.rl.actions import mask_for
from polytrack_agent.rl.checkpoint import bind_runtime, load_actor, read_checkpoint
from polytrack_agent.rl.device import configure_torch, resolve_device
from polytrack_agent.rl.state import LapNumbers
from polytrack_agent.rl.track import game_root, make_track


def track_title(stem: str, override: str | None) -> str:
    """Card title the local client shows for an official track stem."""
    if override:
        return override
    match = re.fullmatch(r"(summer|winter|desert)(\d+)", stem.lower())
    if match is None:
        return stem
    return f"{match.group(1).title()} {match.group(2)}"


def drive(cfg: dict, checkpoint: Path) -> dict:
    """Greedy lap in the worker. Each step is the keys held from that sim time."""
    device = resolve_device(str(cfg["rl"].get("device") or cfg["play"].get("device") or "auto"))
    configure_torch(device)
    payload = read_checkpoint(checkpoint, device)
    cfg = bind_runtime(cfg, payload)
    stem = str(payload.get("track") or Path(cfg["game"]["track"]).stem)
    official = game_root(cfg) / "tracks" / "official" / f"{stem}.track"
    if official.is_file():
        cfg["game"]["track"] = str(official)
    model = load_actor(payload, device)
    env = make_track(cfg)
    numbers = LapNumbers(int(payload["n_actions"]))
    keys = list(cfg["keys"])
    reset_action = bool(cfg["rl"].get("reset_action", False))
    repeat = max(1, int(cfg["rl"].get("action_repeat") or 50))
    steps: list[dict] = []
    try:
        info = env.reset(restart=True)
        numbers.reset()
        numbers.advance(info, -1)
        finished = False
        while True:
            frames = int(info["state"]["frames"])
            observation = torch.from_numpy(numbers.vector()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _logp, _value = model.act(observation, greedy=True)
            action_id = int(action.item())
            mask = mask_for(action_id, keys, reset_action=reset_action)
            steps.append({
                "t": round(frames * 0.001, 3),
                "keys": [key for key in keys if mask.get(key)],
            })
            _frame, _reward, done, info = env.step(action_id)
            numbers.advance(info, action_id)
            if done:
                finished = bool(info.get("finished"))
                break
        state = info["state"]
        seconds = int(state["finishFrames"]) * 0.001 if finished else int(state["frames"]) * 0.001
    finally:
        env.close()
    return {
        "steps": steps,
        "progress": round(float(info.get("progress") or 0.0), 3),
        "finished": finished,
        "seconds": round(seconds, 3),
        "hold_s": round(repeat * 0.001, 3),
        "track": stem,
        "update": payload.get("update"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record a greedy drive of a saved checkpoint in the local PolyTrack 0.6.3 game. "
            "This is a fresh playback of the weights, not the original training episode. "
            "It does not open an external page and it does not submit a time. "
            "Live graphs are python scripts/live.py --run runs/<name>."
        ),
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML overlay merged onto the defaults")
    parser.add_argument("--checkpoint", type=Path, default=None, help="defaults to play.checkpoint")
    parser.add_argument("--out", type=Path, default=None, help="mp4 path. Defaults to the checkpoint with a .mp4 suffix")
    parser.add_argument("--track-name", default=None, help="card title in the local client, such as 'Winter 4'")
    args = parser.parse_args(argv)
    if not os.environ.get("DISPLAY"):
        print(
            "Recording needs DISPLAY and a real GPU. "
            "A virtual framebuffer renders too slowly for a real-time lap.",
            file=sys.stderr,
        )
        return 1
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or Path(cfg["play"]["checkpoint"])
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}. Train first.", file=sys.stderr)
        return 1
    out = (args.out or checkpoint.with_suffix(".mp4")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lap = drive(cfg, checkpoint)
    title = track_title(str(lap["track"]), args.track_name)
    print(
        f"greedy drive  {lap['seconds']:.3f}s  progress {lap['progress']:.3f}  "
        f"finished {lap['finished']}  {len(lap['steps'])} steps  {title}"
    )
    script = Path(__file__).resolve().parent / "record_game.mjs"
    env = os.environ.copy()
    env["POLYTRACK_ROOT"] = str(game_root(cfg))
    handle = tempfile.NamedTemporaryFile(
        prefix="polytrack-drive-",
        suffix=".json",
        delete=False,
    )
    timeline = Path(handle.name)
    handle.close()
    try:
        timeline.write_text(json.dumps(lap), encoding="utf-8")
        completed = subprocess.run(
            ["node", str(script), str(timeline), str(out), title],
            env=env,
        )
    finally:
        timeline.unlink(missing_ok=True)
    if completed.returncode != 0:
        return completed.returncode
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
