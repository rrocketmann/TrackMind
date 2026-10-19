"""Run a fixed number of greedy episodes in the 0.6.3 simulation."""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

import torch

from polytrack_agent.rl.checkpoint import bind_runtime, load_actor, read_checkpoint
from polytrack_agent.rl.device import configure_torch, resolve_device
from polytrack_agent.rl.state import LapNumbers
from polytrack_agent.rl.track import make_track

log = logging.getLogger(__name__)

VIDEO_W = 640
VIDEO_H = 480
VIDEO_FPS = 30
PPM_HEADER_LEN = len(f"P6\n{VIDEO_W} {VIDEO_H}\n255\n".encode("ascii"))


def _encode_video(frames: list[str], out_path: Path) -> None:
    """Encode base64-encoded PPM frames to an MP4 with ffmpeg."""
    if not frames:
        log.warning("No video frames collected; not writing %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_W}x{VIDEO_H}",
        "-r", str(VIDEO_FPS),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for b64 in frames:
            raw = base64.b64decode(b64)
            # Strip PPM header to get raw RGB bytes
            proc.stdin.write(raw[PPM_HEADER_LEN:])
    finally:
        proc.stdin.close()
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {stderr.decode(errors='replace')}")
    log.info("Wrote %s (%d frames)", out_path, len(frames))


def run_episodes(
    cfg: dict,
    checkpoint: Path,
    episodes: int,
    video_out: Path | None = None,
) -> list[dict]:
    device = resolve_device(str(cfg["rl"].get("device") or cfg["play"].get("device") or "auto"))
    configure_torch(device)
    payload = read_checkpoint(Path(checkpoint), device)
    cfg = bind_runtime(cfg, payload)
    model = load_actor(payload, device)
    env = make_track(cfg)
    numbers = LapNumbers(int(payload["n_actions"]))
    rows: list[dict] = []
    try:
        if video_out is not None:
            env.client.start_recording()
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
        if video_out is not None:
            frames = env.client.stop_recording()
            _encode_video(frames, video_out)
        env.close()
    return rows
