"""Talk to the Node host that steps the game's own simulation worker."""

from __future__ import annotations

import base64
import json
import subprocess
import threading
from pathlib import Path


class SimClient:
    def __init__(self, root: Path, parts: Path, track: Path):
        host = Path(__file__).resolve().parent / "host.mjs"
        self._errors: list[str] = []
        self._proc = subprocess.Popen(
            ["node", str(host)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._proc.stderr is not None
        threading.Thread(target=self._read_errors, args=(self._proc.stderr,), daemon=True).start()
        self._send({"cmd": "boot", "root": str(root), "parts": str(parts), "track": str(track)})
        ready = self._expect("ready")
        self.checkpoint_count = int(ready["checkpointCount"])
        self.parts = ready.get("parts") or []
        self.frames = 0
        self._recording = False
        self._video_frames: list[str] = []

    def _read_errors(self, pipe) -> None:
        for line in pipe:
            self._errors.append(line.rstrip())

    def _send(self, message: dict) -> None:
        if self._proc.stdin is None or self._proc.poll() is not None:
            raise RuntimeError(self._failure("simulation host stopped"))
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _failure(self, prefix: str) -> str:
        detail = "\n".join(self._errors[-20:])
        return f"{prefix}\n{detail}" if detail else prefix

    def _expect(self, kind: str) -> dict:
        if self._proc.stdout is None:
            raise RuntimeError("simulation host has no output")
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(self._failure("simulation host closed its output"))
            if not line.startswith("{"):
                continue
            message = json.loads(line)
            if message.get("type") == "error":
                raise RuntimeError(message["error"])
            if message.get("type") == "video-frames":
                # Flushed video frames from the host
                self._video_frames.extend(message.get("frames", []))
                continue
            if message.get("type") == kind:
                if kind == "state":
                    self.frames = int(message["state"]["frames"])
                    if self._recording and message.get("frame"):
                        self._video_frames.append(message["frame"])
                return message

    def start_recording(self) -> None:
        self._video_frames = []
        self._send({"cmd": "record", "enable": True})
        self._recording = True
        # Drain any ack
        self._expect("ack")

    def stop_recording(self) -> list[str]:
        self._send({"cmd": "flush-video"})
        self._recording = False
        self._expect("video-frames")
        frames = list(self._video_frames)
        self._video_frames = []
        return frames

    def reset(self) -> dict:
        self._send({"cmd": "reset"})
        message = self._expect("state")
        return message["state"]

    def step(self, controls: dict, frames: int) -> dict:
        self._send({"cmd": "step", "controls": controls, "target": int(frames)})
        message = self._expect("state")
        return message["state"]

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._send({"cmd": "close"})
        except RuntimeError:
            pass
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
