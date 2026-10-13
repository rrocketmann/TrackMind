"""Small CNN from a stack of RGB frames to one logit per driving key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.out_dim = 64 * 4 * 4

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames)


class DrivingPolicy(nn.Module):
    """Map stacked RGB frames to key logits.

    The only non-pixel input is the previous key mask this agent applied.
    Set `use_prev_action` false for a pure-pixel policy.
    """

    def __init__(
        self,
        stack: int,
        n_actions: int,
        use_prev_action: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stack = int(stack)
        self.n_actions = int(n_actions)
        self.use_prev_action = bool(use_prev_action)
        self.encoder = VisionEncoder(in_channels=self.stack * 3)
        fused = self.encoder.out_dim + (self.n_actions if self.use_prev_action else 0)
        self.head = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, self.n_actions),
        )
        # Start below the press threshold so an untrained net does not mash keys.
        nn.init.constant_(self.head[-1].bias, -1.5)

    def forward(self, frames: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        # frames: (B, T, 3, H, W) in 0..1
        batch, steps, channels, height, width = frames.shape
        if steps != self.stack:
            raise ValueError(f"expected stack {self.stack}, got {steps}")
        if channels != 3:
            raise ValueError(f"expected RGB frames, got {channels} channels")
        flat = frames.reshape(batch, steps * channels, height, width)
        features = self.encoder(flat)
        if self.use_prev_action:
            if prev_action is None:
                prev_action = torch.zeros(
                    batch,
                    self.n_actions,
                    device=frames.device,
                    dtype=frames.dtype,
                )
            features = torch.cat([features, prev_action], dim=-1)
        return self.head(features)


def frames_to_batch(
    stack_uint8: np.ndarray,
    prev_action: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one (T, H, W, 3) uint8 stack into a batch of 1."""
    frames = torch.from_numpy(np.ascontiguousarray(stack_uint8))
    frames = frames.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    frames = frames.to(device=device, dtype=torch.float32).div_(255.0)
    prev = torch.from_numpy(np.ascontiguousarray(prev_action.astype(np.float32, copy=False)))
    prev = prev.view(1, -1).to(device=device, dtype=torch.float32)
    return frames, prev


def save_checkpoint(path: Path, model: nn.Module, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "arch": "vision-cnn-v1", **meta}
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    arch = payload.get("arch")
    if arch == "dinov3-transformer-v1":
        from polytrack_agent.policy.dinov3 import load_dino_policy

        return load_dino_policy(payload, device), payload
    if arch != "vision-cnn-v1":
        raise ValueError(f"{path} is not a vision-cnn-v1 checkpoint")
    model = DrivingPolicy(
        stack=int(payload["stack"]),
        n_actions=len(payload["keys"]),
        use_prev_action=bool(payload["use_prev_action"]),
        dropout=float(payload.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
