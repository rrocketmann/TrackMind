"""Save and load a GRPO actor. DINOv3 weights stay with Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from polytrack_agent.rl.model import TokenActorCritic

ARCH = "dinov3-grpo-v1"


def read_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(path: Path, model: TokenActorCritic, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "arch": ARCH, **meta}
    torch.save(payload, path)


def load_actor(payload: dict[str, Any], device: torch.device) -> TokenActorCritic:
    if payload.get("arch") != ARCH:
        raise ValueError(f"expected {ARCH}, got {payload.get('arch')}")
    model = TokenActorCritic(
        feature_dim=int(payload["feature_dim"]),
        n_tokens=int(payload["n_tokens"]),
        stack=int(payload["stack"]),
        n_actions=int(payload["n_actions"]),
        layers=int(payload.get("transformer_layers") or 2),
        heads=int(payload.get("transformer_heads") or 6),
        dropout=0.0,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
