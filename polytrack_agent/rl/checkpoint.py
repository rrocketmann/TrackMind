"""Save and load a PPO actor. The actor stores no picture encoder."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from polytrack_agent.rl.actions import n_actions
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.state import obs_dim

ARCH = "schoolagy-ppo-0.6.3"


def read_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(path: Path, model: StateActorCritic, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "arch": ARCH, **meta}
    torch.save(payload, path)


def bind_runtime(cfg: dict, payload: dict[str, Any]) -> dict:
    """Use the checkpoint's key list so each action id is the same chord as in training."""
    keys = [str(key) for key in payload["keys"]]
    chords = n_actions(keys)
    if chords != int(payload["n_actions"]):
        raise ValueError(
            f"checkpoint has {payload['n_actions']} chords but keys {keys} make {chords}"
        )
    if int(payload["obs_dim"]) != obs_dim(chords):
        raise ValueError(f"checkpoint obs_dim {payload['obs_dim']} does not match {obs_dim(chords)}")
    bound = copy.deepcopy(cfg)
    bound["keys"] = keys
    return bound


def load_actor(payload: dict[str, Any], device: torch.device) -> StateActorCritic:
    if payload.get("arch") != ARCH:
        raise ValueError(f"expected {ARCH}, got {payload.get('arch')}")
    model = StateActorCritic(
        obs_dim=int(payload["obs_dim"]),
        n_actions=int(payload["n_actions"]),
        hidden=int(payload.get("hidden") or 128),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
