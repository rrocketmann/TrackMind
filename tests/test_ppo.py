"""GRPO pieces that do not download DINOv3 or open Chromium."""

from __future__ import annotations

import numpy as np
import torch

from polytrack_agent.config import load_config
from polytrack_agent.env import PolyTrackVisionEnv
from polytrack_agent.obs.preprocess import rgb_to_tiny_gray
from polytrack_agent.obs.progress import ReferenceProgress
from polytrack_agent.rl.actions import action_masks
from polytrack_agent.rl.device import configure_torch
from polytrack_agent.rl.gae import compute_gae
from polytrack_agent.rl.grpo import group_advantages, grpo_loss, grpo_update
from polytrack_agent.rl.model import TokenActorCritic
from polytrack_agent.rl.trainer import train_grpo


def test_action_table_is_exclusive():
    keys = ["w", "a", "s", "d", "q", "e"]
    masks = action_masks(keys)
    assert len(masks) == 27
    seen = set()
    for mask in masks:
        assert not (mask["w"] and mask["s"])
        assert not (mask["a"] and mask["d"])
        assert not (mask["q"] and mask["e"])
        assert "r" not in mask
        seen.add(tuple(mask[key] for key in keys))
    assert len(seen) == 27


def test_gae_stops_at_done():
    advantages, returns = compute_gae(
        rewards=np.array([1.0, 5.0], dtype=np.float32),
        dones=np.array([1.0, 0.0], dtype=np.float32),
        values=np.zeros(2, dtype=np.float32),
        last_value=0.0,
        gamma=1.0,
        lam=1.0,
    )
    assert abs(float(advantages[0]) - 1.0) < 1e-5
    assert abs(float(advantages[1]) - 5.0) < 1e-5
    assert abs(float(returns[1]) - 5.0) < 1e-5


def test_cpu_setup_does_not_touch_cuda():
    assert configure_torch(torch.device("cpu")) == "cpu"


def test_group_advantages_are_shared_inside_a_lap():
    advantages = group_advantages([0.0, 2.0], [1, 2])
    assert advantages.shape == (3,)
    assert advantages[0] < 0 < advantages[1]
    assert abs(float(advantages[1] - advantages[2])) < 1e-6
    assert np.allclose(group_advantages([1.0, 1.0], [2, 2]), 0.0)


def test_grpo_loss_reaches_the_actor():
    model = TokenActorCritic(feature_dim=16, n_tokens=4, stack=2, n_actions=27, layers=1, heads=4, dropout=0.0)
    tokens = torch.randn(8, 2, 4, 16)
    logits, _values = model(tokens)
    actions = torch.randint(0, 27, (8,))
    old = torch.zeros(8)
    advantages = torch.tensor([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    loss, stats = grpo_loss(logits, actions, old, advantages, 0.2, 0.01, 0.02, 0.01)
    loss.backward()
    assert model.policy.weight.grad is not None
    assert model.value.weight.grad is None
    assert np.isfinite(stats["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.policy.weight.detach().clone()
    grpo_update(
        model,
        optimizer,
        tokens.detach(),
        actions,
        old,
        advantages,
        epochs=1,
        minibatch_size=4,
        clip=0.2,
        entropy_coef=0.01,
        kl_coef=0.02,
        unimix=0.01,
    )
    assert not torch.equal(before, model.policy.weight)


class _Encoder:
    model_id = "fake"
    image_size = (12, 16)
    hidden_size = 16
    n_tokens = 4
    pool_hw = (2, 2)

    def encode_frames(self, frames):
        return torch.zeros(frames.shape[0], self.n_tokens, self.hidden_size, device=frames.device)


class _Cap:
    def __init__(self, frame):
        self.frame = frame

    def grab(self):
        return self.frame.copy()

    def set_region(self, region):
        del region

    def close(self):
        return None


def test_grpo_trainer_updates_on_a_fake_window(tmp_path):
    cfg = load_config(None)
    cfg["capture"]["fps"] = 1000
    cfg["capture"]["region"] = {"left": 0, "top": 0, "width": 32, "height": 24}
    cfg["capture"]["follow_window"] = False
    cfg["play"]["require_focus"] = False
    cfg["train"]["image_size"] = [12, 16]
    cfg["train"]["stack"] = 2
    cfg["rl"]["stack"] = 2
    cfg["rl"]["action_repeat"] = 1
    cfg["rl"]["group_size"] = 2
    cfg["rl"]["epochs"] = 1
    cfg["rl"]["minibatch_size"] = 2
    cfg["rl"]["transformer_layers"] = 1
    cfg["rl"]["transformer_heads"] = 4
    cfg["rl"]["device"] = "cpu"
    cfg["rl"]["stuck_s"] = 30
    cfg["rl"]["timeout_s"] = 0.05
    cfg["rl"]["reset_timeout_s"] = 2
    cfg["rl"]["finish_settle_s"] = 0
    cfg["rl"]["out_dir"] = str(tmp_path / "run")
    start = np.zeros((24, 32, 3), dtype=np.uint8)
    start[:, :16, 1] = 180
    start[:, 16:, 2] = 40
    other = np.zeros_like(start)
    other[:12, :, 0] = 180
    other[12:, :, 1] = 20
    progress = ReferenceProgress(
        np.stack([rgb_to_tiny_gray(start, 8, 8), rgb_to_tiny_gray(other, 8, 8)]),
        search_window=1,
        match_min=0.5,
        finish_hold=100,
        restart_hold=100,
    )
    env = PolyTrackVisionEnv(cfg, capture=_Cap(start), dry_run=True, progress=progress)
    try:
        report = train_grpo(cfg, encoder=_Encoder(), env=env, max_updates=1)
    finally:
        env.close()
    assert report["updates"] == 1
    assert (tmp_path / "run" / "last.pt").is_file()
