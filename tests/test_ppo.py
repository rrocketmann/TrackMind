"""PPO pieces, plus one short lap on the schoolagy simulation when it is present."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from polytrack_agent.config import load_config, validate_output_keys
from polytrack_agent.rl.actions import action_masks, forward_ids, n_actions
from polytrack_agent.rl.checkpoint import ARCH, read_checkpoint
from polytrack_agent.rl.gae import compute_gae
from polytrack_agent.rl.ppo import ppo_loss
from polytrack_agent.rl.state import FEATURES, LapNumbers, obs_dim
from polytrack_agent.rl.track import game_root, make_track
from polytrack_agent.rl.trainer import train_ppo

ROOT = Path(__file__).resolve().parents[1]


def test_wasd_is_nine_exclusive_holds():
    keys = ["w", "a", "s", "d"]
    masks = action_masks(keys)
    assert n_actions(keys) == 9
    seen = set()
    for mask in masks:
        assert not (mask["w"] and mask["s"])
        assert not (mask["a"] and mask["d"])
        seen.add(tuple(mask[key] for key in keys))
    assert len(seen) == 9
    assert len(forward_ids(keys)) == 3
    with pytest.raises(ValueError):
        action_masks(["w", "q"])
    with pytest.raises(ValueError):
        validate_output_keys(["w", "r"])


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


def test_observation_is_car_state_plus_previous_action():
    numbers = LapNumbers(9)
    numbers.advance(
        {
            "state": {
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "speedKmh": 100.0,
                "wheels": [True, True, False, False],
                "nextCheckpointIndex": 1,
                "position": {"y": 10.0},
                "checkpointCount": 4,
            }
        },
        0,
    )
    vector = numbers.vector()
    assert obs_dim(9) == FEATURES + 9
    assert vector.shape == (obs_dim(9),)
    assert vector[0] == pytest.approx(0.5)
    assert vector[FEATURES] == 1.0
    assert float(vector[FEATURES + 1 :].sum()) == 0.0


def test_ppo_loss_is_finite():
    logits = torch.zeros(4, 9)
    values = torch.zeros(4)
    actions = torch.zeros(4, dtype=torch.int64)
    old_logp = torch.full((4,), -np.log(9))
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    returns = torch.zeros(4)
    loss, stats = ppo_loss(logits, values, actions, old_logp, advantages, returns, 0.2, 0.01, 0.5, 0.0)
    assert torch.isfinite(loss)
    assert set(stats) == {"loss", "policy", "value", "entropy"}


def test_missing_game_root_is_an_error(monkeypatch):
    monkeypatch.delenv("POLYTRACK_ROOT", raising=False)
    cfg = load_config(None)
    cfg["game"]["root"] = None
    with pytest.raises(ValueError):
        game_root(cfg)


def _sim_ready() -> bool:
    raw = os.environ.get("POLYTRACK_ROOT")
    if not raw:
        return False
    root = Path(raw)
    parts = ROOT / "polytrack_agent" / "sim" / "cache" / "track_parts.json"
    return (
        shutil.which("node") is not None
        and (root / "simulation_worker.bundle.js").is_file()
        and (root / "polytrack_physics.wasm").is_file()
        and parts.is_file()
    )


def test_throttle_moves_the_sim_car():
    if not _sim_ready():
        pytest.skip("schoolagy simulation or the mesh cache is not available")
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 200
    env = make_track(cfg)
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        info = {}
        for _ in range(4):
            _frame, _reward, done, info = env.step(forward)
            if done:
                break
        assert info["speed"] > 10.0
        assert int(info["state"]["frames"]) > 1
    finally:
        env.close()


def test_one_ppo_update_on_the_sim(tmp_path):
    if not _sim_ready():
        pytest.skip("schoolagy simulation or the mesh cache is not available")
    cfg = load_config(None)
    cfg["rl"]["out_dir"] = str(tmp_path / "run")
    cfg["rl"]["group_size"] = 2
    cfg["rl"]["timeout_s"] = 0.4
    cfg["rl"]["stuck_s"] = 30
    cfg["rl"]["epochs"] = 1
    cfg["rl"]["minibatch_size"] = 4
    cfg["rl"]["hidden"] = 32
    cfg["rl"]["device"] = "cpu"
    cfg["rl"]["resume"] = False
    report = train_ppo(cfg, max_updates=1)
    assert report["updates"] == 1
    payload = read_checkpoint(tmp_path / "run" / "last.pt", torch.device("cpu"))
    assert payload["arch"] == ARCH == "schoolagy-ppo-0.6.3"
    assert payload["track"] == "summer1"
    assert payload["n_actions"] == 9
    assert payload["obs_dim"] == obs_dim(9)
