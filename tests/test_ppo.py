"""PPO pieces that do not download DINOv3 or open Chromium."""

from __future__ import annotations

import numpy as np
import torch

from polytrack_agent.config import load_config
from polytrack_agent.env import PolyTrackVisionEnv
from polytrack_agent.obs.preprocess import rgb_to_tiny_gray
from polytrack_agent.obs.progress import ReferenceProgress
from polytrack_agent.rl.actions import action_masks, forward_ids, mask_for
from polytrack_agent.rl.checkpoint import bind_runtime, load_actor, read_checkpoint
from polytrack_agent.rl.device import configure_torch
from polytrack_agent.rl.gae import compute_gae
from polytrack_agent.rl.headless import HeadlessTrack
from polytrack_agent.rl.summer1 import GATES as SUMMER1_GATES
from polytrack_agent.rl.summer1 import START as SUMMER1_START
from polytrack_agent.rl.summer1 import make_track
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.ppo import ppo_loss, ppo_update
from polytrack_agent.rl.state import LapNumbers, obs_dim
from polytrack_agent.rl.trainer import _episode_gae, train_ppo


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


def test_train_ppo_needs_no_display(tmp_path):
    cfg = load_config(None)
    cfg["rl"]["out_dir"] = str(tmp_path / "run")
    cfg["rl"]["group_size"] = 2
    cfg["rl"]["timeout_s"] = 0.4
    cfg["rl"]["stuck_s"] = 30
    cfg["rl"]["epochs"] = 1
    cfg["rl"]["minibatch_size"] = 4
    cfg["rl"]["hidden"] = 32
    cfg["rl"]["device"] = "cpu"
    cfg["rl"]["finish_settle_s"] = 0
    report = train_ppo(cfg, max_updates=1)
    assert report["updates"] == 1
    assert (tmp_path / "run" / "last.pt").is_file()


def test_headless_throttle_closes_on_the_first_gate():
    cfg = load_config(None)
    env = HeadlessTrack(cfg)
    env.reset()
    forward = forward_ids(env.keys)[0]
    start = env.x
    progress = 0.0
    total = 0.0
    for _ in range(40):
        _obs, reward, done, info = env.step(forward)
        total += reward
        progress = float(info["progress"])
        if done:
            break
    assert env.x > start
    assert progress > 0.5
    assert total > 0.0
    env.close()


def test_cpu_setup_does_not_touch_cuda():
    assert configure_torch(torch.device("cpu")) == "cpu"


def test_ppo_loss_reaches_the_actor_and_value():
    model = StateActorCritic(obs_dim=obs_dim(27), n_actions=27, hidden=32)
    observations = torch.randn(8, obs_dim(27))
    logits, values = model(observations)
    actions = torch.randint(0, 27, (8,))
    old = torch.zeros(8)
    advantages = torch.tensor([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    returns = advantages + values.detach()
    loss, stats = ppo_loss(logits, values, actions, old, advantages, returns, 0.2, 0.01, 0.5, 0.01)
    loss.backward()
    assert model.policy.weight.grad is not None
    assert model.value.weight.grad is not None
    assert np.isfinite(stats["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.policy.weight.detach().clone()
    ppo_update(
        model,
        optimizer,
        observations.detach(),
        actions,
        old,
        advantages,
        returns.detach(),
        epochs=1,
        minibatch_size=4,
        clip=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        unimix=0.01,
    )
    assert not torch.equal(before, model.policy.weight)


class _Cap:
    def __init__(self, frame):
        self.frame = frame

    def grab(self):
        return self.frame.copy()

    def set_region(self, region):
        del region

    def close(self):
        return None


def test_ppo_trainer_updates_on_a_fake_window(tmp_path):
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
    cfg["rl"]["hidden"] = 32
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
        report = train_ppo(cfg, env=env, max_updates=1)
    finally:
        env.close()
    assert report["updates"] == 1
    assert (tmp_path / "run" / "last.pt").is_file()


def test_ppo_credits_the_press_that_moves_progress():
    torch.manual_seed(0)
    keys = ["w", "a", "s", "d", "q", "e"]
    masks = action_masks(keys)
    forward = forward_ids(keys)
    model = StateActorCritic(obs_dim(27), 27, hidden=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def forward_probability() -> float:
        observation = torch.from_numpy(LapNumbers(27).vector()).unsqueeze(0)
        with torch.no_grad():
            logits, _value = model(observation)
            probabilities = torch.softmax(logits, dim=-1)[0]
        return float(probabilities[forward].sum())

    before = forward_probability()
    for _update in range(10):
        observations: list[np.ndarray] = []
        actions: list[int] = []
        logps: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        lengths: list[int] = []
        for _episode in range(8):
            numbers = LapNumbers(27)
            progress = 0.0
            length = 0
            for _step in range(24):
                observation = torch.from_numpy(numbers.vector()).unsqueeze(0)
                with torch.no_grad():
                    action, logp, value = model.act(observation, unimix=0.01)
                action_id = int(action.item())
                delta = 0.08 if masks[action_id]["w"] else -0.01
                progress = float(np.clip(progress + delta, 0.0, 1.0))
                done = progress >= 0.99
                observations.append(numbers.vector())
                actions.append(action_id)
                logps.append(float(logp.item()))
                values.append(float(value.item()))
                rewards.append(delta - 0.002)
                dones.append(1.0 if done else 0.0)
                numbers.advance({"progress": progress, "confidence": 1.0, "dt": 0.1}, action_id)
                length += 1
                if done:
                    break
            lengths.append(length)
        advantages, targets = _episode_gae(rewards, dones, values, lengths, 0.99, 0.95)
        ppo_update(
            model,
            optimizer,
            torch.from_numpy(np.stack(observations)),
            torch.tensor(actions),
            torch.tensor(logps),
            torch.tensor(advantages),
            torch.tensor(targets),
            epochs=4,
            minibatch_size=64,
            clip=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            unimix=0.01,
        )
    after = forward_probability()
    assert after > before + 0.1
    assert after > 0.45


def test_loaded_actor_presses_a_legal_chord(tmp_path):
    cfg = load_config(None)
    cfg["record"]["keys"] = ["w", "a", "s", "d"]
    payload = {
        "arch": "state-ppo-v1",
        "keys": ["w", "a", "s", "d", "q", "e"],
        "n_actions": 27,
        "obs_dim": obs_dim(27),
        "hidden": 32,
        "state_dict": StateActorCritic(obs_dim=obs_dim(27), n_actions=27, hidden=32).state_dict(),
    }
    bound = bind_runtime(cfg, payload)
    assert bound["record"]["keys"] == ["w", "a", "s", "d", "q", "e"]
    assert cfg["record"]["keys"] == ["w", "a", "s", "d"]
    model = load_actor(payload, torch.device("cpu"))
    features = torch.zeros(1, obs_dim(27))
    action, _logp, _value = model.act(features, greedy=True)
    mask = mask_for(int(action.item()), bound["record"]["keys"])
    assert mask in action_masks(bound["record"]["keys"])
    path = tmp_path / "actor.pt"
    torch.save(payload, path)
    assert read_checkpoint(path, torch.device("cpu"))["n_actions"] == 27


def test_summer1_gates_follow_the_official_checkpoints():
    marks = ((36.0, 0.0), (-36.0, 0.0), (-20.0, -16.0), (-60.0, 56.0))
    positions = (SUMMER1_START, *SUMMER1_GATES)
    cursor = 0
    for mark in marks:
        found = next(i for i in range(cursor, len(positions)) if positions[i] == mark)
        cursor = found + 1
    assert positions[-1] == (-60.0, 56.0)
    assert len(SUMMER1_GATES) > len(marks)


def test_summer1_aimed_line_reaches_the_finish():
    import math

    cfg = load_config(None)
    cfg["rl"]["timeout_s"] = 80
    cfg["rl"]["stuck_s"] = 40
    env = make_track(cfg)
    masks = action_masks(env.keys)

    def hold(want: set[str]) -> int:
        for index, mask in enumerate(masks):
            if all(mask[key] == (key in want) for key in mask):
                return index
        raise AssertionError(want)

    actions = {
        "w": hold({"w"}),
        "wa": hold({"w", "a"}),
        "wd": hold({"w", "d"}),
        "sa": hold({"s", "a"}),
        "sd": hold({"s", "d"}),
    }
    info: dict = {}
    for _ in range(2000):
        gx, gy = env.gates[min(env.gate, len(env.gates) - 1)]
        err = (math.atan2(gy - env.y, gx - env.x) - env.heading + math.pi) % (2 * math.pi) - math.pi
        # Positive steer (d) decreases heading. Brake when the nose is far off
        # so the point car can make the U-turn after checkpoint 1.
        if abs(err) > 0.6:
            action = actions["sa"] if err > 0 else actions["sd"]
        elif err > 0.08:
            action = actions["wa"]
        elif err < -0.08:
            action = actions["wd"]
        else:
            action = actions["w"]
        _obs, _reward, done, info = env.step(action)
        if done:
            break
    assert info.get("finished") is True
    assert env.gate == len(env.gates)
    env.close()
