"""PPO pieces, plus one short lap on the polytrack fork simulation when it is present."""

from __future__ import annotations

import copy
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from polytrack_agent.config import load_config, validate_output_keys
from polytrack_agent.rl.actions import action_masks, chord_name, forward_ids, n_actions, throttle_only_ids
from polytrack_agent.rl.checkpoint import ARCH, read_checkpoint
from polytrack_agent.rl.gae import compute_gae
from polytrack_agent.rl.hold import id_named
from polytrack_agent.rl.ppo import ppo_loss
from polytrack_agent.rl.state import FEATURES, NAMES, LapNumbers, obs_dim
from polytrack_agent.rl.track import game_root, make_track, road_border
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.seed import set_seed
from polytrack_agent.rl.trainer import summarize_trials, train_ppo

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
    straight = throttle_only_ids(keys)
    assert straight == [forward_ids(keys)[0]]
    assert chord_name(straight[0], keys) == "w"
    assert chord_name(forward_ids(keys)[1], keys) == "wa"
    with pytest.raises(ValueError):
        action_masks(["w", "q"])
    with pytest.raises(ValueError):
        validate_output_keys(["w", "r"])


def test_forward_bias_is_straight_throttle_only():
    keys = ["w", "a", "s", "d"]
    model = StateActorCritic(obs_dim(n_actions(keys)), n_actions(keys))
    model.bias_toward_forward(keys)
    bias = model.policy.bias.detach()
    straight = throttle_only_ids(keys)
    assert [index for index in range(model.n_actions) if float(bias[index]) != 0.0] == straight
    assert float(bias[straight[0]]) == 1.0
    arrows = ["up", "left", "down", "right"]
    arrow_model = StateActorCritic(obs_dim(n_actions(arrows)), n_actions(arrows))
    arrow_model.bias_toward_forward(arrows)
    arrow_straight = throttle_only_ids(arrows)
    assert chord_name(arrow_straight[0], arrows) == "up"
    assert float(arrow_model.policy.bias[arrow_straight[0]].detach()) == 1.0
    assert int((arrow_model.policy.bias != 0).sum()) == 1


def _actor_weight(seed: int) -> torch.Tensor:
    set_seed(seed)
    model = StateActorCritic(16, 9, hidden=8)
    model.bias_toward_forward(["w", "a", "s", "d"])
    return next(model.net.parameters()).detach().clone()


def test_same_seed_rebuilds_the_same_actor():
    assert torch.equal(_actor_weight(7), _actor_weight(7))
    assert not torch.equal(_actor_weight(7), _actor_weight(8))
    with pytest.raises(ValueError):
        set_seed(-1)


def test_summarize_trials_reports_mean_and_sample_std():
    reports = [
        {
            "seed": 0,
            "updates": 2,
            "best_progress": 0.1,
            "rows": [{"progress": 0.05, "entropy": 2.0}, {"progress": 0.1, "entropy": 1.5}],
            "spawn_chord": "w",
            "spawn_probs": {"w": 0.9},
            "out_dir": "a",
        },
        {
            "seed": 1,
            "updates": 2,
            "best_progress": 0.3,
            "rows": [{"progress": 0.2, "entropy": 1.8}],
            "spawn_chord": "wa",
            "spawn_probs": {"wa": 0.8},
            "out_dir": "b",
        },
    ]
    summary = summarize_trials(reports)
    assert summary["n"] == 2
    assert summary["seeds"] == [0, 1]
    assert summary["best_progress_mean"] == pytest.approx(0.2)
    assert summary["best_progress_std"] == pytest.approx(math.sqrt(0.02))
    assert summary["last_progress_mean"] == pytest.approx(0.15)
    assert summary["trials"][0]["spawn_chord"] == "w"
    assert summary["trials"][1]["last_entropy"] == pytest.approx(1.8)
    one = summarize_trials([reports[0]])
    assert one["best_progress_std"] == 0.0
    with pytest.raises(ValueError):
        summarize_trials([])
    with pytest.raises(ValueError):
        summarize_trials([{"seed": 3, "best_progress": 0.0, "rows": []}])


def test_replay_drops_the_oldest():
    from polytrack_agent.rl.replay import Replay

    buffer = Replay(2, 1)
    for step in range(3):
        sample = np.array([step], dtype=np.float32)
        buffer.add(sample, step, float(step), sample + 1, False)
    assert buffer.size == 2
    _obs, actions, _rewards, _next_obs, _dones = buffer.sample(16, np.random.default_rng(0))
    assert set(np.unique(actions.tolist())).issubset({1, 2})


def test_temperature_rises_when_entropy_is_low():
    from polytrack_agent.rl.sac import temperature_loss

    log_alpha = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([log_alpha], lr=0.5)
    temperature_loss(log_alpha, torch.tensor([0.05]), 1.0).backward()
    optimizer.step()
    assert float(log_alpha.detach()) > 0.0
    peaked = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([peaked], lr=0.5)
    temperature_loss(peaked, torch.tensor([2.0]), 0.1).backward()
    optimizer.step()
    assert float(peaked.detach()) < 0.0


def test_discrete_sac_prefers_the_rewarded_action():
    from polytrack_agent.rl.sac import TwinQ, sac_update

    set_seed(0)
    obs_dim = 4
    actions = 3
    actor = StateActorCritic(obs_dim, actions, hidden=32)
    q = TwinQ(obs_dim, actions, hidden=32)
    target = copy.deepcopy(q)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    log_alpha = nn.Parameter(torch.zeros(()))
    actor_opt = torch.optim.Adam(list(actor.net.parameters()) + list(actor.policy.parameters()), lr=1e-3)
    q_opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    alpha_opt = torch.optim.Adam([log_alpha], lr=1e-3)
    obs = torch.zeros(128, obs_dim)
    chosen = torch.randint(0, actions, (128,))
    rewards = (chosen == 0).float()
    dones = torch.ones(128)
    for _ in range(80):
        sac_update(
            actor,
            q,
            target,
            actor_opt,
            q_opt,
            log_alpha,
            alpha_opt,
            obs,
            chosen,
            rewards,
            obs,
            dones,
            0.99,
            0.05,
            0.2,
        )
    with torch.no_grad():
        probs = torch.softmax(actor(obs[:1])[0], dim=-1)[0]
    assert float(probs[0]) > 0.5


def test_trials_require_max_updates():
    from types import SimpleNamespace

    from polytrack_agent.cli import cmd_train

    cfg = load_config(None)
    args = SimpleNamespace(device=None, out=None, stop_time=None, seed=0, trials=3, max_updates=None)
    with pytest.raises(ValueError, match="max-updates"):
        cmd_train(cfg, args)


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
            "dt": 0.0,
            "state": {
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "velocity": {"x": 20.0, "y": 0.0, "z": 0.0},
                "wheels": [True, True, False, False],
                "nextCheckpointIndex": 1,
                "position": {"x": 0.0, "y": 10.0, "z": 0.0},
                "checkpointCount": 4,
                "border": {"clearance": 2.5, "side": 0.0},
            },
        },
        0,
    )
    vector = numbers.vector()
    assert obs_dim(9) == FEATURES + 9
    assert vector.shape == (obs_dim(9),)
    assert vector[NAMES.index("vx")] == pytest.approx(0.5)
    assert vector[NAMES.index("wheels")] == pytest.approx(0.5)
    assert vector[NAMES.index("border")] == pytest.approx(0.25)
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
        and parts.is_file()
    )


class _FakeClient:
    """Stand-in for SimClient so the finish/lap-time contract is testable."""

    def __init__(self):
        self.checkpoint_count = 2
        self.parts = []
        self.frames = 0
        self._resets = 0

    def reset(self) -> dict:
        self._resets += 1
        self.frames = 0
        return self._state(finish=None)

    def step(self, controls: dict, frames: int) -> dict:
        self.frames = int(frames)
        finish = self.frames if self.frames >= 3000 else None
        return self._state(finish=finish)

    def _state(self, finish):
        return {
            "frames": self.frames,
            "speedKmh": 80.0,
            "finishFrames": finish,
            "nextCheckpointIndex": 0,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": [True, True, True, True],
        }

    def close(self) -> None:
        pass


def test_finish_reports_a_lap_time_for_the_stop_rule():
    if os.environ.get("POLYTRACK_ROOT") is None:
        pytest.skip("POLYTRACK_ROOT is not set")
    parts_cache = ROOT / "polytrack_agent" / "sim" / "cache" / "track_parts.json"
    if not parts_cache.is_file():
        pytest.skip("physics mesh cache is not available")
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 1000
    env = make_track(cfg, client=_FakeClient())
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        info = {}
        for _ in range(5):
            _frame, _reward, done, info = env.step(forward)
            if done:
                break
        assert info["finished"] is True
        assert int(info["state"]["finishFrames"]) * 0.001 == pytest.approx(3.0)
    finally:
        env.close()


def test_reset_action_is_opt_in():
    keys = ["w", "a", "s", "d"]
    assert n_actions(keys) == 9
    assert n_actions(keys, reset_action=True) == 10
    # forward chords are unchanged, so the reset action never biases throttle
    assert forward_ids(keys, reset_action=True) == forward_ids(keys)
    assert throttle_only_ids(keys, reset_action=True) == throttle_only_ids(keys)
    mask = action_masks(keys, reset_action=True)[-1]
    assert mask == {"w": False, "a": False, "s": False, "d": False, "reset": True}


class _StuckClient:
    """Always reports zero speed, so the env sees a wedged car."""

    def __init__(self):
        self.checkpoint_count = 3
        self.parts = []
        self.frames = 0
        self.resets = 0

    def reset(self) -> dict:
        self.resets += 1
        self.frames = 0
        return self._state()

    def step(self, controls: dict, frames: int) -> dict:
        self.frames = int(frames)
        return self._state()

    def _state(self) -> dict:
        return {
            "frames": self.frames,
            "speedKmh": 0.0,
            "finishFrames": None,
            "nextCheckpointIndex": 0,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": [True, True, True, True],
        }

    def close(self) -> None:
        pass


def _respawn_env():
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 10
    cfg["rl"]["stuck_s"] = 1.0
    cfg["rl"]["timeout_s"] = 1e9
    cfg["rl"]["auto_respawn"] = True
    cfg["reward"]["respawn_penalty"] = 0.05
    return make_track(cfg, client=_StuckClient())


def test_stuck_ends_the_episode_without_auto_respawn():
    env = _respawn_env()
    env.auto_respawn = False
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        done = False
        for _ in range(2000):
            _frame, _reward, done, info = env.step(forward)
            if done:
                break
        assert done and info["stuck"] and not info["respawned"]
    finally:
        env.close()


def test_auto_respawn_keeps_the_episode_alive():
    if os.environ.get("POLYTRACK_ROOT") is None:
        pytest.skip("POLYTRACK_ROOT is not set")
    parts_cache = ROOT / "polytrack_agent" / "sim" / "cache" / "track_parts.json"
    if not parts_cache.is_file():
        pytest.skip("physics mesh cache is not available")
    env = _respawn_env()
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        respawned = False
        for _ in range(2000):
            _frame, _reward, done, info = env.step(forward)
            if info["respawned"]:
                respawned = True
                assert not done
                break
        assert respawned
        assert info["respawns"] >= 1
    finally:
        env.close()


class _BeachedClient:
    """Drives for a while, then reports every wheel in the air and no motion.

    This is the failure the fork produces when the car leaves the road: it lands
    on its chassis and no control moves it again, even though speedKmh is stale.
    """

    def __init__(self):
        self.checkpoint_count = 3
        self.parts = []
        self.frames = 0
        self.resets = 0

    def reset(self) -> dict:
        self.resets += 1
        self.frames = 0
        return self._state(beached=False)

    def step(self, controls: dict, frames: int) -> dict:
        self.frames = int(frames)
        return self._state(beached=self.frames >= 300)

    def _state(self, beached: bool) -> dict:
        x = 200.0 if beached else 200.0 - self.frames * 0.4
        return {
            "frames": self.frames,
            "speedKmh": 30.0 if beached else 80.0,
            "finishFrames": None,
            "nextCheckpointIndex": 0,
            "position": {"x": x, "y": 56.0, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": [False, False, False, False] if beached else [True, True, True, True],
        }

    def close(self) -> None:
        pass


def _beach_env():
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 10
    cfg["rl"]["stuck_s"] = 1e9
    cfg["rl"]["timeout_s"] = 1e9
    cfg["rl"]["airborne_grace_s"] = 0.1
    cfg["reward"]["beach_penalty"] = -5.0
    cfg["reward"]["contact_reward"] = 0.01
    return make_track(cfg, client=_BeachedClient())


def test_beaching_is_terminal_and_pays_the_penalty():
    env = _beach_env()
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        last = 0.0
        done = False
        for _ in range(1000):
            _frame, last, done, info = env.step(forward)
            if done:
                break
        assert done and info["beached"]
        assert info["wheels_down"] == 0
        # The step that beaches hands back the penalty, well below one step of
        # contact reward, so the dead end is unmistakable.
        assert last <= -5.0
    finally:
        env.close()


def test_contact_reward_pays_while_on_the_road():
    env = _beach_env()
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        info = {}
        for _ in range(5):
            _frame, reward, _done, info = env.step(forward)
        assert info["wheels_down"] == 4
        assert info["beached"] is False
        # Rolling on the road pays the contact term, minus the small time cost.
        assert reward > 0.0
    finally:
        env.close()


class _ParkedClient:
    """Wheels on the road but the car never moves, so it earns nothing."""

    def __init__(self):
        self.checkpoint_count = 3
        self.parts = []
        self.frames = 0

    def reset(self) -> dict:
        self.frames = 0
        return self._state()

    def step(self, controls: dict, frames: int) -> dict:
        self.frames = int(frames)
        return self._state()

    def _state(self) -> dict:
        return {
            "frames": self.frames,
            "speedKmh": 0.0,
            "finishFrames": None,
            "nextCheckpointIndex": 0,
            "position": {"x": 200.0, "y": 56.0, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": [True, True, True, True],
        }

    def close(self) -> None:
        pass


class _CrawlingClient:
    """Wheels down and creeping forward at about 8 km/h."""

    def __init__(self, speed: float = 2.2):
        self.checkpoint_count = 3
        self.parts = []
        self.frames = 0
        self._speed = float(speed)
        self._z = 0.0

    def reset(self) -> dict:
        self.frames = 0
        self._z = 0.0
        return self._state()

    def step(self, controls: dict, frames: int) -> dict:
        self.frames = int(frames)
        self._z += self._speed * 0.001 * 10  # identity quaternion: +Z is forward
        return self._state()

    def _state(self) -> dict:
        return {
            "frames": self.frames,
            "speedKmh": self._speed * 3.6,
            "velocity": {"x": 0.0, "y": 0.0, "z": self._speed},
            "finishFrames": None,
            "nextCheckpointIndex": 0,
            "position": {"x": 200.0, "y": 56.0, "z": self._z},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": [True, True, True, True],
        }

    def close(self) -> None:
        pass


def test_slow_but_moving_car_is_not_treated_as_stalled():
    """A car at a few km/h covers very little per step at a small repeat.

    A fixed distance floor used to call that parked and charge the stall
    penalty, which punished exactly the careful driving we want.
    """
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 10
    cfg["reward"]["stall_penalty"] = -0.5
    cfg["reward"]["time_penalty"] = 0.0
    cfg["reward"]["speed_reward"] = 0.0
    client = _CrawlingClient()
    env = make_track(cfg, client=client)
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        _frame, reward, _done, info = env.step(forward)
        # ~2.2 m/s is 8 km/h: slow, but the car is definitely driving.
        assert info["ground_speed"] == pytest.approx(8.0, rel=0.05)
        # The stall penalty must not fire, so nothing is subtracted.
        assert reward >= 0.0
    finally:
        env.close()


def test_parking_on_the_road_earns_no_contact_reward():
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 10
    cfg["reward"]["contact_reward"] = 0.01
    env = make_track(cfg, client=_ParkedClient())
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        info = {}
        for _ in range(5):
            _frame, reward, _done, info = env.step(forward)
        assert info["wheels_down"] == 4
        # Only the time penalty lands; a parked car cannot farm the contact term.
        assert reward < 0.0
    finally:
        env.close()


def test_stall_penalty_punishes_sitting_still():
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 10
    cfg["reward"]["contact_reward"] = 0.0
    cfg["reward"]["stall_penalty"] = -0.5
    env = make_track(cfg, client=_ParkedClient())
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        rewards = []
        for _ in range(3):
            _frame, reward, _done, _info = env.step(forward)
            rewards.append(reward)
        # Every idle step on the road pays the stall penalty, not just the clock.
        assert all(r <= -0.5 for r in rewards)
    finally:
        env.close()


class _RollingClient:
    """Wheels down, chassis genuinely moving. Carries a real velocity vector."""

    def __init__(self, speed: float = 25.0):
        self.checkpoint_count = 2
        self.parts = []
        self.frames = 0
        self._speed = float(speed)
        self._x = 0.0

    def reset(self) -> dict:
        self.frames = 0
        self._x = 0.0
        return self._state()

    def step(self, controls: dict, frames: int) -> dict:
        self.frames = int(frames)
        self._x -= self._speed * 0.001 * 10  # travels along -X
        return self._state()

    def _state(self) -> dict:
        return {
            "frames": self.frames,
            "speedKmh": self._speed * 3.6,
            # velocity in m/s; the car's forward axis is -X
            "velocity": {"x": -self._speed, "y": 0.0, "z": 0.0},
            "finishFrames": None,
            "nextCheckpointIndex": 0,
            "position": {"x": self._x, "y": 0.0, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": [True, True, True, True],
        }

    def close(self) -> None:
        pass


class _PinnedClient:
    """Pinned against a wall: wheels spin (speedKmh high) but the chassis is
    stationary and climbing, which is what the straight-line chicane collision
    actually looks like."""

    def __init__(self):
        self.checkpoint_count = 2
        self.parts = []
        self.frames = 0

    def reset(self) -> dict:
        self.frames = 0
        return self._state()

    def step(self, controls: dict, frames: int) -> dict:
        self.frames = int(frames)
        return self._state()

    def _state(self) -> dict:
        return {
            "frames": self.frames,
            "speedKmh": 120.0,
            # essentially no horizontal travel, but climbing
            "velocity": {"x": -0.01, "y": 2.4, "z": 0.0},
            "finishFrames": None,
            "nextCheckpointIndex": 0,
            "position": {"x": 0.0, "y": 1.2, "z": 0.0},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": [False, False, True, False],
        }

    def close(self) -> None:
        pass


def test_speed_reward_pays_on_the_road_only():
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 10
    cfg["reward"]["speed_reward"] = 0.5
    cfg["reward"]["speed_cap_kmh"] = 90.0
    env = make_track(cfg, client=_RollingClient(speed=25.0))
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        _frame, reward, _done, info = env.step(forward)
        # 25 m/s = 90 km/h, exactly the cap, so the reward is paid in full.
        assert info["ground_speed"] == pytest.approx(90.0, rel=1e-3)
        assert reward >= 0.5
    finally:
        env.close()


def test_speed_reward_ignores_wheels_spinning_on_a_wall():
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 10
    cfg["reward"]["speed_reward"] = 0.5
    cfg["reward"]["time_penalty"] = 0.0
    env = make_track(cfg, client=_PinnedClient())
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        _frame, rewarded, _done, info = env.step(forward)
        # speedKmh claims 120 but the chassis is not travelling.
        assert info["ground_speed"] < 1.0
    finally:
        env.close()
    # The same step with the speed term off must pay exactly the same, proving
    # the spinning wheels earned nothing.
    plain = load_config(None)
    plain["rl"]["action_repeat"] = 10
    plain["reward"]["speed_reward"] = 0.0
    plain["reward"]["time_penalty"] = 0.0
    env = make_track(plain, client=_PinnedClient())
    try:
        env.reset()
        forward = forward_ids(env.keys)[0]
        _frame, baseline, _done, _info = env.step(forward)
    finally:
        env.close()
    # The speed term contributes only the sliver for the 0.01 m/s drift
    # (0.5 * 0.25 * 0.036/120), so the two runs agree to well within a percent.
    assert rewarded == pytest.approx(baseline, rel=1e-3)


def test_border_is_the_side_of_the_road_not_the_joint():
    cells = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    clearance, side = road_border(cells, 5.0, 0.0)
    assert clearance == pytest.approx(2.5)
    assert side == pytest.approx(0.0)
    # The shared edge of two tiles is still the middle of the road.
    joint, _side = road_border(cells, 2.5, 0.0)
    assert joint == pytest.approx(2.5)
    clearance, side = road_border(cells, 5.0, 2.0)
    assert clearance == pytest.approx(0.5)
    assert abs(side) == pytest.approx(2.0)
    off, _side = road_border(cells, 5.0, 4.0)
    assert off == pytest.approx(-1.5)


def test_acceleration_is_the_change_in_velocity():
    numbers = LapNumbers(9)
    state = {
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "wheels": [True, True, True, True],
        "nextCheckpointIndex": 0,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "checkpointCount": 3,
        "border": {"clearance": 2.5, "side": 0.0},
    }
    numbers.advance({"dt": 0.05, "state": state}, 0)
    moving = dict(state)
    moving["velocity"] = {"x": 2.0, "y": 0.0, "z": 0.0}
    numbers.advance({"dt": 0.05, "state": moving}, 0)
    vector = numbers.vector()
    assert vector[NAMES.index("vx")] == pytest.approx(2.0 / 40.0)
    assert vector[NAMES.index("ax")] == pytest.approx((2.0 / 0.05) / 40.0)


def test_throttle_moves_the_sim_car():
    if not _sim_ready():
        pytest.skip("polytrack fork simulation or the mesh cache is not available")
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 200
    env = make_track(cfg)
    try:
        start = env.reset()["state"]["position"]
        forward = forward_ids(env.keys)[0]
        info = {"state": {"position": start}}
        moved = 0.0
        for _ in range(15):
            _frame, _reward, done, info = env.step(forward)
            position = info["state"]["position"]
            moved = max(
                moved,
                math.dist(
                    (position["x"], position["y"], position["z"]),
                    (start["x"], start["y"], start["z"]),
                ),
            )
            if done:
                break
        # A throttle-only policy accelerates down the straight and then crashes
        # into the first corner, so check displacement from the grid over the
        # whole run rather than a fixed later window.
        assert moved > 1.0
        assert int(info["state"]["frames"]) > 1
    finally:
        env.close()


def test_steer_lands_on_the_same_frames_after_the_worker_ages():
    """A steer sent now has to apply now, even if the worker has been alive."""
    if not _sim_ready():
        pytest.skip("polytrack fork simulation or the mesh cache is not available")
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 50
    cfg["rl"]["timeout_s"] = 30
    cfg["rl"]["stuck_s"] = 30
    env = make_track(cfg)
    try:
        gas = id_named("w", list(env.keys))
        turn = id_named("wa", list(env.keys))

        def burst(turn_on: bool) -> list[tuple[float, float, float]]:
            env.reset()
            points = []
            for index in range(20):
                action = turn if turn_on and 6 <= index < 14 else gas
                _frame, _reward, done, info = env.step(action)
                position = info["state"]["position"]
                points.append((position["x"], position["y"], position["z"]))
                if done:
                    break
            return points

        fresh = burst(True)
        straight = burst(False)
        time.sleep(1.5)
        aged = burst(True)
        assert len(fresh) == 20
        assert fresh == aged
        assert fresh != straight
    finally:
        env.close()


def test_one_ppo_update_on_the_sim(tmp_path):
    if not _sim_ready():
        pytest.skip("polytrack fork simulation or the mesh cache is not available")
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
    assert payload["arch"] == ARCH == "polytrack-ppo-0.6.3"
    assert payload["track"] == "summer1"
    assert payload["n_actions"] == 9
    assert payload["obs_dim"] == obs_dim(9)
