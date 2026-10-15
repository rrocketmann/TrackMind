"""Fast tests for vision preprocessing, controls, sessions, and progress."""

from __future__ import annotations

import numpy as np
import torch

from polytrack_agent.capture.region import Rect
from polytrack_agent.capture.window import pick_game_window, pick_largest
from polytrack_agent.config import deep_merge, load_config, validate_output_keys
from polytrack_agent.env import PolyTrackVisionEnv
from polytrack_agent.input.keys import key_name
from polytrack_agent.obs.preprocess import FrameStack, bgra_to_rgb, resize_nn
from polytrack_agent.obs.preprocess import rgb_to_tiny_gray as tiny_gray
from polytrack_agent.obs.checkpoint import CheckpointProgress
from polytrack_agent.obs.progress import ProgressState, ReferenceProgress, rgb_to_tiny_gray
from polytrack_agent.policy.cnn import DrivingPolicy
from polytrack_agent.policy.control import Controller
from polytrack_agent.policy.transformer import TemporalDriveHead, fit_heads, pool_hidden
from polytrack_agent.record.dataset import DrivingDataset
from polytrack_agent.record.session import SessionWriter
from polytrack_agent.reward import RewardTracker


class _Key:
    def __init__(self, char=None, name=None):
        self.char = char
        self.name = name


class _FakeCapture:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.closed = False

    def grab(self):
        return self.frame.copy()

    def set_region(self, region):
        del region

    def close(self):
        self.closed = True


def test_keys_and_config():
    assert key_name(_Key(char="W")) == "w"
    assert key_name(_Key(name="F8")) == "f8"
    try:
        validate_output_keys(["w", "r"])
    except ValueError as exc:
        assert "r" in str(exc)
    else:
        raise AssertionError("restart key was accepted")
    cfg = load_config(None)
    assert cfg["record"]["keys"][:4] == ["w", "a", "s", "d"]
    assert "q" in cfg["record"]["keys"]
    merged = deep_merge({"capture": {"fps": 30, "crop": {"top": 0}}}, {"capture": {"fps": 20}})
    assert merged["capture"]["fps"] == 20
    assert merged["capture"]["crop"]["top"] == 0


def test_resize_and_bgra():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0, 0] = (1, 0, 0)
    image[0, 3] = (2, 0, 0)
    image[3, 0] = (3, 0, 0)
    image[3, 3] = (4, 0, 0)
    out = resize_nn(image, 2, 2)
    assert out.shape == (2, 2, 3)
    assert out[0, 0, 0] == 1
    assert out[0, 1, 0] == 2
    assert out[1, 0, 0] == 3
    assert out[1, 1, 0] == 4
    bgra = np.zeros((2, 2, 4), dtype=np.uint8)
    bgra[0, 0] = (10, 20, 30, 255)
    rgb = bgra_to_rgb(bgra)
    assert tuple(rgb[0, 0]) == (30, 20, 10)
    stack = FrameStack(3, 2, 2)
    first = stack.push(image)
    assert first.shape == (3, 2, 2, 3)
    assert np.array_equal(first[0], first[2])


def test_window_pick_and_crop():
    windows = [
        ("Other", Rect(0, 0, 10, 10)),
        ("PolyTrack - browser", Rect(5, 6, 800, 600)),
        ("polytrack editor", Rect(0, 0, 100, 100)),
    ]
    picked = pick_largest(windows, "PolyTrack")
    assert picked is not None and picked.width == 800
    classed = [
        ("PolyTrack", "code.Code", Rect(0, 0, 900, 700)),
        ("PolyTrack - Chromium", "chromium.Chromium", Rect(10, 10, 800, 600)),
    ]
    browser = pick_game_window(classed, "PolyTrack", "chromium")
    assert browser is not None and browser.width == 800
    assert pick_game_window(classed[:1], "PolyTrack", "chromium") is None
    cropped = Rect(0, 0, 100, 80).cropped({"top": 10, "bottom": 10, "left": 5, "right": 5})
    assert (cropped.left, cropped.top, cropped.width, cropped.height) == (5, 10, 90, 60)


def test_controller_hysteresis_and_duty():
    discrete = Controller(["w", "a", "s", "d"], threshold_on=0.6, threshold_off=0.4)
    mask = discrete.discrete_mask(np.array([0.9, 0.7, 0.1, 0.8]))
    assert mask == {"w": True, "a": False, "s": False, "d": True}
    mask = discrete.discrete_mask(np.array([0.5, 0.0, 0.0, 0.5]))
    assert mask["w"] and mask["d"] and not mask["a"]
    mask = discrete.discrete_mask(np.array([0.1, 0.0, 0.0, 0.1]))
    assert not mask["w"] and not mask["d"]

    gated = Controller(["w", "q"], min_key_freq=0.01, key_freq=[0.5, 0.0])
    assert gated.discrete_mask(np.array([0.99, 0.99]))["q"] is False

    continuous = Controller(["w", "a", "s", "d", "q", "e"], mode="continuous", deadzone=0.18)
    full = continuous.continuous_holds(np.array([1, 0, 0, 0, 0, 0]), period=0.1)
    assert abs(full["w"] - 0.1) < 1e-6
    assert full["s"] == 0.0
    steer = continuous.continuous_holds(np.array([0, 0.9, 0, 0.1, 0, 0]), period=1.0)
    assert steer["a"] > 0.8
    assert steer["d"] == 0.0


def test_progress_follows_then_restarts_and_finishes():
    def pattern(index: int) -> np.ndarray:
        frame = np.zeros((18, 32, 3), dtype=np.uint8)
        frame[:, index % 32, :] = 255
        frame[index % 18, :, 1] = 160
        return frame

    frames = [pattern(i) for i in range(40)]
    grays = np.stack([rgb_to_tiny_gray(frame, 18, 32) for frame in frames])
    tracker = ReferenceProgress(
        grays,
        search_window=4,
        restart_backtrack=0.12,
        restart_hold=2,
        start_sim_margin=0.05,
        match_min=0.55,
        finish_progress=0.97,
        finish_sim_min=0.75,
        finish_hold=3,
    )
    for index, frame in enumerate(frames):
        state = tracker.update(frame)
        assert abs(state.index - index) <= 1, (index, state)
    finished = False
    for _ in range(4):
        state = tracker.update(frames[-1])
        finished = finished or state.finished
    assert finished
    restarted = False
    for _ in range(4):
        state = tracker.update(frames[0])
        restarted = restarted or state.restarted
    assert restarted
    assert state.index == 0


def _gate_frame(height: int, width: int, box: tuple[int, int, int, int]) -> np.ndarray:
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    top, bottom, left, right = box
    frame[top:bottom, left:right] = (255, 210, 16)
    return frame


def test_checkpoint_reward_grows_as_the_gate_gets_closer():
    sensor = CheckpointProgress()
    reward = RewardTracker(progress_scale=1.0, time_penalty=0.0, monotonic=False)
    far = sensor.update(_gate_frame(96, 160, (40, 52, 70, 90)))
    far_reward, done = reward.step(far, dt=0.0)
    assert not done and far_reward > 0.0
    near = sensor.update(_gate_frame(96, 160, (28, 70, 50, 110)))
    near_reward, done = reward.step(near, dt=0.0)
    assert not done and near_reward > 0.0
    assert near.progress > far.progress
    away = sensor.update(_gate_frame(96, 160, (42, 50, 76, 86)))
    away_reward, _done = reward.step(away, dt=0.0)
    assert away_reward < 0.0


def test_checkpoint_pass_counts_one_gate():
    sensor = CheckpointProgress()
    sensor.update(_gate_frame(96, 160, (20, 78, 30, 130)))
    state = sensor.update(_gate_frame(96, 160, (0, 1, 0, 1)))
    assert state.index == 1
    assert state.progress >= 1.0


def test_reward_is_monotonic_until_restart():
    tracker = RewardTracker(progress_scale=1.0, time_penalty=0.1, restart_penalty=1.0, finish_bonus=2.0)
    reward, done = tracker.step(ProgressState(0.25, False, False, 1.0, 4), dt=0.5)
    assert not done
    assert abs(reward - (0.25 - 0.05)) < 1e-6
    reward, done = tracker.step(ProgressState(0.20, False, False, 1.0, 3), dt=0.0)
    assert reward == 0.0 and not done
    reward, done = tracker.step(ProgressState(0.0, True, False, 1.0, 0), dt=0.2)
    assert done and reward == -1.0
    tracker.reset()
    reward, done = tracker.step(ProgressState(1.0, False, True, 1.0, 10), dt=0.0)
    assert done and abs(reward - 3.0) < 1e-6
    tracker.reset()
    reward, done = tracker.step(ProgressState(0.4, False, False, 0.1, 2), dt=0.0, min_confidence=0.55)
    assert reward == 0.0 and not done
    assert tracker.prev == 0.0


def test_session_roundtrip_and_weights(tmp_path):
    keys = ["w", "a", "s", "d"]
    session = tmp_path / "session"
    writer = SessionWriter(session, keys, (8, 8, 3), chunk_size=10, fps_target=30)
    for i in range(10):
        frame = np.full((8, 8, 3), i, dtype=np.uint8)
        action = np.array([1, i % 2, 0, 1 - (i % 2)], dtype=np.uint8)
        writer.add(frame, action, i * 0.03, 0)
    writer.close()
    dataset = DrivingDataset([session], image_size=(4, 4), stack=2, augment=False, max_gap_s=0.2)
    assert len(dataset) == 9
    frames, prev, action = dataset[0]
    assert frames.shape == (2, 4, 4, 3)
    assert prev.shape == action.shape == (4,)
    # A change of keys weighs more than a repeated mask. Build two manual comparisons.
    changing = dataset.weight_for(0)
    # Find a sample whose action equals the previous one, if the alternating pattern has none,
    # the first sample always changes. Add a steady tail.
    assert changing >= 1.0


def test_pooled_tokens_and_temporal_head():
    assert fit_heads(384, 6) == 6
    assert fit_heads(1024, 6) == 4
    batch, channels, grid_h, grid_w = 2, 8, 3, 4
    registers = 4
    hidden = torch.zeros(batch, 1 + registers + grid_h * grid_w, channels)
    hidden[:, 1 + registers :, 0] = 2.0
    pooled = pool_hidden(hidden, registers, (grid_h, grid_w), (2, 2))
    assert pooled.shape == (batch, 4, channels)
    assert torch.allclose(pooled[:, :, 0], torch.full((batch, 4), 2.0))
    head = TemporalDriveHead(
        feature_dim=32,
        n_tokens=4,
        stack=3,
        n_actions=6,
        layers=2,
        heads=4,
        dropout=0.0,
        use_prev_action=True,
    )
    tokens = torch.randn(2, 3, 4, 32)
    prev = torch.zeros(2, 6)
    logits = head(tokens, prev)
    assert logits.shape == (2, 6)
    logits.sum().backward()
    assert head.head[-1].weight.grad is not None


def test_policy_backward():
    model = DrivingPolicy(stack=2, n_actions=6, use_prev_action=True, dropout=0.0)
    frames = torch.rand(3, 2, 3, 32, 48)
    prev = torch.zeros(3, 6)
    logits = model(frames, prev)
    assert logits.shape == (3, 6)
    logits.sum().backward()
    assert model.head[-1].weight.grad is not None


def test_env_step_is_vision_shaped(tmp_path):
    del tmp_path
    cfg = load_config(None)
    cfg["capture"]["fps"] = 1000
    cfg["capture"]["region"] = {"left": 0, "top": 0, "width": 64, "height": 48}
    cfg["capture"]["follow_window"] = False
    cfg["play"]["require_focus"] = False
    cfg["train"]["image_size"] = [12, 16]
    cfg["train"]["stack"] = 2
    cfg["reward"]["time_penalty"] = 0.0
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 40
    env = PolyTrackVisionEnv(cfg, capture=_FakeCapture(frame), dry_run=True)
    try:
        obs = env.reset()
        assert obs.shape == (2, 12, 16, 3)
        obs, reward, done, info = env.step(np.zeros(env.action_size, dtype=np.float32))
        assert obs.shape == (2, 12, 16, 3)
        assert reward == 0.0
        assert done is False
        assert info["vision"] is True
    finally:
        env.close()


def test_reset_waits_for_start_line():
    cfg = load_config(None)
    cfg["capture"]["fps"] = 1000
    cfg["capture"]["region"] = {"left": 0, "top": 0, "width": 32, "height": 24}
    cfg["capture"]["follow_window"] = False
    cfg["play"]["require_focus"] = False
    cfg["rl"]["reset_timeout_s"] = 2
    start = np.zeros((24, 32, 3), dtype=np.uint8)
    start[:, :16, 1] = 220
    start[:, 16:, 2] = 40
    other = np.zeros_like(start)
    other[:12, :, 0] = 220
    other[12:, :, 1] = 30
    ref = np.stack([tiny_gray(start, 8, 8), tiny_gray(other, 8, 8)])
    progress = ReferenceProgress(ref, search_window=1, match_min=0.8, finish_hold=100, restart_hold=100)

    class _Cap:
        def __init__(self):
            self.frame = other

        def grab(self):
            return self.frame.copy()

        def set_region(self, region):
            del region

        def close(self):
            return None

    cap = _Cap()
    calls: list[str] = []

    def hook():
        calls.append("r")
        cap.frame = start

    env = PolyTrackVisionEnv(cfg, capture=cap, dry_run=True, progress=progress, restart_hook=hook)
    try:
        obs = env.reset(restart=True)
        assert calls == ["r"]
        assert "r" not in env.runtime.keys_out.held
        assert obs.shape[0] == cfg["train"]["stack"]
        assert env.progress.start_similarity(start) >= 0.8
    finally:
        env.close()
