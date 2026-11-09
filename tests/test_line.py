"""Reference-lap reward: follow the best line, then pay for lap time."""

from __future__ import annotations

import pytest

from polytrack_agent.config import load_config
from polytrack_agent.rl.line import ReferenceLine, ready_for_time
from polytrack_agent.rl.track import make_track


def test_the_cursor_cannot_skip_ahead_of_its_window():
    line = ReferenceLine([(0.0, 0.0), (200.0, 0.0)], seconds=20.0, track="Winter 4")
    along, lateral, _cursor = line.project(10.0, 0.0, 0, 8.0, 40.0)
    assert along == pytest.approx(10.0)
    assert lateral == pytest.approx(0.0)
    along, lateral, _cursor = line.project(10.0, 3.0, 0, 8.0, 40.0)
    assert along == pytest.approx(10.0)
    assert lateral == pytest.approx(3.0)
    along, lateral, _cursor = line.project(150.0, 0.0, 0, 8.0, 40.0)
    assert along == pytest.approx(40.0)
    assert lateral == pytest.approx(110.0)


def test_ready_for_time_waits_for_a_near_lap():
    assert ready_for_time("line", False, 0.89, 0.9) is False
    assert ready_for_time("line", False, 0.9, 0.9) is True
    assert ready_for_time("line", True, 0.2, 0.9) is True
    assert ready_for_time("time", True, 1.0, 0.9) is False
    assert ready_for_time("chord", False, 1.0, 0.9) is False


class _Drive:
    def __init__(self, samples: list[dict]):
        self.samples = samples
        self.checkpoint_count = 2
        self.parts = []
        self.frames = 0
        self.i = 0

    def reset(self) -> dict:
        self.frames = 0
        self.i = 0
        return self._state()

    def step(self, controls: dict, frames: int) -> dict:
        del controls
        self.frames = int(frames)
        self.i += 1
        return self._state()

    def _state(self) -> dict:
        sample = self.samples[min(self.i, len(self.samples) - 1)]
        finish = sample.get("finish")
        return {
            "frames": self.frames,
            "speedKmh": 0.0,
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "finishFrames": None if finish is None else int(finish),
            "nextCheckpointIndex": 0,
            "position": {"x": float(sample.get("x") or 0.0), "y": 0.0, "z": float(sample.get("z") or 0.0)},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "wheels": list(sample.get("wheels") or [True, True, True, True]),
        }

    def close(self) -> None:
        pass


def _env():
    cfg = load_config(None)
    cfg["rl"]["action_repeat"] = 50
    cfg["rl"]["stuck_s"] = 30
    cfg["rl"]["timeout_s"] = 30
    for key in (
        "time_penalty",
        "progress_scale",
        "speed_reward",
        "speed_drop_penalty",
        "stall_penalty",
        "fail_penalty",
        "offtrack_penalty",
        "finish_bonus",
        "line_lateral",
        "time_stage_penalty",
        "time_line_scale",
    ):
        cfg["reward"][key] = 0.0
    cfg["reward"]["line_scale"] = 4.0
    cfg["reward"]["line_lateral_cap"] = 12.0
    cfg["reward"]["line_forward_m"] = 40.0
    cfg["reward"]["line_back_m"] = 8.0
    return cfg


def _follow(cfg, samples, stage="line"):
    env = make_track(cfg, client=_Drive(samples))
    env.use_line(ReferenceLine([(0.0, 0.0), (40.0, 0.0)], seconds=10.0, track="Winter 4"), stage)
    return env


def test_line_reward_pays_along_the_lap_only_on_the_road():
    cfg = _env()
    cfg["reward"]["progress_scale"] = 5.0
    airborne = {"x": 20.0, "wheels": [False, False, False, False]}
    env = _follow(cfg, [{"x": 0.0}, {"x": 10.0}, airborne, {"x": 20.0}, {"x": 30.0}])
    try:
        env.reset()
        _frame, reward, _done, info = env.step(0)
        assert info["line_progress"] == pytest.approx(0.25)
        assert reward == pytest.approx(1.0)
        _frame, flying, _done, info = env.step(0)
        assert info["line_progress"] == pytest.approx(0.5)
        assert flying == pytest.approx(0.0)
        _frame, landed, _done, info = env.step(0)
        assert info["line_progress"] == pytest.approx(0.5)
        assert landed == pytest.approx(0.0)
        _frame, onward, _done, info = env.step(0)
        assert info["line_progress"] == pytest.approx(0.75)
        assert onward == pytest.approx(1.0)
    finally:
        env.close()


def test_a_point_past_the_window_does_not_collect_the_lap():
    cfg = _env()
    env = make_track(cfg, client=_Drive([{"x": 0.0}, {"x": 150.0}]))
    try:
        env.use_line(ReferenceLine([(0.0, 0.0), (200.0, 0.0)], seconds=20.0, track="Winter 4"), "line")
        env.reset()
        _frame, reward, _done, info = env.step(0)
        assert info["line_progress"] == pytest.approx(0.2)
        assert info["line_lateral"] == pytest.approx(110.0)
        assert reward == pytest.approx(0.0)
    finally:
        env.close()


def test_time_stage_pays_more_for_a_faster_finish():
    def once(finish_frames: int) -> float:
        cfg = _env()
        cfg["reward"]["time_stage_bonus"] = 20.0
        cfg["reward"]["time_scale"] = 1.0
        cfg["reward"]["time_stage_penalty"] = 0.0
        env = _follow(cfg, [{"x": 0.0}, {"x": 10.0, "finish": finish_frames}], stage="time")
        try:
            env.reset()
            _frame, reward, done, info = env.step(0)
            assert done
            assert info["finished"]
            return reward
        finally:
            env.close()

    assert once(8000) - once(12000) == pytest.approx(4.0)


def test_leaving_the_line_costs_while_the_car_is_on_the_road():
    cfg = _env()
    cfg["reward"]["line_lateral"] = 10.0
    env = _follow(cfg, [{"x": 0.0, "z": 0.0}, {"x": 0.0, "z": 4.0}])
    try:
        env.reset()
        _frame, reward, _done, info = env.step(0)
        assert info["line_lateral"] == pytest.approx(4.0)
        assert reward == pytest.approx(-2.0)
    finally:
        env.close()
