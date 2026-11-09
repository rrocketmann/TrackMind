"""Behavioral cloning picks the popular track and ignores reset presses."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from polytrack_agent.rl.bc import behavior_clone, load_best_line, load_demos
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.state import NAMES, RECORDED, RECORDED_FEATURES, obs_dim


def _row(action: int, marker: float) -> dict:
    recorded = [0.0] * RECORDED_FEATURES
    recorded[RECORDED.index("vx")] = marker
    return {"action": action, "n": recorded}


def _write_lap(root: Path, name: str, rows: list[dict], *, recovery: bool = False) -> str:
    relative = f"laps/{name}.json.gz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "run",
        "game": "polytrack-0.6.3",
        "determinism": "ok",
        "recovery": recovery,
        "rows": rows,
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return relative


def _index(root: Path, rows: list[dict]) -> None:
    path = root / "laps" / "index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_warmstart_uses_fast_laps_on_the_popular_track(tmp_path: Path):
    winter_fast = _write_lap(tmp_path, "fast", [_row(3, 0.2), _row(5, 0.4), _row(-1, 0.9), _row(3, 0.6)])
    winter_close = _write_lap(tmp_path, "close", [_row(3, 0.1)])
    winter_slow = _write_lap(tmp_path, "slow", [_row(6, 0.8)])
    winter_recovery = _write_lap(tmp_path, "recovery", [_row(3, 0.3)], recovery=True)
    summer = _write_lap(tmp_path, "summer", [_row(3, 0.7)])
    _index(
        tmp_path,
        [
            {"track": "Winter 4", "file": winter_fast, "seconds": 20.46, "game": "polytrack-0.6.3"},
            {"track": "Winter 4", "file": winter_close, "seconds": 20.90, "game": "polytrack-0.6.3"},
            {"track": "Winter 4", "file": winter_slow, "seconds": 22.00, "game": "polytrack-0.6.3"},
            {"track": "Winter 4", "file": winter_recovery, "seconds": 20.50, "game": "polytrack-0.6.3"},
            {"track": "Summer 1", "file": summer, "seconds": 18.00, "game": "polytrack-0.6.3"},
            {"track": "Summer 1", "file": summer, "seconds": 19.00, "game": "polytrack-0.6.3"},
        ],
    )
    demos = load_demos({"root": str(tmp_path), "margin_s": 0.5}, ["w", "a", "s", "d"])
    assert demos.track == "Winter 4"
    assert demos.track_file == "tracks/official/winter4.track"
    assert demos.laps == 2
    assert demos.best_seconds == 20.46
    # The reset press is not a label. The driving row after it is, with no previous chord.
    assert demos.steps == 4
    assert list(demos.actions) == [3, 5, 3, 3]
    vx = NAMES.index("vx")
    assert demos.observations[0, vx] == np.float32(0.2)
    assert demos.observations[1, vx] == np.float32(0.4)
    # The second decision was preceded by straight throttle, action 3.
    assert demos.observations[1, len(NAMES) + 3] == 1.0
    assert float(demos.observations[0, len(NAMES) :].sum()) == 0.0
    assert float(demos.observations[2, len(NAMES) :].sum()) == 0.0


def _at(action: int, x: float, z: float) -> dict:
    recorded = [0.0] * RECORDED_FEATURES
    recorded[RECORDED.index("x")] = x / 200.0
    recorded[RECORDED.index("z")] = z / 200.0
    return {"action": action, "n": recorded}


def test_best_line_is_the_single_fastest_clean_lap(tmp_path: Path):
    fast = _write_lap(tmp_path, "fast", [_at(3, 0.0, 0.0), _at(3, 10.0, 0.0)])
    slow = _write_lap(tmp_path, "slow", [_at(3, 0.0, 0.0), _at(3, 80.0, 0.0)])
    summer = _write_lap(tmp_path, "summer", [_at(3, 0.0, 0.0), _at(3, 5.0, 0.0)])
    _index(
        tmp_path,
        [
            {"track": "Winter 4", "file": fast, "seconds": 20.46, "game": "polytrack-0.6.3"},
            {"track": "Winter 4", "file": slow, "seconds": 22.00, "game": "polytrack-0.6.3"},
            {"track": "Summer 1", "file": summer, "seconds": 18.00, "game": "polytrack-0.6.3"},
            {"track": "Summer 1", "file": summer, "seconds": 19.00, "game": "polytrack-0.6.3"},
        ],
    )
    line = load_best_line({"root": str(tmp_path)})
    assert line.track == "Winter 4"
    assert line.seconds == pytest.approx(20.46)
    assert line.length == pytest.approx(10.0)


def test_clone_loss_raises_the_demo_action():
    from polytrack_agent.rl.ppo import ppo_update

    torch.manual_seed(0)
    width = obs_dim(9)
    model = StateActorCritic(width, 9, hidden=16)
    observations = torch.zeros(8, width)
    actions = torch.zeros(8, dtype=torch.int64)
    zeros = torch.zeros(8)
    demos = torch.zeros(32, width)
    labels = torch.full((32,), 3, dtype=torch.int64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    before = torch.softmax(model(demos[:1])[0], dim=-1)[0, 3].item()
    stats = ppo_update(
        model,
        optimizer,
        observations,
        actions,
        zeros,
        zeros,
        zeros,
        epochs=6,
        minibatch_size=4,
        clip=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        unimix=0.0,
        bc_observations=demos,
        bc_actions=labels,
        bc_coef=2.0,
    )
    after = torch.softmax(model(demos[:1])[0], dim=-1)[0, 3].item()
    assert "bc" in stats
    assert after > before
    assert after > 0.5


def test_behavior_clone_raises_the_labeled_chord():
    width = obs_dim(9)
    observations = np.zeros((32, width), dtype=np.float32)
    observations[:, 0] = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    actions = np.array([3] * 32, dtype=np.int64)
    from polytrack_agent.rl.bc import Demos

    demos = Demos(
        track="Winter 4",
        track_file="tracks/official/winter4.track",
        observations=observations,
        actions=actions,
        laps=1,
        best_seconds=20.0,
        margin_s=0.5,
    )
    model = StateActorCritic(width, 9, hidden=16)
    before = torch.softmax(model(torch.from_numpy(observations[:1]))[0], dim=-1)[0, 3].item()
    stats = behavior_clone(model, demos, epochs=8, batch_size=16, lr=0.05, device=torch.device("cpu"))
    after = torch.softmax(model(torch.from_numpy(observations[:1]))[0], dim=-1)[0, 3].item()
    assert stats["steps"] == 32.0
    assert after > before
    assert after > 0.5
