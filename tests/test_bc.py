"""Behavioral cloning packs the same numbers the PPO actor reads."""

from __future__ import annotations

import numpy as np

from polytrack_agent.rl.bc import pairs_from_rows
from polytrack_agent.rl.state import FEATURES


def test_previous_hold_is_one_hot_and_cleared_after_a_gap():
    rows = [
        {"dt": 0.0, "action": 3, "n": [0.0] * FEATURES},
        {"dt": 0.05, "action": 5, "n": [0.1] * FEATURES},
        {"dt": 0.5, "action": 4, "n": [0.2] * FEATURES},
        {"dt": 0.05, "action": 1, "n": [0.3] * FEATURES},
    ]
    observations, actions = pairs_from_rows(rows, n_actions=9)
    assert actions.tolist() == [3, 5, 4, 1]
    assert observations.shape == (4, FEATURES + 9)
    assert float(observations[0, FEATURES:].sum()) == 0.0
    assert observations[1, FEATURES + 3] == 1.0
    assert float(observations[1, FEATURES:].sum()) == 1.0
    assert float(observations[2, FEATURES:].sum()) == 0.0
    assert observations[3, FEATURES + 4] == 1.0
    assert observations[1, 0] == np.float32(0.1)


def test_bad_rows_do_not_keep_the_previous_hold():
    rows = [
        {"dt": 0.0, "action": 3, "n": [0.0] * FEATURES},
        {"dt": 0.05, "action": 9, "n": [0.0] * FEATURES},
        {"dt": 0.05, "action": 2, "n": [0.0] * FEATURES},
    ]
    observations, actions = pairs_from_rows(rows, n_actions=9)
    assert actions.tolist() == [3, 2]
    assert float(observations[1, FEATURES:].sum()) == 0.0
