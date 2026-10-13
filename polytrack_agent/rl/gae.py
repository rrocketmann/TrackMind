"""Generalized advantage estimation. A done flag cuts the bootstrap."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if rewards.shape != dones.shape or rewards.shape != values.shape:
        raise ValueError("rewards, dones, and values must share a shape")
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = float(last_value) if t == len(rewards) - 1 else float(values[t + 1])
        next_nonterminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + float(gamma) * next_nonterminal * next_value - float(values[t])
        last_gae = delta + float(gamma) * float(lam) * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
