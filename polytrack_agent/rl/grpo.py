"""Group Relative Policy Optimization on whole laps.

A group is several episodes from the current policy. Each lap gets one score,
the sum of its visual rewards. The advantage of every decision in that lap is
the lap score centered and scaled by the group. There is no learned value
and no imagined rollout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from polytrack_agent.rl.model import mix_logits


def group_advantages(episode_returns: list[float], lengths: list[int]) -> np.ndarray:
    """Repeat each episode's group-relative score across its decisions."""
    if len(episode_returns) != len(lengths):
        raise ValueError("episode returns and lengths must match")
    returns = np.asarray(episode_returns, dtype=np.float32)
    if returns.size == 0:
        return np.zeros(0, dtype=np.float32)
    centered = returns - float(returns.mean())
    scale = float(centered.std())
    norm = np.zeros_like(centered) if scale < 1e-6 else centered / scale
    pieces = []
    for score, length in zip(norm, lengths):
        count = int(length)
        if count < 0:
            raise ValueError("episode length must be non-negative")
        if count == 0:
            continue
        pieces.append(np.full(count, float(score), dtype=np.float32))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def grpo_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
    entropy_coef: float,
    kl_coef: float,
    unimix: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    dist = Categorical(logits=mix_logits(logits, unimix))
    logp = dist.log_prob(actions)
    log_ratio = logp - old_logp
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    kl = (ratio - log_ratio - 1.0).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + float(kl_coef) * kl - float(entropy_coef) * entropy
    stats = {
        "loss": float(loss.detach()),
        "policy": float(policy_loss.detach()),
        "kl": float(kl.detach()),
        "entropy": float(entropy.detach()),
    }
    return loss, stats


def grpo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    epochs: int,
    minibatch_size: int,
    clip: float,
    entropy_coef: float,
    kl_coef: float,
    unimix: float,
) -> dict[str, float]:
    count = int(actions.shape[0])
    last_stats = {"loss": 0.0, "policy": 0.0, "kl": 0.0, "entropy": 0.0}
    model.train()
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(count, device=actions.device)
        for start in range(0, count, max(1, int(minibatch_size))):
            batch = order[start : start + minibatch_size]
            logits, _value = model(tokens[batch])
            loss, last_stats = grpo_loss(
                logits,
                actions[batch],
                old_logp[batch],
                advantages[batch],
                clip,
                entropy_coef,
                kl_coef,
                unimix,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return last_stats
