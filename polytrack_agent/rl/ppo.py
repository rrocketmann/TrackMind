"""Clipped PPO update for a discrete actor-critic."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from polytrack_agent.rl.model import mix_logits


def ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip: float,
    entropy_coef: float,
    value_coef: float,
    unimix: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    dist = Categorical(logits=mix_logits(logits, unimix))
    logp = dist.log_prob(actions)
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = (values - returns).pow(2).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy
    stats = {
        "loss": float(loss.detach()),
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
    }
    return loss, stats


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    epochs: int,
    minibatch_size: int,
    clip: float,
    entropy_coef: float,
    value_coef: float,
    unimix: float,
) -> dict[str, float]:
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    count = int(actions.shape[0])
    last_stats = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
    model.train()
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(count, device=actions.device)
        for start in range(0, count, max(1, int(minibatch_size))):
            batch = order[start : start + minibatch_size]
            logits, values = model(tokens[batch])
            loss, last_stats = ppo_loss(
                logits,
                values,
                actions[batch],
                old_logp[batch],
                advantages[batch],
                returns[batch],
                clip,
                entropy_coef,
                value_coef,
                unimix,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return last_stats
