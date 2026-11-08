"""Clipped PPO update for a discrete actor-critic."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from polytrack_agent.rl.model import StateActorCritic, mix_logits


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
    # `returns` is the discounted sum of the environment reward. The critic
    # matches that, not a bonus for staying random.
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
    with torch.no_grad():
        last_stats["entropy"] = _entropy(model, tokens, unimix)
    return last_stats


def _entropy(model: StateActorCritic, observations: torch.Tensor, unimix: float) -> float:
    logits, _values = model(observations)
    return float(Categorical(logits=mix_logits(logits, unimix)).entropy().mean())


def set_entropy(
    model: StateActorCritic,
    observations: torch.Tensor,
    unimix: float,
    target: float,
) -> float:
    """Scale the policy logits so the mean entropy lands on `target`.

    Multiplying every logit by the same positive number keeps the action order
    and only changes how sure the policy is. A smaller scale is a higher entropy.
    """
    ceiling = math.log(int(model.n_actions)) - 1e-4
    goal = min(float(target), ceiling)
    base_weight = model.policy.weight.detach().clone()
    base_bias = model.policy.bias.detach().clone()

    def at(scale: float) -> float:
        model.policy.weight.copy_(base_weight * scale)
        model.policy.bias.copy_(base_bias * scale)
        return _entropy(model, observations, unimix)

    with torch.no_grad():
        # `soft` stays on the high-entropy side of the goal, `sharp` on the low side.
        soft = 1e-3
        sharp = 1.0
        if at(sharp) > goal:
            for _ in range(16):
                sharp *= 2.0
                if at(sharp) <= goal:
                    break
        elif at(soft) < goal:
            return at(soft)
        for _ in range(24):
            mid = (soft + sharp) / 2.0
            if at(mid) > goal:
                soft = mid
            else:
                sharp = mid
        return at(sharp)


def hold_entropy(
    model: StateActorCritic,
    observations: torch.Tensor,
    unimix: float,
    measured: float,
    target: float,
    band: float,
    floor: float,
) -> float:
    """Leave entropy alone inside a band around the schedule. Outside it, move to the edge."""
    width = abs(float(band))
    low = max(float(floor), float(target) - width)
    high = float(target) + width
    if low <= float(measured) <= high:
        return float(measured)
    edge = low if float(measured) < low else high
    return set_entropy(model, observations, unimix, edge)


def clear_policy_moments(optimizer: torch.optim.Optimizer, model: StateActorCritic) -> None:
    """Drop Adam's memory of the policy head after its logits are rescaled."""
    for param in (model.policy.weight, model.policy.bias):
        state = optimizer.state.get(param)
        if not state:
            continue
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
