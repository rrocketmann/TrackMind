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
    update_policy: bool = True,
    bc_observations: torch.Tensor | None = None,
    bc_actions: torch.Tensor | None = None,
    bc_coef: float = 0.0,
) -> dict[str, float]:
    """One PPO step. `update_policy=False` trains the critic and leaves the actor alone.

    AdamW skips a parameter with no gradient, so a value-only step does not
    decay the cloned actor. `bc_coef` adds a clone loss on archived actions
    for the early updates, while the actor is still learning the reference lap.
    """
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    count = int(actions.shape[0])
    last_stats = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0, "bc": 0.0}
    clone = (
        update_policy
        and float(bc_coef) > 0.0
        and bc_observations is not None
        and bc_actions is not None
        and int(bc_actions.shape[0]) > 0
    )
    model.train()
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(count, device=actions.device)
        for start in range(0, count, max(1, int(minibatch_size))):
            batch = order[start : start + minibatch_size]
            logits, values = model(tokens[batch])
            if update_policy:
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
                if clone:
                    pick = torch.randint(0, int(bc_actions.shape[0]), (int(batch.shape[0]),), device=actions.device)
                    demo_logits, _demo_value = model(bc_observations[pick])
                    bc_loss = nn.functional.cross_entropy(demo_logits, bc_actions[pick])
                    loss = loss + float(bc_coef) * bc_loss
                    last_stats = {
                        **last_stats,
                        "loss": float(loss.detach()),
                        "bc": float(bc_loss.detach()),
                    }
            else:
                value_loss = (values - returns[batch]).pow(2).mean()
                loss = float(value_coef) * value_loss
                last_stats = {
                    "loss": float(loss.detach()),
                    "policy": 0.0,
                    "value": float(value_loss.detach()),
                    "entropy": 0.0,
                    "bc": 0.0,
                }
            last_stats.setdefault("bc", 0.0)
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


def entropy_for_progress(progress: float, start: float, floor: float) -> float:
    """Entropy falls in a straight line from `start` to `floor` as a lap fills in.

    `progress` is how much of the lap is already covered: 0 at the start, 1 at
    the finish. Trackmania anneals exploration the same way, off race progress
    rather than off the wall clock. Past the finish the target stays at `floor`.
    """
    covered = min(1.0, max(0.0, float(progress)))
    high = float(start)
    low = min(float(floor), high)
    return low + (high - low) * (1.0 - covered)


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
            set_entropy.scale = soft
            return at(soft)
        for _ in range(24):
            mid = (soft + sharp) / 2.0
            if at(mid) > goal:
                soft = mid
            else:
                sharp = mid
        set_entropy.scale = sharp
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
    """Leave entropy alone inside a band. Outside it, set it to the near edge.

    `hold_entropy.scale` is the logit multiplier that was applied. 1 means the
    weights were left as they were.
    """
    width = abs(float(band))
    low = max(float(floor), float(target) - width)
    high = float(target) + width
    measured = float(measured)
    if low <= measured <= high:
        hold_entropy.scale = 1.0
        return measured
    edge = low if measured < low else high
    landed = set_entropy(model, observations, unimix, edge)
    hold_entropy.scale = float(getattr(set_entropy, "scale", 1.0))
    return landed


def clear_policy_moments(optimizer: torch.optim.Optimizer, model: StateActorCritic) -> None:
    """Drop Adam's memory of the policy head after its logits are rescaled."""
    for param in (model.policy.weight, model.policy.bias):
        state = optimizer.state.get(param)
        if not state:
            continue
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
