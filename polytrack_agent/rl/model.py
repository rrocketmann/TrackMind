"""Actor-critic over lap numbers. The picture is not an input."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def mix_logits(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Logits of a distribution mixed with a uniform action prior."""
    if unimix <= 0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    mixed = (1.0 - float(unimix)) * probs + float(unimix) / float(probs.shape[-1])
    return mixed.clamp_min(1e-8).log()


class StateActorCritic(nn.Module):
    """MLP over a numeric lap vector. Categorical chord, scalar value."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        if int(obs_dim) < 1 or int(n_actions) < 1 or int(hidden) < 1:
            raise ValueError("obs_dim, n_actions, and hidden must be positive")
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )
        self.policy = nn.Linear(self.hidden, self.n_actions)
        self.value = nn.Linear(self.hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`obs` is (B, obs_dim). Returns action logits (B, A) and value (B,)."""
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"expected obs (B, {self.obs_dim}), got {tuple(obs.shape)}")
        hidden = self.net(obs)
        return self.policy(hidden), self.value(hidden).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        unimix: float = 0.0,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        mixed = mix_logits(logits, 0.0 if greedy else unimix)
        dist = Categorical(logits=mixed)
        action = mixed.argmax(dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value
