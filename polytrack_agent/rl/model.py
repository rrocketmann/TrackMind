"""Actor-critic over a short stack of frozen visual tokens."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from polytrack_agent.policy.transformer import fit_heads


def mix_logits(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Logits of a distribution mixed with a uniform action prior."""
    if unimix <= 0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    mixed = (1.0 - float(unimix)) * probs + float(unimix) / float(probs.shape[-1])
    return mixed.clamp_min(1e-8).log()


class TokenActorCritic(nn.Module):
    """Temporal transformer trunk, categorical policy, scalar value."""

    def __init__(
        self,
        feature_dim: int,
        n_tokens: int,
        stack: int,
        n_actions: int,
        layers: int = 2,
        heads: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(feature_dim, n_tokens, stack, n_actions) < 1:
            raise ValueError("feature_dim, n_tokens, stack, and n_actions must be positive")
        self.feature_dim = int(feature_dim)
        self.n_tokens = int(n_tokens)
        self.stack = int(stack)
        self.n_actions = int(n_actions)
        self.layers = int(layers)
        self.heads = fit_heads(self.feature_dim, heads)
        self.pos_time = nn.Parameter(torch.zeros(1, self.stack, 1, self.feature_dim))
        self.pos_space = nn.Parameter(torch.zeros(1, 1, self.n_tokens, self.feature_dim))
        nn.init.normal_(self.pos_time, std=0.02)
        nn.init.normal_(self.pos_space, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.heads,
            dim_feedforward=self.feature_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=self.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.policy = nn.Linear(self.feature_dim, self.n_actions)
        self.value = nn.Linear(self.feature_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`tokens` is (B, T, N, D). Returns action logits (B, A) and value (B,)."""
        batch, steps, count, dim = tokens.shape
        if steps != self.stack or count != self.n_tokens or dim != self.feature_dim:
            raise ValueError(
                f"expected tokens (B, {self.stack}, {self.n_tokens}, {self.feature_dim}), got {tuple(tokens.shape)}"
            )
        x = tokens + self.pos_time + self.pos_space
        x = self.encoder(x.reshape(batch, steps * count, dim))
        pooled = self.norm(x.mean(dim=1))
        return self.policy(pooled), self.value(pooled).squeeze(-1)

    def act(
        self,
        tokens: torch.Tensor,
        unimix: float = 0.0,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(tokens)
        mixed = mix_logits(logits, 0.0 if greedy else unimix)
        dist = Categorical(logits=mixed)
        action = mixed.argmax(dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value
