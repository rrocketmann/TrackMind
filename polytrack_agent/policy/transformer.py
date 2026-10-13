"""Temporal transformer over frozen visual tokens.

DINOv3 supplies one token grid per frame. This module attends across that
short stack and predicts the driving-key logits. It does not load weights.
"""

from __future__ import annotations

import torch
from torch import nn


def fit_heads(feature_dim: int, requested: int) -> int:
    """Pick an attention-head count that divides `feature_dim`."""
    requested = max(1, int(requested))
    if feature_dim % requested == 0:
        return requested
    for heads in range(min(requested, feature_dim), 0, -1):
        if feature_dim % heads == 0:
            return heads
    return 1


def pool_hidden(
    hidden: torch.Tensor,
    n_registers: int,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> torch.Tensor:
    """Turn a DINOv3 hidden sequence into a coarse spatial token grid.

    `hidden` is (B, 1 + registers + H*W, C): class token, register tokens, then
    one vector per patch, in row-major order. The result is (B, pool_h*pool_w, C).
    """
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    pool_h, pool_w = int(pool_hw[0]), int(pool_hw[1])
    patches = hidden[:, 1 + int(n_registers) :, :]
    batch, count, channels = patches.shape
    expected = grid_h * grid_w
    if count != expected:
        raise ValueError(f"expected {expected} patch tokens, got {count}")
    grid = patches.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (pool_h, pool_w))
    return pooled.flatten(2).transpose(1, 2).contiguous()


class TemporalDriveHead(nn.Module):
    """Attend over a (time, space) token grid and emit one logit per key."""

    def __init__(
        self,
        feature_dim: int,
        n_tokens: int,
        stack: int,
        n_actions: int,
        layers: int = 2,
        heads: int = 6,
        dropout: float = 0.1,
        use_prev_action: bool = True,
    ):
        super().__init__()
        if feature_dim < 1 or n_tokens < 1 or stack < 1 or n_actions < 1:
            raise ValueError("feature_dim, n_tokens, stack, and n_actions must be positive")
        self.feature_dim = int(feature_dim)
        self.n_tokens = int(n_tokens)
        self.stack = int(stack)
        self.n_actions = int(n_actions)
        self.layers = int(layers)
        self.heads = fit_heads(self.feature_dim, heads)
        self.use_prev_action = bool(use_prev_action)
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
        fused = self.feature_dim + (self.n_actions if self.use_prev_action else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 256),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, self.n_actions),
        )
        nn.init.constant_(self.head[-1].bias, -1.5)

    def forward(self, tokens: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: (B, T, N, D)
        batch, steps, count, dim = tokens.shape
        if steps != self.stack or count != self.n_tokens or dim != self.feature_dim:
            raise ValueError(
                f"expected tokens (B, {self.stack}, {self.n_tokens}, {self.feature_dim}), got {tuple(tokens.shape)}"
            )
        x = tokens + self.pos_time + self.pos_space
        x = self.encoder(x.reshape(batch, steps * count, dim))
        pooled = x.mean(dim=1)
        if self.use_prev_action:
            if prev_action is None:
                prev_action = torch.zeros(batch, self.n_actions, device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, prev_action], dim=-1)
        return self.head(pooled)
