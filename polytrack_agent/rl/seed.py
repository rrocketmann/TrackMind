"""Fix the RNGs a training run draws from."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. The same seed rebuilds the same actor."""
    value = int(seed)
    if value < 0:
        raise ValueError(f"seed must be >= 0, got {value}")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
