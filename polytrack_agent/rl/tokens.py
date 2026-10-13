"""Turn one captured frame into the token stack the actor reads."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


def encode_frame(encoder, frame_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """One HWC uint8 frame to one token row (N, D) on `device`."""
    image = np.ascontiguousarray(frame_hwc)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    return encoder.encode_frames(tensor)[0].detach()


def preview_stack(history: deque, token: torch.Tensor, stack: int) -> torch.Tensor:
    """Stack as if `token` were appended, without changing `history`."""
    items = list(history) + [token]
    if len(items) < stack:
        items = [items[0]] * (stack - len(items)) + items
    return torch.stack(items[-stack:], dim=0)


def stack_history(history: deque, token: torch.Tensor, stack: int) -> torch.Tensor:
    """Append `token` and return (stack, N, D), repeating the first frame to fill."""
    history.append(token)
    items = list(history)
    if len(items) < stack:
        items = [items[0]] * (stack - len(items)) + items
    return torch.stack(items[-stack:], dim=0)
