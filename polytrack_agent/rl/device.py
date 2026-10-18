"""Pick a torch device and enable the fast CUDA kernels when one is present."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_torch(device: torch.device) -> str:
    """Enable fast CUDA kernels. Returns a short description of the device."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except (AttributeError, RuntimeError):
        pass
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    name = torch.cuda.get_device_name(index)
    log.info("%s has %d MiB.", name, props.total_memory // (1024**2))
    return name
