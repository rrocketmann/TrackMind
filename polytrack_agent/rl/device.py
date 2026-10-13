"""CUDA setup for the desktop RTX 3050 in the custom PC.

That card has 8GB. Chromium's WebGL view of the game shares it with PyTorch,
so the frozen ViT stays in float16 and about 2GB is left for the browser.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

# 8GB desktop 3050, with a little room above the reported total.
_SHARED_VRAM_BYTES = 10 * 1024**3
_PYTORCH_FRACTION = 0.75


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
    if props.total_memory <= _SHARED_VRAM_BYTES:
        try:
            torch.cuda.set_per_process_memory_fraction(_PYTORCH_FRACTION, index)
            log.info(
                "%s has %d MiB. PyTorch will use about %d%% so Chromium can keep the game.",
                name,
                props.total_memory // (1024**2),
                int(_PYTORCH_FRACTION * 100),
            )
        except RuntimeError as exc:
            log.info("Could not cap CUDA memory (%s). Close other GPU apps if Chromium stutters.", exc)
    else:
        log.info("%s has %d MiB.", name, props.total_memory // (1024**2))
    return name
