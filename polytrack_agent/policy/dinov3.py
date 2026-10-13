"""Frozen DINOv3 encoder plus the temporal driving head.

Weights come from Hugging Face (`transformers`), not from the game. The ViT
stays frozen. Training updates only the transformer that reads its patch tokens.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from polytrack_agent.obs.preprocess import resize_nn
from polytrack_agent.policy.transformer import TemporalDriveHead, fit_heads, pool_hidden

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _load_automodel(model_id: str) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "DINOv3 needs Hugging Face transformers>=4.56. "
            "Install it with: pip install 'transformers>=4.56'"
        ) from exc
    log.info("Loading frozen DINOv3 %s", model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class FrozenDinoEncoder(nn.Module):
    """ImageNet-normalized RGB in, pooled DINOv3 patch tokens out."""

    def __init__(
        self,
        model_id: str,
        image_size: tuple[int, int],
        pool_hw: tuple[int, int],
        model: nn.Module | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.pool_hw = (int(pool_hw[0]), int(pool_hw[1]))
        if self.pool_hw[0] < 1 or self.pool_hw[1] < 1:
            raise ValueError(f"pool must be positive, got {self.pool_hw}")
        self.model = model if model is not None else _load_automodel(model_id)
        self.hidden_size = int(self.model.config.hidden_size)
        self.patch_size = int(self.model.config.patch_size)
        self.n_registers = int(getattr(self.model.config, "num_register_tokens", 0))
        height, width = self.image_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv3 image size {height}x{width} must be divisible by patch size {self.patch_size}"
            )
        self.grid_hw = (height // self.patch_size, width // self.patch_size)
        self.n_tokens = self.pool_hw[0] * self.pool_hw[1]
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def encode_frames(self, frames_01: torch.Tensor) -> torch.Tensor:
        """`frames_01` is (N, 3, H, W) in 0..1. Returns (N, n_tokens, C), detached."""
        if frames_01.ndim != 4 or frames_01.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tuple(frames_01.shape)}")
        height, width = int(frames_01.shape[-2]), int(frames_01.shape[-1])
        if (height, width) != self.image_size:
            raise ValueError(f"expected {self.image_size[0]}x{self.image_size[1]}, got {height}x{width}")
        pixels = (frames_01.float() - self.mean) / self.std
        if next(self.model.parameters()).dtype == torch.float16:
            pixels = pixels.to(dtype=torch.float16)
        with torch.no_grad():
            hidden = self.model(pixel_values=pixels).last_hidden_state
            tokens = pool_hidden(hidden.float(), self.n_registers, self.grid_hw, self.pool_hw)
        return tokens.detach()

    def prepare_inference(self) -> FrozenDinoEncoder:
        """Float16 weights on CUDA. CPU stays float32, which is faster there."""
        if next(self.model.parameters()).is_cuda:
            self.model.to(dtype=torch.float16)
            log.info("DINOv3 weights are float16 on CUDA")
        self.model.eval()
        return self


class DinoTransformerPolicy(nn.Module):
    """Frozen DINOv3 plus a trained temporal head. `forward` matches the CNN policy."""

    def __init__(self, encoder: FrozenDinoEncoder, head: TemporalDriveHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.stack = head.stack
        self.n_actions = head.n_actions
        self.use_prev_action = head.use_prev_action
        self._stream: deque[torch.Tensor] = deque(maxlen=self.stack)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(False)
        return self

    def forward(self, frames: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        """`frames` is (B, T, 3, H, W) in 0..1."""
        batch, steps, channels, height, width = frames.shape
        if channels != 3:
            raise ValueError(f"expected RGB, got {channels} channels")
        flat = frames.reshape(batch * steps, channels, height, width)
        tokens = self.encoder.encode_frames(flat)
        tokens = tokens.reshape(batch, steps, self.encoder.n_tokens, self.encoder.hidden_size)
        return self.head(tokens, prev_action)

    def forward_frame(self, frame_hwc: np.ndarray, prev_action: torch.Tensor) -> torch.Tensor:
        """Encode one new RGB frame and attend over the cached token stack.

        Play calls this so each tick pays for one ViT forward, not one per stacked frame.
        """
        device = next(self.head.parameters()).device
        image = np.ascontiguousarray(frame_hwc)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
        token = self.encoder.encode_frames(tensor)[0]
        if not self._stream:
            for _ in range(self.stack):
                self._stream.append(token)
        else:
            self._stream.append(token)
        stacked = torch.stack(list(self._stream), dim=0).unsqueeze(0)
        prev = prev_action
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
        return self.head(stacked, prev)


class EncodedDrivingDataset(Dataset):
    """Windows of precomputed DINOv3 tokens aligned with the pixel dataset samples."""

    def __init__(self, pixel_dataset, encoded: list[np.ndarray]):
        if len(encoded) != len(pixel_dataset.frames):
            raise ValueError("encoded chunks do not match the pixel dataset")
        self.base = pixel_dataset
        self.encoded = encoded
        self.samples = pixel_dataset.samples
        self.keys = pixel_dataset.keys
        self.n_actions = pixel_dataset.n_actions
        self.actions = pixel_dataset.actions
        self.stack = pixel_dataset.stack

    def __len__(self) -> int:
        return len(self.samples)

    def weight_for(self, index: int) -> float:
        return self.base.weight_for(index)

    def __getitem__(self, index: int):
        chunk, t, _episode = self.samples[index]
        window = self.encoded[chunk][t - self.stack + 1 : t + 1]
        tokens = torch.from_numpy(np.array(window, dtype=np.float32, copy=True))
        prev = (
            np.zeros(self.n_actions, dtype=np.uint8)
            if t == 0
            else np.array(self.actions[chunk][t - 1], dtype=np.uint8, copy=True)
        )
        action = np.array(self.actions[chunk][t], dtype=np.uint8, copy=True)
        return tokens, prev, action


def collate_tokens(batch):
    tokens, prevs, actions = zip(*batch)
    stacked = torch.stack(tokens, dim=0)
    prev = torch.from_numpy(np.ascontiguousarray(np.stack(prevs))).float()
    action = torch.from_numpy(np.ascontiguousarray(np.stack(actions))).float()
    return stacked, prev, action


def encode_chunks(dataset, encoder: FrozenDinoEncoder, device: torch.device, batch_size: int = 8) -> list[np.ndarray]:
    """Run the frozen ViT once per recorded frame. Later epochs train the transformer only."""
    height, width = encoder.image_size
    encoded: list[np.ndarray] = []
    total = sum(len(frames) for frames in dataset.frames)
    done = 0
    log.info(
        "Encoding %s frames with %s into %sx%s tokens of dim %s",
        total,
        encoder.model_id,
        encoder.pool_hw[0],
        encoder.pool_hw[1],
        encoder.hidden_size,
    )
    encoder.to(device)
    encoder.eval()
    for frames in dataset.frames:
        count = len(frames)
        out = np.empty((count, encoder.n_tokens, encoder.hidden_size), dtype=np.float16)
        step = max(1, int(batch_size))
        for start in range(0, count, step):
            stop = min(count, start + step)
            batch = np.stack([resize_nn(np.asarray(frames[i]), height, width) for i in range(start, stop)])
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2)
            tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
            tokens = encoder.encode_frames(tensor).cpu().numpy().astype(np.float16)
            out[start:stop] = tokens
            done += stop - start
            if done == total or done % 500 < step:
                log.info("Encoded %s/%s frames", done, total)
        encoded.append(out)
    return encoded


def build_dino_policy(cfg: dict[str, Any], n_actions: int, device: torch.device) -> tuple[FrozenDinoEncoder, TemporalDriveHead]:
    train_cfg = cfg["train"]
    image_size = (int(train_cfg["image_size"][0]), int(train_cfg["image_size"][1]))
    pool = train_cfg.get("pool") or [4, 6]
    pool_hw = (int(pool[0]), int(pool[1]))
    model_id = str(train_cfg.get("dino_model") or DEFAULT_DINO_MODEL)
    encoder = FrozenDinoEncoder(model_id, image_size, pool_hw).to(device)
    heads = fit_heads(encoder.hidden_size, int(train_cfg.get("transformer_heads") or 6))
    head = TemporalDriveHead(
        feature_dim=encoder.hidden_size,
        n_tokens=encoder.n_tokens,
        stack=int(train_cfg["stack"]),
        n_actions=n_actions,
        layers=int(train_cfg.get("transformer_layers") or 2),
        heads=heads,
        dropout=float(train_cfg.get("dropout") or 0.0),
        use_prev_action=bool(train_cfg.get("use_prev_action", True)),
    ).to(device)
    log.info(
        "DINOv3 %s  grid %sx%s  pooled tokens %s  transformer layers %s heads %s",
        model_id,
        encoder.grid_hw[0],
        encoder.grid_hw[1],
        encoder.n_tokens,
        head.layers,
        head.heads,
    )
    return encoder, head


def load_dino_policy(payload: dict[str, Any], device: torch.device) -> DinoTransformerPolicy:
    image_size = (int(payload["image_size"][0]), int(payload["image_size"][1]))
    pool = payload.get("pool") or [4, 6]
    encoder = FrozenDinoEncoder(
        str(payload["dino_model"]),
        image_size,
        (int(pool[0]), int(pool[1])),
    ).to(device)
    head = TemporalDriveHead(
        feature_dim=int(payload["feature_dim"]),
        n_tokens=int(payload["n_tokens"]),
        stack=int(payload["stack"]),
        n_actions=len(payload["keys"]),
        layers=int(payload.get("transformer_layers") or 2),
        heads=int(payload.get("transformer_heads") or 6),
        dropout=float(payload.get("dropout") or 0.0),
        use_prev_action=bool(payload.get("use_prev_action", True)),
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device)
    head.eval()
    policy = DinoTransformerPolicy(encoder, head)
    policy.eval()
    return policy
