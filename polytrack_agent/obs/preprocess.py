"""Resize and stack RGB frames. The hot path stays in NumPy."""

from __future__ import annotations

from collections import deque

import numpy as np

_INDEX_CACHE: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def bgra_to_rgb(bgra: np.ndarray) -> np.ndarray:
    """Convert an mss BGRA image to contiguous RGB."""
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"expected BGRA HxWx4, got {bgra.shape}")
    return np.ascontiguousarray(bgra[:, :, 2::-1])


def as_rgb(frame: np.ndarray) -> np.ndarray:
    """Accept RGB or RGBA. RGBA drops the alpha channel and keeps RGB order."""
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 or HxWx4, got {getattr(frame, 'shape', None)}")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def resize_nn(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize. Index arrays are cached for the capture size."""
    if out_h < 1 or out_w < 1:
        raise ValueError(f"bad output size {out_h}x{out_w}")
    if image.shape[0] == out_h and image.shape[1] == out_w:
        return np.ascontiguousarray(image)
    key = (int(image.shape[0]), int(image.shape[1]), int(out_h), int(out_w))
    cached = _INDEX_CACHE.get(key)
    if cached is None:
        ys = np.linspace(0, image.shape[0] - 1, out_h).astype(np.int32)
        xs = np.linspace(0, image.shape[1] - 1, out_w).astype(np.int32)
        cached = (ys, xs)
        _INDEX_CACHE[key] = cached
    ys, xs = cached
    return np.ascontiguousarray(image[ys][:, xs])


def rgb_to_tiny_gray(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Downsample an RGB frame to a float32 grayscale picture for lap alignment."""
    small = resize_nn(as_rgb(frame), out_h, out_w).astype(np.float32)
    return 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]


def augment_stack(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild brightness, contrast, and a 0-2 px shift. No horizontal flip.

    A flip would swap left and right steering, so it is not a valid driving augment.
    `frames` is uint8 with shape (T, H, W, 3).
    """
    out = frames.astype(np.float32)
    out *= float(rng.uniform(0.9, 1.1))
    out += float(rng.uniform(-8.0, 8.0))
    np.clip(out, 0, 255, out)
    out_u8 = out.astype(np.uint8)
    dy = int(rng.integers(-2, 3))
    dx = int(rng.integers(-2, 3))
    if dy or dx:
        out_u8 = np.roll(out_u8, shift=(dy, dx), axis=(1, 2))
    return np.ascontiguousarray(out_u8)


class FrameStack:
    """Fixed-length stack of resized RGB frames. The first frame is repeated."""

    def __init__(self, stack: int, height: int, width: int):
        if stack < 1:
            raise ValueError("stack must be >= 1")
        self.stack = int(stack)
        self.height = int(height)
        self.width = int(width)
        self.frames: deque[np.ndarray] = deque(maxlen=self.stack)

    def reset(self, frame: np.ndarray | None = None) -> np.ndarray | None:
        self.frames.clear()
        if frame is None:
            return None
        return self.push(frame)

    def push(self, frame: np.ndarray) -> np.ndarray:
        small = resize_nn(as_rgb(frame), self.height, self.width)
        if not self.frames:
            for _ in range(self.stack):
                self.frames.append(small)
        else:
            self.frames.append(small)
        return np.stack(list(self.frames), axis=0)

    def latest(self) -> np.ndarray:
        if len(self.frames) != self.stack:
            raise RuntimeError("frame stack is empty")
        return np.stack(list(self.frames), axis=0)
