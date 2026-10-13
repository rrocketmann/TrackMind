"""Key-name helpers and the driving action layout.

Actions are multi-hot key holds, because PolyTrack steering is digital and
W+A is a normal input. Continuous mode turns those probabilities into axes.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

ACCEL_NAMES = ("w", "up")
BRAKE_NAMES = ("s", "down")
LEFT_NAMES = ("a", "left")
RIGHT_NAMES = ("d", "right")
AIR_LEFT_NAMES = ("q",)
AIR_RIGHT_NAMES = ("e",)


def key_name(key: object) -> str | None:
    """Normalize a pynput key, or any object with `char` / `name`, to a lower-case name."""
    char = getattr(key, "char", None)
    if isinstance(char, str) and char and len(char) == 1 and char.isprintable():
        return char.lower()
    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    return None


def pynput_key(name: str):
    """Build a pynput key. Imported lazily so tests do not need an X display."""
    from pynput.keyboard import Key, KeyCode

    special = {
        "space": Key.space,
        "enter": Key.enter,
        "return": Key.enter,
        "esc": Key.esc,
        "escape": Key.esc,
        "tab": Key.tab,
        "shift": Key.shift,
        "ctrl": Key.ctrl,
        "alt": Key.alt,
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
        "backspace": Key.backspace,
        "caps_lock": Key.caps_lock,
    }
    if name.startswith("f") and name[1:].isdigit():
        return getattr(Key, name)
    if name in special:
        return special[name]
    if len(name) == 1:
        return KeyCode.from_char(name)
    raise ValueError(f"Unsupported key {name!r}")


def axis_indices(keys: Sequence[str]) -> dict[str, tuple[list[int], list[int]]]:
    def idx(names: Iterable[str]) -> list[int]:
        wanted = set(names)
        return [i for i, key in enumerate(keys) if key in wanted]

    return {
        "throttle": (idx(ACCEL_NAMES), idx(BRAKE_NAMES)),
        "steer": (idx(LEFT_NAMES), idx(RIGHT_NAMES)),
        "air": (idx(AIR_LEFT_NAMES), idx(AIR_RIGHT_NAMES)),
    }


def format_mask(mask: dict[str, bool], keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        parts.append(key.upper() if mask.get(key) else ".")
    return " ".join(parts)


def zeros_action(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.uint8)
