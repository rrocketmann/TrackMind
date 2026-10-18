"""Discrete holds of the 0.6.3 car. Opposite keys never share a mask."""

from __future__ import annotations

_THROTTLE = ("w", "up")
_BRAKE = ("s", "down")
_LEFT = ("a", "left")
_RIGHT = ("d", "right")
_ALLOWED = frozenset((*_THROTTLE, *_BRAKE, *_LEFT, *_RIGHT))


def _named(keyset: set[str], options: tuple[str, ...]) -> str | None:
    found = [name for name in options if name in keyset]
    return found[0] if found else None


def action_masks(keys: list[str]) -> list[dict[str, bool]]:
    """Every legal hold of up/down/left/right, in a stable order.

    PolyTrack 0.6.3 has those four controls. Air pitch and roll reuse them
    while the car is off the ground, so q and e are not actions.
    """
    keyset = set(keys)
    unknown = sorted(keyset - _ALLOWED)
    if unknown:
        raise ValueError(
            f"PolyTrack 0.6.3 only accepts {sorted(_ALLOWED)}. Refusing {unknown}."
        )
    throttle_axis: list[str | None] = [None]
    steer_axis: list[str | None] = [None]
    throttle = _named(keyset, _THROTTLE)
    brake = _named(keyset, _BRAKE)
    left = _named(keyset, _LEFT)
    right = _named(keyset, _RIGHT)
    if throttle is not None:
        throttle_axis.append(throttle)
    if brake is not None:
        throttle_axis.append(brake)
    if left is not None:
        steer_axis.append(left)
    if right is not None:
        steer_axis.append(right)
    masks: list[dict[str, bool]] = []
    for throttle_key in throttle_axis:
        for steer_key in steer_axis:
            mask = {key: False for key in keys}
            if throttle_key is not None:
                mask[throttle_key] = True
            if steer_key is not None:
                mask[steer_key] = True
            masks.append(mask)
    return masks


def mask_for(action_id: int, keys: list[str]) -> dict[str, bool]:
    masks = action_masks(keys)
    if action_id < 0 or action_id >= len(masks):
        raise ValueError(f"action id {action_id} is outside 0..{len(masks) - 1}")
    return dict(masks[action_id])


def n_actions(keys: list[str]) -> int:
    return len(action_masks(keys))


def forward_ids(keys: list[str]) -> list[int]:
    """Chords that hold throttle and do not brake."""
    ids = []
    for index, mask in enumerate(action_masks(keys)):
        throttle = bool(mask.get("w") or mask.get("up"))
        brake = bool(mask.get("s") or mask.get("down"))
        if throttle and not brake:
            ids.append(index)
    return ids
