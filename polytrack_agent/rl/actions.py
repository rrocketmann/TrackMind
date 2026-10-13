"""Discrete driving actions. Opposite keys never share a mask, and restart is absent."""

from __future__ import annotations


def _pick(keys: set[str], options: tuple[str, ...]) -> list[str | None]:
    found = [name for name in options if name in keys]
    return [None, *found]


def action_masks(keys: list[str]) -> list[dict[str, bool]]:
    """Every legal hold, in a stable order."""
    keyset = set(keys)
    throttle = _pick(keyset, ("w", "up"))
    steer = _pick(keyset, ("a", "left"))
    # Brake and the other steer direction are separate axes.
    brake = _pick(keyset, ("s", "down"))
    right = _pick(keyset, ("d", "right"))
    air_left = _pick(keyset, ("q",))
    air_right = _pick(keyset, ("e",))
    # Rebuild axes as {off, positive, negative} without duplicating off.
    throttle_axis: list[str | None] = [None]
    if len(throttle) > 1:
        throttle_axis.append(throttle[1])
    if len(brake) > 1:
        throttle_axis.append(brake[1])
    steer_axis: list[str | None] = [None]
    if len(steer) > 1:
        steer_axis.append(steer[1])
    if len(right) > 1:
        steer_axis.append(right[1])
    air_axis: list[str | None] = [None]
    if len(air_left) > 1:
        air_axis.append(air_left[1])
    if len(air_right) > 1:
        air_axis.append(air_right[1])

    masks: list[dict[str, bool]] = []
    for throttle_key in throttle_axis:
        for steer_key in steer_axis:
            for air_key in air_axis:
                mask = {key: False for key in keys}
                for name in (throttle_key, steer_key, air_key):
                    if name is not None:
                        mask[name] = True
                masks.append(mask)
    return masks


def mask_for(action_id: int, keys: list[str]) -> dict[str, bool]:
    masks = action_masks(keys)
    if action_id < 0 or action_id >= len(masks):
        raise ValueError(f"action id {action_id} is outside 0..{len(masks) - 1}")
    return dict(masks[action_id])


def n_actions(keys: list[str]) -> int:
    return len(action_masks(keys))
