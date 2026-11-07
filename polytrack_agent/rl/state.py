"""Car-state observation from the PolyTrack 0.6.3 simulation. No pixels.

Each decision sees a short list of numbers: where the car is, how it is moving,
how that motion is changing, and how much road is left to the side. A future
run can also append the rest of the car and the shape of the road ahead. That
block stays off unless `rl.road_window` is set, so an old checkpoint still
matches.
"""

from __future__ import annotations

import math

import numpy as np

# Order is the observation layout. Scales below keep a Summer 1 lap near ±1.
NAMES = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "forward_x",
    "forward_z",
    "upright",
    "wheels",
    "border",
    "border_side",
    "gate_right",
    "gate_forward",
    "gate_distance",
    "checkpoint",
)
FEATURES = len(NAMES)
# Only present when the road window is on, between the 19 and the road slots.
# Window 0 stays the nineteen numbers plus the previous button.
EXTRA_NAMES = (
    "nose_up",
    "roll",
    "v_forward",
    "v_left",
    "v_up",
    "a_forward",
    "a_left",
    "a_up",
    "w_forward",
    "w_left",
    "w_up",
    "wheel_0",
    "wheel_1",
    "wheel_2",
    "wheel_3",
    "speed_kmh",
    "skid_0",
    "skid_1",
    "skid_2",
    "skid_3",
    "suspension_0",
    "suspension_1",
    "suspension_2",
    "suspension_3",
    "suspension_v0",
    "suspension_v1",
    "suspension_v2",
    "suspension_v3",
    "above",
    "line_left",
    "ground_bank",
    "ground_grade",
)
_POS = 200.0
_VEL = 40.0
_ACC = 40.0
_BORDER = 10.0
_GATE = 100.0
_RATE = 4.0  # rad/s. A hard yaw is a few rad/s, so this sits near 1.
_SPEED = 200.0  # km/h
_SUSP_V = 5.0  # m/s
_OFFSET = 10.0  # metres above the road, and metres left of the racing line


def extra_width(road_window: int) -> int:
    """How many extra car numbers a window adds. Zero keeps the old width."""
    return len(EXTRA_NAMES) if int(road_window) > 0 else 0


def obs_dim(n_actions: int, road_window: int = 0) -> int:
    if int(n_actions) < 1:
        raise ValueError("n_actions must be positive")
    from polytrack_agent.rl.road import road_width

    window = int(road_window)
    return FEATURES + extra_width(window) + road_width(window) + int(n_actions)


def _quaternion(state: dict) -> tuple[float, float, float, float]:
    quat = state.get("quaternion") or {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    return float(quat["x"]), float(quat["y"]), float(quat["z"]), float(quat["w"])


def _axes(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Body axes in world coordinates. +Z is forward, +X is the driver's left, +Y is up.

    Identity faces +z, so left is +x. One source for the nineteen numbers and
    for the extra block, so the two cannot disagree about which way is left.
    """
    forward = (
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    left = (
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qw * qz),
        2.0 * (qx * qz - qw * qy),
    )
    up = (
        2.0 * (qx * qy - qw * qz),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz + qw * qx),
    )
    return forward, left, up


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _vec(raw) -> tuple[float, float, float]:
    if not isinstance(raw, dict):
        return (0.0, 0.0, 0.0)
    return (_num(raw.get("x")), _num(raw.get("y")), _num(raw.get("z")))


def _four(state: dict, key: str) -> list[float]:
    raw = state.get(key) or ()
    if not isinstance(raw, (list, tuple)):
        return [0.0, 0.0, 0.0, 0.0]
    return [_num(raw[index]) if index < len(raw) else 0.0 for index in range(4)]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _ground_tilt(
    state: dict,
    forward: tuple[float, float, float],
    left: tuple[float, float, float],
) -> tuple[float, float]:
    """Bank and grade of the ground under the tyres that are touching.

    A contact normal straight up is flat. Left side of the road up is a
    positive bank. Uphill along the nose is a positive grade. Wheels in the
    air are left out of the average.
    """
    raw = state.get("wheelNormal") or ()
    if not isinstance(raw, (list, tuple)):
        return 0.0, 0.0
    bank = 0.0
    grade = 0.0
    count = 0
    for item in list(raw)[:4]:
        if not isinstance(item, dict):
            continue
        normal = _vec(item)
        if normal == (0.0, 0.0, 0.0):
            continue
        # Normal leans away from the high side. Flip it so high-on-the-left
        # and uphill are positive, matching roll and the road slope.
        bank += -_dot(normal, left)
        grade += -_dot(normal, forward)
        count += 1
    if count == 0:
        return 0.0, 0.0
    return bank / count, grade / count


def _checkpoint_index(state: dict) -> int | None:
    checkpoint = state.get("nextCheckpointIndex")
    try:
        return None if checkpoint is None else int(checkpoint)
    except (TypeError, ValueError):
        return None


def checkpoint_from_scaled(scaled: float, gate_count: int) -> int | None:
    """Turn the checkpoint fraction in a TrackData row back into an index."""
    gates = int(gate_count or 0)
    if gates <= 0:
        return None
    return int(round(float(scaled) * gates))


def _features(state: dict, acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    forward, left, up = _axes(*_quaternion(state))
    wheels = state.get("wheels") or []
    contact = sum(1.0 for wheel in wheels if wheel) / 4.0
    checkpoints = float(state.get("checkpointCount") or 3)
    position = state.get("position") or {}
    px = float(position.get("x") or 0.0)
    py = float(position.get("y") or 0.0)
    pz = float(position.get("z") or 0.0)
    target = state.get("target") or {"x": px, "z": pz}
    dx = float(target["x"]) - px
    dz = float(target["z"]) - pz
    # The body's +Z is forward, and its +X points to the driver's left.
    # Gate-right is the opposite of that left, which is what the name says.
    local_right = -(dx * left[0] + dz * left[2])
    local_forward = dx * forward[0] + dz * forward[2]
    velocity = state.get("velocity") or {}
    vx = float(velocity.get("x") or 0.0)
    vy = float(velocity.get("y") or 0.0)
    vz = float(velocity.get("z") or 0.0)
    ax, ay, az = acceleration
    border = state.get("border") or {}
    clearance = float(border.get("clearance") or 0.0)
    side = float(border.get("side") or 0.0)
    return [
        px / _POS,
        py / _POS,
        pz / _POS,
        vx / _VEL,
        vy / _VEL,
        vz / _VEL,
        ax / _ACC,
        ay / _ACC,
        az / _ACC,
        forward[0],
        forward[2],
        up[1],
        contact,
        clearance / _BORDER,
        side / _BORDER,
        local_right / _GATE,
        local_forward / _GATE,
        math.hypot(dx, dz) / _GATE,
        float(state.get("nextCheckpointIndex") or 0.0) / max(1.0, checkpoints),
    ]


def _extra(
    state: dict,
    acceleration: tuple[float, float, float],
    course,
) -> np.ndarray:
    """The rest of the car. Same order as EXTRA_NAMES.

    Nose-up is forward · world up. Roll is left · world up, so the driver's
    left side being high is positive. w_left is the pitch rate: a positive
    spin about the left axis (right-hand, thumb to the driver's left) pitches
    the nose down. Wheel numbers stay in the game's own 0..3 order.
    """
    forward, left, up = _axes(*_quaternion(state))
    velocity = _vec(state.get("velocity"))
    angular = _vec(state.get("angularVelocity"))
    wheels = state.get("wheels") or []
    wheel_bits = [1.0 if index < len(wheels) and wheels[index] else 0.0 for index in range(4)]
    skid = [_clamp(value, -1.0, 2.0) for value in _four(state, "wheelSkidInfo")]
    suspension = [_clamp(value, -1.0, 2.0) for value in _four(state, "wheelSuspensionLength")]
    suspension_v = [value / _SUSP_V for value in _four(state, "wheelSuspensionVelocity")]
    above_m, left_m = 0.0, 0.0
    bank, grade = _ground_tilt(state, forward, left)
    if course is not None:
        position = state.get("position") or {}
        above_m, left_m = course.car_offsets(
            float(position.get("x") or 0.0),
            float(position.get("y") or 0.0),
            float(position.get("z") or 0.0),
            _checkpoint_index(state),
        )
    values = [
        forward[1],
        left[1],
        _dot(velocity, forward) / _VEL,
        _dot(velocity, left) / _VEL,
        _dot(velocity, up) / _VEL,
        _dot(acceleration, forward) / _ACC,
        _dot(acceleration, left) / _ACC,
        _dot(acceleration, up) / _ACC,
        _dot(angular, forward) / _RATE,
        _dot(angular, left) / _RATE,
        _dot(angular, up) / _RATE,
        *wheel_bits,
        _num(state.get("speedKmh")) / _SPEED,
        *skid,
        *suspension,
        *suspension_v,
        above_m / _OFFSET,
        left_m / _OFFSET,
        bank,
        grade,
    ]
    if len(values) != len(EXTRA_NAMES):
        raise RuntimeError(f"expected {len(EXTRA_NAMES)} extra car numbers, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def extra_from_scaled(features, course) -> np.ndarray:
    """Best-effort extra block from one TrackData row of nineteen numbers.

    The row has no spin, skid, suspension, nose, roll, or ground tilt, so those stay 0.
    Height above the road and the offset from the racing line come from the
    course. Speed in the car's axes uses the flat heading in forward_x and
    forward_z, which is the whole heading a row can still show.
    """
    row = np.asarray(features, dtype=np.float64)
    out = np.zeros(len(EXTRA_NAMES), dtype=np.float32)
    names = {name: index for index, name in enumerate(EXTRA_NAMES)}
    fx = float(row[NAMES.index("forward_x")])
    fz = float(row[NAMES.index("forward_z")])
    flat = math.hypot(fx, fz)
    vx = float(row[NAMES.index("vx")])
    vy = float(row[NAMES.index("vy")])
    vz = float(row[NAMES.index("vz")])
    ax = float(row[NAMES.index("ax")])
    ay = float(row[NAMES.index("ay")])
    az = float(row[NAMES.index("az")])
    if flat > 1e-3:
        hx, hz = fx / flat, fz / flat
        # Left of a flat heading (hx, hz) is (hz, -hx). Facing +z, left is +x.
        out[names["v_forward"]] = np.float32(vx * hx + vz * hz)
        out[names["v_left"]] = np.float32(vx * hz - vz * hx)
        out[names["v_up"]] = np.float32(vy)
        out[names["a_forward"]] = np.float32(ax * hx + az * hz)
        out[names["a_left"]] = np.float32(ax * hz - az * hx)
        out[names["a_up"]] = np.float32(ay)
    out[names["speed_kmh"]] = np.float32(math.hypot(vx, vz) * _VEL * 3.6 / _SPEED)
    touched = float(row[NAMES.index("wheels")])
    for name in ("wheel_0", "wheel_1", "wheel_2", "wheel_3"):
        out[names[name]] = np.float32(touched)
    if course is not None:
        index = checkpoint_from_scaled(float(row[NAMES.index("checkpoint")]), getattr(course, "gate_count", 0))
        above_m, left_m = course.car_offsets(
            float(row[NAMES.index("x")]) * _POS,
            float(row[NAMES.index("y")]) * _POS,
            float(row[NAMES.index("z")]) * _POS,
            index,
        )
        out[names["above"]] = np.float32(above_m / _OFFSET)
        out[names["line_left"]] = np.float32(left_m / _OFFSET)
    return out


class LapNumbers:
    """The numbers a decision is allowed to see."""

    def __init__(self, n_actions: int, road=None, road_window: int = 0):
        self.n_actions = int(n_actions)
        self.road = road
        self.road_window = int(road_window)
        if self.road_window < 0:
            raise ValueError("road_window must be >= 0")
        self._vector = np.zeros(obs_dim(self.n_actions, self.road_window), dtype=np.float32)
        self._velocity = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._vector[:] = 0.0
        self._velocity = (0.0, 0.0, 0.0)

    def vector(self) -> np.ndarray:
        return self._vector.copy()

    def advance(self, info: dict, action_id: int) -> None:
        """Store the car state that the next decision will read."""
        state = info["state"]
        velocity = state.get("velocity") or {}
        current = (
            float(velocity.get("x") or 0.0),
            float(velocity.get("y") or 0.0),
            float(velocity.get("z") or 0.0),
        )
        dt = float(info.get("dt") or 0.0)
        if dt > 1e-6:
            previous = self._velocity
            acceleration = tuple((current[i] - previous[i]) / dt for i in range(3))
        else:
            acceleration = (0.0, 0.0, 0.0)
        self._velocity = current
        features = _features(state, acceleration)
        if len(features) != FEATURES:
            raise RuntimeError(f"expected {FEATURES} car features, got {len(features)}")
        self._vector[:FEATURES] = np.asarray(features, dtype=np.float32)
        self._vector[FEATURES:] = 0.0
        cursor = FEATURES
        if self.road_window:
            extra = _extra(state, acceleration, self.road)
            self._vector[cursor : cursor + extra.shape[0]] = extra
            cursor += int(extra.shape[0])
            block = self._road_block(state)
            self._vector[cursor : cursor + block.shape[0]] = block
            cursor += int(block.shape[0])
        if 0 <= int(action_id) < self.n_actions:
            self._vector[cursor + int(action_id)] = 1.0

    def _road_block(self, state: dict) -> np.ndarray:
        from polytrack_agent.rl.road import road_width

        width = road_width(self.road_window)
        if self.road is None:
            return np.zeros(width, dtype=np.float32)
        position = state.get("position") or {}
        index = _checkpoint_index(state)
        return self.road.ahead(
            float(position.get("x") or 0.0),
            float(position.get("y") or 0.0),
            float(position.get("z") or 0.0),
            self.road_window,
            checkpoint=index,
        )
