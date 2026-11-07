"""The road ahead is length, turn, slope, and the two side distances."""

import math

import numpy as np
import pytest

from polytrack_agent.rl.road import ROAD_AHEAD, RoadCourse, _project, course_from_parts, rotate_tile, side_clearance
from polytrack_agent.rl.state import EXTRA_NAMES, FEATURES, NAMES, LapNumbers, obs_dim

_START = 5
_STRAIGHT = 0
_FINISH = 6


def _part(x, y, z, kind, order=-1):
    return [x, y, z, kind, 0, 0, order]


def test_tile_rotation_matches_the_worker():
    assert rotate_tile(0, 0, 0, 0, 0) == (0, 0, 0)
    # Y-up, rotation 1: (x, z) = (z, -x - 1).
    assert rotate_tile(0, 0, 0, 1, 0) == (0, 0, -1)
    assert rotate_tile(-2, 0, -2, 1, 0) == (-2, 0, 1)
    # Y-down uses the other yaw table, then flips x and y.
    assert rotate_tile(0, 0, 0, 1, 1) == (0, -1, 0)


def test_a_straight_then_a_left_turn_is_two_shapes():
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, 30.0), (0.0, 0.0, 60.0), (30.0, 0.0, 60.0), (60.0, 0.0, 60.0)]
    course = RoadCourse(points)
    view = course.ahead(0.0, 0.0, 0.0, 4)
    assert view[0] == pytest.approx(60.0 / 50.0)
    assert view[1] == pytest.approx(0.0, abs=1e-6)
    assert view[2] == pytest.approx(0.0, abs=1e-6)
    # The bend itself is one metre. Left, toward +X, is positive.
    assert view[5] == pytest.approx(1.0 / 50.0)
    assert view[6] == pytest.approx(0.5, abs=0.05)
    assert view[7] == pytest.approx(0.0, abs=1e-6)
    assert view[10] == pytest.approx(59.0 / 50.0, abs=0.05)
    assert view[11] == pytest.approx(0.0, abs=1e-6)
    assert float(view[15:].sum()) == 0.0


def test_halfway_along_a_straight_reports_what_is_left():
    points = [(0.0, 0.0, float(z)) for z in range(0, 61, 10)]
    course = RoadCourse(points)
    view = course.ahead(0.0, 0.0, 25.0, 2)
    assert view[0] == pytest.approx(35.0 / 50.0, abs=0.02)
    assert view[1] == pytest.approx(0.0, abs=1e-5)
    assert float(view[3:].sum()) == 0.0


def test_a_drop_is_a_negative_slope():
    points = [(0.0, 10.0, 0.0), (0.0, 10.0, 20.0), (0.0, 0.0, 40.0)]
    view = RoadCourse(points).ahead(0.0, 10.0, 0.0, 2)
    assert view[0] == pytest.approx(20.0 / 50.0)
    assert view[2] == pytest.approx(0.0, abs=1e-6)
    assert view[5] == pytest.approx(20.0 / 50.0)
    assert view[7] == pytest.approx(-0.5)


def test_a_gentle_left_curve_stays_one_piece():
    radius = 40.0
    points = []
    for step in range(17):
        angle = step * (math.pi / 2.0) / 16.0
        points.append((radius - radius * math.cos(angle), 0.0, radius * math.sin(angle)))
    course = RoadCourse(points)
    assert len(course.segments) <= 3
    assert sum(segment.turn for segment in course.segments) == pytest.approx(math.pi / 2.0, abs=0.15)
    view = course.ahead(*points[0], 4)
    assert float(view[1::5].sum()) > 0.4


def test_the_same_shape_moved_and_turned_looks_the_same():
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, 40.0), (0.0, -8.0, 70.0), (25.0, -8.0, 70.0)]
    course = RoadCourse(points)

    def moved(point):
        x, y, z = point
        return (-z + 100.0, y - 20.0, x + 50.0)

    other = RoadCourse([moved(point) for point in points])
    here = course.ahead(*points[0], ROAD_AHEAD)
    there = other.ahead(*moved(points[0]), ROAD_AHEAD)
    assert here == pytest.approx(there, abs=1e-5)


def test_pieces_ahead_of_a_straight_track_do_not_turn():
    parts = [
        _part(0, 0, 0, _START),
        _part(0, 0, -4, _STRAIGHT),
        _part(0, 0, -8, _STRAIGHT),
        _part(0, 0, -12, _FINISH),
    ]
    course = course_from_parts(parts)
    assert course.length > 30.0
    assert course.points[0][2] > course.points[-1][2]
    view = course.ahead(0.0, 0.35, -1.35, 4)
    assert view[0] > 0.4
    assert abs(float(view[1])) < 0.05
    assert abs(float(view[2])) < 0.05


def test_a_bend_to_the_right_is_negative():
    # Identity start faces -Z. Leaving that line toward +X is a right turn.
    parts = [
        _part(0, 0, 0, _START),
        _part(0, 0, -4, _STRAIGHT),
        _part(0, 0, -8, _STRAIGHT),
        _part(4, 0, -10, _STRAIGHT),
        _part(8, 0, -10, _STRAIGHT),
        _part(12, 0, -10, _FINISH),
    ]
    course = course_from_parts(parts)
    assert sum(segment.turn for segment in course.segments) == pytest.approx(-math.pi / 2.0, abs=0.5)
    view = course.ahead(0.0, 0.35, -1.35, 8)
    assert float(view[1::5].min()) < -0.2


def test_a_lower_piece_is_a_downhill():
    parts = [
        _part(0, 0, 0, _START),
        _part(0, 0, -4, _STRAIGHT),
        _part(0, -2, -8, _STRAIGHT),
        _part(0, -2, -12, _FINISH),
    ]
    course = course_from_parts(parts)
    drop = sum(segment.slope * segment.length for segment in course.segments)
    assert drop == pytest.approx(-10.0, abs=3.0)
    view = course.ahead(0.0, 0.35, -1.35, 8)
    assert float(view[2::5].min()) < -0.2


def test_default_observation_width_is_unchanged():
    assert obs_dim(9) == FEATURES + 9
    assert obs_dim(9, 0) == FEATURES + 9
    assert obs_dim(9, 8) == FEATURES + len(EXTRA_NAMES) + 40 + 9


def test_road_numbers_sit_in_front_of_the_previous_action():
    points = [(0.0, 0.0, float(z)) for z in range(0, 61, 10)]
    numbers = LapNumbers(9, road=RoadCourse(points), road_window=4)
    numbers.advance(
        {
            "dt": 0.0,
            "state": {
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "wheels": [True, True, True, True],
                "nextCheckpointIndex": 0,
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "checkpointCount": 3,
            },
        },
        3,
    )
    vector = numbers.vector()
    road_at = FEATURES + len(EXTRA_NAMES)
    assert vector.shape == (obs_dim(9, 4),)
    assert vector[road_at] > 0.5
    assert vector[EXTRA_NAMES.index("wheel_0") + FEATURES] == 1.0
    assert vector[road_at + 20 + 3] == 1.0
    assert float(vector[road_at + 20 : road_at + 20 + 3].sum()) == 0.0
    assert float(vector[road_at + 20 + 4 :].sum()) == 0.0


def test_cloned_rows_can_carry_the_same_road_block():
    from polytrack_agent.rl.bc import pairs_from_rows

    course = RoadCourse([(0.0, 0.0, float(z)) for z in range(0, 81, 10)])
    rows = [
        {"dt": 0.0, "action": 3, "n": [0.0] * FEATURES},
        {"dt": 0.05, "action": 5, "n": [0.0] * FEATURES},
    ]
    observations, actions = pairs_from_rows(rows, n_actions=9, road_window=4, course=course)
    road_at = FEATURES + len(EXTRA_NAMES)
    assert observations.shape == (2, obs_dim(9, 4))
    assert observations[0, road_at] > 0.0
    assert observations[1, road_at + 20 + 3] == np.float32(1.0)
    assert actions.tolist() == [3, 5]


def test_left_is_the_drivers_left_when_the_road_faces_plus_z():
    # One row of cells. The sample sits near the +x edge, so most of the road
    # is toward -x. Facing +z, -x is the driver's right.
    surface = {}
    for iz in range(8):
        surface[(0, iz)] = (0, 2.5, 0.0, (iz + 0.5) * 5.0)
    left, right = side_clearance(surface, 4.0, 0.0, 10.0, 0.0, 1.0)
    assert left < 2.0
    assert right > 3.0


def test_a_straight_reports_both_sides():
    parts = [
        _part(0, 0, 0, _START),
        _part(0, 0, -4, _STRAIGHT),
        _part(0, 0, -8, _STRAIGHT),
        _part(0, 0, -12, _FINISH),
    ]
    course = course_from_parts(parts)
    view = course.ahead(0.0, 0.35, -1.35, 4)
    assert course.segments[0].left == pytest.approx(10.0, abs=3.0)
    assert course.segments[0].right == pytest.approx(10.0, abs=3.0)
    assert view[3] == pytest.approx(course.segments[0].left / 10.0)
    assert view[4] == pytest.approx(course.segments[0].right / 10.0)
    assert course.surface
    assert all(height == pytest.approx(0.0) for height in course.surface.values())


def test_height_above_the_road_and_which_side_of_the_line():
    # Point-only line along +z at y=0. The cell under the car wins when it exists.
    course = RoadCourse([(0.0, 0.0, 0.0), (0.0, 0.0, 40.0)], surface={(0, 2): 20.0})
    above, left = course.car_offsets(2.5, 25.0, 12.5)
    assert above == pytest.approx(5.0)
    assert left == pytest.approx(2.5)
    # No cell here, so the height is the line's. +x of a +z line is the driver's left.
    above, left = course.car_offsets(5.0, 10.0, 10.0)
    assert above == pytest.approx(10.0)
    assert left == pytest.approx(5.0)
    above, right = course.car_offsets(-4.0, 0.0, 10.0)
    assert right == pytest.approx(-4.0)
    # Facing +x, left is -z.
    along_x = RoadCourse([(0.0, 0.0, 0.0), (40.0, 0.0, 0.0)])
    _above, left = along_x.car_offsets(10.0, 0.0, -5.0)
    assert left == pytest.approx(5.0)


def test_the_extra_block_is_the_rest_of_the_car():
    # Identity faces +z, so +x is the driver's left. World spin about +y is yaw.
    # A gate out along +x is to the left, so gate_right is negative.
    course = RoadCourse([(0.0, 0.0, 0.0), (0.0, 0.0, 40.0)])
    numbers = LapNumbers(9, road=course, road_window=1)
    state = {
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "velocity": {"x": 40.0, "y": 0.0, "z": 0.0},
        "angularVelocity": {"x": 4.0, "y": 2.0, "z": 0.0},
        "wheels": [True, False, False, False],
        "nextCheckpointIndex": 0,
        "position": {"x": 5.0, "y": 10.0, "z": 10.0},
        "checkpointCount": 3,
        "target": {"x": 105.0, "y": 10.0, "z": 10.0},
        "speedKmh": 100.0,
        "wheelSkidInfo": [1.5, -0.2, 3.0, 0.0],
        "wheelSuspensionLength": [0.2, -2.0, 0.0, 0.0],
        "wheelSuspensionVelocity": [5.0, 0.0, 0.0, 0.0],
    }
    numbers.advance({"dt": 0.0, "state": state}, 0)
    vector = numbers.vector()

    def at(name):
        return float(vector[FEATURES + EXTRA_NAMES.index(name)])

    assert EXTRA_NAMES == (
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
    assert at("nose_up") == pytest.approx(0.0, abs=1e-6)
    assert at("roll") == pytest.approx(0.0, abs=1e-6)
    assert at("v_left") == pytest.approx(1.0)
    assert at("v_forward") == pytest.approx(0.0, abs=1e-6)
    assert at("w_left") == pytest.approx(1.0)
    assert at("w_up") == pytest.approx(0.5)
    assert at("w_forward") == pytest.approx(0.0, abs=1e-6)
    assert at("wheel_0") == 1.0
    assert at("wheel_1") == 0.0
    assert at("speed_kmh") == pytest.approx(0.5)
    assert at("skid_0") == pytest.approx(1.5)
    assert at("skid_2") == pytest.approx(2.0)
    assert at("suspension_0") == pytest.approx(0.2)
    assert at("suspension_1") == pytest.approx(-1.0)
    assert at("suspension_v0") == pytest.approx(1.0)
    assert at("above") == pytest.approx(1.0)
    assert at("line_left") == pytest.approx(0.5)
    assert vector[NAMES.index("gate_right")] == pytest.approx(-1.0)
    assert vector[NAMES.index("forward_z")] == pytest.approx(1.0)
    assert vector[NAMES.index("upright")] == pytest.approx(1.0)
    # The nineteen stay in front. Window 0 does not grow when the same car is seen.
    plain = LapNumbers(9)
    plain.advance({"dt": 0.0, "state": state}, 0)
    assert plain.vector().shape == (FEATURES + 9,)


def test_nose_up_and_roll_use_the_body_axes():
    # Right-hand about +X pitches the nose down, so nose-up is the opposite angle.
    half = math.pi / 4.0
    nose = {
        "quaternion": {"x": -math.sin(half), "y": 0.0, "z": 0.0, "w": math.cos(half)},
        "velocity": {"x": 0.0, "y": 40.0, "z": 0.0},
        "wheels": [False, False, False, False],
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    numbers = LapNumbers(9, road=RoadCourse([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]), road_window=1)
    numbers.advance({"dt": 0.0, "state": nose}, 0)
    vector = numbers.vector()
    assert vector[FEATURES + EXTRA_NAMES.index("nose_up")] == pytest.approx(1.0)
    assert vector[FEATURES + EXTRA_NAMES.index("v_forward")] == pytest.approx(1.0)
    assert vector[FEATURES + EXTRA_NAMES.index("v_up")] == pytest.approx(0.0, abs=1e-5)
    # Right-hand about +Z lifts the driver's left (+x) toward +y.
    rolled = {
        "quaternion": {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "wheels": [False, False, False, False],
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    numbers.advance({"dt": 0.0, "state": rolled}, 0)
    vector = numbers.vector()
    assert vector[FEATURES + EXTRA_NAMES.index("roll")] == pytest.approx(1.0)
    assert vector[FEATURES + EXTRA_NAMES.index("nose_up")] == pytest.approx(0.0, abs=1e-5)


def test_a_copied_row_rebuilds_height_side_and_flat_speed():
    from polytrack_agent.rl.bc import pairs_from_rows

    course = RoadCourse([(0.0, 0.0, 0.0), (0.0, 0.0, 40.0)])
    numbers = LapNumbers(9, road=course, road_window=1)
    numbers.advance(
        {
            "dt": 0.0,
            "state": {
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "velocity": {"x": 40.0, "y": 0.0, "z": 0.0},
                "angularVelocity": {"x": 0.0, "y": 2.0, "z": 0.0},
                "wheels": [True, True, True, True],
                "position": {"x": 5.0, "y": 10.0, "z": 10.0},
                "checkpointCount": 3,
                "nextCheckpointIndex": 0,
            },
        },
        3,
    )
    live = numbers.vector()
    observations, _actions = pairs_from_rows(
        [{"dt": 0.0, "action": 3, "n": live[:FEATURES].tolist()}],
        n_actions=9,
        road_window=1,
        course=course,
    )
    copied = observations[0]

    def at(vector, name):
        return float(vector[FEATURES + EXTRA_NAMES.index(name)])

    assert at(copied, "above") == pytest.approx(at(live, "above"))
    assert at(copied, "line_left") == pytest.approx(at(live, "line_left"))
    assert at(copied, "v_left") == pytest.approx(at(live, "v_left"))
    # A row has no gyro, so the copied spin stays 0 while the live car has yaw.
    assert at(live, "w_up") == pytest.approx(0.5)
    assert at(copied, "w_up") == pytest.approx(0.0)
    # Nose straight up: the live car calls +y speed "forward". The row cannot.
    half = math.pi / 4.0
    numbers.advance(
        {
            "dt": 0.0,
            "state": {
                "quaternion": {"x": -math.sin(half), "y": 0.0, "z": 0.0, "w": math.cos(half)},
                "velocity": {"x": 0.0, "y": 40.0, "z": 0.0},
                "wheels": [False, False, False, False],
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        },
        0,
    )
    pitched = numbers.vector()
    copied_pitch, _actions = pairs_from_rows(
        [{"dt": 0.0, "action": 0, "n": pitched[:FEATURES].tolist()}],
        n_actions=9,
        road_window=1,
        course=course,
    )
    assert at(pitched, "v_forward") == pytest.approx(1.0)
    assert at(copied_pitch[0], "v_forward") == pytest.approx(0.0, abs=1e-5)
    assert NAMES.index("y") == 1


def test_ground_tilt_is_left_side_up_and_uphill():
    # Two tyres down. One normal points back (uphill). One points to the
    # driver's right, which is the ground with the left side raised.
    course = RoadCourse([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])
    numbers = LapNumbers(9, road=course, road_window=1)
    numbers.advance(
        {
            "dt": 0.0,
            "state": {
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "wheels": [True, True, False, False],
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "wheelNormal": [
                    {"x": 0.0, "y": 0.0, "z": -1.0},
                    {"x": -1.0, "y": 0.0, "z": 0.0},
                    None,
                    None,
                ],
            },
        },
        0,
    )
    vector = numbers.vector()
    assert vector[FEATURES + EXTRA_NAMES.index("ground_grade")] == pytest.approx(0.5)
    assert vector[FEATURES + EXTRA_NAMES.index("ground_bank")] == pytest.approx(0.5)


def test_summer1_road_drops_after_the_opening_straight():
    import os
    import shutil
    from pathlib import Path

    from polytrack_agent.config import load_config
    from polytrack_agent.rl.track import make_track

    root = os.environ.get("POLYTRACK_ROOT")
    parts = Path("polytrack_agent/sim/cache/track_parts.json")
    if not root or shutil.which("node") is None or not parts.is_file():
        pytest.skip("polytrack fork simulation is not available")
    if not (Path(root) / "simulation_worker.bundle.js").is_file():
        pytest.skip("polytrack fork simulation is not available")
    cfg = load_config(None)
    env = make_track(cfg)
    try:
        course = env.road_course()
        marks = list(env.gates)
        if env.finish is not None:
            marks.append(env.finish)
    finally:
        env.close()
    assert course.length > 200.0
    assert course.gate_count >= 1
    assert course.finish_s is not None
    assert course.finish_s > course.gate_s[-1]
    assert course.segments[0].length > 15.0
    assert abs(course.segments[0].turn) < 0.25
    assert course.segments[0].left > 4.0
    assert course.segments[0].right > 4.0
    drop = min(segment.slope for segment in course.segments)
    assert drop < -0.15
    # A branch that has to be driven out and back still belongs on the line.
    for mark in marks:
        assert _nearest_line(course, mark) < 15.0


def _nearest_line(course: RoadCourse, point: tuple[float, float, float]) -> float:
    best = 1e18
    points = course.points
    for index in range(len(points) - 1):
        _t, distance = _project(point[0], point[1], point[2], points[index], points[index + 1])
        if distance < best:
            best = distance
    return best
