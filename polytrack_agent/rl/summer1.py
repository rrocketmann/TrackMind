"""Summer 1 centerline from the official PolyTrack 0.6.3 track file.

The 48 points are road samples, not the three yellow checkpoint gates.
The car is a point on this grid. It is not the game's physics, and a
simulated second is not a lap time. The next trainer should drive the
official window, the way Trackmania agents use the real game.
"""

from __future__ import annotations

from polytrack_agent.rl.headless import HeadlessTrack

TRACK = "summer1"
START = (64.0, 4.0)
HEADING = -2.819842
GATES = (
    (52.0, 0.0),
    (48.0, 0.0),
    (44.0, 0.0),
    (40.0, 0.0),
    (36.0, 0.0),
    (24.0, 0.0),
    (20.0, 0.0),
    (16.0, 0.0),
    (12.0, 0.0),
    (8.0, 0.0),
    (4.0, 0.0),
    (0.0, 0.0),
    (-4.0, 0.0),
    (-8.0, 0.0),
    (-12.0, 0.0),
    (-16.0, 0.0),
    (-20.0, 0.0),
    (-24.0, 0.0),
    (-28.0, 0.0),
    (-32.0, 0.0),
    (-36.0, 0.0),
    (-32.0, 0.0),
    (-28.0, 0.0),
    (-24.0, -4.0),
    (-20.0, -8.0),
    (-20.0, -12.0),
    (-20.0, -16.0),
    (-20.0, -12.0),
    (-20.0, -8.0),
    (-20.0, -4.0),
    (-20.0, 0.0),
    (-24.0, 12.0),
    (-24.0, 16.0),
    (-24.0, 20.0),
    (-24.0, 24.0),
    (-24.0, 28.0),
    (-24.0, 32.0),
    (-24.0, 36.0),
    (-24.0, 40.0),
    (-28.0, 44.0),
    (-32.0, 48.0),
    (-36.0, 48.0),
    (-40.0, 48.0),
    (-44.0, 48.0),
    (-48.0, 52.0),
    (-52.0, 52.0),
    (-56.0, 56.0),
    (-60.0, 56.0),
)


def make_track(cfg: dict) -> HeadlessTrack:
    return HeadlessTrack(
        cfg,
        gates=GATES,
        start=START,
        heading=HEADING,
        gate_radius=4.0,
        reach=24.0,
    )
