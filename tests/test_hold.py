"""Whole-race hold search, without the simulator."""

from polytrack_agent.rl.actions import chord_name
from polytrack_agent.rl.hold import (
    HoldPlan,
    better,
    better_correction,
    correction_windows,
    id_named,
    improves,
    improves_correction,
    saves_the_car,
    search_correction,
    search_hold,
    steer_windows,
)


KEYS = ["w", "a", "s", "d"]


def test_steer_windows_cover_the_opening_without_gaps():
    tiles = [(start, stop) for start, stop in steer_windows() if abs((stop - start) - 0.05) < 1e-9]
    assert tiles[0][0] == 0.0
    for (_start, stop), (next_start, _next_stop) in zip(tiles, tiles[1:]):
        assert abs(stop - next_start) < 1e-9
    assert abs(tiles[-1][1] - 0.6) < 1e-9


def test_plan_uses_the_steer_only_inside_the_window():
    assert id_named("w", KEYS) == id_named(chord_name(id_named("w", KEYS), KEYS), KEYS)
    plan = HoldPlan("w", steer="wa", start=0.2, stop=0.25)
    assert plan.label().startswith("wa while 0.20-0.25")
    try:
        HoldPlan("w", steer="wa", start=0.4, stop=0.4)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero-width steer window should be refused")


def test_better_prefers_farther_progress_and_a_finish():
    short = {"finished": False, "progress": 0.2, "return": 5.0, "seconds": 10.0}
    far = {"finished": False, "progress": 0.5, "return": -1.0, "seconds": 30.0}
    done = {"finished": True, "progress": 4.0, "return": 1.0, "seconds": 40.0}
    faster = {"finished": True, "progress": 4.0, "return": 0.0, "seconds": 25.0}
    assert better(far, short)
    assert not better(short, far)
    assert better(done, far)
    assert better(faster, done)
    assert not improves(short, far)
    assert improves(far, short)
    assert improves(done, far)


class _Table:
    """One step, then the episode ends at a progress that depends on the chord."""

    def __init__(self, scores: dict[str, float]):
        self.keys = list(KEYS)
        self.reset_action = False
        self.scores = scores
        self.frames = 0

    def reset(self, restart: bool = True):
        del restart
        self.frames = 0
        return {"progress": 0.0, "finished": False, "counted": True, "state": {"frames": 0}}

    def step(self, action_id: int):
        name = chord_name(action_id, self.keys)
        self.frames += 50
        progress = float(self.scores.get(name, 0.0))
        info = {
            "progress": progress,
            "finished": False,
            "counted": True,
            "stuck": True,
            "state": {"frames": self.frames},
        }
        return None, progress, True, info


def test_whole_race_keeps_straight_throttle():
    scores = {name: 0.0 for name in ("none", "a", "d", "w", "wa", "wd", "s", "as", "sd")}
    scores["w"] = 0.54
    scores["wa"] = 0.04
    scores["wd"] = 0.05
    report = search_hold(_Table(scores), max_steps=5, windows=[])
    assert report["plan"]["chord"] == "w"
    assert report["plan"]["steer"] is None
    assert report["kept_steer"] is False
    assert report["champion"]["progress"] == 0.54


class _Road:
    """Straight gas reaches 0.5. One tap of wa at progress 0.2 unlocks 0.8."""

    def __init__(self, unlock: bool):
        self.keys = list(KEYS)
        self.reset_action = False
        self.unlock = unlock
        self.progress = 0.0
        self.frames = 0
        self.good_tap = False

    def reset(self, restart: bool = True):
        del restart
        self.progress = 0.0
        self.frames = 0
        self.good_tap = False
        return {"progress": 0.0, "finished": False, "counted": True, "state": {"frames": 0}}

    def step(self, action_id: int):
        name = chord_name(action_id, self.keys)
        self.frames += 50
        done = False
        reward = 0.0
        if name == "w":
            self.progress = round(self.progress + 0.1, 10)
            reward = 0.1
            if self.progress >= 0.5 - 1e-9:
                if self.unlock and self.good_tap:
                    self.progress = 0.8
                    reward += 0.3
                done = True
        elif name == "wa" and self.unlock and 0.2 - 1e-9 <= self.progress < 0.25:
            self.good_tap = True
            self.progress = round(self.progress + 0.02, 10)
            reward = 0.02
        else:
            reward = -0.5
            done = True
        info = {
            "progress": self.progress,
            "finished": False,
            "counted": True,
            "stuck": done,
            "state": {"frames": self.frames},
        }
        return None, reward, done, info


def test_steer_is_kept_only_when_the_race_goes_farther():
    found = search_hold(_Road(unlock=True), max_steps=20, windows=[(0.0, 0.05), (0.2, 0.25)])
    assert found["kept_steer"] is True
    assert found["plan"]["steer"] == "wa"
    assert found["plan"]["start"] == 0.2
    assert found["chosen"]["progress"] == 0.8
    assert found["champion"]["progress"] == 0.5

    plain = search_hold(_Road(unlock=False), max_steps=20, windows=[(0.2, 0.25)])
    assert plain["kept_steer"] is False
    assert plain["plan"]["steer"] is None
    assert plain["plan"]["chord"] == "w"


def test_correction_windows_tile_after_the_first_tap():
    tiles = [(start, stop) for start, stop in correction_windows(0.10, 0.30) if abs((stop - start) - 0.05) < 1e-9]
    assert abs(tiles[0][0] - 0.10) < 1e-9
    for (_start, stop), (next_start, _next_stop) in zip(tiles, tiles[1:]):
        assert abs(stop - next_start) < 1e-9
    assert tiles[-1][1] > 0.25


def test_a_second_tap_must_stop_the_fall_or_go_clearly_farther():
    base = {"finished": False, "fell": True, "progress": 1.15, "return": 1.0, "seconds": 7.0}
    stuck = {"finished": False, "fell": False, "progress": 0.04, "return": -1.0, "seconds": 10.0}
    tiny = {"finished": False, "fell": True, "progress": 1.16, "return": 1.0, "seconds": 7.2}
    farther_fall = {"finished": False, "fell": True, "progress": 1.40, "return": 1.2, "seconds": 8.0}
    saved = {"finished": False, "fell": False, "progress": 1.20, "return": 1.1, "seconds": 9.0}
    assert not improves_correction(stuck, base)
    assert not improves_correction(tiny, base)
    assert improves_correction(farther_fall, base)
    assert improves_correction(saved, base)
    assert better_correction(saved, farther_fall)


class _AfterFall:
    """The first plan falls at 1.15. A right tap at progress 0.4 stays up to 1.4."""

    def __init__(self):
        self.keys = list(KEYS)
        self.reset_action = False
        self.progress = 0.0
        self.frames = 0
        self.saved = False

    def reset(self, restart: bool = True):
        del restart
        self.progress = 0.0
        self.frames = 0
        self.saved = False
        return {"progress": 0.0, "finished": False, "counted": True, "state": {"frames": 0}}

    def step(self, action_id: int):
        name = chord_name(action_id, self.keys)
        at = self.progress
        self.frames += 50
        fell = False
        beached = False
        done = False
        reward = 0.1
        if name == "wd" and 0.4 - 1e-9 <= at < 0.5:
            self.saved = True
        if name == "wa" and at >= 0.2 - 1e-9:
            fell = True
            done = True
            reward = -0.5
        else:
            self.progress = round(at + 0.1, 10)
            if self.saved and self.progress >= 1.2 - 1e-9:
                self.progress = 1.4
                beached = True
                done = True
            elif not self.saved and self.progress >= 1.1 - 1e-9:
                self.progress = 1.15
                fell = True
                done = True
        info = {
            "progress": self.progress,
            "finished": False,
            "counted": True,
            "stuck": False,
            "beached": beached,
            "fell": fell,
            "state": {"frames": self.frames, "position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        }
        return None, reward, done, info


def test_a_crash_tap_has_to_stay_up_through_the_fall():
    base = {"finished": False, "fell": True, "progress": 1.24, "return": 1.0, "seconds": 7.0}
    farther_fall = {"finished": False, "fell": True, "progress": 1.50, "return": 1.2, "seconds": 8.0}
    short = {"finished": False, "fell": False, "progress": 1.04, "return": 0.5, "seconds": 12.0}
    saved = {"finished": False, "fell": False, "progress": 1.30, "return": 1.1, "seconds": 9.0}
    assert not saves_the_car(farther_fall, base)
    assert not saves_the_car(short, base)
    assert saves_the_car(saved, base)


class _Launch:
    """Two taps still fall at 1.24. A left tap at progress 0.9 stays up to 1.3.

    A later left tap falls farther, at 1.5. That one must not be kept.
    """

    def __init__(self):
        self.keys = list(KEYS)
        self.reset_action = False
        self.progress = 0.0
        self.frames = 0
        self.saved = False
        self.overshoot = False

    def reset(self, restart: bool = True):
        del restart
        self.progress = 0.0
        self.frames = 0
        self.saved = False
        self.overshoot = False
        return {"progress": 0.0, "finished": False, "counted": True, "state": {"frames": 0}}

    def step(self, action_id: int):
        name = chord_name(action_id, self.keys)
        at = self.progress
        self.frames += 50
        fell = False
        beached = False
        done = False
        reward = 0.1
        if name == "wa" and 0.9 - 1e-9 <= at < 1.0:
            self.saved = True
        if name == "wa" and 1.05 - 1e-9 <= at < 1.15:
            self.overshoot = True
        self.progress = round(at + 0.05, 10)
        if self.overshoot and self.progress >= 1.2 - 1e-9:
            self.progress = 1.5
            fell = True
            done = True
            reward = -0.5
        elif self.saved and self.progress >= 1.25 - 1e-9:
            self.progress = 1.3
            beached = True
            done = True
        elif not self.saved and self.progress >= 1.2 - 1e-9:
            self.progress = 1.24
            fell = True
            done = True
            reward = -0.5
        info = {
            "progress": self.progress,
            "finished": False,
            "counted": True,
            "stuck": False,
            "beached": beached,
            "fell": fell,
            "state": {"frames": self.frames, "position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        }
        return None, reward, done, info


def test_third_tap_is_kept_only_when_it_stays_up():
    base = HoldPlan("w", steer="wa", start=0.35, stop=0.40, steer2="wa", start2=0.55, stop2=0.70)
    report = search_correction(_Launch(), base, max_steps=80, windows=[(0.90, 1.00), (1.05, 1.15)])
    assert report["kept_steer"] is True
    assert report["plan"]["steer2"] == "wa"
    assert report["plan"]["steer3"] == "wa"
    assert report["plan"]["start3"] == 0.9
    assert report["chosen"]["fell"] is False
    assert report["chosen"]["progress"] == 1.3
    assert report["champion"]["fell"] is True
    assert any(row["plan"].get("steer3") == "s" for row in report["steers"])


def test_second_tap_is_kept_when_it_stops_the_fall():
    base = HoldPlan("w", steer="wa", start=0.05, stop=0.10)
    report = search_correction(_AfterFall(), base, max_steps=40, windows=[(0.0, 0.05), (0.40, 0.50)])
    assert report["kept_steer"] is True
    assert report["plan"]["steer"] == "wa"
    assert report["plan"]["steer2"] == "wd"
    assert report["plan"]["start2"] == 0.4
    assert report["chosen"]["progress"] == 1.4
    assert report["chosen"]["fell"] is False
    assert report["champion"]["fell"] is True
