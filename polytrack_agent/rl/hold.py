"""Pick a button hold by the whole race, then try one short steer.

Gas-plus-turn looks better for the first moment and then spins off. A whole
race of straight throttle goes farther. This search drives each hold to the
end of the episode and keeps the one that gets farther. A steer is kept only
when a full race with that tap beats straight throttle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from polytrack_agent.rl.actions import action_masks, chord_name

log = logging.getLogger(__name__)


def id_named(name: str, keys: list[str], reset_action: bool = False) -> int:
    """Action id whose chord name is `name`."""
    for index in range(len(action_masks(keys, reset_action=reset_action))):
        if chord_name(index, keys, reset_action=reset_action) == name:
            return index
    raise ValueError(f"no chord named {name!r} for keys {keys}")


def id_holding(keys: list[str], held: frozenset[str], reset_action: bool = False) -> int:
    """Action id that holds exactly these keys."""
    for index, mask in enumerate(action_masks(keys, reset_action=reset_action)):
        on = frozenset(key for key in keys if mask.get(key))
        if on == held:
            return index
    raise ValueError(f"no chord holds {sorted(held)}")


def _throttle_key(keys: list[str]) -> str | None:
    for key in keys:
        if key in ("w", "up"):
            return key
    return None


def _steer_keys(keys: list[str]) -> list[str]:
    return [key for key in keys if key in ("a", "left", "d", "right")]


def steer_windows() -> list[tuple[float, float]]:
    """Short taps tiled across the opening, plus a few longer corrections.

    Straight throttle clips the wall around progress 0.54. The tiles cover
    every moment up to 0.60 so a tap cannot fall in a gap. The longer windows
    are there in case the car needs a steer that lasts more than one tap.
    """
    short = [(index / 20.0, index / 20.0 + 0.05) for index in range(12)]
    longer = [(start, start + 0.15) for start in (0.0, 0.15, 0.30, 0.45)]
    return short + longer


def _check_window(steer: str | None, start: float, stop: float) -> None:
    if steer is not None and not (float(start) < float(stop)):
        raise ValueError(f"steer window must have start < stop, got {start} .. {stop}")


@dataclass(frozen=True)
class HoldPlan:
    """Hold `chord`, plus up to three steer taps while progress is in a window.

    A later tap wins if the windows overlap. The searches place each tap after
    the one before it, so that does not come up in a saved plan.
    """

    chord: str
    steer: str | None = None
    start: float = 0.0
    stop: float = 0.0
    steer2: str | None = None
    start2: float = 0.0
    stop2: float = 0.0
    steer3: str | None = None
    start3: float = 0.0
    stop3: float = 0.0

    def __post_init__(self) -> None:
        _check_window(self.steer, self.start, self.stop)
        _check_window(self.steer2, self.start2, self.stop2)
        _check_window(self.steer3, self.start3, self.stop3)

    def label(self) -> str:
        bits = []
        if self.steer is not None:
            bits.append(f"{self.steer} while {self.start:.2f}-{self.stop:.2f}")
        if self.steer2 is not None:
            bits.append(f"{self.steer2} while {self.start2:.2f}-{self.stop2:.2f}")
        if self.steer3 is not None:
            bits.append(f"{self.steer3} while {self.start3:.2f}-{self.stop3:.2f}")
        if not bits:
            return self.chord
        return ", ".join(bits) + f", else {self.chord}"

    def to_dict(self) -> dict:
        return {
            "chord": self.chord,
            "steer": self.steer,
            "start": float(self.start),
            "stop": float(self.stop),
            "steer2": self.steer2,
            "start2": float(self.start2),
            "stop2": float(self.stop2),
            "steer3": self.steer3,
            "start3": float(self.start3),
            "stop3": float(self.stop3),
        }


def load_plan(path: Path) -> HoldPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    return HoldPlan(
        chord=str(plan["chord"]),
        steer=None if plan.get("steer") is None else str(plan["steer"]),
        start=float(plan.get("start") or 0.0),
        stop=float(plan.get("stop") or 0.0),
        steer2=None if plan.get("steer2") is None else str(plan["steer2"]),
        start2=float(plan.get("start2") or 0.0),
        stop2=float(plan.get("stop2") or 0.0),
        steer3=None if plan.get("steer3") is None else str(plan["steer3"]),
        start3=float(plan.get("start3") or 0.0),
        stop3=float(plan.get("stop3") or 0.0),
    )


def better(row: dict, other: dict) -> bool:
    """True when `row` is the better whole race. Farther progress wins."""
    if bool(row["finished"]) != bool(other["finished"]):
        return bool(row["finished"])
    if float(row["progress"]) != float(other["progress"]):
        return float(row["progress"]) > float(other["progress"])
    if bool(row["finished"]) and float(row["seconds"]) != float(other["seconds"]):
        return float(row["seconds"]) < float(other["seconds"])
    return float(row["return"]) > float(other["return"])


def improves(row: dict, baseline: dict) -> bool:
    """True when a steer actually gets the car farther than holding the champion."""
    if bool(row["finished"]) and not bool(baseline["finished"]):
        return True
    if bool(row["finished"]) and bool(baseline["finished"]):
        return float(row["seconds"]) + 0.05 < float(baseline["seconds"])
    return float(row["progress"]) > float(baseline["progress"]) + 1e-3


def improves_correction(row: dict, baseline: dict) -> bool:
    """A later tap has to stop the fall, or clearly get farther while still falling.

    A car that sticks at the start did not fall, but it also did not fix the race.
    It only counts as staying up if it still reaches the baseline.
    """
    if bool(row["finished"]) and not bool(baseline["finished"]):
        return True
    if bool(row["finished"]) and bool(baseline["finished"]):
        return float(row["seconds"]) + 0.05 < float(baseline["seconds"])
    if bool(row.get("fell")) == bool(baseline.get("fell")):
        return float(row["progress"]) > float(baseline["progress"]) + 0.02
    if baseline.get("fell") and not row.get("fell"):
        return float(row["progress"]) + 0.02 >= float(baseline["progress"])
    return float(row["progress"]) > float(baseline["progress"]) + 0.05


def saves_the_car(row: dict, baseline: dict) -> bool:
    """A tap at the crash counts only when the car stays up and still reaches it.

    Falling farther down the same drop is the same crash. A car that sticks at
    the start stayed up, but it did not get through the place where the baseline
    fell.
    """
    if bool(row["finished"]) and not bool(baseline["finished"]):
        return True
    if bool(row["finished"]) and bool(baseline["finished"]):
        return float(row["seconds"]) + 0.05 < float(baseline["seconds"])
    if row.get("fell"):
        return False
    if baseline.get("fell"):
        return float(row["progress"]) + 0.02 >= float(baseline["progress"])
    return float(row["progress"]) > float(baseline["progress"]) + 0.02


def better_correction(row: dict, other: dict) -> bool:
    """Prefer a finish, then staying on the road, then distance."""
    if bool(row["finished"]) != bool(other["finished"]):
        return bool(row["finished"])
    if bool(row.get("fell")) != bool(other.get("fell")):
        return not bool(row.get("fell"))
    if float(row["progress"]) != float(other["progress"]):
        return float(row["progress"]) > float(other["progress"])
    if bool(row["finished"]) and float(row["seconds"]) != float(other["seconds"]):
        return float(row["seconds"]) < float(other["seconds"])
    return float(row["return"]) > float(other["return"])


def correction_windows(after: float, until: float) -> list[tuple[float, float]]:
    """Short taps tiled from the end of the first steer to just past the fall."""
    start_c = int(round(float(after) * 100))
    until_c = int(round(float(until) * 100))
    if until_c <= start_c + 5:
        until_c = start_c + 40
    short = [(c / 100.0, (c + 5) / 100.0) for c in range(start_c, until_c, 5)]
    longer = [(c / 100.0, (c + 15) / 100.0) for c in range(start_c, max(start_c, until_c - 10), 15)]
    return short + longer


def _end(row: dict) -> str:
    if row["finished"]:
        return "finish"
    for name in ("fell", "stuck", "beached", "timeout", "capped"):
        if row.get(name):
            return name
    return "stop"


def run_plan(env, plan: HoldPlan, max_steps: int) -> dict:
    """Drive one episode. The button can change only when the plan says to steer."""
    keys = list(env.keys)
    reset_action = bool(getattr(env, "reset_action", False))
    base = id_named(plan.chord, keys, reset_action)
    steer = None if plan.steer is None else id_named(plan.steer, keys, reset_action)
    steer2 = None if plan.steer2 is None else id_named(plan.steer2, keys, reset_action)
    steer3 = None if plan.steer3 is None else id_named(plan.steer3, keys, reset_action)
    env.reset(restart=True)
    total = 0.0
    progress = 0.0
    steps = 0
    info: dict = {"progress": 0.0, "finished": False, "state": {"frames": 0}}
    done = False
    while not done and steps < int(max_steps):
        action = base
        if steer is not None and plan.start <= progress < plan.stop:
            action = steer
        if steer2 is not None and plan.start2 <= progress < plan.stop2:
            action = steer2
        if steer3 is not None and plan.start3 <= progress < plan.stop3:
            action = steer3
        _frame, reward, done, info = env.step(action)
        if not info.get("counted", True):
            continue
        total += float(reward)
        progress = float(info.get("progress") or 0.0)
        steps += 1
    state = info.get("state") or {}
    frames = int(state.get("frames") or 0)
    finished = bool(info.get("finished"))
    position = state.get("position") or {}
    row = {
        "plan": plan.to_dict(),
        "label": plan.label(),
        "progress": progress,
        "return": total,
        "steps": steps,
        "seconds": (int(state["finishFrames"]) if finished else frames) * 0.001,
        "finished": finished,
        "stuck": bool(info.get("stuck")),
        "beached": bool(info.get("beached")),
        "fell": bool(info.get("fell")),
        "timeout": bool(info.get("timeout")),
        "capped": steps >= int(max_steps) and not done,
        "checkpoint": int(state.get("nextCheckpointIndex") or 0),
        "position": {axis: float(position.get(axis) or 0.0) for axis in "xyz"},
    }
    log.info(
        "%s  progress %.3f  return %.2f  %.2fs  %s steps  %s",
        row["label"],
        row["progress"],
        row["return"],
        row["seconds"],
        row["steps"],
        _end(row),
    )
    return row


def _driving_plans(keys: list[str], reset_action: bool) -> list[HoldPlan]:
    plans = []
    for index, mask in enumerate(action_masks(keys, reset_action=reset_action)):
        if mask.get("reset"):
            continue
        plans.append(HoldPlan(chord_name(index, keys, reset_action=reset_action)))
    return plans


def _steer_plans(keys: list[str], chord: str, windows: list[tuple[float, float]]) -> list[HoldPlan]:
    throttle = _throttle_key(keys)
    if throttle is None:
        return []
    plans = []
    for steer_key in _steer_keys(keys):
        name = chord_name(id_holding(keys, frozenset({throttle, steer_key})), keys)
        for start, stop in windows:
            plans.append(HoldPlan(chord, steer=name, start=start, stop=stop))
    return plans


def search_hold(
    env,
    max_steps: int,
    windows: list[tuple[float, float]] | None = None,
) -> dict:
    """Race every hold, keep the farthest, then keep one steer only if it goes farther."""
    keys = list(env.keys)
    reset_action = bool(getattr(env, "reset_action", False))
    chords = [run_plan(env, plan, max_steps) for plan in _driving_plans(keys, reset_action)]
    if not chords:
        raise ValueError("no driving holds to race")
    champion = chords[0]
    for row in chords[1:]:
        if better(row, champion):
            champion = row
    plan = HoldPlan(champion["plan"]["chord"])
    throttle = _throttle_key(keys)
    straight = None if throttle is None else chord_name(id_holding(keys, frozenset({throttle})), keys)
    steers: list[dict] = []
    chosen = champion
    kept = False
    if straight is not None and plan.chord == straight:
        candidates = _steer_plans(keys, plan.chord, steer_windows() if windows is None else windows)
        for candidate in candidates:
            row = run_plan(env, candidate, max_steps)
            steers.append(row)
            if improves(row, champion) and (not kept or better(row, chosen)):
                chosen = row
                plan = candidate
                kept = True
    else:
        log.info("Champion is %s, so no steer is tried on top of gas.", plan.chord)
    return {
        "plan": plan.to_dict(),
        "kept_steer": kept,
        "champion": champion,
        "chosen": chosen,
        "chords": sorted(chords, key=lambda row: (row["progress"], row["return"]), reverse=True),
        "steers": steers,
    }


def _tap_names(keys: list[str], release: bool) -> list[str]:
    """Chords to try on top of a plan.

    A later tap on a jump has to be allowed to lift off the gas. Steering with
    the throttle held open cannot drop the car onto a landing below.
    """
    throttle = _throttle_key(keys)
    if throttle is None:
        return []
    if not release:
        return [
            chord_name(id_holding(keys, frozenset({throttle, steer_key})), keys)
            for steer_key in _steer_keys(keys)
        ]
    straight = chord_name(id_holding(keys, frozenset({throttle})), keys)
    names = []
    for index, mask in enumerate(action_masks(keys)):
        if mask.get("reset"):
            continue
        name = chord_name(index, keys)
        if name != straight:
            names.append(name)
    return names


def _with_tap(plan: HoldPlan, which: int, steer: str, start: float, stop: float) -> HoldPlan:
    fields = dict(
        chord=plan.chord,
        steer=plan.steer,
        start=plan.start,
        stop=plan.stop,
        steer2=plan.steer2,
        start2=plan.start2,
        stop2=plan.stop2,
        steer3=plan.steer3,
        start3=plan.start3,
        stop3=plan.stop3,
    )
    if which == 3:
        fields.update(steer3=steer, start3=start, stop3=stop)
    else:
        fields.update(steer2=steer, start2=start, stop2=stop)
    return HoldPlan(**fields)


def search_correction(
    env,
    plan: HoldPlan,
    max_steps: int,
    windows: list[tuple[float, float]] | None = None,
) -> dict:
    """Try one more tap after `plan`.

    The second tap may fall farther and still count. A third tap is at the
    crash. It may coast or brake, and it counts only when the car stays up and
    still reaches the line where the baseline fell.
    """
    keys = list(env.keys)
    baseline = run_plan(env, plan, max_steps)
    third = plan.steer2 is not None
    if third:
        after = float(plan.stop2)
        accept = saves_the_car
    else:
        after = float(plan.stop) if plan.steer is not None else 0.0
        accept = improves_correction
    until = max(after + 0.4, float(baseline["progress"]) + 0.25)
    chosen_windows = correction_windows(after, until) if windows is None else windows
    throttle = _throttle_key(keys)
    steers: list[dict] = []
    chosen = baseline
    kept = False
    best = plan
    if throttle is None:
        log.info("No throttle key, so no further tap is tried.")
    else:
        covered = float(plan.stop2) if third else float(plan.stop)
        for name in _tap_names(keys, release=third):
            for start, stop in chosen_windows:
                if plan.steer is not None and start < covered - 1e-9:
                    continue
                candidate = _with_tap(plan, 3 if third else 2, name, start, stop)
                row = run_plan(env, candidate, max_steps)
                steers.append(row)
                if accept(row, baseline) and (not kept or better_correction(row, chosen)):
                    chosen = row
                    best = candidate
                    kept = True
    if not kept:
        log.info("No further tap beat %s.", plan.label())
    return {
        "plan": best.to_dict(),
        "kept_steer": kept,
        "champion": baseline,
        "chosen": chosen,
        "chords": [],
        "steers": steers,
    }
