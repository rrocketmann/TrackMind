"""Online PPO on the PolyTrack 0.6.3 fork simulation."""

from __future__ import annotations

import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from polytrack_agent.rl.bc import behavior_clone, load_best_line, load_demos
from polytrack_agent.rl.line import ready_for_time
from polytrack_agent.rl.device import resolve_device
from polytrack_agent.rl.track import SchoolagyTrack, make_track
from polytrack_agent.rl.actions import chord_name, n_actions
from polytrack_agent.rl.checkpoint import ARCH, read_checkpoint, save_checkpoint
from polytrack_agent.rl.device import configure_torch
from polytrack_agent.rl.gae import compute_gae
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.ppo import clear_policy_moments, hold_entropy, ppo_update, set_entropy
from polytrack_agent.rl.seed import set_seed
from polytrack_agent.rl.state import LapNumbers, obs_dim

log = logging.getLogger(__name__)


def _meta(model: StateActorCritic, keys: list[str], track: str) -> dict:
    return {
        "obs_dim": int(model.obs_dim),
        "hidden": int(model.hidden),
        "n_actions": int(model.n_actions),
        "keys": list(keys),
        "track": track,
    }


def _episode_gae(
    rewards: list[float],
    dones: list[float],
    values: list[float],
    lengths: list[int],
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advantage for each decision. A lap boundary does not bootstrap into the next lap."""
    reward_arr = np.asarray(rewards, dtype=np.float32)
    done_arr = np.asarray(dones, dtype=np.float32)
    value_arr = np.asarray(values, dtype=np.float32)
    advantages: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        count = int(length)
        window = slice(offset, offset + count)
        advantage, target = compute_gae(
            reward_arr[window],
            done_arr[window],
            value_arr[window],
            0.0,
            gamma,
            lam,
        )
        advantages.append(advantage)
        returns.append(target)
        offset += count
    if offset != len(rewards):
        raise RuntimeError(f"GAE saw {offset} steps and the rollout has {len(rewards)}")
    return np.concatenate(advantages), np.concatenate(returns)


def _checkpoint_matches(path: Path, track: str, chords: int, width: int, hidden: int) -> bool:
    """True when last.pt is this actor, so a warm start must not replace it."""
    if not path.is_file():
        return False
    try:
        payload = read_checkpoint(path, torch.device("cpu"))
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        payload.get("track") == track
        and payload.get("arch") == ARCH
        and int(payload.get("n_actions") or -1) == chords
        and int(payload.get("obs_dim") or -1) == width
        and int(payload.get("hidden") or -1) == hidden
    )


def _resume(
    path: Path, model: StateActorCritic, device: torch.device, track: str
) -> tuple[int, float | None, str | None] | None:
    if not path.is_file():
        return None
    payload = read_checkpoint(path, device)
    if payload.get("track") != track:
        log.warning("Ignoring %s because it was trained on %s.", path, payload.get("track"))
        return None
    if payload.get("arch") != ARCH or int(payload.get("n_actions") or -1) != model.n_actions:
        log.warning("Ignoring %s because it is a different actor.", path)
        return None
    if int(payload.get("obs_dim") or -1) != model.obs_dim or int(payload.get("hidden") or -1) != model.hidden:
        log.warning("Ignoring %s because the observation layout changed.", path)
        return None
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        log.warning("Ignoring %s (%s).", path, exc)
        return None
    model.eval()
    update = int(payload.get("update") or 0)
    raw_target = payload.get("entropy_target")
    target = None if raw_target is None else float(raw_target)
    raw_stage = payload.get("reward_stage")
    stage = None if raw_stage is None else str(raw_stage)
    log.info("Resumed %s from update %s", path, update)
    return update, target, stage


def _begin(env: SchoolagyTrack, numbers: LapNumbers) -> None:
    numbers.reset()
    numbers.advance(env.reset(restart=True), -1)


def _spawn_policy(model: StateActorCritic, env: SchoolagyTrack, device: torch.device) -> tuple[str, dict[str, float]]:
    """Softmax over chords at the spawn, after a fresh restart. No unimix."""
    numbers = LapNumbers(model.n_actions)
    numbers.reset()
    numbers.advance(env.reset(restart=True), -1)
    observation = torch.from_numpy(numbers.vector()).unsqueeze(0).to(device)
    reset_action = bool(getattr(env, "reset_action", False))
    with torch.no_grad():
        logits, _value = model(observation)
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
    named = {
        chord_name(index, env.keys, reset_action): float(probabilities[index])
        for index in range(model.n_actions)
    }
    top = max(named, key=named.get)
    return top, named


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if int(values.shape[0]) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def summarize_trials(reports: list[dict]) -> dict:
    """Mean and sample standard deviation of progress across independent runs."""
    if not reports:
        raise ValueError("need at least one trial")
    best = np.asarray([float(report["best_progress"]) for report in reports], dtype=np.float64)
    last_values = []
    trials = []
    for report in reports:
        rows = list(report.get("rows") or [])
        if not rows:
            raise ValueError(f"seed {report.get('seed')} finished with no updates")
        last_progress = float(rows[-1]["progress"])
        last_values.append(last_progress)
        trials.append(
            {
                "seed": report.get("seed"),
                "updates": int(report.get("updates") or 0),
                "best_progress": float(report["best_progress"]),
                "last_progress": last_progress,
                "last_entropy": float(rows[-1]["entropy"]) if rows[-1].get("entropy") is not None else 0.0,
                "spawn_chord": report.get("spawn_chord"),
                "spawn_probs": report.get("spawn_probs") or {},
                "out_dir": report.get("out_dir"),
            }
        )
    last = np.asarray(last_values, dtype=np.float64)
    best_mean, best_std = _mean_std(best)
    last_mean, last_std = _mean_std(last)
    return {
        "n": len(reports),
        "seeds": [report.get("seed") for report in reports],
        "best_progress_mean": best_mean,
        "best_progress_std": best_std,
        "last_progress_mean": last_mean,
        "last_progress_std": last_std,
        "trials": trials,
    }


def _rank_core_cpus(rows: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """`(core, cpu, max_khz)` rows. One cpu per core, highest clock first.

    The second logical cpu of a core is a hyperthread and is dropped.
    """
    found: dict[int, tuple[int, int]] = {}
    for core, cpu, freq in rows:
        core_id = int(core)
        previous = found.get(core_id)
        if previous is None or int(cpu) < previous[1]:
            found[core_id] = (int(freq), int(cpu))
    return sorted(found.values(), key=lambda item: (-item[0], item[1]))


def _fast_cpus(ranked: list[tuple[int, int]]) -> list[int]:
    """Cpus tied for the highest clock. Empty when the machine could not be read."""
    if not ranked:
        return []
    top = ranked[0][0]
    return [cpu for freq, cpu in ranked if freq == top]


def _read_core_cpus() -> list[tuple[int, int]]:
    rows: list[tuple[int, int, int]] = []
    for cpu in range(os.cpu_count() or 1):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id")
        freq_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
        try:
            core = int(topo.read_text())
        except OSError:
            continue
        try:
            freq = int(freq_path.read_text())
        except OSError:
            freq = 0
        rows.append((core, cpu, freq))
    return _rank_core_cpus(rows)


def _worker_count(raw: object, group_size: int, fast: list[int]) -> int:
    """How many physics processes to boot.

    A missing value is one process per performance core, and never more than
    the lap group. An explicit number is used as given, at least one.
    """
    if raw is None:
        return max(1, min(int(group_size), len(fast) or 1))
    return max(1, int(raw))


def _pin_physics(envs: list[SchoolagyTrack], ranked: list[tuple[int, int]]) -> list[int]:
    """Pin each owned process to its own core, fastest cores first."""
    cpus = [cpu for _freq, cpu in ranked]
    if not cpus:
        return []
    pinned: list[int] = []
    for index, env in enumerate(envs):
        pin = getattr(getattr(env, "client", None), "pin", None)
        if pin is None:
            continue
        cpu = cpus[index] if index < len(cpus) else cpus[index % len(cpus)]
        try:
            pin(cpu)
        except OSError:
            break
        pinned.append(cpu)
    return pinned


def train_ppo(
    cfg: dict,
    env: SchoolagyTrack | list[SchoolagyTrack] | tuple[SchoolagyTrack, ...] | None = None,
    max_updates: int | None = None,
) -> dict:
    """Update the actor and the reward critic on one 0.6.3 track.

    The actor sees velocity, heading, the road edge, and the next gate. Unless
    a checkpoint resumes, it first clones the fastest TrackData laps of the
    most common track, and the early PPO updates keep a clone loss. The reward
    then follows that track's single best lap. Once a car finishes the lap or
    covers most of it, the reward becomes lap time. The critic is trained on
    the discounted return. A checkpoint resumes only when its track matches.
    Several cars step at once, one physics process each. A lap stays on its
    own car until it ends, and only then joins the update.
    """
    rl_cfg = cfg["rl"]
    seed = rl_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))
    device = resolve_device(str(rl_cfg.get("device") or "auto"))
    hardware = configure_torch(device)
    if seed is not None and device.type == "cuda":
        # benchmark picks kernels from shapes, which changes the result across runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    bc_cfg = dict(rl_cfg.get("bc") or {})
    bc_on = bool(bc_cfg.get("enabled", True))
    out_dir = Path(rl_cfg.get("out_dir") or "runs/winter4")
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(cfg.get("keys") or ["w", "a", "s", "d"])
    hidden = int(rl_cfg.get("hidden") or 128)
    reset_action = bool(rl_cfg.get("reset_action"))
    chords_planned = n_actions(keys, reset_action=reset_action)
    # A resumed actor already has weights. Cloning again would overwrite them.
    resuming = bool(rl_cfg.get("resume", True)) and _checkpoint_matches(
        out_dir / "last.pt",
        Path(cfg["game"]["track"]).stem,
        chords_planned,
        obs_dim(chords_planned),
        hidden,
    )
    bc_epochs = int(bc_cfg.get("epochs") or 25)
    bc_coef = float(0.0 if bc_cfg.get("coef") is None else bc_cfg.get("coef"))
    bc_updates = int(150 if bc_cfg.get("updates") is None else bc_cfg.get("updates"))
    resumed_update_hint = None
    if resuming:
        try:
            hint = read_checkpoint(out_dir / "last.pt", torch.device("cpu"))
            resumed_update_hint = int(hint.get("update") or 0)
        except (OSError, RuntimeError, ValueError, TypeError):
            resumed_update_hint = None
    reference = None
    demos = None
    if bc_on and env is None:
        reference = load_best_line(bc_cfg)
        log.info(
            "Reference lap is %s at %.3fs, %.0fm.",
            reference.track,
            reference.seconds,
            reference.length,
        )
        keep_cloning = bc_coef > 0.0 and bc_updates > 0 and (
            not resuming or (resumed_update_hint is not None and resumed_update_hint < bc_updates)
        )
        if not resuming or keep_cloning:
            demos = load_demos(bc_cfg, keys)
            if not resuming:
                cfg["game"]["track"] = demos.track_file
                log.info(
                    "BC warm start from %s laps of %s at or under %.3fs (best %.3fs, margin %.3fs, %s decisions).",
                    demos.laps,
                    demos.track,
                    demos.best_seconds + demos.margin_s,
                    demos.best_seconds,
                    demos.margin_s,
                    demos.steps,
                )
    track = Path(cfg["game"]["track"]).stem
    log.info("PPO device %s (%s) track %s seed %s", device, hardware, track, "unset" if seed is None else int(seed))
    passed_envs = list(env) if isinstance(env, (list, tuple)) else None
    if passed_envs is not None:
        if not passed_envs:
            raise ValueError("train_ppo needs an environment")
        env = passed_envs[0]
        own_env = False
    else:
        own_env = env is None
        if env is None:
            env = make_track(cfg)
    chords = n_actions(env.keys, reset_action=getattr(env, "reset_action", False))
    model = StateActorCritic(obs_dim=obs_dim(chords), n_actions=chords, hidden=hidden).to(device)
    resumed_pair = _resume(out_dir / "last.pt", model, device, track) if bool(rl_cfg.get("resume", True)) else None
    resumed = resumed_pair is not None
    resumed_update = None if resumed_pair is None else resumed_pair[0]
    entropy_target = None if resumed_pair is None else resumed_pair[1]
    resumed_stage = None if resumed_pair is None else resumed_pair[2]
    entropy_cfg = dict(rl_cfg.get("entropy") or {})
    entropy_bump = float(0.5 if entropy_cfg.get("bump") is None else entropy_cfg.get("bump"))
    entropy_drop = float(0 if entropy_cfg.get("drop") is None else entropy_cfg.get("drop"))
    entropy_floor = float(0.08 if entropy_cfg.get("floor") is None else entropy_cfg.get("floor"))
    entropy_wobble = float(0.04 if entropy_cfg.get("wobble") is None else entropy_cfg.get("wobble"))
    if resumed:
        demos = None
        if entropy_drop == 0:
            log.info("Entropy held at %.3f, wobble %.3f.", float(entropy_target), entropy_wobble)
        else:
            log.info(
                "Entropy target %.3f, down %.4f each update, wobble %.3f, floor %.3f.",
                float(entropy_target),
                entropy_drop,
                entropy_wobble,
                entropy_floor,
            )
    elif demos is not None:
        stats = behavior_clone(
            model,
            demos,
            epochs=bc_epochs,
            batch_size=int(bc_cfg.get("batch_size") or 256),
            lr=float(bc_cfg.get("lr") or 1e-3),
            device=device,
        )
        log.info(
            "Cloned %s decisions from %s laps. Loss %.4f, accuracy %.3f.",
            int(stats["steps"]),
            int(stats["laps"]),
            stats["loss"],
            stats["accuracy"],
        )
    else:
        model.bias_toward_forward(env.keys, reset_action=getattr(env, "reset_action", False))
    model.eval()
    bc_obs_t = None
    bc_act_t = None
    if demos is not None and bc_coef > 0.0 and bc_updates > 0:
        bc_obs_t = torch.as_tensor(demos.observations, dtype=torch.float32, device=device)
        bc_act_t = torch.as_tensor(demos.actions, dtype=torch.int64, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rl_cfg.get("lr") or 3e-4))
    group_size = max(2, int(rl_cfg.get("group_size") or 8))
    unimix = float(rl_cfg.get("unimix") or 0.0)
    gamma = float(rl_cfg.get("gamma") or 0.99)
    lam = float(rl_cfg.get("lam") or 0.95)
    value_coef = float(rl_cfg.get("value_coef") or 0.5)
    meta = _meta(model, env.keys, track)
    reward_stage = "chord"
    book = {**meta, "reward_stage": reward_stage}
    raw_close = (cfg.get("reward") or {}).get("line_close")
    line_close = 0.9 if raw_close is None else float(raw_close)
    log_path = out_dir / "log.csv"
    best_progress = -1.0
    best_line = -1.0
    best_seconds: float | None = None
    best_path = out_dir / "best.pt"
    if best_path.is_file():
        try:
            saved = read_checkpoint(best_path, device)
            same_actor = (
                saved.get("arch") == ARCH
                and int(saved.get("obs_dim") or -1) == model.obs_dim
                and int(saved.get("n_actions") or -1) == model.n_actions
            )
            if same_actor:
                raw_progress = saved.get("progress")
                best_progress = -1.0 if raw_progress is None else float(raw_progress)
                raw_seconds = saved.get("finished_seconds")
                best_seconds = None if raw_seconds is None else float(raw_seconds)
                raw_line = saved.get("line")
                best_line = -1.0 if raw_line is None else float(raw_line)
        except (OSError, KeyError, ValueError, TypeError):
            best_progress = -1.0
            best_seconds = None
            best_line = -1.0
    stop_progress = rl_cfg.get("stop_progress")
    stop_at = None if stop_progress is None else float(stop_progress)
    stop_time = rl_cfg.get("stop_time")
    stop_time_s = None if stop_time is None else float(stop_time)
    if stop_time_s is None and demos is not None:
        stop_time_s = float(demos.best_seconds)
    if stop_time_s is not None:
        log.info("Stopping once a lap finishes at or below %.3fs.", stop_time_s)

    class _Car:
        """One physics process and the lap it is driving. Steps stay here until the lap ends."""

        def __init__(self, one: SchoolagyTrack):
            self.env = one
            self.numbers = LapNumbers(chords)
            self.obs: list[np.ndarray] = []
            self.actions: list[int] = []
            self.logp: list[float] = []
            self.values: list[float] = []
            self.rewards: list[float] = []
            self.dones: list[float] = []
            self.progress: list[float] = []
            self.ret = 0.0
            self.length = 0
            self.trace: list[dict] = []
            self.passed_at: int | None = None

        def clear(self) -> None:
            self.obs.clear()
            self.actions.clear()
            self.logp.clear()
            self.values.clear()
            self.rewards.clear()
            self.dones.clear()
            self.progress.clear()
            self.ret = 0.0
            self.length = 0
            self.trace.clear()
            self.passed_at = None

    rows: list[dict] = []
    episode_log = out_dir / "episodes.csv"
    episode_index = 0

    def write_episode(
        seconds: float,
        progress: float,
        finished: bool,
        episode_return: float,
        steps: int,
        line: float = 0.0,
    ) -> None:
        nonlocal episode_index
        episode_index += 1
        record = {
            "episode": episode_index,
            "seconds": round(float(seconds), 3),
            "progress": round(float(progress), 4),
            "line": round(float(line), 4),
            "finished": int(bool(finished)),
            "return": round(float(episode_return), 4),
            "steps": int(steps),
        }
        fresh_episode = not episode_log.is_file()
        with episode_log.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
            if fresh_episode:
                writer.writeheader()
            writer.writerow(record)

    updates = 0 if resumed_update is None else int(resumed_update)
    stored_obs: list[np.ndarray] = []
    stored_actions: list[int] = []
    stored_logp: list[float] = []
    stored_values: list[float] = []
    stored_rewards: list[float] = []
    stored_dones: list[float] = []
    stored_progress: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_ends: list[tuple[float, float | None, float]] = []
    # Flush only sees finished laps. A car still driving keeps its steps, so this
    # stays zero and the update does not invent a partial episode.
    current_return = 0.0
    current_length = 0
    best_lap: float | None = None
    spawn_chord: str | None = None
    spawn_probs: dict[str, float] = {}

    def save_pass(progress: float, car: _Car) -> None:
        one = car.env
        gates = [list(point) for point in getattr(one, "gates", ())]
        finish = list(getattr(one, "finish", (0.0, 0.0, 0.0)))
        payload = {
            "track": track,
            "progress": progress,
            "gates": gates,
            "finish": finish,
            "parts": getattr(one.client, "parts", []),
            "steps": car.trace,
        }
        path = out_dir / "pass.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info("Passed the gate. Wrote %s (%s steps).", path, len(payload["steps"]))

    # A resumed critic has already seen returns. A fresh run fits the critic
    # once before the cloned actor takes a policy step.
    critic_ready = resumed

    def _better(progress: float, seconds: float | None, line: float) -> bool:
        if seconds is not None and best_seconds is None:
            return True
        if seconds is not None and best_seconds is not None:
            return seconds < best_seconds
        if seconds is None and best_seconds is not None:
            return False
        # Before a finish, the reference lap is the score. Chord progress is
        # what a jump off the road used to win with.
        if reward_stage == "line":
            return line > best_line
        return progress > best_progress

    def flush() -> None:
        nonlocal updates, best_progress, best_line, best_seconds, current_return, current_length, entropy_target, critic_ready
        lengths = list(episode_lengths)
        returns_ep = list(episode_returns)
        ends = list(episode_ends)
        if current_length > 0:
            lengths.append(current_length)
            returns_ep.append(current_return)
            if stored_progress:
                ends.append((float(stored_progress[-1]), None, 0.0))
            current_return = 0.0
            current_length = 0
        if sum(lengths) < 2:
            return
        advantages, targets = _episode_gae(stored_rewards, stored_dones, stored_values, lengths, gamma, lam)
        observations = torch.from_numpy(np.stack(stored_obs).astype(np.float32)).to(device)
        stats = ppo_update(
            model,
            optimizer,
            observations,
            torch.tensor(stored_actions, dtype=torch.int64, device=device),
            torch.tensor(stored_logp, dtype=torch.float32, device=device),
            torch.tensor(advantages, dtype=torch.float32, device=device),
            torch.tensor(targets, dtype=torch.float32, device=device),
            epochs=int(rl_cfg.get("epochs") or 4),
            minibatch_size=int(rl_cfg.get("minibatch_size") or 64),
            clip=float(rl_cfg.get("clip") or 0.2),
            entropy_coef=float(0 if rl_cfg.get("entropy_coef") is None else rl_cfg.get("entropy_coef")),
            value_coef=value_coef,
            unimix=unimix,
            update_policy=critic_ready,
            bc_observations=bc_obs_t if critic_ready and reward_stage == "line" and updates < bc_updates else None,
            bc_actions=bc_act_t if critic_ready and reward_stage == "line" and updates < bc_updates else None,
            bc_coef=bc_coef if critic_ready and reward_stage == "line" and updates < bc_updates else 0.0,
        )
        measured = float(stats["entropy"])
        if not critic_ready or entropy_target is None:
            if entropy_drop == 0:
                raw_target = entropy_cfg.get("target")
                entropy_target = 0.6 if raw_target is None else float(raw_target)
                log.info("Entropy held at %.3f, wobble %.3f.", float(entropy_target), entropy_wobble)
            else:
                entropy_target = measured + entropy_bump
                log.info(
                    "Entropy bump from %.3f to %.3f, then down %.4f each update, wobbling by %.3f.",
                    measured,
                    entropy_target,
                    entropy_drop,
                    entropy_wobble,
                )
            stats["entropy"] = set_entropy(model, observations, unimix, entropy_target)
            clear_policy_moments(optimizer, model)
            critic_ready = True
        else:
            entropy_target = max(entropy_floor, entropy_target - entropy_drop)
            held = hold_entropy(
                model,
                observations,
                unimix,
                measured,
                entropy_target,
                entropy_wobble,
                entropy_floor,
            )
            scale = float(getattr(hold_entropy, "scale", 1.0))
            if scale < 0.5 or scale > 2.0:
                clear_policy_moments(optimizer, model)
            stats["entropy"] = held
        if ends:
            mean_progress = float(np.mean([item[0] for item in ends]))
            mean_line = float(np.mean([item[2] for item in ends]))
            finishes = [item for item in ends if item[1] is not None]
            if finishes:
                chosen = min(finishes, key=lambda item: float(item[1]))
            elif reward_stage == "line":
                chosen = max(ends, key=lambda item: float(item[2]))
            else:
                chosen = max(ends, key=lambda item: float(item[0]))
            candidate = float(chosen[0])
            candidate_seconds = None if chosen[1] is None else float(chosen[1])
            candidate_line = float(chosen[2])
        else:
            mean_progress = 0.0
            mean_line = 0.0
            candidate = 0.0
            candidate_seconds = None
            candidate_line = 0.0
        mean_return = float(np.mean(returns_ep))
        updates += 1
        log.info(
            "update %s  laps %s  return %.3f  progress %.3f  line %.3f  policy %.4f  value %.4f  entropy %.3f  %s",
            updates,
            len(returns_ep),
            mean_return,
            mean_progress,
            mean_line,
            stats["policy"],
            stats["value"],
            stats["entropy"],
            reward_stage,
        )
        payload = {
            **book,
            "update": updates,
            "progress": float(candidate),
            "line": float(candidate_line),
            "entropy_target": float(entropy_target),
        }
        if candidate_seconds is not None:
            payload["finished_seconds"] = float(candidate_seconds)
        save_checkpoint(out_dir / "last.pt", model, payload)
        if _better(float(candidate), candidate_seconds, float(candidate_line)):
            best_progress = float(candidate)
            best_line = float(candidate_line)
            best_seconds = None if candidate_seconds is None else float(candidate_seconds)
            save_checkpoint(best_path, model, payload)
        row = {
            "update": updates,
            **stats,
            "progress": mean_progress,
            "line": mean_line,
            "return": mean_return,
            "stage": reward_stage,
        }
        rows.append(row)
        fresh = not log_path.is_file()
        with log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if fresh:
                writer.writeheader()
            writer.writerow(row)
        stored_obs.clear()
        stored_actions.clear()
        stored_logp.clear()
        stored_values.clear()
        stored_rewards.clear()
        stored_dones.clear()
        stored_progress.clear()
        episode_returns.clear()
        episode_lengths.clear()
        episode_ends.clear()

    envs = list(passed_envs) if passed_envs is not None else [env]
    pool: ThreadPoolExecutor | None = None

    def step_car(job: tuple[_Car, int]):
        one, action_id = job
        return one.env.step(action_id)

    def begin_car(one: _Car) -> None:
        _begin(one.env, one.numbers)

    try:
        # Own the extra processes only when this call opened the first one.
        # A caller-supplied env, or a list of them, stays as given.
        ranked = _read_core_cpus() if own_env else []
        if own_env:
            worker_n = _worker_count(rl_cfg.get("workers"), group_size, _fast_cpus(ranked))
            while len(envs) < worker_n:
                envs.append(make_track(cfg))
        cars = [_Car(one) for one in envs]
        if reference is not None:
            reward_stage = resumed_stage if resumed_stage in ("line", "time") else "line"
            book["reward_stage"] = reward_stage
            for one in cars:
                one.env.use_line(reference, reward_stage)
            log.info(
                "Reward stage %s. The reference lap pays until %.0f%% or a finish, then lap time.",
                reward_stage,
                line_close * 100.0,
            )
        pinned = _pin_physics(envs, ranked) if own_env else []
        # readline on the host pipe releases the GIL, so these overlap.
        if len(cars) > 1:
            pool = ThreadPoolExecutor(max_workers=len(cars))
        where = ",".join(str(cpu) for cpu in pinned) if pinned else "any"
        log.info("Stepping %s physics worker(s) on cpu %s.", len(cars), where)
        if pool is None:
            for one in cars:
                begin_car(one)
        else:
            list(pool.map(begin_car, cars))
        while max_updates is None or updates < max_updates:
            observation = torch.from_numpy(np.stack([one.numbers.vector() for one in cars])).to(device)
            with torch.no_grad():
                action, logp, value = model.act(observation, unimix=unimix, greedy=False)
            jobs = [(one, int(action[index].item())) for index, one in enumerate(cars)]
            if pool is None:
                results = [step_car(jobs[0])]
            else:
                results = list(pool.map(step_car, jobs))
            stopping = False
            promote = False
            reset_cars: list[_Car] = []
            for index, one in enumerate(cars):
                _frame, reward, done, info = results[index]
                if not info.get("counted", False):
                    continue
                if ready_for_time(
                    reward_stage,
                    bool(info.get("finished")),
                    float(info.get("line_progress") or 0.0),
                    line_close,
                ):
                    promote = True
                action_id = jobs[index][1]
                one.obs.append(one.numbers.vector())
                one.numbers.advance(info, action_id)
                one.actions.append(action_id)
                one.logp.append(float(logp[index].item()))
                one.values.append(float(value[index].item()))
                reward_t = float(reward)
                # A timeout is a cut, not a death. Fold one bootstrap into the last
                # reward and keep done set, so the next lap does not leak in.
                if (
                    done
                    and info.get("timeout")
                    and not info.get("finished")
                    and not info.get("fell")
                    and not info.get("stuck")
                    and not info.get("beached")
                ):
                    post = torch.from_numpy(one.numbers.vector()).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _logits, post_value = model(post)
                    reward_t += gamma * float(post_value.item())
                one.rewards.append(reward_t)
                one.dones.append(1.0 if done else 0.0)
                one.progress.append(float(info.get("progress") or 0.0))
                one.ret += float(reward)
                one.length += 1
                if stop_at is not None:
                    state = info["state"]
                    one.trace.append(
                        {
                            "frames": int(state["frames"]),
                            "progress": float(info.get("progress") or 0.0),
                            "checkpoint": int(state["nextCheckpointIndex"]),
                            "speed": float(state["speedKmh"]),
                            "action": action_id,
                            "position": {axis: float(state["position"][axis]) for axis in "xyz"},
                            "quaternion": {axis: float(state["quaternion"][axis]) for axis in "xyzw"},
                        }
                    )
                    if one.passed_at is None and float(info.get("progress") or 0.0) >= stop_at:
                        one.passed_at = int(state["frames"])
                        log.info(
                            "Cleared the gate at %.2fs, progress %.2f.",
                            one.passed_at * 0.001,
                            info.get("progress") or 0.0,
                        )
                    if one.passed_at is not None and (done or int(state["frames"]) >= one.passed_at + 2000):
                        save_checkpoint(
                            out_dir / "last.pt",
                            model,
                            {
                                **book,
                                "update": updates,
                                "progress": float(info.get("progress") or 0.0),
                                "line": float(info.get("line_progress") or 0.0),
                                **({} if entropy_target is None else {"entropy_target": float(entropy_target)}),
                            },
                        )
                        save_pass(float(info.get("progress") or 0.0), one)
                        stopping = True
                        continue
                if not done:
                    continue
                stored_obs.extend(one.obs)
                stored_actions.extend(one.actions)
                stored_logp.extend(one.logp)
                stored_values.extend(one.values)
                stored_rewards.extend(one.rewards)
                stored_dones.extend(one.dones)
                stored_progress.extend(one.progress)
                episode_returns.append(one.ret)
                episode_lengths.append(one.length)
                finished = bool(info.get("finished"))
                state = info["state"]
                seconds = int(state["finishFrames"]) * 0.001 if finished else int(state["frames"]) * 0.001
                line_progress = float(info.get("line_progress") or 0.0)
                write_episode(
                    seconds,
                    float(info.get("progress") or 0.0),
                    finished,
                    one.ret,
                    one.length,
                    line_progress,
                )
                episode_ends.append(
                    (float(info.get("progress") or 0.0), seconds if finished else None, line_progress)
                )
                one.clear()
                if finished:
                    lap_s = int(info["state"]["finishFrames"]) * 0.001
                    if best_lap is None or lap_s < best_lap:
                        best_lap = lap_s
                    log.info(
                        "Finish at %.2fs (progress %.2f), best %.2fs.",
                        lap_s,
                        info.get("progress") or 0.0,
                        best_lap,
                    )
                    if stop_time_s is not None and lap_s <= stop_time_s:
                        log.info("Lap is at or under the %.2fs target. Stopping.", stop_time_s)
                        save_checkpoint(
                            out_dir / "last.pt",
                            model,
                            {
                                **book,
                                "update": updates,
                                "progress": float(info.get("progress") or 0.0),
                                "line": float(info.get("line_progress") or 0.0),
                                **({} if entropy_target is None else {"entropy_target": float(entropy_target)}),
                            },
                        )
                        stopping = True
                        continue
                if not stopping:
                    reset_cars.append(one)
            if promote and reward_stage == "line":
                reward_stage = "time"
                book["reward_stage"] = "time"
                for one in cars:
                    one.env.reward_stage = "time"
                log.info(
                    "A car finished or reached %.0f%% of the reference lap. Time reward is on.",
                    line_close * 100.0,
                )
            if stopping:
                break
            if len(episode_returns) >= group_size:
                flush()
            if max_updates is not None and updates >= max_updates:
                break
            if reset_cars:
                if pool is None or len(reset_cars) == 1:
                    for one in reset_cars:
                        begin_car(one)
                else:
                    list(pool.map(begin_car, reset_cars))
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        try:
            spawn_chord, spawn_probs = _spawn_policy(model, envs[0], device)
            throttle_name = "w" if "w" in spawn_probs else "up"
            log.info(
                "spawn policy top %s (%.2f), straight throttle %.2f",
                spawn_chord,
                spawn_probs.get(spawn_chord, 0.0),
                spawn_probs.get(throttle_name, 0.0),
            )
        except Exception:
            log.warning("Could not read the policy at the spawn.", exc_info=True)
        if own_env:
            for one in envs:
                one.close()
    return {
        "updates": updates,
        "best_progress": best_progress,
        "rows": rows,
        "out_dir": str(out_dir),
        "seed": None if seed is None else int(seed),
        "spawn_chord": spawn_chord,
        "spawn_probs": spawn_probs,
    }
