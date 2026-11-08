"""Online PPO on the PolyTrack 0.6.3 fork simulation."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from polytrack_agent.rl.bc import behavior_clone, load_demos
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


def _resume(path: Path, model: StateActorCritic, device: torch.device, track: str) -> tuple[int, float | None] | None:
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
    log.info("Resumed %s from update %s", path, update)
    return update, target


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


def train_ppo(
    cfg: dict,
    env: SchoolagyTrack | None = None,
    max_updates: int | None = None,
) -> dict:
    """Update the actor and the reward critic on one 0.6.3 track.

    The actor sees velocity, heading, the road edge, and the next gate. Unless
    a checkpoint resumes, it first clones the fastest TrackData laps of the
    most common track. The critic is trained on the discounted return of the
    environment reward. A checkpoint resumes only when its track matches.
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
    demos = None
    if bc_on and not resuming and env is None:
        demos = load_demos(bc_cfg, keys)
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
    own_env = env is None
    if env is None:
        env = make_track(cfg)
    chords = n_actions(env.keys, reset_action=getattr(env, "reset_action", False))
    model = StateActorCritic(obs_dim=obs_dim(chords), n_actions=chords, hidden=hidden).to(device)
    resumed_pair = _resume(out_dir / "last.pt", model, device, track) if bool(rl_cfg.get("resume", True)) else None
    resumed = resumed_pair is not None
    resumed_update = None if resumed_pair is None else resumed_pair[0]
    entropy_target = None if resumed_pair is None else resumed_pair[1]
    entropy_cfg = dict(rl_cfg.get("entropy") or {})
    entropy_bump = float(0.5 if entropy_cfg.get("bump") is None else entropy_cfg.get("bump"))
    entropy_drop = float(0.003 if entropy_cfg.get("drop") is None else entropy_cfg.get("drop"))
    entropy_floor = float(0.08 if entropy_cfg.get("floor") is None else entropy_cfg.get("floor"))
    entropy_wobble = float(0.04 if entropy_cfg.get("wobble") is None else entropy_cfg.get("wobble"))
    if resumed:
        demos = None
    elif demos is not None:
        stats = behavior_clone(
            model,
            demos,
            epochs=int(bc_cfg.get("epochs") or 5),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rl_cfg.get("lr") or 3e-4))
    group_size = max(2, int(rl_cfg.get("group_size") or 8))
    unimix = float(rl_cfg.get("unimix") or 0.0)
    gamma = float(rl_cfg.get("gamma") or 0.99)
    lam = float(rl_cfg.get("lam") or 0.95)
    value_coef = float(rl_cfg.get("value_coef") or 0.5)
    meta = _meta(model, env.keys, track)
    log_path = out_dir / "log.csv"
    best_progress = -1.0
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
                best_progress = float(saved.get("progress") or -1.0)
        except (OSError, KeyError, ValueError, TypeError):
            best_progress = -1.0
    stop_progress = rl_cfg.get("stop_progress")
    stop_at = None if stop_progress is None else float(stop_progress)
    stop_time = rl_cfg.get("stop_time")
    stop_time_s = None if stop_time is None else float(stop_time)
    if stop_time_s is None and demos is not None:
        stop_time_s = float(demos.best_seconds)
    if stop_time_s is not None:
        log.info("Stopping once a lap finishes at or below %.3fs.", stop_time_s)

    numbers = LapNumbers(chords)
    rows: list[dict] = []
    episode_log = out_dir / "episodes.csv"
    episode_index = 0

    def write_episode(seconds: float, progress: float, finished: bool, episode_return: float, steps: int) -> None:
        nonlocal episode_index
        episode_index += 1
        record = {
            "episode": episode_index,
            "seconds": round(float(seconds), 3),
            "progress": round(float(progress), 4),
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

    _begin(env, numbers)
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
    current_return = 0.0
    current_length = 0
    episode_trace: list[dict] = []
    passed_at: int | None = None
    best_lap: float | None = None
    spawn_chord: str | None = None
    spawn_probs: dict[str, float] = {}

    def save_pass(progress: float) -> None:
        gates = [list(point) for point in getattr(env, "gates", ())]
        finish = list(getattr(env, "finish", (0.0, 0.0, 0.0)))
        payload = {
            "track": track,
            "progress": progress,
            "gates": gates,
            "finish": finish,
            "parts": getattr(env.client, "parts", []),
            "steps": episode_trace,
        }
        path = out_dir / "pass.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info("Passed the gate. Wrote %s (%s steps).", path, len(episode_trace))

    def flush() -> None:
        nonlocal updates, best_progress, current_return, current_length, entropy_target
        lengths = list(episode_lengths)
        returns_ep = list(episode_returns)
        if current_length > 0:
            lengths.append(current_length)
            returns_ep.append(current_return)
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
        )
        measured = float(stats["entropy"])
        if entropy_target is None:
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
            if abs(held - measured) > 1e-6:
                clear_policy_moments(optimizer, model)
            stats["entropy"] = held
        mean_progress = float(np.mean(stored_progress)) if stored_progress else 0.0
        mean_return = float(np.mean(returns_ep))
        updates += 1
        log.info(
            "update %s  laps %s  return %.3f  progress %.3f  policy %.4f  value %.4f  entropy %.3f",
            updates,
            len(returns_ep),
            mean_return,
            mean_progress,
            stats["policy"],
            stats["value"],
            stats["entropy"],
        )
        payload = {**meta, "update": updates, "progress": mean_progress, "entropy_target": float(entropy_target)}
        save_checkpoint(out_dir / "last.pt", model, payload)
        if mean_progress > best_progress:
            best_progress = mean_progress
            save_checkpoint(best_path, model, payload)
        row = {"update": updates, **stats, "progress": mean_progress, "return": mean_return}
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

    try:
        while max_updates is None or updates < max_updates:
            observation = torch.from_numpy(numbers.vector()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logp, value = model.act(observation, unimix=unimix, greedy=False)
            action_id = int(action.item())
            _frame, reward, done, info = env.step(action_id)
            if not info.get("counted", False):
                continue
            stored_obs.append(numbers.vector())
            numbers.advance(info, action_id)
            stored_actions.append(action_id)
            stored_logp.append(float(logp.item()))
            stored_values.append(float(value.item()))
            stored_rewards.append(float(reward))
            stored_dones.append(1.0 if done else 0.0)
            stored_progress.append(float(info.get("progress") or 0.0))
            current_return += float(reward)
            current_length += 1
            if stop_at is not None:
                car = info["state"]
                episode_trace.append(
                    {
                        "frames": int(car["frames"]),
                        "progress": float(info.get("progress") or 0.0),
                        "checkpoint": int(car["nextCheckpointIndex"]),
                        "speed": float(car["speedKmh"]),
                        "action": action_id,
                        "position": {axis: float(car["position"][axis]) for axis in "xyz"},
                        "quaternion": {axis: float(car["quaternion"][axis]) for axis in "xyzw"},
                    }
                )
                if passed_at is None and float(info.get("progress") or 0.0) >= stop_at:
                    passed_at = int(car["frames"])
                    log.info("Cleared the gate at %.2fs, progress %.2f.", passed_at * 0.001, info.get("progress") or 0.0)
                if passed_at is not None and (done or int(car["frames"]) >= passed_at + 2000):
                    save_checkpoint(out_dir / "last.pt", model, {**meta, "update": updates, "progress": float(info.get("progress") or 0.0)})
                    save_pass(float(info.get("progress") or 0.0))
                    break
            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                finished = bool(info.get("finished"))
                state = info["state"]
                seconds = (
                    int(state["finishFrames"]) * 0.001
                    if finished
                    else int(state["frames"]) * 0.001
                )
                write_episode(
                    seconds,
                    float(info.get("progress") or 0.0),
                    finished,
                    current_return,
                    current_length,
                )
                current_return = 0.0
                current_length = 0
                if info.get("finished"):
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
                            {**meta, "update": updates, "progress": float(info.get("progress") or 0.0)},
                        )
                        break
                episode_trace.clear()
                passed_at = None
                _begin(env, numbers)
                if len(episode_returns) >= group_size:
                    flush()
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        try:
            spawn_chord, spawn_probs = _spawn_policy(model, env, device)
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
            env.close()
    return {
        "updates": updates,
        "best_progress": best_progress,
        "rows": rows,
        "out_dir": str(out_dir),
        "seed": None if seed is None else int(seed),
        "spawn_chord": spawn_chord,
        "spawn_probs": spawn_probs,
    }
