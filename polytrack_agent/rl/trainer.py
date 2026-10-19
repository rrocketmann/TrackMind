"""Online PPO on the schoolagy PolyTrack 0.6.3 simulation."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from polytrack_agent.rl.device import resolve_device
from polytrack_agent.rl.track import SchoolagyTrack, make_track
from polytrack_agent.rl.actions import n_actions
from polytrack_agent.rl.checkpoint import ARCH, read_checkpoint, save_checkpoint
from polytrack_agent.rl.device import configure_torch
from polytrack_agent.rl.gae import compute_gae
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.ppo import ppo_update
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


def _resume(path: Path, model: StateActorCritic, device: torch.device, track: str) -> bool:
    if not path.is_file():
        return False
    payload = read_checkpoint(path, device)
    if payload.get("track") != track:
        log.warning("Ignoring %s because it was trained on %s.", path, payload.get("track"))
        return False
    if payload.get("arch") != ARCH or int(payload.get("n_actions") or -1) != model.n_actions:
        log.warning("Ignoring %s because it is a different actor.", path)
        return False
    if int(payload.get("obs_dim") or -1) != model.obs_dim or int(payload.get("hidden") or -1) != model.hidden:
        log.warning("Ignoring %s because the observation layout changed.", path)
        return False
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        log.warning("Ignoring %s (%s).", path, exc)
        return False
    model.eval()
    log.info("Resumed %s from update %s", path, payload.get("update"))
    return True


def _begin(env: SchoolagyTrack, numbers: LapNumbers) -> None:
    numbers.reset()
    numbers.advance(env.reset(restart=True), -1)


def train_ppo(
    cfg: dict,
    env: SchoolagyTrack | None = None,
    max_updates: int | None = None,
) -> dict:
    """Update the policy and the value head on the 0.6.3 car.

    The actor sees the simulation's car state. Each press is scored with its
    own advantage, and the PPO step trains both heads. A checkpoint resumes
    only when its track matches.
    """
    rl_cfg = cfg["rl"]
    device = resolve_device(str(rl_cfg.get("device") or "auto"))
    hardware = configure_torch(device)
    track = Path(cfg["game"]["track"]).stem
    log.info("PPO device %s (%s) track %s", device, hardware, track)
    own_env = env is None
    if env is None:
        env = make_track(cfg)
    chords = n_actions(env.keys)
    hidden = int(rl_cfg.get("hidden") or 128)
    model = StateActorCritic(obs_dim=obs_dim(chords), n_actions=chords, hidden=hidden).to(device)
    out_dir = Path(rl_cfg.get("out_dir") or "runs/ppo")
    out_dir.mkdir(parents=True, exist_ok=True)
    resumed = bool(rl_cfg.get("resume", True)) and _resume(out_dir / "last.pt", model, device, track)
    if not resumed:
        model.bias_toward_forward(env.keys)
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

    numbers = LapNumbers(chords)
    rows: list[dict] = []
    _begin(env, numbers)
    updates = 0
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
        nonlocal updates, best_progress, current_return, current_length
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
            entropy_coef=float(rl_cfg.get("entropy_coef") or 0.01),
            value_coef=value_coef,
            unimix=unimix,
        )
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
        payload = {**meta, "update": updates, "progress": mean_progress}
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
                current_return = 0.0
                current_length = 0
                if info.get("finished"):
                    log.info("Finish at progress %.2f.", info.get("progress") or 0.0)
                episode_trace.clear()
                passed_at = None
                _begin(env, numbers)
                if len(episode_returns) >= group_size:
                    flush()
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        if own_env:
            env.close()
    return {"updates": updates, "best_progress": best_progress, "rows": rows, "out_dir": str(out_dir)}
