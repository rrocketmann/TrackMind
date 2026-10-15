"""Online PPO. A group of real laps is collected, then the keys are released for the update."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch

from polytrack_agent.env import PolyTrackVisionEnv
from polytrack_agent.obs.progress import load_reference
from polytrack_agent.policy.cnn import resolve_device
from polytrack_agent.rl.actions import n_actions
from polytrack_agent.rl.checkpoint import ARCH, read_checkpoint, save_checkpoint
from polytrack_agent.rl.device import configure_torch
from polytrack_agent.rl.gae import compute_gae
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.ppo import ppo_update
from polytrack_agent.rl.state import LapNumbers, obs_dim

log = logging.getLogger(__name__)


def _meta(model: StateActorCritic, keys: list[str]) -> dict:
    return {
        "obs_dim": int(model.obs_dim),
        "hidden": int(model.hidden),
        "n_actions": int(model.n_actions),
        "keys": list(keys),
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


def _resume(path: Path, model: StateActorCritic, device: torch.device) -> bool:
    if not path.is_file():
        return False
    payload = read_checkpoint(path, device)
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


def train_grpo(
    cfg: dict,
    reference: Path | None = None,
    env: PolyTrackVisionEnv | None = None,
    max_updates: int | None = None,
) -> dict:
    """Drive Chromium and update the actor from a group of real laps.

    The actor sees lap numbers only. Each press is scored with its own
    advantage from the value head. The reference lap is the progress sensor.
    Pass `env` only from tests. A real run opens polytrack.dev in Chromium
    when that window is not already up. A reference lap is required.
    `runs/grpo/last.pt` is loaded when it matches this actor.
    """
    rl_cfg = cfg["rl"]
    device = resolve_device(str(rl_cfg.get("device") or "auto"))
    hardware = configure_torch(device)
    log.info("PPO device %s (%s)", device, hardware)
    own_env = env is None
    if env is None:
        if reference is None or not Path(reference).is_file():
            raise FileNotFoundError(
                "Training needs a reference lap of the track (play.reference or --reference)."
            )
        from polytrack_agent.capture.launch import ensure_chromium_window

        ensure_chromium_window(cfg)
        progress = load_reference(Path(reference), cfg["progress"])
        env = PolyTrackVisionEnv(cfg, dry_run=False, progress=progress)
    chords = n_actions(env.keys)
    hidden = int(rl_cfg.get("hidden") or 128)
    model = StateActorCritic(obs_dim=obs_dim(chords), n_actions=chords, hidden=hidden).to(device)
    out_dir = Path(rl_cfg.get("out_dir") or "runs/grpo")
    out_dir.mkdir(parents=True, exist_ok=True)
    resumed = bool(rl_cfg.get("resume", True)) and _resume(out_dir / "last.pt", model, device)
    if not resumed:
        model.bias_toward_forward(env.keys)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rl_cfg.get("lr") or 3e-4))
    group_size = max(2, int(rl_cfg.get("group_size") or 8))
    unimix = float(rl_cfg.get("unimix") or 0.0)
    gamma = float(rl_cfg.get("gamma") or 0.99)
    lam = float(rl_cfg.get("lam") or 0.95)
    value_coef = float(rl_cfg.get("value_coef") or 0.5)
    settle = float(rl_cfg.get("finish_settle_s") or 0.0)
    meta = _meta(model, env.keys)
    log_path = out_dir / "log.csv"
    best_progress = -1.0
    best_path = out_dir / "best.pt"
    if best_path.is_file():
        try:
            best_progress = float(read_checkpoint(best_path, device).get("progress") or -1.0)
        except (OSError, KeyError, ValueError):
            best_progress = -1.0

    numbers = LapNumbers(chords)
    rows: list[dict] = []
    env.reset(restart=True)
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
        env.runtime.release_keys()
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
            "update %s  laps %s  return %.3f  progress %.3f  loss %.4f  value %.4f  entropy %.3f",
            updates,
            len(returns_ep),
            mean_return,
            mean_progress,
            stats["loss"],
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
            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                current_return = 0.0
                current_length = 0
                if info.get("finished") and not env.runtime.dry_run and settle > 0:
                    env.runtime.release_keys()
                    log.info(
                        "Finish at progress %.2f. Waiting %.1fs so Chromium can submit the run.",
                        info.get("progress") or 0.0,
                        settle,
                    )
                    time.sleep(settle)
                restart = not bool(info.get("restarted"))
                numbers.reset()
                env.reset(restart=restart)
                if len(episode_returns) >= group_size:
                    flush()
    except KeyboardInterrupt:
        log.info("Stopped. Keys released.")
    finally:
        env.runtime.release_keys()
        if own_env:
            env.close()
    return {"updates": updates, "best_progress": best_progress, "rows": rows, "out_dir": str(out_dir)}
