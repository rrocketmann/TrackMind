"""Online GRPO. A group of real laps is scored, then the keys are released for the update."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from polytrack_agent.env import PolyTrackVisionEnv
from polytrack_agent.obs.progress import load_reference
from polytrack_agent.policy.cnn import resolve_device
from polytrack_agent.rl.actions import n_actions
from polytrack_agent.rl.device import configure_torch
from polytrack_agent.rl.checkpoint import save_checkpoint
from polytrack_agent.rl.grpo import group_advantages, grpo_update
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.state import LapNumbers, obs_dim

log = logging.getLogger(__name__)


def _meta(model: StateActorCritic, keys: list[str]) -> dict:
    return {
        "obs_dim": int(model.obs_dim),
        "hidden": int(model.hidden),
        "n_actions": int(model.n_actions),
        "keys": list(keys),
    }


def train_grpo(
    cfg: dict,
    reference: Path | None = None,
    env: PolyTrackVisionEnv | None = None,
    max_updates: int | None = None,
) -> dict:
    """Drive Chromium and update the actor from a group of real laps.

    The actor sees lap numbers only: progress, its change, match confidence,
    time, stall, and the previous chord. The reference lap is the progress
    sensor. Pass `env` only from tests. A real run opens polytrack.dev in
    Chromium when that window is not already up. A reference lap is required.
    """
    rl_cfg = cfg["rl"]
    device = resolve_device(str(rl_cfg.get("device") or "auto"))
    hardware = configure_torch(device)
    log.info("GRPO device %s (%s)", device, hardware)
    own_env = env is None
    if env is None:
        if reference is None or not Path(reference).is_file():
            raise FileNotFoundError(
                "GRPO needs a reference lap of the track (play.reference or --reference)."
            )
        from polytrack_agent.capture.launch import ensure_chromium_window

        ensure_chromium_window(cfg)
        progress = load_reference(Path(reference), cfg["progress"])
        env = PolyTrackVisionEnv(cfg, dry_run=False, progress=progress)
    chords = n_actions(env.keys)
    model = StateActorCritic(
        obs_dim=obs_dim(chords),
        n_actions=chords,
        hidden=int(rl_cfg.get("hidden") or 128),
    ).to(device)
    model.eval()
    policy_params = [param for name, param in model.named_parameters() if not name.startswith("value")]
    optimizer = torch.optim.AdamW(policy_params, lr=float(rl_cfg.get("lr") or 3e-4))
    group_size = max(2, int(rl_cfg.get("group_size") or 8))
    unimix = float(rl_cfg.get("unimix") or 0.0)
    settle = float(rl_cfg.get("finish_settle_s") or 0.0)
    out_dir = Path(rl_cfg.get("out_dir") or "runs/grpo")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta(model, env.keys)

    numbers = LapNumbers(chords)
    rows: list[dict] = []
    env.reset(restart=True)
    updates = 0
    best_progress = -1.0
    stored_obs: list[np.ndarray] = []
    stored_actions: list[int] = []
    stored_logp: list[float] = []
    stored_progress: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    current_return = 0.0
    current_length = 0

    def flush() -> None:
        nonlocal updates, best_progress, current_return, current_length
        lengths = list(episode_lengths)
        returns = list(episode_returns)
        if current_length > 0:
            lengths.append(current_length)
            returns.append(current_return)
            current_return = 0.0
            current_length = 0
        if sum(lengths) < 2 or len(returns) < 2:
            return
        env.runtime.release_keys()
        advantages = group_advantages(returns, lengths)
        if advantages.shape[0] != len(stored_actions):
            raise RuntimeError(
                f"GRPO group has {advantages.shape[0]} advantages and {len(stored_actions)} decisions"
            )
        observations = torch.from_numpy(np.stack(stored_obs).astype(np.float32)).to(device)
        stats = grpo_update(
            model,
            optimizer,
            observations,
            torch.tensor(stored_actions, dtype=torch.int64, device=device),
            torch.tensor(stored_logp, dtype=torch.float32, device=device),
            torch.tensor(advantages, dtype=torch.float32, device=device),
            epochs=int(rl_cfg.get("epochs") or 4),
            minibatch_size=int(rl_cfg.get("minibatch_size") or 64),
            clip=float(rl_cfg.get("clip") or 0.2),
            entropy_coef=float(rl_cfg.get("entropy_coef") or 0.01),
            kl_coef=float(rl_cfg.get("kl_coef") or 0.0),
            unimix=unimix,
        )
        mean_progress = float(np.mean(stored_progress)) if stored_progress else 0.0
        mean_return = float(np.mean(returns))
        updates += 1
        log.info(
            "group %s  laps %s  return %.3f  progress %.3f  loss %.4f  kl %.4f",
            updates,
            len(returns),
            mean_return,
            mean_progress,
            stats["loss"],
            stats["kl"],
        )
        save_checkpoint(out_dir / "last.pt", model, {**meta, "update": updates, "progress": mean_progress})
        if mean_progress > best_progress:
            best_progress = mean_progress
            save_checkpoint(out_dir / "best.pt", model, {**meta, "update": updates, "progress": mean_progress})
        rows.append({"update": updates, **stats, "progress": mean_progress, "return": mean_return})
        stored_obs.clear()
        stored_actions.clear()
        stored_logp.clear()
        stored_progress.clear()
        episode_returns.clear()
        episode_lengths.clear()

    try:
        while max_updates is None or updates < max_updates:
            observation = torch.from_numpy(numbers.vector()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logp, _value = model.act(observation, unimix=unimix, greedy=False)
            action_id = int(action.item())
            _frame, reward, done, info = env.step(action_id)
            if not info.get("counted", False):
                continue
            stored_obs.append(numbers.vector())
            numbers.advance(info, action_id)
            stored_actions.append(action_id)
            stored_logp.append(float(logp.item()))
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
