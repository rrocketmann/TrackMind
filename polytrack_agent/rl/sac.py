"""Discrete soft actor-critic on the 0.6.3 car.

PPO sees each lap once and its entropy bonus is a fixed coefficient, so the
policy collapses onto a steering chord before straight throttle's return shows
up. SAC keeps a replay buffer and tunes the temperature so the entropy stays
near a target. Actions are a categorical distribution over the nine chords.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from polytrack_agent.rl.checkpoint import SAC_ARCH, read_checkpoint, save_checkpoint
from polytrack_agent.rl.device import configure_torch, resolve_device
from polytrack_agent.rl.model import StateActorCritic
from polytrack_agent.rl.replay import Replay
from polytrack_agent.rl.seed import set_seed
from polytrack_agent.rl.state import LapNumbers, obs_dim
from polytrack_agent.rl.track import SchoolagyTrack, make_track

log = logging.getLogger(__name__)


def _mlp(obs_dim: int, n_actions: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, n_actions),
    )


class TwinQ(nn.Module):
    """Two Q heads over a discrete chord. The backup uses the smaller one."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        if int(obs_dim) < 1 or int(n_actions) < 1 or int(hidden) < 1:
            raise ValueError("obs_dim, n_actions, and hidden must be positive")
        self.q1 = _mlp(int(obs_dim), int(n_actions), int(hidden))
        self.q2 = _mlp(int(obs_dim), int(n_actions), int(hidden))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs), self.q2(obs)


def temperature_loss(log_alpha: torch.Tensor, entropy: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """Loss whose minimum raises alpha when entropy is below the target."""
    return (log_alpha * (entropy - float(target_entropy))).mean()


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    rate = float(tau)
    with torch.no_grad():
        for online, frozen in zip(source.parameters(), target.parameters()):
            frozen.mul_(1.0 - rate).add_(online, alpha=rate)


def _distribution(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.exp(), log_probs


def sac_update(
    actor: StateActorCritic,
    q: TwinQ,
    q_target: TwinQ,
    actor_opt: torch.optim.Optimizer,
    q_opt: torch.optim.Optimizer,
    log_alpha: nn.Parameter,
    alpha_opt: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    tau: float,
    target_entropy: float,
) -> dict[str, float]:
    """One discrete SAC step. The actor is a chord distribution, not a squashed Gaussian."""
    alpha = log_alpha.detach().exp()
    with torch.no_grad():
        next_probs, next_log = _distribution(actor(next_obs)[0])
        next_q = torch.minimum(*q_target(next_obs))
        next_value = (next_probs * (next_q - alpha * next_log)).sum(dim=-1)
        target = rewards + (1.0 - dones) * float(gamma) * next_value

    q1, q2 = q(obs)
    chosen = actions.long().unsqueeze(-1)
    q_loss = (q1.gather(-1, chosen).squeeze(-1) - target).pow(2).mean()
    q_loss = q_loss + (q2.gather(-1, chosen).squeeze(-1) - target).pow(2).mean()
    q_opt.zero_grad(set_to_none=True)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
    q_opt.step()

    probs, log_probs = _distribution(actor(obs)[0])
    q_min = torch.minimum(*q(obs)).detach()
    actor_loss = (probs * (alpha * log_probs - q_min)).sum(dim=-1).mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(actor.net.parameters()) + list(actor.policy.parameters()), 1.0)
    actor_opt.step()

    with torch.no_grad():
        updated_probs, updated_log = _distribution(actor(obs)[0])
        entropy = -(updated_probs * updated_log).sum(dim=-1)
    alpha_loss = temperature_loss(log_alpha, entropy, target_entropy)
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()
    with torch.no_grad():
        log_alpha.clamp_(-10.0, 2.0)

    soft_update(q, q_target, tau)
    return {
        "actor": float(actor_loss.detach()),
        "q": float(q_loss.detach()),
        "alpha": float(log_alpha.detach().exp()),
        "entropy": float(entropy.mean().detach()),
    }


def _batch(arrays: tuple[np.ndarray, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    obs, actions, rewards, next_obs, dones = arrays
    return (
        torch.from_numpy(np.ascontiguousarray(obs)).to(device),
        torch.from_numpy(np.ascontiguousarray(actions)).to(device),
        torch.from_numpy(np.ascontiguousarray(rewards)).to(device),
        torch.from_numpy(np.ascontiguousarray(next_obs)).to(device),
        torch.from_numpy(np.ascontiguousarray(dones)).to(device),
    )


def train_sac(
    cfg: dict,
    env: SchoolagyTrack | None = None,
    max_updates: int | None = None,
) -> dict:
    """Off-policy SAC. One logged update is `group_size` laps, same as PPO."""
    from polytrack_agent.rl.actions import n_actions
    from polytrack_agent.rl.trainer import _begin, _spawn_policy

    rl_cfg = cfg["rl"]
    sac_cfg = rl_cfg.get("sac") or {}
    seed = rl_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))
    rng = np.random.default_rng(None if seed is None else int(seed))
    device = resolve_device(str(rl_cfg.get("device") or "auto"))
    hardware = configure_torch(device)
    if seed is not None and device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    track = Path(cfg["game"]["track"]).stem
    log.info("SAC device %s (%s) track %s seed %s", device, hardware, track, "unset" if seed is None else int(seed))
    own_env = env is None
    if env is None:
        env = make_track(cfg)
    chords = n_actions(env.keys, reset_action=getattr(env, "reset_action", False))
    hidden = int(rl_cfg.get("hidden") or 128)
    width = obs_dim(chords)
    actor = StateActorCritic(obs_dim=width, n_actions=chords, hidden=hidden).to(device)
    q = TwinQ(width, chords, hidden).to(device)
    q_target = copy.deepcopy(q).to(device)
    for parameter in q_target.parameters():
        parameter.requires_grad_(False)
    init_alpha = float(sac_cfg.get("init_alpha") or 0.2)
    log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), device=device))
    out_dir = Path(rl_cfg.get("out_dir") or "runs/sac")
    out_dir.mkdir(parents=True, exist_ok=True)
    gamma = float(rl_cfg.get("gamma") or 0.99)
    tau = float(sac_cfg.get("tau") or 0.005)
    batch_size = int(sac_cfg.get("batch_size") or 256)
    learning_starts = max(batch_size, int(sac_cfg.get("learning_starts") or 1000))
    updates_per_step = max(1, int(sac_cfg.get("updates_per_step") or 1))
    target_entropy = sac_cfg.get("target_entropy")
    if target_entropy is None:
        # Just under a uniform policy, which is the usual discrete-SAC target.
        # A lower target drops the temperature while Q is still flat, and the
        # chord distribution then locks onto whichever action was sampled.
        target_entropy = 0.98 * math.log(chords)
    target_entropy = float(target_entropy)
    lr = float(rl_cfg.get("lr") or 3e-4)
    # No weight decay. AdamW's default decay shrinks Q toward zero and pulls alpha toward 1.
    actor_opt = torch.optim.Adam(list(actor.net.parameters()) + list(actor.policy.parameters()), lr=lr)
    q_opt = torch.optim.Adam(q.parameters(), lr=lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)
    meta = {
        "obs_dim": width,
        "hidden": hidden,
        "n_actions": chords,
        "keys": list(env.keys),
        "track": track,
    }
    resumed = False
    last_path = out_dir / "last.pt"
    if bool(rl_cfg.get("resume", True)) and last_path.is_file():
        payload = read_checkpoint(last_path, device)
        same = (
            payload.get("arch") == SAC_ARCH
            and payload.get("track") == track
            and int(payload.get("obs_dim") or -1) == width
            and int(payload.get("n_actions") or -1) == chords
            and "q" in payload
        )
        if same:
            actor.load_state_dict(payload["state_dict"])
            q.load_state_dict(payload["q"])
            q_target.load_state_dict(payload.get("q_target") or payload["q"])
            with torch.no_grad():
                log_alpha.fill_(float(payload.get("log_alpha") or math.log(init_alpha)))
            resumed = True
            log.info("Resumed %s from update %s", last_path, payload.get("update"))
        else:
            log.warning("Ignoring %s because it is not this SAC actor.", last_path)
    if not resumed:
        actor.bias_toward_forward(env.keys, reset_action=getattr(env, "reset_action", False))
    actor.eval()
    q.eval()
    buffer = Replay(int(sac_cfg.get("buffer_size") or 100_000), width)
    group_size = max(2, int(rl_cfg.get("group_size") or 8))
    log_path = out_dir / "log.csv"
    best_progress = -1.0
    best_path = out_dir / "best.pt"
    numbers = LapNumbers(chords)
    rows: list[dict] = []
    updates = 0
    episode_returns: list[float] = []
    window_progress: list[float] = []
    window_actor: list[float] = []
    window_q: list[float] = []
    window_alpha: list[float] = []
    window_entropy: list[float] = []
    current_return = 0.0
    learned = False
    spawn_chord: str | None = None
    spawn_probs: dict[str, float] = {}
    stop_time = rl_cfg.get("stop_time")
    stop_time_s = None if stop_time is None else float(stop_time)
    best_lap: float | None = None
    _begin(env, numbers)

    def snapshot(progress: float) -> dict:
        return {
            **meta,
            "update": updates,
            "progress": progress,
            "q": q.state_dict(),
            "q_target": q_target.state_dict(),
            "log_alpha": float(log_alpha.detach()),
        }

    def flush() -> None:
        nonlocal updates, best_progress
        if len(episode_returns) < group_size:
            return
        updates += 1
        mean_progress = float(np.mean(window_progress)) if window_progress else 0.0
        mean_return = float(np.mean(episode_returns))
        stats = {
            "actor": float(np.mean(window_actor)) if window_actor else 0.0,
            "q": float(np.mean(window_q)) if window_q else 0.0,
            "alpha": float(np.mean(window_alpha)) if window_alpha else float(log_alpha.detach().exp()),
            "entropy": float(np.mean(window_entropy)) if window_entropy else 0.0,
        }
        log.info(
            "update %s  laps %s  return %.3f  progress %.3f  actor %.4f  q %.4f  alpha %.3f  entropy %.3f",
            updates,
            len(episode_returns),
            mean_return,
            mean_progress,
            stats["actor"],
            stats["q"],
            stats["alpha"],
            stats["entropy"],
        )
        payload = snapshot(mean_progress)
        save_checkpoint(last_path, actor, payload, arch=SAC_ARCH)
        if mean_progress > best_progress:
            best_progress = mean_progress
            save_checkpoint(best_path, actor, payload, arch=SAC_ARCH)
        row = {"update": updates, **stats, "progress": mean_progress, "return": mean_return}
        rows.append(row)
        fresh = not log_path.is_file()
        with log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if fresh:
                writer.writeheader()
            writer.writerow(row)
        episode_returns.clear()
        window_progress.clear()
        window_actor.clear()
        window_q.clear()
        window_alpha.clear()
        window_entropy.clear()

    try:
        while max_updates is None or updates < max_updates:
            observation = torch.from_numpy(numbers.vector()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _logp, _value = actor.act(observation, greedy=False)
            action_id = int(action.item())
            before = numbers.vector()
            _frame, reward, done, info = env.step(action_id)
            if not info.get("counted", False):
                continue
            numbers.advance(info, action_id)
            buffer.add(before, action_id, float(reward), numbers.vector(), bool(done))
            if buffer.size >= learning_starts:
                if not learned:
                    learned = True
                    log.info("SAC learning starts at %s transitions.", buffer.size)
                actor.train()
                q.train()
                for _ in range(updates_per_step):
                    stats = sac_update(
                        actor,
                        q,
                        q_target,
                        actor_opt,
                        q_opt,
                        log_alpha,
                        alpha_opt,
                        *_batch(buffer.sample(batch_size, rng), device),
                        gamma,
                        tau,
                        target_entropy,
                    )
                    window_actor.append(stats["actor"])
                    window_q.append(stats["q"])
                    window_alpha.append(stats["alpha"])
                    window_entropy.append(stats["entropy"])
            else:
                with torch.no_grad():
                    probs, log_probs = _distribution(actor(observation)[0])
                    window_entropy.append(float((-(probs * log_probs).sum()).item()))
            current_return += float(reward)
            window_progress.append(float(info.get("progress") or 0.0))
            if done:
                episode_returns.append(current_return)
                current_return = 0.0
                if info.get("finished"):
                    lap_s = int(info["state"]["finishFrames"]) * 0.001
                    if best_lap is None or lap_s < best_lap:
                        best_lap = lap_s
                    log.info("Finish at %.2fs (progress %.2f), best %.2fs.", lap_s, info.get("progress") or 0.0, best_lap)
                    if stop_time_s is not None and lap_s <= stop_time_s:
                        log.info("Lap is at or under the %.2fs target. Stopping.", stop_time_s)
                        save_checkpoint(last_path, actor, snapshot(float(info.get("progress") or 0.0)), arch=SAC_ARCH)
                        break
                _begin(env, numbers)
                if len(episode_returns) >= group_size:
                    flush()
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        try:
            spawn_chord, spawn_probs = _spawn_policy(actor, env, device)
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
