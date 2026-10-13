"""Online GRPO. A group of real laps is scored, then the keys are released for the update."""

from __future__ import annotations

import logging
import time
from collections import deque
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
from polytrack_agent.rl.model import TokenActorCritic
from polytrack_agent.rl.tokens import encode_frame, stack_history

log = logging.getLogger(__name__)


def _meta(cfg: dict, encoder, model: TokenActorCritic, keys: list[str]) -> dict:
    image = encoder.image_size
    pool = getattr(encoder, "pool_hw", None) or cfg["train"].get("pool") or [4, 6]
    return {
        "dino_model": str(getattr(encoder, "model_id", None) or cfg["train"].get("dino_model")),
        "image_size": [int(image[0]), int(image[1])],
        "pool": [int(pool[0]), int(pool[1])],
        "feature_dim": int(model.feature_dim),
        "n_tokens": int(model.n_tokens),
        "stack": int(model.stack),
        "n_actions": int(model.n_actions),
        "transformer_layers": int(model.layers),
        "transformer_heads": int(model.heads),
        "keys": list(keys),
    }


def _build_encoder(cfg: dict, device: torch.device):
    from polytrack_agent.policy.dinov3 import FrozenDinoEncoder

    train_cfg = cfg["train"]
    image = (int(train_cfg["image_size"][0]), int(train_cfg["image_size"][1]))
    pool = train_cfg.get("pool") or [4, 6]
    model_id = str(train_cfg.get("dino_model") or "facebook/dinov3-vits16-pretrain-lvd1689m")
    encoder = FrozenDinoEncoder(model_id, image, (int(pool[0]), int(pool[1]))).to(device)
    return encoder.prepare_inference()


def train_grpo(
    cfg: dict,
    reference: Path | None = None,
    encoder=None,
    env: PolyTrackVisionEnv | None = None,
    max_updates: int | None = None,
) -> dict:
    """Drive Chromium and update the actor from a group of real laps.

    Pass `encoder` and `env` only from tests. A real run opens polytrack.dev
    in Chromium when that window is not already up, then builds the frozen
    DINOv3 encoder. A reference lap is required.
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
    if encoder is None:
        encoder = _build_encoder(cfg, device)
    stack = int(rl_cfg.get("stack") or cfg["train"]["stack"])
    model = TokenActorCritic(
        feature_dim=int(encoder.hidden_size),
        n_tokens=int(encoder.n_tokens),
        stack=stack,
        n_actions=n_actions(env.keys),
        layers=int(rl_cfg.get("transformer_layers") or 2),
        heads=int(rl_cfg.get("transformer_heads") or 6),
        dropout=0.0,
    ).to(device)
    model.eval()
    policy_params = [param for name, param in model.named_parameters() if not name.startswith("value")]
    optimizer = torch.optim.AdamW(policy_params, lr=float(rl_cfg.get("lr") or 3e-4))
    group_size = max(2, int(rl_cfg.get("group_size") or 8))
    unimix = float(rl_cfg.get("unimix") or 0.0)
    settle = float(rl_cfg.get("finish_settle_s") or 0.0)
    out_dir = Path(rl_cfg.get("out_dir") or "runs/grpo")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta(cfg, encoder, model, env.keys)

    history: deque = deque(maxlen=stack)
    rows: list[dict] = []
    obs = env.reset(restart=True)
    updates = 0
    best_progress = -1.0
    stored_tokens: list[np.ndarray] = []
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
        tokens = torch.from_numpy(np.stack(stored_tokens).astype(np.float32)).to(device)
        stats = grpo_update(
            model,
            optimizer,
            tokens,
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
        stored_tokens.clear()
        stored_actions.clear()
        stored_logp.clear()
        stored_progress.clear()
        episode_returns.clear()
        episode_lengths.clear()

    try:
        while max_updates is None or updates < max_updates:
            token = encode_frame(encoder, obs[-1], device)
            stacked = stack_history(history, token, stack).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logp, _value = model.act(stacked, unimix=unimix, greedy=False)
            action_id = int(action.item())
            obs, reward, done, info = env.step(action_id)
            if not info.get("counted", False):
                if history:
                    history.pop()
                continue
            stored_tokens.append(stacked.squeeze(0).cpu().numpy().astype(np.float16))
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
                history.clear()
                obs = env.reset(restart=restart)
                if len(episode_returns) >= group_size:
                    flush()
    except KeyboardInterrupt:
        log.info("Stopped. Keys released.")
    finally:
        env.runtime.release_keys()
        if own_env:
            env.close()
    return {"updates": updates, "best_progress": best_progress, "rows": rows, "out_dir": str(out_dir)}
