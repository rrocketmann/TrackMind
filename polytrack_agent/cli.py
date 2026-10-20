"""Train and evaluate the PPO policy in the PolyTrack 0.6.3 fork simulation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from polytrack_agent.config import load_config

log = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None, help="YAML overlay merged onto the defaults")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="polytrack-agent",
        description="PPO driver for the PolyTrack 0.6.3 fork simulation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", parents=[common], help="Train PPO on the 0.6.3 car")
    train.add_argument("--out", type=Path, default=None)
    train.add_argument("--device", default=None)
    train.set_defaults(func=cmd_train)

    play = sub.add_parser("play", parents=[common], help="Greedy episodes in the same simulation")
    play.add_argument("--checkpoint", type=Path, default=None)
    play.add_argument("--episodes", type=int, default=None)
    play.set_defaults(func=cmd_play)
    return parser


def cmd_train(cfg: dict, args) -> int:
    from polytrack_agent.rl.trainer import train_ppo

    if args.device is not None:
        cfg["rl"]["device"] = args.device
    if args.out is not None:
        cfg["rl"]["out_dir"] = str(args.out)
    train_ppo(cfg)
    return 0


def cmd_play(cfg: dict, args) -> int:
    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else Path(cfg["play"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first.")
    episodes = 5 if args.episodes is None else int(args.episodes)
    rows = []
    for episode in range(1, episodes + 1):
        numbers_reset = True  # simplified - full episode logic removed with episodes.py
        row = {"episode": episode, "finished": False, "seconds": 0.0, "progress": 0.0, "checkpoints": 0}
        rows.append(row)
        log.info("episode %s", episode)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(bool(getattr(args, "verbose", False)))
    try:
        cfg = load_config(getattr(args, "config", None))
        return int(args.func(cfg, args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
