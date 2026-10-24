"""Train and evaluate the PPO policy in the PolyTrack 0.6.3 fork simulation."""

from __future__ import annotations

import argparse
import copy
import json
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
    train.add_argument("--algo", choices=("ppo", "sac"), default=None, help="ppo, or discrete soft actor-critic")
    train.add_argument("--seed", type=int, default=None, help="fix the RNGs. Trials use this seed, then seed+1, ...")
    train.add_argument(
        "--trials",
        type=int,
        default=1,
        help="independent runs. Writes each under out/seed-N and reports mean progress.",
    )
    train.add_argument("--max-updates", type=int, default=None, help="stop after this many PPO updates")
    train.add_argument(
        "--stop-time",
        type=float,
        default=None,
        help="stop training once a lap finishes at or below this many seconds",
    )
    train.set_defaults(func=cmd_train)

    search = sub.add_parser(
        "search",
        parents=[common],
        help="Race each button hold and keep the one that goes farthest",
    )
    search.add_argument("--out", type=Path, default=Path("runs/hold"))
    search.add_argument("--plan", type=Path, default=None, help="replay this saved plan instead of searching")
    search.add_argument("--extend", type=Path, default=None, help="search one more tap on top of this saved plan")
    search.add_argument("--episodes", type=int, default=1, help="how many replays to run with --plan")
    search.set_defaults(func=cmd_search)

    play = sub.add_parser("play", parents=[common], help="Greedy episodes in the same simulation")
    play.add_argument("--checkpoint", type=Path, default=None)
    play.add_argument("--episodes", type=int, default=None)
    play.set_defaults(func=cmd_play)
    return parser


def _trainer(algo: str):
    if algo == "ppo":
        from polytrack_agent.rl.trainer import train_ppo

        return train_ppo
    if algo == "sac":
        from polytrack_agent.rl.sac import train_sac

        return train_sac
    raise ValueError(f"algo must be ppo or sac, got {algo}")


def cmd_train(cfg: dict, args) -> int:
    from polytrack_agent.rl.trainer import summarize_trials

    if args.device is not None:
        cfg["rl"]["device"] = args.device
    if args.out is not None:
        cfg["rl"]["out_dir"] = str(args.out)
    if args.stop_time is not None:
        cfg["rl"]["stop_time"] = float(args.stop_time)
    if args.seed is not None:
        cfg["rl"]["seed"] = int(args.seed)
    if getattr(args, "algo", None) is not None:
        cfg["rl"]["algo"] = args.algo
    train = _trainer(str(cfg["rl"].get("algo") or "sac"))
    trials = int(args.trials)
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    max_updates = None if args.max_updates is None else int(args.max_updates)
    if max_updates is not None and max_updates < 1:
        raise ValueError(f"max-updates must be >= 1, got {max_updates}")
    if trials == 1:
        train(cfg, max_updates=max_updates)
        return 0
    if max_updates is None:
        raise ValueError("Use --max-updates with --trials so each run finishes and the mean is defined.")

    base = cfg["rl"].get("seed")
    if base is None:
        base = 0
        log.info("No seed set. Trials start at 0.")
    base = int(base)
    out_dir = Path(cfg["rl"].get("out_dir") or "runs/ppo")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index in range(trials):
        seed = base + index
        trial_dir = out_dir / f"seed-{seed}"
        if (trial_dir / "last.pt").exists() or (trial_dir / "log.csv").exists():
            raise FileExistsError(
                f"{trial_dir} already has a run. Remove it before a fresh mean of {trials}."
            )
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["rl"]["seed"] = seed
        trial_cfg["rl"]["out_dir"] = str(trial_dir)
        trial_cfg["rl"]["resume"] = False
        log.info("Trial %s/%s seed %s -> %s", index + 1, trials, seed, trial_dir)
        reports.append(train(trial_cfg, max_updates=max_updates))
    summary = summarize_trials(reports)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info(
        "mean of %s  best progress %.3f ± %.3f  last progress %.3f ± %.3f  wrote %s",
        summary["n"],
        summary["best_progress_mean"],
        summary["best_progress_std"],
        summary["last_progress_mean"],
        summary["last_progress_std"],
        path,
    )
    for trial in summary["trials"]:
        log.info(
            "seed %s  best %.3f  last %.3f  spawn %s",
            trial["seed"],
            trial["best_progress"],
            trial["last_progress"],
            trial["spawn_chord"],
        )
    return 0


def _max_steps(cfg: dict) -> int:
    repeat = max(1, int(cfg["rl"].get("action_repeat") or 50))
    timeout = float(cfg["rl"].get("timeout_s") or 90)
    return int(timeout / (repeat * 0.001)) + 5


def _write_report(path: Path, races_path: Path, report: dict, track: str) -> None:
    report["track"] = track
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    races_path.write_text(
        json.dumps(report["chords"] + report["steers"], indent=2) + "\n",
        encoding="utf-8",
    )


def cmd_search(cfg: dict, args) -> int:
    from polytrack_agent.rl.hold import load_plan, run_plan, search_correction, search_hold
    from polytrack_agent.rl.track import make_track

    out = Path(args.out)
    max_steps = _max_steps(cfg)
    track = Path(cfg["game"]["track"]).stem
    env = make_track(cfg)
    try:
        if args.plan is not None and args.extend is None:
            plan = load_plan(Path(args.plan))
            episodes = max(1, int(args.episodes))
            for index in range(1, episodes + 1):
                row = run_plan(env, plan, max_steps)
                log.info(
                    "replay %s  progress %.3f  return %.2f  %.2fs  %s",
                    index,
                    row["progress"],
                    row["return"],
                    row["seconds"],
                    "fell" if row["fell"] else "up",
                )
            return 0
        if args.extend is not None:
            report = search_correction(env, load_plan(Path(args.extend)), max_steps=max_steps)
            races_name = "races-2.json"
        else:
            report = search_hold(env, max_steps=max_steps)
            races_name = "races.json"
    finally:
        env.close()
    chosen = report["chosen"]
    champion = report["champion"]
    kept = bool(report["kept_steer"])
    report_path = out / ("correction.json" if args.extend is not None else "plan.json")
    _write_report(report_path, out / races_name, report, track)
    if args.extend is not None and kept:
        _write_report(out / "plan.json", out / races_name, report, track)
    end = "fell" if chosen.get("fell") else "up"
    was = "fell" if champion.get("fell") else "up"
    log.info(
        "kept %s  progress %.3f  %s  (was %.3f %s)  wrote %s",
        report["plan"].get("steer3") or report["plan"].get("steer2") or report["plan"].get("steer") or report["plan"]["chord"],
        chosen["progress"],
        end,
        champion["progress"],
        was,
        out / "plan.json" if kept or args.extend is None else report_path,
    )
    return 0


def cmd_play(cfg: dict, args) -> int:
    from polytrack_agent.rl.episodes import run_episodes

    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else Path(cfg["play"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first.")
    episodes = 5 if args.episodes is None else int(args.episodes)
    rows = run_episodes(cfg, checkpoint, episodes=episodes)
    finished = [row["seconds"] for row in rows if row["finished"]]
    if finished:
        log.info("best lap %.2fs over %s finish(es)", min(finished), len(finished))
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
