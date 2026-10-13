"""Command line for calibrate, record, train, reference, and play."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

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
        description="Vision-based agent for the official PolyTrack window. Pixels in, keys out.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", parents=[common], help="Grab one frame and print a capture region")
    calibrate.add_argument("--preview", type=Path, default=Path("shots/preview.png"))
    calibrate.add_argument("--write", type=Path, default=None, help="Write a partial config with a fixed region")
    calibrate.set_defaults(func=cmd_calibrate)

    record = sub.add_parser("record", parents=[common], help="Record frames and your key holds")
    record.add_argument("--out", type=Path, default=None, help="Directory that will contain session-* folders")
    record.set_defaults(func=cmd_record)

    train = sub.add_parser("train", parents=[common], help="Train the GRPO policy in the Chromium window")
    train.add_argument("--reference", type=Path, default=None, help="reference.npz for the track, default play.reference")
    train.add_argument("--out", type=Path, default=None)
    train.add_argument("--device", default=None)
    train.set_defaults(func=cmd_train)

    reference = sub.add_parser("reference", parents=[common], help="Build a visual lap reference")
    reference.add_argument("--session", type=Path, required=True)
    reference.add_argument("--out", type=Path, required=True)
    reference.add_argument("--episode", type=int, default=None)
    reference.set_defaults(func=cmd_reference)

    play = sub.add_parser("play", parents=[common], help="Drive, or score a recording offline")
    play.add_argument("--checkpoint", type=Path, default=None)
    play.add_argument("--dry-run", action="store_true", help="Predict keys and press nothing")
    play.add_argument("--offline", type=Path, default=None, help="Score this recorded session. Does not open the game.")
    play.add_argument("--max-frames", type=int, default=0)
    play.add_argument("--reference", type=Path, default=None)
    play.add_argument("--no-log", action="store_true")
    play.add_argument("--episodes", type=int, default=None, help="GRPO eval episodes. Default 5.")
    play.set_defaults(func=cmd_play)
    return parser


def cmd_calibrate(cfg: dict, args) -> int:
    from PIL import Image

    from polytrack_agent.capture.factory import make_capture
    from polytrack_agent.capture.window import find_window, list_monitors

    try:
        for index, mon in enumerate(list_monitors()):
            log.info("monitor %s: %s,%s %sx%s", index, mon.left, mon.top, mon.width, mon.height)
    except Exception as exc:
        log.warning("Could not list monitors: %s", exc)
    title = str(cfg["capture"].get("window_title") or "PolyTrack")
    found = find_window(title, str(cfg["capture"].get("browser") or "") or None)
    if found is None:
        log.info("No window titled %r yet. The preview is whatever resolve_region captures.", title)
    else:
        log.info("Window %r at %s,%s %sx%s", title, found.left, found.top, found.width, found.height)

    capture = make_capture(cfg)
    try:
        frame = capture.grab()
    finally:
        region = capture.region
        capture.close()
    if frame is None:
        raise RuntimeError("Capture returned no frame. Is a display connected?")
    preview = Path(args.preview)
    preview.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGB").save(preview)
    log.info("Wrote %s (%sx%s)", preview, frame.shape[1], frame.shape[0])
    snippet = {
        "capture": {
            "region": {
                "left": int(region.left),
                "top": int(region.top),
                "width": int(region.width),
                "height": int(region.height),
            },
            "follow_window": False,
        }
    }
    text = yaml.safe_dump(snippet, sort_keys=False)
    log.info("Fixed region for competitive play (stops window-lookup jitter):\n%s", text)
    if args.write is not None:
        Path(args.write).write_text(text, encoding="utf-8")
        log.info("Wrote %s. Use --config %s on record and play.", args.write, args.write)
    return 0


def cmd_record(cfg: dict, args) -> int:
    from polytrack_agent.record.recorder import record_session

    record_session(cfg, out_root=args.out)
    return 0


def cmd_train(cfg: dict, args) -> int:
    from polytrack_agent.rl.trainer import train_grpo

    if args.device is not None:
        cfg["rl"]["device"] = args.device
    if args.out is not None:
        cfg["rl"]["out_dir"] = str(args.out)
    reference = args.reference
    if reference is None and cfg["play"].get("reference"):
        reference = Path(cfg["play"]["reference"])
    train_grpo(cfg, reference)
    return 0


def cmd_reference(cfg: dict, args) -> int:
    from polytrack_agent.obs.progress import build_reference_from_session, save_reference

    progress = cfg["progress"]
    strip, info = build_reference_from_session(
        Path(args.session),
        downsample=(int(progress["downsample"][0]), int(progress["downsample"][1])),
        stride=int(progress.get("stride") or 1),
        max_frames=int(progress.get("max_frames") or 4000),
        episode=args.episode,
    )
    save_reference(Path(args.out), strip, info)
    log.info(
        "Wrote %s from episode %s (%s pictures, stride %s)",
        args.out,
        info["episode"],
        info["frames"],
        info["stride"],
    )
    log.info("Point play.reference at that file, or pass --reference when you play.")
    return 0


def cmd_play(cfg: dict, args) -> int:
    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else Path(cfg["play"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first.")
    if args.offline is not None:
        from polytrack_agent.eval_offline import evaluate_session, format_report

        stats = evaluate_session(
            checkpoint,
            Path(args.offline),
            device=str(cfg["play"].get("device") or "auto"),
            max_frames=int(args.max_frames or 0),
            threshold_on=float(cfg["play"]["threshold_on"]),
            threshold_off=float(cfg["play"]["threshold_off"]),
            max_gap_s=float(cfg["record"].get("max_gap_s") or 0.25),
        )
        print(format_report(stats))
        return 0 if stats["steer_f1"] >= 0.6 else 3

    import torch

    from polytrack_agent.rl.checkpoint import ARCH, read_checkpoint

    payload = read_checkpoint(checkpoint, torch.device("cpu"))
    reference = args.reference
    if reference is None and cfg["play"].get("reference"):
        reference = Path(cfg["play"]["reference"])
    log_path = None if args.no_log else Path(cfg["play"]["log"])
    if payload.get("arch") == ARCH:
        from polytrack_agent.rl.episodes import run_episodes

        episodes = 5 if args.episodes is None else int(args.episodes)
        run_episodes(
            cfg,
            checkpoint=checkpoint,
            episodes=episodes,
            dry_run=bool(args.dry_run),
            reference=reference,
            log_path=log_path,
        )
        return 0

    from polytrack_agent.play_loop import play_live

    play_live(
        cfg,
        checkpoint=checkpoint,
        dry_run=bool(args.dry_run),
        reference=reference,
        log_path=log_path,
    )
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
