"""Train the vision CNN on a synthetic video and check that it imitates steering."""

from __future__ import annotations

from polytrack_agent.config import load_config
from polytrack_agent.eval_offline import evaluate_session
from polytrack_agent.synthetic import write_synthetic_session
from polytrack_agent.train.bc import train_behavior_cloning


def test_synthetic_vision_learns_steering(tmp_path):
    cfg = load_config(None)
    session = write_synthetic_session(tmp_path / "session", keys=list(cfg["record"]["keys"]), n=200)
    cfg["train"]["backbone"] = "cnn"
    cfg["train"]["image_size"] = [48, 80]
    cfg["train"]["stack"] = 4
    cfg["train"]["batch_size"] = 32
    cfg["train"]["epochs"] = 6
    cfg["train"]["lr"] = 1e-3
    cfg["train"]["dropout"] = 0.0
    cfg["train"]["ema_decay"] = 0.0
    cfg["train"]["augment"] = False
    cfg["train"]["sampler"] = "shuffle"
    cfg["train"]["device"] = "cpu"
    cfg["train"]["patience"] = 10
    cfg["train"]["out_dir"] = str(tmp_path / "run")
    cfg["train"]["num_workers"] = 0
    report = train_behavior_cloning(cfg, [session])
    assert report["best_score"] > 0.7, report
    stats = evaluate_session(tmp_path / "run" / "best.pt", session, device="cpu")
    assert stats["per_key"]["a"]["accuracy"] > 0.8, stats
    assert stats["per_key"]["d"]["accuracy"] > 0.8, stats
    assert stats["per_key"]["q"]["accuracy"] > 0.95, stats
