#!/usr/bin/env python3
"""Draw the live graphs for a specialist run.

    python scripts/live.py --run runs/winter4

The window tails that directory's episodes.csv and log.csv and redraws lap
progress, lap time, update progress, and entropy. A chase-camera video of a
saved checkpoint is scripts/record.py, played in the local game.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    try:
        return list(csv.DictReader(lines))
    except csv.Error:
        return []


def _column(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            values.append(float("nan"))
    return values


def _mean(values: list[float], window: int) -> list[float]:
    """Trailing mean. A gap in the raw series does not break the average."""
    averaged = []
    for index in range(len(values)):
        chunk = [value for value in values[max(0, index - window + 1) : index + 1] if value == value]
        averaged.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return averaged


def _fit(axis, values: list[float]) -> None:
    finite = [value for value in values if value == value]
    count = len(values)
    if count == 0 or not finite:
        return
    low = min(finite)
    high = max(finite)
    pad = (high - low) * 0.15 or 0.05
    axis.set_xlim(0.5, count + 0.5)
    axis.set_ylim(low - pad, high + pad)


def _paint(raw_line, mean_line, values: list[float], window: int) -> list[float]:
    xs = list(range(1, len(values) + 1))
    averaged = _mean(values, window)
    raw_line.set_data(xs, values)
    mean_line.set_data(xs, averaged)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Live graphs for a specialist training run.")
    parser.add_argument("--run", type=Path, default=Path("runs/winter4"))
    args = parser.parse_args()
    run = args.run
    episodes_path = run / "episodes.csv"
    updates_path = run / "log.csv"

    plt.ion()
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    figure.canvas.manager.set_window_title("Winter 4 training")
    panels = (
        (axes[0, 0], "Lap progress", "#1f4b99", 12),
        (axes[0, 1], "Lap time", "#0b6e4f", 12),
        (axes[1, 0], "Update progress", "#9a3412", 5),
        (axes[1, 1], "Entropy", "#1d4e89", 5),
    )
    artists = []
    for axis, title, color, _window in panels:
        raw, = axis.plot([], [], color=color, linewidth=1.0, alpha=0.28)
        mean, = axis.plot([], [], color=color, linewidth=2.2)
        axis.set_title(title)
        axis.grid(True, color="#d9d9d9", linewidth=0.6)
        axis.tick_params(labelsize=8)
        artists.append((axis, raw, mean))
    axes[0, 0].set_ylabel("progress")
    axes[0, 1].set_ylabel("seconds")
    axes[1, 0].set_xlabel("update")
    axes[1, 0].set_ylabel("progress")
    axes[1, 1].set_xlabel("update")
    axes[1, 1].set_ylabel("entropy")
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    while plt.fignum_exists(figure.number):
        episodes = _rows(episodes_path)
        updates = _rows(updates_path)
        series = (
            _column(episodes, "progress"),
            _column(episodes, "seconds"),
            _column(updates, "progress"),
            _column(updates, "entropy"),
        )
        latest = []
        for (axis, raw, mean), values, panel in zip(artists, series, panels):
            averaged = _paint(raw, mean, values, panel[3])
            _fit(axis, averaged)
            latest.append(averaged[-1] if averaged else float("nan"))
        entropy = latest[3]
        entropy_text = f"{entropy:.2f}" if entropy == entropy else "—"
        figure.suptitle(
            f"{len(episodes)} laps   {len(updates)} updates   entropy {entropy_text}",
            fontsize=12,
        )
        figure.canvas.draw_idle()
        plt.pause(1.0)


if __name__ == "__main__":
    main()
