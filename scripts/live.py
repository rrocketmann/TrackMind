#!/usr/bin/env python3
"""Draw the live graphs for a specialist run.

    python scripts/live.py --run runs/winter4-follow

The window tails that directory's episodes.csv and log.csv and redraws lap
progress, lap time, reference-lap fraction (or update progress), and entropy.
With no DISPLAY, the same
four panels are served on every network address of this machine, port 8765.
A chase-camera video of a saved checkpoint is scripts/record.py, played in
the local game.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
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


def _draw(run: Path, figure, axes, artists, panels) -> None:
    episodes = _rows(run / "episodes.csv")
    updates = _rows(run / "log.csv")
    reference = bool(updates) and "line" in updates[0]
    series = (
        _column(episodes, "progress"),
        _column(episodes, "seconds"),
        _column(updates, "line" if reference else "progress"),
        _column(updates, "entropy"),
    )
    if reference:
        axes[1, 0].set_title("Reference lap")
        axes[1, 0].set_ylabel("fraction")
    latest = []
    for (axis, raw, mean), values, panel in zip(artists, series, panels):
        averaged = _paint(raw, mean, values, panel[3])
        _fit(axis, averaged)
        latest.append(averaged[-1] if averaged else float("nan"))
    entropy = latest[3]
    entropy_text = f"{entropy:.2f}" if entropy == entropy else "—"
    stage = str(updates[-1].get("stage") or "") if updates else ""
    title = f"{len(episodes)} laps   {len(updates)} updates   entropy {entropy_text}"
    if stage:
        title += f"   {stage}"
    figure.suptitle(title, fontsize=12)
    return episodes, updates


def _build(run: Path):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(run.name)
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
    return figure, axes, artists, panels


def _http_server(host: str, port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    """Listen on every interface when host is '::', including IPv4."""
    if host in ("::", ""):
        class Dual(ThreadingHTTPServer):
            address_family = socket.AF_INET6

            def server_bind(self) -> None:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                super().server_bind()

        try:
            return Dual(("::", port), handler)
        except OSError:
            host = "0.0.0.0"
    return ThreadingHTTPServer((host, port), handler)


def _graph_urls(port: int) -> list[str]:
    """Pages a browser on another machine on this network can open."""
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        if url not in seen:
            seen.add(url)
            urls.append(url)

    add(f"http://127.0.0.1:{port}/")
    try:
        raw = subprocess.check_output(["ip", "-json", "addr"], text=True)
        ifaces = json.loads(raw)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        ifaces = []
    for iface in ifaces:
        if iface.get("operstate") == "DOWN":
            continue
        for item in iface.get("addr_info") or []:
            ip = str(item.get("local") or "")
            if not ip or ip.startswith("127.") or ip == "::1" or ip.startswith("fe80:"):
                continue
            if item.get("family") == "inet6":
                add(f"http://[{ip}]:{port}/")
            else:
                add(f"http://{ip}:{port}/")
    return urls


def _serve(run: Path, host: str, port: int) -> None:
    """Redraw the four panels once a second and serve the PNG."""
    figure, axes, artists, panels = _build(run)
    png = run / "live.png"
    page = (
        "<!doctype html><meta charset=utf-8>"
        "<title>Winter 4 training</title>"
        "<style>body{margin:0;background:#111;color:#eee;font:14px sans-serif}"
        "img{width:100%;height:auto;background:#fff}</style>"
        f"<p style='padding:8px 12px'>{run}</p>"
        "<img id=g alt='training graphs'>"
        "<script>const g=document.getElementById('g');"
        "function tick(){g.src='/live.png?t='+Date.now()} tick(); setInterval(tick,1000)</script>"
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body, kind = page, "text/html; charset=utf-8"
            elif path == "/live.png":
                body, kind = png.read_bytes() if png.is_file() else b"", "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = _http_server(host, port, Handler)
    for url in _graph_urls(port):
        print(f"graphs at {url}", flush=True)
    server.timeout = 0.2

    def redraw() -> None:
        _draw(run, figure, axes, artists, panels)
        figure.savefig(png, dpi=110)
        figure.canvas.draw_idle()

    redraw()
    drawn = time.monotonic()
    try:
        while True:
            server.handle_request()
            now = time.monotonic()
            if now - drawn >= 1.0:
                redraw()
                drawn = now
    except KeyboardInterrupt:
        return
    finally:
        server.server_close()
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live graphs for a specialist training run.")
    parser.add_argument("--run", type=Path, default=Path("runs/winter4-follow"))
    parser.add_argument("--host", default="::", help="bind address; '::' is every IPv4 and IPv6 interface")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    if matplotlib.get_backend().lower() == "agg":
        _serve(run, args.host, args.port)
        return

    plt.ion()
    figure, axes, artists, panels = _build(run)
    while plt.fignum_exists(figure.number):
        _draw(run, figure, axes, artists, panels)
        figure.canvas.draw_idle()
        plt.pause(1.0)


if __name__ == "__main__":
    main()
