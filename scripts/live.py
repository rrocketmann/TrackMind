#!/usr/bin/env python3
"""Draw the live graphs for a specialist run.

    python scripts/live.py --run runs/winter4-follow

The window tails that directory's episodes.csv and log.csv and redraws lap
progress, lap time, reference-lap fraction (or update progress), and entropy.
Each panel opens on the latest points, scaled to that stretch, and scrolls
back through the rest. With no DISPLAY, the page is served on every network
address of this machine, port 8765.
A chase-camera video of a saved checkpoint is scripts/record.py, played in
the local game.
"""

from __future__ import annotations

import argparse
import csv
import html
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


def _fit(axis, values: list[float], span: int) -> None:
    """Scale the saved image to the latest points. The page can scroll further back."""
    count = len(values)
    if count == 0:
        return
    start = max(0, count - max(1, int(span)))
    finite = [value for value in values[start:] if value == value]
    if not finite:
        return
    low = min(finite)
    high = max(finite)
    pad = (high - low) * 0.15 or 0.05
    axis.set_xlim(start + 0.5, count + 0.5)
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
        _fit(axis, averaged, panel[4])
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
        (axes[0, 0], "Lap progress", "#1f4b99", 12, 100),
        (axes[0, 1], "Lap time", "#0b6e4f", 12, 100),
        (axes[1, 0], "Update progress", "#9a3412", 5, 80),
        (axes[1, 1], "Entropy", "#1d4e89", 5, 80),
    )
    artists = []
    for axis, title, color, _window, _span in panels:
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


def _number(value: float) -> float | None:
    if value != value:
        return None
    return round(float(value), 4)


def _payload(run: Path) -> dict:
    """Full series for the page. `span` is how many recent points open in view."""
    episodes = _rows(run / "episodes.csv")
    updates = _rows(run / "log.csv")
    reference = bool(updates) and "line" in updates[0]
    specs = (
        ("Lap progress", "lap", "progress", "#1f4b99", 100, 12, _column(episodes, "progress")),
        ("Lap time", "lap", "seconds", "#0b6e4f", 100, 12, _column(episodes, "seconds")),
        (
            "Reference lap" if reference else "Update progress",
            "update",
            "fraction" if reference else "progress",
            "#9a3412",
            80,
            5,
            _column(updates, "line" if reference else "progress"),
        ),
        ("Entropy", "update", "entropy", "#1d4e89", 80, 5, _column(updates, "entropy")),
    )
    panels = []
    for title, xlabel, ylabel, color, span, smooth, values in specs:
        panels.append(
            {
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "color": color,
                "span": span,
                "y": [_number(value) for value in values],
                "mean": [_number(value) for value in _mean(values, smooth)],
            }
        )
    entropy = panels[3]["mean"][-1] if panels[3]["mean"] else None
    return {
        "run": str(run),
        "episodes": len(episodes),
        "updates": len(updates),
        "stage": str(updates[-1].get("stage") or "") if updates else "",
        "entropy": entropy,
        "panels": panels,
    }


_PAGE = """<!doctype html>
<meta charset=utf-8>
<title>__TITLE__</title>
<style>
  body { margin: 0; background: #111; color: #eee; font: 14px sans-serif; }
  header { padding: 10px 16px 0; }
  p { margin: 4px 0 0; color: #aaa; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px; }
  article { background: #fff; color: #222; border-radius: 8px; padding: 8px 10px 10px; }
  h2 { margin: 0 0 4px; font-size: 14px; }
  canvas { width: 100%; height: 220px; display: block; touch-action: none; }
  label { display: flex; gap: 8px; align-items: center; color: #555; font-size: 12px; }
  input[type=range] { flex: 1; }
  @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
</style>
<header>
  <strong id=meta>loading</strong>
  <p>Each graph opens on the latest points. Scroll a bar, or the wheel, to look further back.</p>
</header>
<main id=grid></main>
<script>
const charts = [];
function finite(values) {
  return values.filter((value) => value != null && value === value);
}
function draw(chart) {
  const panel = chart.panel;
  const canvas = chart.canvas;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const count = panel.y.length;
  const span = count === 0 ? 1 : Math.min(panel.span, count);
  const end = count === 0 ? 0 : Math.min(count, Math.max(span, chart.end || count));
  const start = Math.max(0, end - span);
  const left = 48, right = 12, top = 12, bottom = 26;
  const plotW = Math.max(1, width - left - right);
  const plotH = Math.max(1, height - top - bottom);
  ctx.strokeStyle = "#e4e4e4";
  ctx.strokeRect(left, top, plotW, plotH);
  chart.caption.textContent = count
    ? panel.xlabel + " " + (start + 1) + "–" + end + " of " + count
    : "no points yet";
  if (!count) return;
  const sliceY = panel.y.slice(start, end);
  const sliceM = panel.mean.slice(start, end);
  const vals = finite(sliceY.concat(sliceM));
  if (!vals.length) return;
  let low = Math.min.apply(null, vals);
  let high = Math.max.apply(null, vals);
  const pad = (high - low) * 0.15 || 0.05;
  low -= pad;
  high += pad;
  const xOf = (index) => left + ((index - start) / Math.max(1, end - start - 1)) * plotW;
  const yOf = (value) => top + (1 - (value - low) / (high - low)) * plotH;
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#666";
  ctx.textAlign = "right";
  ctx.fillText(high.toFixed(2), left - 6, top + 10);
  ctx.fillText(low.toFixed(2), left - 6, top + plotH);
  ctx.textAlign = "left";
  ctx.fillText(panel.ylabel, left + 6, top + 12);
  function stroke(series, alpha, widthPx) {
    ctx.beginPath();
    let moved = false;
    series.forEach((value, index) => {
      if (value == null || value !== value) { moved = false; return; }
      const x = xOf(start + index);
      const y = yOf(value);
      if (!moved) { ctx.moveTo(x, y); moved = true; }
      else ctx.lineTo(x, y);
    });
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = panel.color;
    ctx.lineWidth = widthPx;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  stroke(sliceY, 0.35, 1.2);
  stroke(sliceM, 1, 2.3);
}
function build(panel) {
  const article = document.createElement("article");
  const title = document.createElement("h2");
  title.textContent = panel.title;
  const canvas = document.createElement("canvas");
  const label = document.createElement("label");
  const caption = document.createElement("span");
  const range = document.createElement("input");
  range.type = "range";
  range.min = "1";
  range.value = "1";
  label.append(range, caption);
  article.append(title, canvas, label);
  document.getElementById("grid").append(article);
  const chart = { panel, canvas, range, caption, end: null, count: 0 };
  range.addEventListener("input", () => {
    chart.end = Number(range.value);
    draw(chart);
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const span = chart.count === 0 ? 1 : Math.min(chart.panel.span, chart.count);
    const step = Math.max(1, Math.round(span / 8));
    const next = (chart.end || chart.count) + (event.deltaY > 0 ? -step : step);
    chart.end = Math.max(span, Math.min(chart.count || span, next));
    range.value = String(chart.end);
    draw(chart);
  }, { passive: false });
  charts.push(chart);
}
async function tick() {
  const data = await fetch("/data.json", { cache: "no-store" }).then((response) => response.json());
  const entropy = data.entropy == null ? "—" : Number(data.entropy).toFixed(2);
  document.getElementById("meta").textContent =
    data.episodes + " laps   " + data.updates + " updates   entropy " + entropy +
    (data.stage ? "   " + data.stage : "");
  if (!charts.length) data.panels.forEach(build);
  data.panels.forEach((panel, index) => {
    const chart = charts[index];
    const follow = chart.end == null || chart.end >= chart.count;
    chart.panel = panel;
    chart.count = panel.y.length;
    const span = chart.count === 0 ? 1 : Math.min(panel.span, chart.count);
    chart.range.min = String(span);
    chart.range.max = String(Math.max(chart.count, span));
    chart.range.disabled = chart.count <= span;
    if (follow) chart.end = chart.count || span;
    chart.range.value = String(chart.end);
    draw(chart);
  });
}
tick();
setInterval(tick, 1000);
window.addEventListener("resize", () => charts.forEach(draw));
</script>
"""


def _page(run: Path) -> bytes:
    title = html.escape(run.name)
    return _PAGE.replace("__TITLE__", title).encode()


def _serve(run: Path, host: str, port: int) -> None:
    """Redraw the four panels once a second and serve a scrollable page."""
    figure, axes, artists, panels = _build(run)
    png = run / "live.png"
    page = _page(run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body, kind = page, "text/html; charset=utf-8"
            elif path == "/data.json":
                body = json.dumps(_payload(run), allow_nan=False).encode()
                kind = "application/json"
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
