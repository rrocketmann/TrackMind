// Play a greedy key timeline in the local PolyTrack 0.6.3 client and save an mp4.
//
// The page is the checkout in POLYTRACK_ROOT, served on 127.0.0.1. Fetch blocks
// every other host before navigation, so this does not open an external page and
// it does not submit a time. The timeline comes from scripts/record.py. It is a
// fresh greedy playback of a checkpoint, not the original training episode.
//
// A real display and GPU are required. The game clock follows requestAnimationFrame
// and does not catch up, so a software renderer (Xvfb, SwiftShader) is rejected
// when it stays under 25 fps. This machine's ffmpeg has no libx264. The encode
// uses libopenh264, then mpeg4.

import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import net from "node:net";
import { execFileSync, spawn, spawnSync } from "node:child_process";

const drivePath = process.argv[2];
const outPath = process.argv[3];
const trackName = process.argv[4];

if (!drivePath || !outPath || !trackName) {
  console.error("usage: node scripts/record_game.mjs <drive.json> <out.mp4> <track name>");
  process.exit(1);
}
if (!process.env.DISPLAY) {
  console.error("Recording needs DISPLAY and a real GPU. A virtual framebuffer renders too slowly for a real-time lap.");
  process.exit(1);
}

const root = path.resolve(process.env.POLYTRACK_ROOT || "");
if (!process.env.POLYTRACK_ROOT) {
  console.error("Set POLYTRACK_ROOT to the polytrack fork directory.");
  process.exit(1);
}
if (!fs.existsSync(path.join(root, "index.html"))) {
  console.error(`${root} has no index.html.`);
  process.exit(1);
}

const drive = JSON.parse(fs.readFileSync(drivePath, "utf8"));
if (!Array.isArray(drive.steps) || drive.steps.length === 0) {
  console.error("The drive has no steps.");
  process.exit(1);
}
const holdS = Number(drive.hold_s) || 0.05;

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const names = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"];
  for (const name of names) {
    try {
      const found = execFileSync("which", [name], { encoding: "utf8" }).trim();
      if (found) return found;
    } catch {
      // try the next name
    }
  }
  const home = os.homedir();
  const candidates = [path.join(home, ".cache/ms-playwright")];
  if (process.platform === "darwin") {
    candidates.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
    candidates.push("/Applications/Chromium.app/Contents/MacOS/Chromium");
  }
  if (process.platform === "win32") {
    const local = process.env.LOCALAPPDATA || "";
    const programFiles = process.env.PROGRAMFILES || "C:\\Program Files";
    candidates.push(path.join(programFiles, "Google/Chrome/Application/chrome.exe"));
    candidates.push(path.join(local, "Google/Chrome/Application/chrome.exe"));
  }
  for (const candidate of candidates) {
    if (candidate.endsWith("chrome") || candidate.endsWith("Chromium") || candidate.endsWith("chrome.exe")) {
      if (fs.existsSync(candidate)) return candidate;
      continue;
    }
    if (!fs.existsSync(candidate)) continue;
    const hits = fs.readdirSync(candidate)
      .filter((name) => name.startsWith("chromium-"))
      .map((name) => path.join(candidate, name, "chrome-linux64/chrome"))
      .filter((file) => fs.existsSync(file))
      .sort();
    if (hits.length) return hits[hits.length - 1];
  }
  throw new Error("Set CHROME to a Chromium or Chrome binary.");
}

function contentType(file) {
  if (file.endsWith(".js") || file.endsWith(".mjs")) return "text/javascript";
  if (file.endsWith(".wasm")) return "application/wasm";
  if (file.endsWith(".json")) return "application/json";
  if (file.endsWith(".html")) return "text/html";
  if (file.endsWith(".css")) return "text/css";
  if (file.endsWith(".svg")) return "image/svg+xml";
  if (file.endsWith(".woff2")) return "font/woff2";
  if (file.endsWith(".jpg") || file.endsWith(".jpeg")) return "image/jpeg";
  if (file.endsWith(".png")) return "image/png";
  if (file.endsWith(".glb")) return "model/gltf-binary";
  if (file.endsWith(".ogg")) return "audio/ogg";
  if (file.endsWith(".mp3")) return "audio/mpeg";
  return "application/octet-stream";
}

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close((error) => (error ? reject(error) : resolve(port)));
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isLocal(url) {
  try {
    const parsed = new URL(url);
    return parsed.hostname === "127.0.0.1" || parsed.hostname === "localhost";
  } catch {
    return false;
  }
}

const KEYS = {
  w: { key: "w", code: "KeyW", vk: 87 },
  a: { key: "a", code: "KeyA", vk: 65 },
  s: { key: "s", code: "KeyS", vk: 83 },
  d: { key: "d", code: "KeyD", vk: 68 },
  up: { key: "ArrowUp", code: "ArrowUp", vk: 38 },
  down: { key: "ArrowDown", code: "ArrowDown", vk: 40 },
  left: { key: "ArrowLeft", code: "ArrowLeft", vk: 37 },
  right: { key: "ArrowRight", code: "ArrowRight", vk: 39 },
};

const chrome = findChrome();

const server = http.createServer((req, res) => {
  const url = new URL(req.url || "/", "http://127.0.0.1");
  const rel = decodeURIComponent(url.pathname).replace(/^\/+/, "");
  const file = path.resolve(root, rel === "" ? "index.html" : rel);
  const inside = file === root || file.startsWith(root + path.sep);
  if (!inside || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404);
    res.end("missing");
    return;
  }
  res.writeHead(200, { "content-type": contentType(file) });
  fs.createReadStream(file).pipe(res);
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const pagePort = server.address().port;
const debugPort = await freePort();
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "polytrack-record-"));
const child = spawn(chrome, [
  "--no-sandbox",
  "--disable-setuid-sandbox",
  "--ignore-gpu-blocklist",
  "--disable-gpu-sandbox",
  "--enable-webgl",
  "--enable-gpu",
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--disable-software-rasterizer",
  "--force-device-scale-factor=1",
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-dev-shm-usage",
  "--disable-sync",
  "--disable-background-timer-throttling",
  "--disable-backgrounding-occluded-windows",
  "--disable-renderer-backgrounding",
  "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling",
  "--autoplay-policy=no-user-gesture-required",
  `--remote-debugging-port=${debugPort}`,
  `--user-data-dir=${profile}`,
  "--window-position=80,40",
  "--window-size=1600,960",
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });

let chromeLog = "";
child.stderr.on("data", (chunk) => {
  chromeLog = (chromeLog + chunk.toString()).slice(-4000);
});

let frameDir = "";
let ws;

function fail(error) {
  console.error(error.message || error);
  if (chromeLog.trim()) console.error(chromeLog.trim());
  cleanup();
  process.exit(1);
}

function cleanup() {
  try { child.kill("SIGKILL"); } catch { /* already gone */ }
  try { if (ws) ws.close(); } catch { /* already gone */ }
  try { server.close(); } catch { /* already gone */ }
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch { /* already gone */ }
  if (frameDir) {
    try { fs.rmSync(frameDir, { recursive: true, force: true }); } catch { /* already gone */ }
  }
}

process.on("SIGINT", () => fail(new Error("interrupted")));
process.on("SIGTERM", () => fail(new Error("interrupted")));

async function browserSocket() {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${debugPort}/json/version`);
      const info = await response.json();
      if (info.webSocketDebuggerUrl) return info.webSocketDebuggerUrl;
    } catch {
      // Chrome is still starting.
    }
    await sleep(200);
  }
  throw new Error("Chrome did not open a debugging port");
}

try {
  ws = new WebSocket(await browserSocket());
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", () => reject(new Error("debugging socket failed")), { once: true });
  });

  let nextId = 0;
  const pending = new Map();
  const shots = [];
  let recording = false;

  function send(method, params = {}, sessionId) {
    const id = ++nextId;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      pending.set(id, (message) => {
        if (message.error) reject(new Error(`${method}: ${message.error.message}`));
        else resolve(message);
      });
      ws.send(JSON.stringify(payload));
    });
  }

  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data.toString());
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
      return;
    }
    if (message.method === "Fetch.requestPaused") {
      const paused = message.params;
      const local = isLocal(paused.request.url);
      const method = local ? "Fetch.continueRequest" : "Fetch.failRequest";
      const params = local
        ? { requestId: paused.requestId }
        : { requestId: paused.requestId, errorReason: "BlockedByClient" };
      send(method, params, message.sessionId).catch(() => {});
      if (!local) console.log(`blocked ${paused.request.method} ${paused.request.url}`);
      return;
    }
    if (message.method === "Page.screencastFrame") {
      const frame = message.params;
      send("Page.screencastFrameAck", { sessionId: frame.sessionId }, message.sessionId).catch(() => {});
      if (recording) {
        const stamp = Number(frame.metadata && frame.metadata.timestamp);
        shots.push({
          buf: Buffer.from(frame.data, "base64"),
          t: Number.isFinite(stamp) ? stamp : performance.now() / 1000,
        });
      }
    }
  });

  const targets = await send("Target.getTargets");
  const page = targets.result.targetInfos.find((item) => item.type === "page");
  if (!page) throw new Error("Chrome opened no page");
  const attached = await send("Target.attachToTarget", { targetId: page.targetId, flatten: true });
  const sessionId = attached.result.sessionId;

  async function placeWindow() {
    const found = await send("Browser.getWindowForTarget", { targetId: page.targetId });
    await send("Browser.setWindowBounds", {
      windowId: found.result.windowId,
      bounds: { left: 80, top: 40, width: 1600, height: 960, windowState: "normal" },
    });
  }

  async function evaluate(expression, awaitPromise = false) {
    const response = await send("Runtime.evaluate", {
      expression,
      awaitPromise,
      returnByValue: true,
    }, sessionId);
    const details = response.result && response.result.exceptionDetails;
    if (details) {
      const text = (details.exception && details.exception.description) || details.text || "evaluate failed";
      throw new Error(text);
    }
    return response.result && response.result.result ? response.result.result.value : undefined;
  }

  async function waitFor(expression, timeoutMs, label) {
    const deadline = Date.now() + timeoutMs;
    let lastError = null;
    while (Date.now() < deadline) {
      try {
        if (await evaluate(expression)) return;
      } catch (error) {
        lastError = error;
      }
      await sleep(200);
    }
    const extra = lastError ? `: ${lastError.message}` : "";
    throw new Error(`timed out waiting for ${label}${extra}`);
  }

  // Block non-local requests before the game page loads.
  await send("Page.enable", {}, sessionId);
  await send("Runtime.enable", {}, sessionId);
  await send("Fetch.enable", { patterns: [{ urlPattern: "*" }] }, sessionId);
  await placeWindow();
  await send("Page.navigate", { url: `http://127.0.0.1:${pagePort}/index.html` }, sessionId);

  const menuPlay = `(() => Boolean([...document.querySelectorAll("button")].find((node) => node.innerText.trim() === "Play")))()`;
  await waitFor(menuPlay, 30000, "the menu Play button");
  // The track list stays empty if Play is clicked as soon as the button exists.
  await sleep(8000);
  await send("Page.bringToFront", {}, sessionId);
  const opened = await evaluate(`(() => {
    const button = [...document.querySelectorAll("button")].find((node) => node.innerText.trim() === "Play");
    if (!button) return false;
    button.click();
    return true;
  })()`);
  if (!opened) throw new Error("the menu Play button disappeared");

  const quoted = JSON.stringify(trackName);
  await waitFor(`(() => Boolean([...document.querySelectorAll(".track-title p")].find((node) => node.textContent.trim() === ${quoted})))()`, 20000, trackName);
  const picked = await evaluate(`(() => {
    const title = [...document.querySelectorAll(".track-title p")].find((node) => node.textContent.trim() === ${quoted});
    const button = title && title.closest("button");
    if (!button) return false;
    button.click();
    return true;
  })()`);
  if (!picked) throw new Error(`could not open ${trackName}`);

  await waitFor(`(() => Boolean(document.querySelector(".track-info .button.play")))()`, 20000, "the track Play button");
  const started = await evaluate(`(() => {
    const button = document.querySelector(".track-info .button.play");
    if (!button) return false;
    button.click();
    return true;
  })()`);
  if (!started) throw new Error("the track Play button disappeared");
  await waitFor(`(() => Boolean(document.querySelector(".timer") && !document.querySelector(".track-info")))()`, 30000, "the race timer");
  await sleep(1200);

  const renderer = await evaluate(`(() => {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl");
    if (!gl) return "no webgl";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  })()`);
  const fps = await evaluate(`new Promise((resolve) => {
    let frames = 0;
    const start = performance.now();
    function tick() {
      frames += 1;
      if (performance.now() - start >= 1000) resolve(frames);
      else requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  })`, true);
  if (fps < 25) {
    throw new Error(`game is rendering at ${fps} fps, the lap would not play in real time`);
  }

  const point = await evaluate(`(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
  })()`);
  if (!point) throw new Error("the race has no canvas");
  await send("Input.dispatchMouseEvent", {
    type: "mousePressed", x: point.x, y: point.y, button: "left", clickCount: 1,
  }, sessionId);
  await send("Input.dispatchMouseEvent", {
    type: "mouseReleased", x: point.x, y: point.y, button: "left", clickCount: 1,
  }, sessionId);
  await placeWindow();
  await send("Page.startScreencast", {
    format: "jpeg",
    quality: 80,
    maxWidth: 1600,
    maxHeight: 900,
    everyNthFrame: 1,
  }, sessionId);
  recording = true;

  let held = new Set();
  async function hold(keys) {
    const next = new Set(keys);
    for (const key of held) {
      if (!next.has(key)) await press("keyUp", key);
    }
    for (const key of next) {
      if (!held.has(key)) await press("keyDown", key);
    }
    held = next;
  }
  function press(type, name) {
    const spec = KEYS[name];
    if (!spec) throw new Error(`no key mapping for ${name}`);
    return send("Input.dispatchKeyEvent", {
      type,
      key: spec.key,
      code: spec.code,
      windowsVirtualKeyCode: spec.vk,
      nativeVirtualKeyCode: spec.vk,
    }, sessionId);
  }

  const clock = performance.now();
  for (const step of drive.steps) {
    const delay = clock + Number(step.t) * 1000 - performance.now();
    if (delay > 2) await sleep(delay);
    await hold(Array.isArray(step.keys) ? step.keys : []);
  }
  const last = drive.steps[drive.steps.length - 1];
  const tail = clock + (Number(last.t) + holdS) * 1000 - performance.now();
  if (tail > 2) await sleep(tail);
  await hold([]);
  await sleep(600);
  recording = false;
  await send("Page.stopScreencast", {}, sessionId);

  const hud = await evaluate(`(() => {
    const timer = document.querySelector(".timer");
    return timer ? timer.innerText.replace(/\\s+/g, " ").trim() : "";
  })()`);
  if (shots.length < 30) throw new Error(`only captured ${shots.length} frames`);

  frameDir = fs.mkdtempSync(path.join(os.tmpdir(), "polytrack-frames-"));
  const lines = ["ffconcat version 1.0"];
  for (let index = 0; index < shots.length; index += 1) {
    const name = `frame-${String(index).padStart(5, "0")}.jpg`;
    fs.writeFileSync(path.join(frameDir, name), shots[index].buf);
    const next = index + 1 < shots.length ? shots[index + 1].t : shots[index].t + 0.04;
    const span = next > shots[index].t ? next - shots[index].t : 0.04;
    lines.push(`file ${name}`);
    lines.push(`duration ${Math.max(0.01, span).toFixed(3)}`);
  }
  const lastName = `frame-${String(shots.length - 1).padStart(5, "0")}.jpg`;
  lines.push(`file ${lastName}`);
  const listFile = path.join(frameDir, "list.txt");
  fs.writeFileSync(listFile, `${lines.join("\n")}\n`);
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });

  const attempts = [
    ["-c:v", "libopenh264", "-b:v", "8M"],
    ["-c:v", "mpeg4", "-q:v", "3"],
  ];
  let encoded = "";
  let ffmpegLog = "";
  for (const codec of attempts) {
    const result = spawnSync("ffmpeg", [
      "-y", "-hide_banner", "-loglevel", "error",
      "-f", "concat", "-safe", "0", "-i", listFile,
      "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
      ...codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
      path.resolve(outPath),
    ], { encoding: "utf8" });
    if (result.status === 0) {
      encoded = codec[1];
      break;
    }
    ffmpegLog = `${result.stderr || ""}${result.stdout || ""}`.slice(-1500);
  }
  if (!encoded) throw new Error(`ffmpeg failed to encode the lap\n${ffmpegLog}`);

  console.log(`renderer: ${renderer}`);
  console.log(`fps: ${fps}`);
  console.log(`hud: ${hud}`);
  console.log(`frames: ${shots.length}`);
  console.log(`encoder: ${encoded}`);
  console.log(path.resolve(outPath));
  cleanup();
} catch (error) {
  fail(error);
}
