// Record the real PolyTrack (kodub.com/apps/polytrack) into an MP4.
//
// Runs Chromium on a virtual X display (Xvfb) and grabs that display with ffmpeg,
// so it works from an SSH session with no physical monitor. The car is driven
// by holding W (throttle); steer manually or extend the drive loop below.
//
// Prerequisites:
//   - google-chrome / chromium / chrome binary (set CHROME if not on PATH)
//   - ffmpeg, xvfb-run, Xvfb  (all present on this machine)
//
// Usage:
//   CHROME=/usr/bin/google-chrome node scripts/record.mjs runs/real_polytrack.mp4
//
// Press Ctrl+C to stop recording.

import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync, spawn } from "node:child_process";

const URL = process.env.POLYTRACK_URL || "https://www.kodub.com/apps/polytrack";
const OUT = process.argv[2] || path.resolve("runs/real_polytrack.mp4");
const DISPLAY = ":99";
const WIDTH = 1280;
const HEIGHT = 720;
const FPS = 30;

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
  const candidates = [
    path.join(home, ".cache/ms-playwright"),
  ];
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
    if (!fs.existsSync(candidate)) continue;
    if (candidate.endsWith("chrome") || candidate.endsWith("Chromium") || candidate.endsWith("chrome.exe")) return candidate;
    const hits = fs.readdirSync(candidate)
      .filter((name) => name.startsWith("chromium-"))
      .map((name) => path.join(candidate, name, "chrome-linux64/chrome"))
      .filter((file) => fs.existsSync(file))
      .sort();
    if (hits.length) return hits[hits.length - 1];
  }
  throw new Error("Set CHROME to a Chromium or Chrome binary.");
}

const chrome = findChrome();
fs.mkdirSync(path.dirname(OUT), { recursive: true });

let xvfb, ffmpeg, browser;
let ws;
let nextId = 0;
const pending = new Map();

function send(method, params = {}, sessionId) {
  const id = ++nextId;
  const payload = { id, method, params };
  if (sessionId) payload.sessionId = sessionId;
  ws.send(JSON.stringify(payload));
  return new Promise((resolve) => pending.set(id, resolve));
}

function cleanup() {
  console.log("\nStopping recorder...");
  // Close the WebDriver key first so the car stops, then shut down ffmpeg
  // cleanly so it can finalize the MP4 moov atom.
  if (ws && sessionId) {
    try {
      send("Input.dispatchKeyEvent", {
        type: "keyUp",
        key: "w",
        code: "KeyW",
        windowsVirtualKeyCode: 87,
        nativeVirtualKeyCode: 87,
      }, sessionId);
    } catch {}
  }
  for (const proc of [browser, ffmpeg, xvfb]) {
    if (proc && !proc.killed) {
      try { proc.kill("SIGTERM"); } catch {}
    }
  }
  if (ws) ws.close();
  console.log("Recorder stopped.");
}

process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);

(async () => {
  // 1. Virtual display.
  console.log(`Starting Xvfb on ${DISPLAY} ...`);
  xvfb = spawn("Xvfb", [DISPLAY, "-screen", "0", `${WIDTH}x${HEIGHT}x24`, "-nolisten", "tcp"], {
    stdio: "ignore",
    env: { ...process.env },
  });
  xvfb.on("exit", (code) => console.log(`Xvfb exited with ${code}`));

  // Wait for Xvfb to be ready (X socket must exist).
  console.log(`Waiting for Xvfb display ${DISPLAY} ...`);
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      fs.accessSync(`/tmp/.X11-unix/X${DISPLAY.replace(":", "")}`);
      break;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }

  // 2. ffmpeg screen grab of the virtual display.
  console.log(`Recording to ${OUT} ...`);
  ffmpeg = spawn("ffmpeg", [
    "-f", "x11grab",
    "-framerate", String(FPS),
    "-video_size", `${WIDTH}x${HEIGHT}`,
    "-i", DISPLAY + ".0",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "23",
    "-movflags", "frag_keyframe+empty_moov",
    "-y",
    OUT,
  ], {
    stdio: ["ignore", "ignore", "pipe"],
    env: { ...process.env, DISPLAY },
  });
  ffmpeg.stderr.on("data", (chunk) => {
    // ffmpeg logs progress to stderr; keep it quiet unless it errors.
    const text = chunk.toString();
    if (text.includes("Error") || text.includes("error")) console.error(text.trim());
  });

  // 3. Chromium on the virtual display, headed so it renders.
  //    PolyTrack is a WebGL game: --disable-gpu breaks it. Use swiftshader
  //    software rendering instead so WebGL works on a headless Xvfb display.
  const debugPort = 9222;
  console.log(`Launching Chromium at ${URL} ...`);
  browser = spawn(chrome, [
    `--display=${DISPLAY}`,
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-dev-shm-usage",
    `--remote-debugging-port=${debugPort}`,
    "--window-size=" + WIDTH + "," + HEIGHT,
    URL,
  ], {
    stdio: ["ignore", "ignore", "pipe"],
    env: { ...process.env, DISPLAY },
  });
  browser.stderr.on("data", (chunk) => {
    const text = chunk.toString().trim();
    if (text) console.error(`[chrome] ${text}`);
  });

  // Wait for the debugging port.
  let debugUrl;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const res = await fetch(`http://127.0.0.1:${debugPort}/json/version`);
      const info = await res.json();
      debugUrl = info.webSocketDebuggerUrl;
      break;
    } catch {
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  if (!debugUrl) throw new Error("Chromium did not open a debugging port");

  ws = new WebSocket(debugUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve);
    ws.addEventListener("error", () => reject(new Error("debugging socket failed")));
  });

  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
    // Surface PolyTrack / page console errors so we can see what went wrong.
    if (message.method === "Runtime.consoleAPICalled") {
      const args = (message.params.args || []).map((a) => a.value || a.description || String(a));
      if (message.params.executionContextId) {
        console.log(`[page console ${message.params.type}] ${args.join(" ")}`);
      }
    }
    if (message.method === "Runtime.exceptionThrown") {
      console.error(`[page exception] ${message.params.exceptionDetails.text}`);
    }
  });

  await send("Page.enable");
  await send("Runtime.enable");

  // Attach to the page target to get a real CDP session id.
  const targets = await fetch(`http://127.0.0.1:${debugPort}/json`);
  const targetList = await targets.json();
  const pageTarget = targetList.find((t) => t.type === "page");
  if (!pageTarget) throw new Error("No page target found");
  const attached = await send("Target.attachToTarget", { targetId: pageTarget.id, flatten: true });
  const sessionId = attached.result.sessionId;

  await send("Page.enable", {}, sessionId);
  await send("Runtime.enable", {}, sessionId);

  // 4. Drive: hold W so the car moves forward. Steering is manual.
  //    Send keyDown once; the game repeats while held.
  console.log("Driving: holding W (throttle). Steer manually if needed.");
  await send("Input.dispatchKeyEvent", {
    type: "keyDown",
    key: "w",
    code: "KeyW",
    windowsVirtualKeyCode: 87,
    nativeVirtualKeyCode: 87,
    autoRepeat: false,
    isKeypad: false,
    isSystemKey: false,
  }, sessionId);

  console.log("Recording. Press Ctrl+C to stop.");
  // Keep the process alive until interrupted.
  await new Promise(() => {});
})().catch((error) => {
  console.error(error);
  cleanup();
  process.exit(1);
});
