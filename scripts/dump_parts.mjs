// Ask the schoolagy PolyTrack page for the physics meshes it already builds.
// The simulation worker receives those meshes in its Init message. This saves
// that payload so training can start the worker without opening the page again.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const root = process.argv[2] || "/home/martinmalyshau/dev/schoolagy/polytrack";
const out = process.argv[3] || path.resolve("polytrack_agent/sim/cache/track_parts.json");
const chrome =
  process.env.CHROME ||
  "/home/martinmalyshau/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";

function contentType(file) {
  if (file.endsWith(".js")) return "text/javascript";
  if (file.endsWith(".wasm")) return "application/wasm";
  if (file.endsWith(".json")) return "application/json";
  if (file.endsWith(".html")) return "text/html";
  if (file.endsWith(".svg")) return "image/svg+xml";
  if (file.endsWith(".woff2")) return "font/woff2";
  return "application/octet-stream";
}

const dumped = new Promise((resolve, reject) => {
  const server = http.createServer((req, res) => {
    if (req.method === "POST" && req.url === "/__dump") {
      const chunks = [];
      req.on("data", (chunk) => chunks.push(chunk));
      req.on("end", () => {
        const body = Buffer.concat(chunks);
        fs.mkdirSync(path.dirname(out), { recursive: true });
        fs.writeFileSync(out, body);
        res.writeHead(204);
        res.end();
        resolve(body.length);
      });
      return;
    }
    const url = new URL(req.url, "http://127.0.0.1");
    const rel = decodeURIComponent(url.pathname);
    const file = path.join(root, rel === "/" ? "index.html" : rel);
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404);
      res.end("missing");
      return;
    }
    res.writeHead(200, { "content-type": contentType(file) });
    fs.createReadStream(file).pipe(res);
  });
  server.listen(0, "127.0.0.1", () => {
    const port = server.address().port;
    dumped.server = server;
    dumped.port = port;
  });
  setTimeout(() => reject(new Error("timed out waiting for physics meshes")), 90000);
});

await new Promise((resolve) => {
  const timer = setInterval(() => {
    if (dumped.port) {
      clearInterval(timer);
      resolve();
    }
  }, 20);
});

const pagePort = dumped.port;
const debugPort = 9333;
const child = spawn(
  chrome,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    `--remote-debugging-port=${debugPort}`,
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "about:blank",
  ],
  { stdio: "ignore" },
);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function browserSocket() {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${debugPort}/json/version`);
      const info = await response.json();
      return info.webSocketDebuggerUrl;
    } catch {
      await sleep(200);
    }
  }
  throw new Error("Chrome did not open a debugging port");
}

const ws = new WebSocket(await browserSocket());
await new Promise((resolve, reject) => {
  ws.addEventListener("open", resolve);
  ws.addEventListener("error", () => reject(new Error("debugging socket failed")));
});

let nextId = 0;
const pending = new Map();
ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
});

function send(method, params = {}, sessionId) {
  const id = ++nextId;
  const payload = { id, method, params };
  if (sessionId) payload.sessionId = sessionId;
  ws.send(JSON.stringify(payload));
  return new Promise((resolve) => pending.set(id, resolve));
}

const hook = `(() => {
  const Orig = window.Worker;
  window.Worker = function(url, opts) {
    const worker = new Orig(url, opts);
    const orig = worker.postMessage.bind(worker);
    worker.postMessage = function(message, transfer) {
      if (message && message.messageType === 0 && message.trackParts && !window.__partsSent) {
        window.__partsSent = true;
        const parts = message.trackParts.map((part) => ({
          id: part.id,
          vertices: Array.from(part.vertices),
          detector: part.detector == null ? null : {
            type: part.detector.type,
            center: Array.isArray(part.detector.center) ? Array.from(part.detector.center) : [part.detector.center.x, part.detector.center.y, part.detector.center.z],
            size: Array.isArray(part.detector.size) ? Array.from(part.detector.size) : [part.detector.size.x, part.detector.size.y, part.detector.size.z],
          },
          startOffset: part.startOffset == null ? null : Array.from(part.startOffset),
        }));
        fetch("/__dump", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(parts) });
      }
      return orig(message, transfer);
    };
    return worker;
  };
})();`;

const created = await send("Target.createTarget", { url: "about:blank" });
const targetId = created.result.targetId;
const attached = await send("Target.attachToTarget", { targetId, flatten: true });
const sessionId = attached.result.sessionId;
await send("Page.enable", {}, sessionId);
await send("Page.addScriptToEvaluateOnNewDocument", { source: hook }, sessionId);
await send("Page.navigate", { url: `http://127.0.0.1:${pagePort}/index.html` }, sessionId);

const bytes = await dumped;
child.kill();
ws.close();
dumped.server.close();
console.log(`Wrote ${out} (${bytes} bytes)`);
