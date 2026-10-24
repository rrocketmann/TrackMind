// Drive the PolyTrack 0.6.3 simulation worker. The worker steps
// polytrack_physics.wasm. This process loads it, feeds a track, and reads
// the car state bytes the worker posts.

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import vm from "node:vm";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const MASS_OFFSET = 0.6;
const PART_SIZE = 5;

function write(message) {
  fs.writeSync(1, `${JSON.stringify(message)}\n`);
}

function fail(error) {
  write({ type: "error", error: String(error && error.stack ? error.stack : error) });
  process.exit(1);
}

function patchWorker(source) {
  const replacements = [
    [
      "case X_.CreateCar:\n                        !function(e) {\n                            const i = sw.fromSaveString(e.data.trackData);",
      `case X_.CreateCar: {
                        let trackText = e.data.trackData;
                        let track = sw.fromSaveString(trackText);
                        if (null == track) {
                          const fromExport = sw.fromExportString(trackText);
                          if (null == fromExport) throw new Error("Failed to load track");
                          track = fromExport.trackData;
                          trackText = track.toSaveString();
                        }
                        const bounds = track.getBounds();
                        const seen = new Set();
                        const placed = [];
                        track.forEachPart((x, y, z, type, rotation, axis, color, checkpointOrder) => {
                          if (checkpointOrder != null) seen.add(checkpointOrder);
                          placed.push([x, y, z, type, rotation, axis, checkpointOrder == null ? -1 : checkpointOrder]);
                        });
                        postMessage({
                          messageType: 11,
                          bounds: { minX: bounds.min.x, minY: bounds.min.y, maxX: bounds.max.x, maxY: bounds.max.y },
                          checkpointCount: seen.size,
                          parts: placed,
                          save: trackText
                        });
                        if (e.data.carId == null) break;
                        !function(e) {
                            const i = sw.fromSaveString(e.data.trackData);`,
    ],
    [
      "                        }(e);\n                        break;\n                    case X_.DeleteCar:",
      "                        }(e);\n                        break;\n                    }\n                    case X_.DeleteCar:",
    ],
    [
      "setInterval(c)",
      "void 0;globalThis.__step=c;",
    ],
    [
      "            getPosition() {\n                return S_(this, a_, \"f\")\n            }",
      `            getVelocity() {
                const v = S_(this, JA, "f").getLinearVelocity();
                return { x: v.x(), y: v.y(), z: v.z() };
            }
            getPosition() {
                return S_(this, a_, "f")
            }`,
    ],
    [
      "                        speedKmh: r.getSpeedKmh(),\n                        hasStarted: r.hasStarted(),",
      `                        speedKmh: r.getSpeedKmh(),
                        // getSpeedKmh() is derived from wheel rotation, so it
                        // reads like real speed even while the wheels spin
                        // freely in the air. The chassis velocity is the honest
                        // one; the slip angle the steering assist keys on needs
                        // it. wheelSkidInfo is how much each tyre is sliding.
                        velocity: (() => {
                          const v = r.getVelocity && r.getVelocity();
                          return v ? { x: v.x, y: v.y, z: v.z } : null;
                        })(),
                        hasStarted: r.hasStarted(),`,
    ],
    [
      "                function c() {\n                    const e = performance.now()\n                      , t = [];\n                    if (n.length > 0) {",
      `                function c() {
                    // The offscreen worker only flushes control samples in its
                    // realtime loop. This loop does not, so copy the latest
                    // sample that is already due onto the live controls. Leave
                    // a future sample queued. Applying it early holds the first
                    // button of the race and ignores every later command.
                    if (n.length > 0) {
                      for (const e of n) if (null != e.userControls && e.userControls.length > 0) {
                        const frame = e.model.getTime().numberOfFrames;
                        let latest = null;
                        let cursor = 0;
                        for (const sample of e.userControls) {
                            if (sample.frame > frame) break;
                            latest = sample;
                            cursor += 1;
                        }
                        if (latest != null) {
                          e.userControls.splice(0, cursor);
                          const on = e.model.controls;
                          on.up = latest.up, on.right = latest.right, on.down = latest.down,
                          on.left = latest.left, on.reset = latest.reset;
                        }
                      }
                    }
                    const e = performance.now()
                      , t = [];
                    if (n.length > 0) {`,
    ],
    [
      "                                    const n = Math.max(0, t - s);\n                                    let i = r.model.getTime().numberOfFrames + n;\n                                    if (r.model.hasStarted() || (i = 0),\n                                    0 == r.userControls.length)",
      `                                    // Stamp the sample on this simulation frame. The stock
                                    // line adds the milliseconds since boot, so once the worker
                                    // has been alive a steer is scheduled thousands of frames
                                    // later and the same plan drives a different line.
                                    let i = r.model.hasStarted() ? r.model.getTime().numberOfFrames : 0;
                                    if (0 == r.userControls.length)`,
    ],
    [
      "onmessage = i;",
      `onmessage = (event) => { try { i(event) } catch (error) { postMessage({ messageType: -1, error: String(error && error.stack || error) }) } }; globalThis.__step = c; postMessage({ messageType: -2 });`,
    ],
  ];
  let patched = `var onmessage;\nvar postMessage = globalThis.postMessage;\nvar self = globalThis;\n${source}`;
  for (const [needle, replacement] of replacements) {
    if (!patched.includes(needle)) {
      throw new Error(`simulation worker no longer contains ${JSON.stringify(needle.slice(0, 80))}`);
    }
    patched = patched.replace(needle, replacement);
  }
  return patched;
}

function rngTable(mainBundle) {
  const text = fs.readFileSync(mainBundle, "utf8");
  const valueAt = text.indexOf("value: [.120477647");
  if (valueAt < 0) throw new Error("mountain random table is missing from main.bundle.js");
  const end = text.indexOf("]", valueAt);
  return vm.runInThisContext(`(${text.slice(valueAt + "value:".length, end + 1)})`);
}

function mountains(bounds, table) {
  let cursor = 0;
  const next = () => {
    cursor += 1;
    if (cursor >= table.length) cursor = 0;
    return table[cursor];
  };
  const span = Math.max(
    Math.abs(bounds.maxX - bounds.minX) * PART_SIZE / 2 * Math.SQRT2,
    Math.abs(bounds.maxY - bounds.minY) * PART_SIZE / 2 * Math.SQRT2,
  );
  const radius = Math.max(200, 160 + span);
  const offset = {
    x: (bounds.minX + (bounds.maxX - bounds.minX) / 2) * PART_SIZE,
    y: 0,
    z: (bounds.minY + (bounds.maxY - bounds.minY) / 2) * PART_SIZE,
  };
  if (radius > 4500) return { vertices: [], offset };
  const rings = Math.floor(radius / 10);
  const columns = [];
  for (let ring = 0; ring < rings; ring += 1) {
    const column = [];
    for (let sample = 0; sample < 8; sample += 1) {
      column.push(sample === 0 || sample === 7 || (sample === 1 && next() < 0.5) ? 0 : next());
    }
    columns.push(column);
  }
  const vertices = [];
  for (let ring = 0; ring < columns.length; ring += 1) {
    const left = (ring / columns.length) * Math.PI * 2;
    const right = ((ring + 1) / columns.length) * Math.PI * 2;
    const here = columns[ring];
    const nextColumn = ring + 1 < columns.length ? columns[ring + 1] : columns[0];
    for (let band = 0; band < here.length - 1; band += 1) {
      const inner = radius + 100 * band;
      const outer = radius + 100 * (band + 1);
      vertices.push(
        Math.cos(left) * inner, here[band] * 100, Math.sin(left) * inner,
        Math.cos(right) * inner, nextColumn[band] * 100, Math.sin(right) * inner,
        Math.cos(right) * outer, nextColumn[band + 1] * 100, Math.sin(right) * outer,
        Math.cos(left) * inner, here[band] * 100, Math.sin(left) * inner,
        Math.cos(right) * outer, nextColumn[band + 1] * 100, Math.sin(right) * outer,
        Math.cos(left) * outer, here[band + 1] * 100, Math.sin(left) * outer,
      );
    }
  }
  return { vertices, offset };
}

function collisionPoints(file) {
  const data = fs.readFileSync(file);
  const jsonLength = data.readUInt32LE(12);
  const document = JSON.parse(data.subarray(20, 20 + jsonLength).toString("utf8"));
  const mesh = document.meshes.find((item) => item.name === "Collision");
  if (!mesh) throw new Error("car.glb has no Collision mesh");
  const accessor = document.accessors[mesh.primitives[0].attributes.POSITION];
  if (accessor.bufferView == null) {
    throw new Error("car collision mesh is compressed. Re-run node scripts/dump_parts.mjs against the 0.6.3 page.");
  }
  const view = document.bufferViews[accessor.bufferView];
  const binStart = 20 + jsonLength;
  const binHeader = 8;
  const start = binStart + binHeader + (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const points = [];
  for (let index = 0; index < accessor.count; index += 1) {
    points.push(
      data.readFloatLE(start + index * 12),
      data.readFloatLE(start + index * 12 + 4),
      data.readFloatLE(start + index * 12 + 8),
    );
  }
  return points;
}

function typedParts(parts) {
  return parts.map((part) => ({
    id: part.id,
    vertices: Float32Array.from(part.vertices),
    detector: part.detector,
    startOffset: part.startOffset,
  }));
}

function decodeCarState(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let cursor = 4;
  const read24 = () => {
    const value = bytes[cursor] | (bytes[cursor + 1] << 8) | (bytes[cursor + 2] << 16);
    cursor += 3;
    return value;
  };
  const frames = read24();
  const speedKmh = view.getFloat32(cursor, true);
  cursor += 4;
  const flags = bytes[cursor];
  cursor += 1;
  const finishFrames = flags & 2 ? read24() : null;
  const nextCheckpointIndex = view.getUint16(cursor, true);
  cursor += 2;
  const position = {
    x: view.getFloat32(cursor, true),
    y: view.getFloat32(cursor + 4, true),
    z: view.getFloat32(cursor + 8, true),
  };
  cursor += 12;
  const quaternion = {
    x: view.getFloat32(cursor, true),
    y: view.getFloat32(cursor + 4, true),
    z: view.getFloat32(cursor + 8, true),
    w: view.getFloat32(cursor + 12, true),
  };
  return {
    frames,
    speedKmh,
    velocity: null,
    finishFrames,
    nextCheckpointIndex,
    position,
    quaternion,
    wheels: [!!(flags & 8), !!(flags & 16), !!(flags & 32), !!(flags & 64)],
  };
}

function carStateFromObject(raw) {
  const contact = raw.wheelContact || [];
  const position = raw.position || {};
  const quaternion = raw.quaternion || {};
  const velocity = raw.velocity || null;
  return {
    frames: raw.frames,
    speedKmh: raw.speedKmh,
    velocity: velocity ? { x: velocity.x, y: velocity.y, z: velocity.z } : null,
    finishFrames: raw.finishFrames === undefined ? null : raw.finishFrames,
    nextCheckpointIndex: raw.nextCheckpointIndex,
    position: { x: position.x, y: position.y, z: position.z },
    quaternion: { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w },
    wheels: [0, 1, 2, 3].map((index) => !!contact[index]),
  };
}

let workerMessage = null;
let simContext = null;

function pump() {
  const step = simContext && simContext.__step;
  if (typeof step !== "function") fail("simulation step is not ready");
  const started = Date.now();
  while (waiting != null && Date.now() - started < 5000) step();
  if (waiting != null) fail(`simulation stopped at frame ${waiting}`);
}
let waiting = null;
let boundsMessage = null;
let booted = null;
let carId = 0;
let trackText = "";
let mountain = null;
let collision = null;
let checkpointCount = 0;

function postMessage(message) {
  if (message.messageType === -2) {
    booted = true;
    return;
  }
  if (message.messageType === -1) {
    fail(message.error);
  }
  if (message.messageType === 11) {
    boundsMessage = message;
    return;
  }
  if (message.messageType === 10) {
    // The 0.6.3 worker posts `carStates` objects. Older workers posted packed
    // `carStateBuffers`, which decodeCarState reads.
    const states = message.carStates;
    const buffers = message.carStateBuffers;
    const state = Array.isArray(states) && states.length
      ? carStateFromObject(states[states.length - 1])
      : Array.isArray(buffers) && buffers.length
        ? decodeCarState(buffers[buffers.length - 1])
        : null;
    if (!state) return;
    if (waiting && state.frames >= waiting) {
      waiting = null;
      const out = { type: "state", state, checkpointCount };
      write(out);
    }
  }
}

function boot(command) {
  const root = path.resolve(command.root);
  const partsFile = path.resolve(command.parts);
  const trackFile = path.resolve(command.track);
  process.chdir(root);
  const cached = JSON.parse(fs.readFileSync(partsFile, "utf8"));
  const parts = Array.isArray(cached) ? cached : cached.parts;
  if (!Array.isArray(parts)) fail("physics mesh cache has no parts");
  trackText = fs.readFileSync(trackFile, "utf8").trim();
  collision = Array.isArray(cached.collision) && cached.collision.length
    ? cached.collision
    : collisionPoints(path.join(root, "models/car.glb"));
  const massOffset = typeof cached.massOffset === "number" ? cached.massOffset : MASS_OFFSET;
  const table = rngTable(path.join(root, "main.bundle.js"));
  const context = {
    console,
    process,
    Buffer,
    require: createRequire(import.meta.url),
    __dirname: root,
    __filename: path.join(root, "simulation_worker.bundle.js"),
    setInterval,
    clearInterval,
    setTimeout,
    clearTimeout,
    performance,
    atob,
    TextDecoder,
    TextEncoder,
    WebAssembly,
    URL,
    queueMicrotask,
    JS_SHA256_NO_NODE_JS: true,
    postMessage,
    location: { href: pathToFileURL(path.join(root, "simulation_worker.bundle.js")).href },
    fetch: async (url) => {
      const target = fileURLToPath(url);
      const body = fs.readFileSync(target);
      const copy = new Uint8Array(body.byteLength);
      copy.set(body);
      return {
        ok: true,
        status: 200,
        url: String(url),
        arrayBuffer: async () => copy.buffer,
      };
    },
  };
  simContext = context;
  vm.createContext(context);
  context.global = context;
  context.self = context;
  context.window = context;
  context.globalThis = context;
  const listeners = new Map();
  context.addEventListener = (type, fn) => {
    const list = listeners.get(type) || [];
    list.push(fn);
    listeners.set(type, list);
  };
  context.removeEventListener = (type, fn) => {
    const list = listeners.get(type) || [];
    const index = list.indexOf(fn);
    if (index >= 0) list.splice(index, 1);
  };
  context.dispatchEvent = (event) => {
    for (const fn of listeners.get(event.type) || []) fn(event);
  };
  context.document = {
    createElementNS: () => ({ style: {}, setAttribute() {}, appendChild() {} }),
  };
  context.importScripts = (...urls) => {
    const saved = context.importScripts;
    delete context.importScripts;
    try {
      for (const url of urls) {
        const file = path.resolve(root, url);
        vm.runInContext(fs.readFileSync(file, "utf8"), context, { filename: file });
      }
    } finally {
      context.importScripts = saved;
    }
  };
  const source = patchWorker(fs.readFileSync(path.join(root, "simulation_worker.bundle.js"), "utf8"));
  vm.runInContext(source, context, { filename: path.join(root, "simulation_worker.bundle.js") });
  const started = Date.now();
  const timer = setInterval(() => {
    if (!booted) {
      if (Date.now() - started > 20000) fail("simulation worker did not start");
      return;
    }
    clearInterval(timer);
    workerMessage = context.onmessage;
    workerMessage({
      data: {
        messageType: 0,
        version: "0.6.3",
        isRealtime: false,
        trackParts: typedParts(parts),
        carCollisionShapeVertices: collision,
        carMassOffset: massOffset,
      },
    });
    workerMessage({ data: { messageType: 3, trackData: trackText } });
    if (!boundsMessage) fail("track did not report its bounds");
    mountain = mountains(boundsMessage.bounds, table);
    checkpointCount = boundsMessage.checkpointCount;
    trackText = boundsMessage.save;
    write({
      type: "ready",
      checkpointCount,
      bounds: boundsMessage.bounds,
      parts: boundsMessage.parts,
    });
  }, 10);
}

function spawnCar() {
  carId += 1;
  const id = carId;
  workerMessage({
    data: {
      messageType: 3,
      mountainVertices: mountain.vertices,
      mountainOffset: mountain.offset,
      trackData: trackText,
      carId: id,
      carCollisionShapeVertices: collision,
      carMassOffset: MASS_OFFSET,
    },
  });
  waiting = 1;
  workerMessage({ data: { messageType: 5, carId: id, targetSimulationTimeFrames: 1 } });
  pump();
  return id;
}

function commandReset() {
  if (carId > 0) {
    workerMessage({ data: { messageType: 4, carId } });
  }
  spawnCar();
}

function commandStep(command) {
  const controls = command.controls;
  workerMessage({
    data: {
      messageType: 6,
      carId,
      up: !!controls.up,
      right: !!controls.right,
      down: !!controls.down,
      left: !!controls.left,
      reset: !!controls.reset,
    },
  });
  waiting = command.target;
  workerMessage({ data: { messageType: 5, carId, targetSimulationTimeFrames: command.target } });
  pump();
}

const commands = readline.createInterface({ input: process.stdin });
commands.on("line", (line) => {
  if (!line.trim()) return;
  let command;
  try {
    command = JSON.parse(line);
  } catch (error) {
    fail(error);
    return;
  }
  try {
    if (command.cmd === "boot") boot(command);
    else if (command.cmd === "reset") commandReset();
    else if (command.cmd === "step") commandStep(command);
    else if (command.cmd === "close") process.exit(0);
    else fail(`unknown command ${command.cmd}`);
  } catch (error) {
    fail(error);
  }
});
