// Drive the PolyTrack 0.5.2 simulation worker. The worker is the game's own
// Ammo car. This process only loads it, feeds a track, and steps it.

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import vm from "node:vm";
import { createRequire } from "node:module";

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
      "case X_.CreateCar:",
      `case 11: {
        const exported = sw.fromExportString(e.data.trackData);
        const editor = exported && exported.trackData && exported.trackData.toSaveString ? exported.trackData : null;
        const save = editor ? editor.toSaveString() : e.data.trackData;
        const track = sw.fromSaveString(save);
        if (null == track || null == editor) throw new Error("Failed to load track");
        const bounds = editor.getBounds();
        const seen = new Set();
        editor.forEachPart((x, y, z, type, rotation, axis, color, checkpointOrder) => {
          if (checkpointOrder != null) seen.add(checkpointOrder);
        });
        postMessage({
          messageType: 11,
          bounds: { minX: bounds.min.x, minY: bounds.min.y, maxX: bounds.max.x, maxY: bounds.max.y },
          checkpointCount: seen.size,
          save
        });
        break;
      }
      case X_.CreateCar:`,
    ],
    [
      "const n = Math.max(0, t - s);",
      "const n = 0;",
    ],
    [
      "let i = r.model.getTime().numberOfFrames + n;",
      "let i = r.model.getTime().numberOfFrames;",
    ],
    [
      "!e.isPaused && (t.push(a(e)),",
      `!e.isPaused && ((car => {
        const frame = car.model.getTime().numberOfFrames;
        if (car.userControls != null) {
          while (car.userControls.length > 0 && car.userControls[0].frame <= frame) {
            const ctl = car.userControls.shift();
            if (ctl != null) {
              const controls = car.model.controls;
              controls.up = ctl.up;
              controls.right = ctl.right;
              controls.down = ctl.down;
              controls.left = ctl.left;
              controls.reset = ctl.reset;
            }
          }
        }
      })(e), t.length = 0, t.push(a(e)),`,
    ],
    [
      `} while (Math.max(0, performance.now() - e) / 1e3 < .01 && !i)
                    }
                    postMessage({
                        messageType: X_.UpdateResult,
                        carStates: t
                    })`,
      `} while (Math.max(0, performance.now() - e) / 1e3 < .01 && !i)
                    }
                    t.length > 0 && postMessage({
                        messageType: X_.UpdateResult,
                        carStates: t
                    })`,
    ],
    ["setInterval(c)", "void 0"],
    [
      "cw.length = 0,\n                onmessage = i;",
      `cw.length = 0,
                onmessage = (event) => {
                  try { i(event); }
                  catch (error) { postMessage({ messageType: -1, error: String(error && error.stack || error) }); }
                };
                globalThis.__step = c;
                postMessage({ messageType: -2 });`,
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
  const marker = text.indexOf("const MR = SR");
  const valueAt = text.lastIndexOf("value: [", marker);
  if (marker < 0 || valueAt < 0) throw new Error("mountain random table is missing from main.bundle.js");
  const end = text.indexOf("]", valueAt);
  return vm.runInThisContext(`(${text.slice(valueAt + "value: ".length, end + 1)})`);
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

function slim(state) {
  return {
    frames: state.frames,
    speedKmh: state.speedKmh,
    finishFrames: state.finishFrames,
    nextCheckpointIndex: state.nextCheckpointIndex,
    position: state.position,
    quaternion: state.quaternion,
    wheels: state.wheelContact.map((item) => item != null),
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
  if (message.messageType === 10 && message.carStates && message.carStates.length > 0) {
    const state = slim(message.carStates[message.carStates.length - 1]);
    if (waiting && state.frames >= waiting) {
      waiting = null;
      write({ type: "state", state, checkpointCount });
    }
  }
}

function boot(command) {
  const root = path.resolve(command.root);
  const partsFile = path.resolve(command.parts);
  const trackFile = path.resolve(command.track);
  process.chdir(root);
  const parts = JSON.parse(fs.readFileSync(partsFile, "utf8"));
  trackText = fs.readFileSync(trackFile, "utf8").trim();
  collision = collisionPoints(path.join(root, "models/car.glb"));
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
  };
  simContext = context;
  vm.createContext(context);
  context.global = context;
  context.self = context;
  context.window = context;
  context.globalThis = context;
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
    workerMessage({ data: { messageType: 0, isRealtime: false, trackParts: parts } });
    workerMessage({ data: { messageType: 11, trackData: trackText } });
    if (!boundsMessage) fail("track did not report its bounds");
    mountain = mountains(boundsMessage.bounds, table);
    checkpointCount = boundsMessage.checkpointCount;
    trackText = boundsMessage.save;
    write({ type: "ready", checkpointCount, bounds: boundsMessage.bounds });
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
