# PolyTrack agent

PPO on the PolyTrack 0.6.3 car in the schoolagy web build at `polytrack-0.6.3`. The game's own `polytrack_physics.wasm` is the environment. The policy reads the car state and picks one key hold. That is the same shape as Linesight: a real simulation, numeric state in, a discrete action out, stepped faster than the screen.

`POLYTRACK_ROOT` points at that 0.6.3 directory. It contains `simulation_worker.bundle.js` and `polytrack_physics.wasm`. The official page is [kodub.com/apps/polytrack](https://www.kodub.com/apps/polytrack). This program does not open that page, read its memory, or submit a leaderboard time.

## What a step is

`polytrack_agent/sim/host.mjs` loads `simulation_worker.bundle.js` in a Node vm and steps `polytrack_physics.wasm`. Patches stay in memory. The schoolagy files on disk are not edited. The fast worker is pumped from Python, so a decision does not wait on a timer.

Each decision holds W, A, S, and D for 50 physics frames (50 ms). The worker ticks at 1 ms. Those four keys are the whole control set. Air pitch and roll reuse them while the car is off the ground. The nine actions are the exclusive combinations: throttle or brake or neither, and left or right or straight. Restart is not an action. An episode reset deletes the car and creates it at the start. The in-game respawn key would put the car back on the last checkpoint, so the trainer does not send it.

The observation is fifteen numbers plus a one-hot of the previous action:

- speed, quaternion, upright, and the body's forward direction on the ground plane
- fraction of wheels in contact, and whether the car is airborne
- next checkpoint divided by the checkpoint count
- height
- where the next gate sits in the car frame: right, forward, and distance

Reward is the change in progress, minus a small time penalty, plus a bonus on the finish. Progress is the checkpoint index plus how far the car has driven since that checkpoint, capped just short of the next index. A finish scores one past the last checkpoint. An episode ends on a finish, a fall, a stall after the car has had time to leave the grid, or the time limit.

Checkpoints use arch `schoolagy-ppo-0.6.3`. Older checkpoints do not load. `last.pt` resumes only when the track name matches. `best.pt` is the highest mean progress in an update.

## Setup

You need Python 3.10+, PyTorch, Node, and a Chromium or Chrome binary. `POLYTRACK_ROOT` is the schoolagy `polytrack-0.6.3` directory. Copy that directory onto the training machine. The mesh dump has to be taken from that same build. The cache is not in git.

```bash
export POLYTRACK_ROOT=/path/to/schoolagy/polytrack-0.6.3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
node scripts/dump_parts.mjs
python -m pytest
```

`scripts/dump_parts.mjs` serves that checkout once in headless Chromium and saves the physics meshes from the worker's Init message. The file is `polytrack_agent/sim/cache/track_parts.json`, about 14 MB, and it is gitignored. Training after that dump does not open a window. The script looks for `google-chrome`, `chromium`, or a Playwright Chromium under the home directory. Set `CHROME` when the binary is somewhere else.

```bash
node scripts/dump_parts.mjs "$POLYTRACK_ROOT" /path/to/track_parts.json
```

## Train and play

```bash
python scripts/train.py
python scripts/play.py --episodes 5
```

`scripts/train.py` runs until you stop it. It writes `runs/summer1/last.pt`, `best.pt`, and `log.csv`. Play loads `runs/summer1/best.pt` and prints finish, time, progress, and checkpoints for each greedy episode.

On an SSH machine with no display, `play.py` cannot open a window. Watch the chase-camera gate pass with `ffplay` instead:

```bash
ffplay -autoexit runs/summer1/gate.mp4
```

From another computer, without cloning:

```bash
ffplay -autoexit https://github.com/rrocketmann/TrackMind/raw/refs/heads/main/runs/summer1/gate.mp4
```

## First checkpoint

[runs/summer1/gate.mp4](runs/summer1/gate.mp4) is the chase-camera lap that first cleared Summer 1 checkpoint 1. It is 30 seconds. The car leaves the start, takes the S-bend, and goes under the yellow gate at 28.2 s.

This video was recorded from the **schoolagy PolyTrack 0.6.3 simulation** that this agent drives, not from the official kodub page. The program does not open `kodub.com/apps/polytrack`, read its memory, or submit a leaderboard time — so a "real polytrack" recording has to be made on a machine with a display and a browser pointed at that page.

There is no display on the training machine, so play the file where you have a screen:

```bash
ffplay -autoexit runs/summer1/gate.mp4
```

From another computer, without cloning:

```bash
ffplay -autoexit https://github.com/rrocketmann/TrackMind/raw/refs/heads/main/runs/summer1/gate.mp4
```

To record the real PolyTrack yourself, use `scripts/record.mjs`. It launches Chromium on a virtual X display (Xvfb) and grabs that display with ffmpeg, so it works from an SSH session with no physical monitor. The car drives forward by holding W; steer manually if needed.

```bash
export POLYTRACK_URL=https://www.kodub.com/apps/polytrack   # optional, defaults to kodub
CHROME=/usr/bin/google-chrome node scripts/record.mjs runs/real_polytrack.mp4
```

Press Ctrl+C to stop. The script writes `runs/real_polytrack.mp4` (or the path you pass). Play it where you have a screen:

```bash
ffplay -autoexit runs/real_polytrack.mp4
```

A sample real-PolyTrack recording is already committed at `runs/real_polytrack.mp4` (15 s). The schoolagy simulation gate pass remains at `runs/summer1/gate.mp4`:

```bash
ffplay -autoexit runs/summer1/gate.mp4
```

The recorder needs `ffmpeg`, `xvfb-run`/`Xvfb`, and a Chromium or Chrome binary. It does not open or read the local schoolagy simulation — it records the official page directly. The program still does not submit a leaderboard time.

If PolyTrack shows "encountered an unexpected error":
- Make sure Chrome is **not** launched with `--disable-gpu` — PolyTrack needs WebGL. The recorder uses `--use-gl=swiftshader` for software rendering.
- Run from a machine with a real display and GPU if possible; Xvfb software rendering can be unstable.
- Check the console output — the recorder prints PolyTrack's page errors under `[page console error]` / `[page exception]`.

`runs/summer1/last.pt` is the actor saved with that lap. It is `schoolagy-ppo-v1`, 24 inputs, progress 1.03, and it had not taken a PPO step yet. `runs/summer1/best.pt` is an earlier 21-input actor from update 18. The current observation does not load it.

`POLYTRACK_ROOT` overrides `game.root`. Both commands also take `--config some.yaml`, merged onto `polytrack_agent/default_config.yaml`. Useful fields:

| Field | Meaning |
| --- | --- |
| `game.root` | schoolagy `polytrack-0.6.3` directory, when `POLYTRACK_ROOT` is unset |
| `game.track` | `.track` export, relative to `game.root` unless absolute |
| `game.parts` | mesh cache. Empty uses `polytrack_agent/sim/cache/track_parts.json` |
| `keys` | subset of `w a s d` |
| `rl.action_repeat` | physics frames per decision |
| `rl.timeout_s` / `rl.stuck_s` | episode limits |
| `rl.group_size` | laps collected before one PPO update |
| `rl.out_dir` | checkpoint directory |
| `reward.progress_scale` / `time_penalty` / `finish_bonus` | reward terms |

The default track is `tracks/official/summer1.track`. Any other official or community `.track` in that checkout works after the same parts dump. The dump is the page's track-piece catalog, shared across tracks. It also stores the car collision hull and the mass offset from that same Init message. The 0.6.3 hull in `models/car.glb` is Draco-compressed, so the host cannot read it from the file. Mountains still come from the game's random table in `main.bundle.js`.

## Layout

```
polytrack_agent/sim/     Node host, Python client, generated mesh cache
polytrack_agent/rl/      observation, track env, PPO
scripts/dump_parts.mjs   one-time mesh dump
scripts/record.mjs       record the real PolyTrack page (Xvfb + ffmpeg)
scripts/train.py
scripts/play.py
```
