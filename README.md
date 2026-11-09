# PolyTrack specialist

PPO on one PolyTrack 0.6.3 track in the polytrack fork at `polytrack-0.6.3`. The game's own `polytrack_physics.wasm` is the environment. Before the first PPO update, the actor clones the fastest TrackData laps of the most common track. A separate critic scores the discounted return of the driving reward. That is the same shape as Linesight: a real simulation, numeric state in, a discrete action out, stepped faster than the screen.

`POLYTRACK_ROOT` points at that 0.6.3 directory. It contains `simulation_worker.bundle.js` and `polytrack_physics.wasm`. This program does not open any external page, read its memory, or submit a leaderboard time.

## What a step is

`polytrack_agent/sim/host.mjs` loads `simulation_worker.bundle.js` in a Node vm and steps `polytrack_physics.wasm`. Patches stay in memory. The fork files on disk are not edited. The fast worker is pumped from Python, so a decision does not wait on a timer. Each decision steps the physics for that hold and reads the car once at the end.

Each decision holds W, A, S, and D for 50 physics frames (50 ms). The worker ticks at 1 ms. Training steps one car on each performance core at the same time, and pins each physics process to its own core. Those four keys are the whole control set. Air pitch and roll reuse them while the car is off the ground. The nine actions are the exclusive combinations: throttle or brake or neither, and left or right or straight. Restart is not an action. An episode reset deletes the car and creates it at the start. The in-game respawn key would put the car back on the last checkpoint, so the trainer does not send it.

The observation is twelve numbers plus a one-hot of the previous action:

- velocity
- heading on the ground plane, uprightness, and how many wheels are down
- meters of road left to the side, and which side the edge is on
- where the next gate sits in the car frame: right and forward
- the next checkpoint divided by the checkpoint count

World position, acceleration, and gate distance are in the TrackData rows and are not given to this policy.

`speedKmh` is not in the observation. It comes from wheel rotation, so a car spinning its wheels against a wall or in mid-air still reads as moving. The velocity vector is the motion the policy sees, and it also drives the stall timer.

Reward is forward travel along the chord from the previous gate to the next one, plus speed toward that gate. Any movement forward along the track pays, and going back costs. Speed toward the gate pays up to a cap. A sudden drop in chassis speed costs, so a hit is worse than lifting off the throttle. Discount is 0.998 per decision, about 25 seconds at a 50 ms hold. Passing a gate and finishing the lap are that forward number moving on. An episode ends on a finish, a fall, a stall after the car has had time to leave the grid, no progress along the chord for the stall window, or the time limit.

PPO checkpoints use arch `polytrack-specialist-0.6.3`. SAC checkpoints use `polytrack-sac-0.6.3`. A checkpoint loads only when the arch, the track name, and the observation width match. `best.pt` is the furthest episode in an update. A finished lap replaces an unfinished one, and a faster finish replaces a slower one. The log's progress column is the mean of those episode ends. The measurements are in [DISCOVERIES.md](DISCOVERIES.md).

## Setup

You need Python 3.10+, PyTorch, Node, and a Chromium or Chrome binary.

The polytrack fork (0.6.3) is included in this repo at `polytrack-0.6.3/`. It contains `simulation_worker.bundle.js`, tracks, models, and the car mesh. Point `POLYTRACK_ROOT` at it:

```bash
export POLYTRACK_ROOT=$PWD/polytrack-0.6.3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
node scripts/dump_parts.mjs "$POLYTRACK_ROOT"
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
python scripts/live.py --run runs/winter4
python scripts/record.py --checkpoint runs/winter4/best.pt --out runs/winter4/best.mp4
```

`scripts/train.py` clones the best TrackData laps, then runs PPO until a lap matches that time or you stop it. It writes `runs/winter4/last.pt`, `best.pt`, and `log.csv`. Play loads `runs/winter4/best.pt` and prints finish, time, progress, and checkpoints for each greedy episode. The fly-over run stays in `runs/ppo-reward`. The SAC runs stay in `runs/sac` and `runs/sac-blowup`. The older PPO run stays in `runs/summer1`.

Training has not yet completed a lap. The env and the host are correct; the policy collapses onto the wrong action instead of holding throttle, which a constant action proves is worth far more. The measurements, what has been ruled out, and the plan are in [TROUBLESHOOTING.md](TROUBLESHOOTING.md), under "the learner does not find the good action".

Every finish logs the lap time and the best lap of the run. To train until a lap drops under a target time instead of running forever, pass `--stop-time`:

```bash
python scripts/train.py --stop-time 25
```

Training saves `last.pt` and exits on the first lap at or below 25 s. The same field is `rl.stop_time` in a config overlay.

`scripts/play.py --episodes 5` loads `runs/winter4/best.pt` and prints finish, lap time, progress, and checkpoints for each greedy episode.

## Watch a run

`scripts/live.py` opens the graphs. It tails `episodes.csv` and `log.csv` in the run directory and redraws four panels once a second: lap progress, lap time, update progress, and entropy. Training does not open this window.

```bash
python scripts/live.py --run runs/winter4
```

`scripts/record.py` writes a chase-camera mp4 of a saved checkpoint. The drive is greedy. It is a fresh playback of the weights, not the stochastic episode that produced the checkpoint. The picture is the local checkout in `POLYTRACK_ROOT`, served on 127.0.0.1. Every other request is blocked. The script does not open an external page and does not submit a time.

```bash
python scripts/record.py --checkpoint runs/winter4/best.pt --out runs/winter4/best.mp4
```

The game clock follows the display frame rate and does not catch up, so the recording needs `DISPLAY` and a GPU that holds about 60 fps. The script stops when the game renders under 25 fps. A virtual framebuffer is too slow. This machine's ffmpeg has no libx264, so the file is encoded with libopenh264. `runs/winter4-live/best.mp4` is one such recording. The browser half is `scripts/record_game.mjs`. The old `scripts/record.mjs`, which opened an external page, is gone.

`runs/summer1/last.pt` is an older `polytrack-ppo-v1` actor, 24 inputs, progress 1.03, and it had not taken a PPO step yet. `runs/summer1/best.pt` is an earlier 21-input actor from update 18. The current observation does not load it.

`POLYTRACK_ROOT` overrides `game.root`. Train, play, and record also take `--config some.yaml`, merged onto `polytrack_agent/default_config.yaml`. Useful fields:

| Field | Meaning |
| --- | --- |
| `game.root` | polytrack fork directory, when `POLYTRACK_ROOT` is unset |
| `game.track` | `.track` export, relative to `game.root` unless absolute |
| `game.parts` | mesh cache. Empty uses `polytrack_agent/sim/cache/track_parts.json` |
| `keys` | subset of `w a s d` |
| `rl.action_repeat` | physics frames per decision |
| `rl.timeout_s` / `rl.stuck_s` | episode limits |
| `rl.airborne_grace_s` | how long the car may have no wheel on the track before it counts as beached |
| `rl.reset_action` / `rl.auto_respawn` | opt-in 10th reset action, and respawning at a checkpoint on beaching or a stall |
| `rl.stop_time` | stop training once a lap finishes at or below this many seconds |
| `rl.group_size` | laps collected before one PPO update |
| `rl.workers` | physics processes stepped together. Empty uses one per performance core, capped at the lap group |
| `rl.out_dir` | checkpoint directory |
| `reward.progress_scale` | pay per gate of forward travel along the chord |
| `reward.speed_reward` / `speed_cap_kmh` | pay per decision for speed toward the next gate, and where that pay stops growing |
| `reward.speed_drop_penalty` / `speed_drop_margin` | cost for a sudden loss of chassis speed, and the km/h per second that is still free |
| `reward.time_penalty` / `finish_bonus` | optional per-second time cost, finish bonus |
| `reward.fail_penalty` / `offtrack_penalty` | optional one cost for a failed episode, optional per-second cost past the road edge |
| `reward.contact_reward` / `respawn_penalty` / `beach_penalty` | per-step pay for wheels on the road, cost per respawn, one-off cost for leaving the road |
| `reward.stall_penalty` | per-step cost for sitting on the road without covering ground |

The default track is `tracks/official/winter4.track`, the official track with the most laps in TrackData. Training replaces that path when another track becomes the most common. Any other official or community `.track` in that checkout works after the same parts dump. The dump is the page's track-piece catalog, shared across tracks. It also stores the car collision hull and the mass offset from that same Init message. The 0.6.3 hull in `models/car.glb` is Draco-compressed, so the host cannot read it from the file. Mountains still come from the game's random table in `main.bundle.js`.

## Layout

```
polytrack_agent/sim/     Node host, Python client, generated mesh cache
polytrack_agent/rl/      observation, track env, PPO
scripts/dump_parts.mjs   one-time mesh dump
scripts/train.py         PPO. See the docstring for the graph and video commands
scripts/play.py          greedy laps in the simulation, numbers only
scripts/live.py          four live graphs for a run directory
scripts/record.py        chase-camera video of a checkpoint in the local game
scripts/record_game.mjs  local page, capture, and encode used by record.py
```
