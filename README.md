# PolyTrack agent

Discrete SAC on the PolyTrack 0.6.3 car in the polytrack fork at `polytrack-0.6.3`. The game's own `polytrack_physics.wasm` is the environment. The policy reads the car state and picks one key hold. That is the same shape as Linesight: a real simulation, numeric state in, a discrete action out, stepped faster than the screen.

`POLYTRACK_ROOT` points at that 0.6.3 directory. It contains `simulation_worker.bundle.js` and `polytrack_physics.wasm`. This program does not open any external page, read its memory, or submit a leaderboard time.

## What a step is

`polytrack_agent/sim/host.mjs` loads `simulation_worker.bundle.js` in a Node vm and steps `polytrack_physics.wasm`. Patches stay in memory. The fork files on disk are not edited. The fast worker is pumped from Python, so a decision does not wait on a timer.

Each decision holds W, A, S, and D for 50 physics frames (50 ms). The worker ticks at 1 ms. Those four keys are the whole control set. Air pitch and roll reuse them while the car is off the ground. The nine actions are the exclusive combinations: throttle or brake or neither, and left or right or straight. Restart is not an action. An episode reset deletes the car and creates it at the start. The in-game respawn key would put the car back on the last checkpoint, so the trainer does not send it.

The observation is nineteen numbers plus a one-hot of the previous action:

- position, velocity, and acceleration
- heading on the ground plane, uprightness, and how many wheels are down
- meters of road left to the side, and which side the edge is on
- where the next gate sits in the car frame: right, forward, and distance
- the next checkpoint divided by the checkpoint count

`speedKmh` is not in the observation. It comes from wheel rotation, so a car spinning its wheels against a wall or in mid-air still reads as moving. The velocity vector is the motion the policy sees, and it also drives the stall timer.

Reward is the change in progress, minus a small time penalty, plus a bonus on the finish. Progress is the checkpoint index plus how far the car has driven since that checkpoint, capped just short of the next index. A finish scores one past the last checkpoint. An episode ends on a finish, a fall, a stall after the car has had time to leave the grid, or the time limit.

SAC checkpoints use arch `polytrack-sac-0.6.3`. PPO checkpoints use `polytrack-ppo-0.6.3`. A checkpoint loads only when the arch, the track name, and the observation width match. `best.pt` is the highest mean progress in an update.

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
```

`scripts/train.py` runs SAC until you stop it. It writes `runs/sac/last.pt`, `best.pt`, and `log.csv`. Play loads `runs/sac/best.pt` and prints finish, time, progress, and checkpoints for each greedy episode. The older PPO run stays in `runs/summer1`.

Training has not yet completed a lap. The env and the host are correct; the policy collapses onto the wrong action instead of holding throttle, which a constant action proves is worth far more. The measurements, what has been ruled out, and the plan are in [TROUBLESHOOTING.md](TROUBLESHOOTING.md), under "the learner does not find the good action".

Every finish logs the lap time and the best lap of the run. To train until a lap drops under a target time instead of running forever, pass `--stop-time`:

```bash
python scripts/train.py --stop-time 25
```

Training saves `last.pt` and exits on the first lap at or below 25 s. The same field is `rl.stop_time` in a config overlay.

`scripts/play.py --episodes 5` loads `runs/sac/best.pt` and prints finish, lap time, progress, and checkpoints for each greedy episode.

## First checkpoint

[runs/summer1/gate.mp4](runs/summer1/gate.mp4) is the chase-camera lap that first cleared Summer 1 checkpoint 1. It is 30 seconds. The car leaves the start, takes the S-bend, and goes under the yellow gate at 28.2 s.

This video was recorded from the polytrack fork simulation that this agent drives, not from any external page. The program does not open external pages, read their memory, or submit a leaderboard time.

There is no display on the training machine, so play the file where you have a screen:

```bash
ffplay -autoexit runs/summer1/gate.mp4
```

From another computer, without cloning:

```bash
ffplay -autoexit https://github.com/rrocketmann/TrackMind/raw/refs/heads/main/runs/summer1/gate.mp4
```

A sample real-PolyTrack recording is committed at `runs/real_polytrack.mp4` (15 s). The polytrack fork gate pass remains at `runs/summer1/gate.mp4`:

```bash
ffplay -autoexit runs/summer1/gate.mp4
```

The recorder script (`scripts/record.mjs`) that captured the real PolyTrack page was removed; the committed MP4s remain as samples. The program does not open external pages, read their memory, or submit a leaderboard time.

`runs/summer1/last.pt` is the actor saved with that lap. It is `polytrack-ppo-v1`, 24 inputs, progress 1.03, and it had not taken a PPO step yet. `runs/summer1/best.pt` is an earlier 21-input actor from update 18. The current observation does not load it.

`POLYTRACK_ROOT` overrides `game.root`. Both commands also take `--config some.yaml`, merged onto `polytrack_agent/default_config.yaml`. Useful fields:

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
| `rl.out_dir` | checkpoint directory |
| `reward.progress_scale` / `time_penalty` / `finish_bonus` | reward terms |
| `reward.contact_reward` / `respawn_penalty` / `beach_penalty` | per-step pay for wheels on the road, cost per respawn, one-off cost for leaving the road |
| `reward.stall_penalty` | per-step cost for sitting on the road without covering ground |
| `reward.speed_reward` / `speed_cap_kmh` | per-step pay for ground speed while the wheels are down, and the speed where it stops growing |

The default track is `tracks/official/summer1.track`. Any other official or community `.track` in that checkout works after the same parts dump. The dump is the page's track-piece catalog, shared across tracks. It also stores the car collision hull and the mass offset from that same Init message. The 0.6.3 hull in `models/car.glb` is Draco-compressed, so the host cannot read it from the file. Mountains still come from the game's random table in `main.bundle.js`.

## Layout

```
polytrack_agent/sim/     Node host, Python client, generated mesh cache
polytrack_agent/rl/      observation, track env, PPO
scripts/dump_parts.mjs   one-time mesh dump
scripts/train.py
scripts/play.py
```
