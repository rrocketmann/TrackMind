# PolyTrack agent

PPO on the car that ships in a schoolagy PolyTrack 0.5.2 checkout. The game's own Ammo worker is the environment. The policy reads the car state and picks one key hold. That is the same shape as Linesight: a real simulation, numeric state in, a discrete action out, stepped faster than the screen.

This checkout is PolyTrack 0.5.2. The official game is [kodub.com/apps/polytrack](https://www.kodub.com/apps/polytrack), version 0.6.3, with a different C++ simulation inside the page. A policy or a press sequence from this trainer does not transfer to that client. This program does not open the official page, read its memory, or submit a leaderboard time. The schoolagy build itself says unofficial versions cannot use the leaderboard.

## What a step is

`polytrack_agent/sim/host.mjs` loads `simulation_worker.bundle.js` in a Node vm. Patches stay in memory. The schoolagy files on disk are not edited. The fast worker is pumped from Python, so a decision does not wait on a timer.

Each decision holds W, A, S, and D for 50 physics frames (50 ms). The worker ticks at 1 ms. Those four keys are the whole 0.5.2 control set. Air pitch and roll reuse them while the car is off the ground. The nine actions are the exclusive combinations: throttle or brake or neither, and left or right or straight. Restart is not an action. An episode reset deletes the car and creates it at the start. The in-game respawn key would put the car back on the last checkpoint, so the trainer does not send it.

The observation is twelve numbers plus a one-hot of the previous action:

- speed, quaternion, upright, and the body's forward direction on the ground plane
- fraction of wheels in contact, and whether the car is airborne
- next checkpoint divided by the checkpoint count
- height

Reward is the change in progress, minus a small time penalty, plus a bonus on the finish. Progress is the checkpoint index plus how far the car has driven since that checkpoint, capped just short of the next index. A finish scores one past the last checkpoint. An episode ends on a finish, a fall, a stall after the car has had time to leave the grid, or the time limit.

Checkpoints use arch `schoolagy-ppo-v1`. Older `state-ppo-v1` files do not load. `last.pt` resumes only when the track name matches. `best.pt` is the highest mean progress in an update.

## Setup

You need Python 3.10+, PyTorch, Node, and a Chromium or Chrome binary. `POLYTRACK_ROOT` is the schoolagy directory that contains `simulation_worker.bundle.js` and `tracks/`. Copy that checkout onto the training machine. The mesh cache is built there and is not in git.

```bash
export POLYTRACK_ROOT=/path/to/schoolagy/polytrack
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
node scripts/dump_parts.mjs
python -m pytest
```

`scripts/dump_parts.mjs` serves that checkout once in headless Chromium and saves the physics meshes from the worker's Init message. The file is `polytrack_agent/sim/cache/track_parts.json`, about 8 MB, and it is gitignored. Training after that dump does not open a window. The script looks for `google-chrome`, `chromium`, or a Playwright Chromium under the home directory. Set `CHROME` when the binary is somewhere else.

```bash
node scripts/dump_parts.mjs "$POLYTRACK_ROOT" /path/to/track_parts.json
```

## Train and play

```bash
python scripts/train.py
python scripts/play.py --episodes 5
```

`scripts/train.py` runs until you stop it. It writes `runs/summer1/last.pt`, `best.pt`, and `log.csv`. Play loads `runs/summer1/best.pt` and prints finish, time, progress, and checkpoints for each greedy episode.

`POLYTRACK_ROOT` overrides `game.root`. Both commands also take `--config some.yaml`, merged onto `polytrack_agent/default_config.yaml`. Useful fields:

| Field | Meaning |
| --- | --- |
| `game.root` | schoolagy `polytrack` directory, when `POLYTRACK_ROOT` is unset |
| `game.track` | `.track` export, relative to `game.root` unless absolute |
| `game.parts` | mesh cache. Empty uses `polytrack_agent/sim/cache/track_parts.json` |
| `keys` | subset of `w a s d` |
| `rl.action_repeat` | physics frames per decision |
| `rl.timeout_s` / `rl.stuck_s` | episode limits |
| `rl.group_size` | laps collected before one PPO update |
| `rl.out_dir` | checkpoint directory |
| `reward.progress_scale` / `time_penalty` / `finish_bonus` | reward terms |

The default track is `tracks/official/summer1.track`. Any other official or community `.track` in that checkout works after the same parts dump. The dump is the page's track-piece catalog, shared across tracks. Mountains and the car hull are built by the host from the game's own table and `models/car.glb`.

## Layout

```
polytrack_agent/sim/     Node host, Python client, generated mesh cache
polytrack_agent/rl/      observation, track env, PPO
scripts/dump_parts.mjs   one-time mesh dump
scripts/train.py
scripts/play.py
```
