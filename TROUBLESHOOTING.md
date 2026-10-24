# Troubleshooting (for the next session)

Everything is pushed to `main`. To continue on another machine:

```bash
git pull origin main
export POLYTRACK_ROOT=$PWD/polytrack-0.6.3
export PATH="$HOME/.local/node/bin:$PATH"   # node must be on PATH for the host
```

## Current plan

The trainer is discrete SAC (`python scripts/train.py`, checkpoints in `runs/sac`).
Each decision sees position, velocity, acceleration, meters of road left to the
side, and the next gate. PPO and the hold-search numbers below are the record
of what already failed. Do not resume `runs/summer1` into this SAC run.

## Open problem: the learner does not find the good action

No lap has ever finished. Train and play work, the env is correct, and the tests
pass, but PPO plateaus well short of checkpoint 0. This section records what has
actually been measured, so the next session does not repeat it.

### The core measurement

From the spawn, drive each action as a constant and score the whole episode
(`/tmp` probe, 3000-step cap, config `assist.yaml`):

| action | total reward | best progress | end state |
| --- | --- | --- | --- |
| `w` (throttle only) | **+8.38** | **0.539** | drives the chicane, clips the wall at x~245 |
| `wa` | -1.56 | 0.036 | stuck at x~313 |
| `wd` | -1.55 | 0.045 | stuck at x~313 |
| everything else | -3.0 .. +1.4 | 0.000 | stuck or reversed |

A constant `w` is worth **+10 reward more** than any other action, and PPO can
see that gap. The current policy nonetheless settles on `wa` with probability
0.87 and spins off the near edge. So this is not a reward-shaping problem and
not a missing-observation problem: **the learner is collapsing onto the wrong
action and not exploring the right one.**

The original simple run (before any of the shaping below) reached 0.519, which
is essentially what a constant `w` gets. That run was working by finding "hold
throttle", which is the ceiling of a straight line. Later reward work made this
worse, not better.

### What has been ruled out

- **Track geometry.** An earlier note claimed the opening chicane was
  undrivable. That was wrong and has been retracted. A straight `w` takes the
  whole chicane at 136 km/h with all four wheels down; it only clips the barrier
  because the straight line drifts to the lane edge (`z ~ -4` in a `z -5..+5`
  lane). The track is drivable.
- **Reward scale.** The returns that looked like they "improved" from -17 to
  -1.5 only shrank because the penalty coefficients were reduced. Coefficients
  are now sane for the value head, but that did not buy progress.
- **Missing slip signal.** The slip feature and `speed_reward` were added and are
  correct (slip reaches -34.7 deg on curb contact, verified live), but 280
  updates with them produced no progress.
- **`action_repeat`.** Switching 50 -> 10 was done on a "finer control" theory
  that a fixed-drive probe disproved. It also measurably hurt learning: 80
  updates gave 0.205 at repeat 50 and 25, but 0.159 at repeat 10. **Revert to 50.**

### Suspected causes, in order

1. **Forward-bias init ties the wrong actions.** Fixed. `bias_toward_forward`
   now raises only the straight-throttle logit. `wa` and `wd` stay at zero.
   The re-run below did not move the plateau, so the init was not sufficient.
2. **Run-to-run variance swamps short comparisons.** Fixed. `rl.seed` defaults
   to 0, and `--trials N` with `--max-updates` writes `summary.json` with the
   mean and sample standard deviation. The same config scored 0.125 in one
   60-update test and 0.205 in another before this.
3. **Credit assignment over long noisy episodes.** Episodes run ~1000 steps and
   end in different ways (beach at x~244, or stall at x~313). With `gamma=0.99`
   and `group_size=8`, PPO sees each trajectory once and discards it. On-policy
   is a poor fit for this.
4. **Exploration is not enforced.** `entropy_coef` is a fixed nudge the optimizer
   overpowers: entropy fell 2.2 -> 1.9 while collapsing onto `wa`.

### Algorithm notes (for the next session)

- **GRPO is probably not the fix.** It is PPO with a group-relative baseline
  instead of a value network: it changes advantage normalization, not
  exploration. It would inherit the same collapse. It also assumes a decent
  policy to start from, which is the thing we do not have.
- **SAC is the reasonable escalation.** Its entropy term lives in the objective
  with an auto-tuned temperature, rather than being a fixed coefficient, and
  off-policy replay reuses each transition many times, which directly attacks
  the credit-assignment problem in (3). Caveat: it is built for continuous
  actions and needs the discrete variant here, so it is more moving parts.
- Only `torch` and `numpy` are installed. There is no gymnasium, SB3, or ray.
  Anything else is either written here or added as a dependency.

### Suggested order

Done, and it did not move the plateau. `action_repeat` was already 50.

1. Straight-throttle bias only.
2. Fixed seeds and mean-of-N reporting.
3. Re-ran PPO, then discrete SAC. See the measurements below.

### Measured after the bias and seed change

Three seeds (0, 1, 2), 60 updates, `group_size` 4, summer1. The number is the
mean progress of the steps in one update. "Best" is the highest update, "last"
is update 60. Both are mean ± sample standard deviation across the three seeds.

| run | best | last | spawn policy |
| --- | --- | --- | --- |
| PPO, straight-throttle bias | 0.199 ± 0.022 | 0.081 ± 0.023 | w 0.30, wa 0.54, wd 0.23 |
| SAC, target entropy `0.5 log\|A\|` | 0.158 ± 0.021 | 0.017 ± 0.022 | peaked on `as` (0.84) or `wd` (0.51, 0.97). Critic stayed finite. |
| SAC, target entropy `0.98 log\|A\|` (the default) | 0.197 ± 0.062 | 0.024 ± 0.013 | near uniform, straight throttle about 0.11–0.14. Critic loss grew past 800. |

Constant `w` is still +8.38 and progress 0.539. Logs and checkpoints:

- `runs/w-bias-seeded`
- `runs/sac-seeded` (the lower entropy target)
- `runs/sac-target98`

SAC did not beat PPO. The default trainer is still PPO. `--algo sac` is the
discrete actor-critic. Do not treat another 60-update sweep of the same setup
as a new question.

## Whole-race hold search

`python scripts/search.py` drives each chord for a whole episode and keeps the
one that gets farther. A steer tap is kept only when that full race beats the
champion. On summer1, with the default limits:

| hold | progress | how it ended |
| --- | --- | --- |
| `w` | 0.539 | beached at 5.0 s |
| `wd` | 0.045 | stuck |
| `wa` | 0.036 | stuck |
| everything else | 0.000 | stuck or fell |

The worker used to stamp each button with wall-clock time since boot. After a
few races a steer was scheduled thousands of frames late, so the same plan
drove a different line. Samples are stamped on the simulation frame now. Replay
the same plan three times and the progress matches to the reported digits.
`runs/hold/plan.json` is from before that fix. Do not replay it.

With the stamp fixed, straight `w` is still the best constant hold (0.539,
beached at 5.0 s). One tap beats it: hold `wa` while progress is 0.35–0.40,
then `w`. That race reaches progress 1.155 (past checkpoint 1) in 7.40 s and
falls at x≈126, y≈47. The early window `wa` at 0.05–0.10, which the old search
credited with progress 1.15, now stalls at 0.147. That 1.15 was the late stamp
landing the steer near 0.35.

A second tap does not stop the fall. The farthest one is another `wa`, while
progress is 0.55–0.70. The full race reaches progress 1.237 in 7.10 s and
falls at x≈94, y≈47. Nothing that stayed on the road got as far as 1.155.
Right-hand taps do not beat the one-tap line. The saved plan is
`runs/hold-frames/plan.json`. Replay it with
`python scripts/search.py --plan runs/hold-frames/plan.json`.

The fall is a jump, not a corner missed by one more steer. Gate 0 is at
x=180, y=55, z=0. The next road piece is at x=140, y=45, z=0. The saved line
leaves the lip at about 190 km/h with the wheels still down through x≈165,
then flies. At x≈140 it is still near y=55 and z≈5, about 10 m above the
road piece and off to the side, so it passes over the landing and drops to
the fall line at x≈94, y≈47.

A third tap on that line does not land the jump. Steering, coasting, and
braking were tried from progress 0.70 to 1.50. Anything after progress 1.05
matches the untapped fall, because the car is already in the air. On the lip,
a short brake or coast still falls around 1.16–1.23. A longer brake
(`s` while 0.70–0.85) sticks at 0.685 and never reaches the jump. No replay
stayed up and still reached 1.24. The plan file was left as the two-tap line.
The attempts are in `runs/hold-frames/correction.json`.

Do not add a fourth tap on this plan. The next lesson is a slower arrival at
the lip (x≈180, z≈0, well under 190 km/h) that still has the speed to reach
x=140. Keep it only when a replay has the wheels down again after x=140.
Do not go back to PPO until that sticks.

The env work below (velocity, slip, contact, the stall-floor bug) is sound and
worth keeping regardless of which learner is used.

## Train until a lap is under a target time

```bash
python scripts/train.py --stop-time 25
```

Training logs every finish with its lap time and the best lap of the run, saves
`last.pt`, and exits on the first lap at or below 25 s. The same field is
`rl.stop_time` in a config overlay. Without `--stop-time`, training runs until
you stop it.

## The car drives off the racing line and hits a wall

A throttle-only policy tracks the lane through the opening chicane at over
100 km/h with all four wheels down, then hits the barrier at `x ~ 249`. The
crash is not a fall: speed collapses from ~137 to ~38 km/h in one step while
the car climbs, and the chassis velocity gains a large upward component. The
straight line has drifted to `z ~ -4` and the lane there spans `z -5..+5`, so
the car is riding the outermost edge and clips the wall. The track is drivable;
the policy just needs to hold the lane, not merely hold throttle.

`speedKmh` cannot show this. It is derived from wheel rotation
(`getCurrentSpeedKmHour`), so it still reads tens of km/h while the car is
pinned against a wall or spinning in mid-air. The host now also reports the
chassis velocity from `getLinearVelocity`, which is the honest motion, and it
drives the stall timer and the observation's ground-speed feature.

The game's own steering is a slip-cancelling assist, not raw input: it
counter-steers by `slip * min(1, speed / 30)` and is disabled (`f = 0`) whenever
a front wheel leaves the ground. That is why the car steers itself at speed but
nothing helps once a wheel is on a curb, and why speed with wheels down is the
regime worth rewarding:

- `reward.speed_reward` pays each step for ground speed while the wheels are
  down, capped by `reward.speed_cap_kmh`, so the policy is pushed toward the
  speed where the assist actually works. It cannot be collected while beached.

## The car leaves the road and cannot move again

When the car drifts off the track it lands on its chassis with every wheel in
the air. Throttle, brake, and steering all do nothing from there, and the
`speedKmh` field still reads tens of km/h, so speed is not a usable signal for
"is it moving".

The env reads wheel contact from the car state:

- `reward.contact_reward` pays each step, per wheel on the surface, scaled by the
  distance the car actually travelled, so parking on the grid earns nothing.
- `reward.beach_penalty` is charged once when every wheel has left the ground for
  `rl.airborne_grace_s` and the car is not moving. That state also ends the
  episode, unless `rl.auto_respawn` is on.
- `reward.stall_penalty` costs each step spent on the road without covering
  ground, so sitting still is worse than driving.
- The stuck timer uses ground displacement plus wheel contact, not `speedKmh`.

All of these default to off, so existing runs behave exactly as before until a
config overlay turns them on.

## Fixed in this session

- `polytrack_agent/sim/client.py` had been deleted while `rl/track.py` still
  imported `SimClient`; it has been restored. Training and the test suite work
  again.
- `polytrack_agent/rl/episodes.py` had been deleted, leaving `play.py` with a
  stub that always reported `0.0` seconds. It has been restored (without video),
  so `play.py` reports real finish, time, progress, and checkpoints again.
- The host's `patchWorker` now matches the 0.6.3 worker: `e.data.trackData`, a
  `fromExportString` fallback + `toSaveString()` for raw `.track` files, no car
  spawn when `carId == null`, and the `UpdateResult` (messageType 10)
  `carStates` objects instead of `carStateBuffers`.

## Car did not move (fixed)

The offscreen worker flushes control samples only inside its realtime loop,
keyed by wall-clock frames. The host pumps the loop faster than wall clock
advances, so samples piled up at future frames and the car never moved.
`patchWorker` now applies the latest pending sample directly to the live
controls holder before each state read, mirroring the realtime loop. A
throttle-only policy reaches ~100 km/h and passes the first checkpoints.

## Commands on a fresh machine

```bash
export POLYTRACK_ROOT=$PWD/polytrack-0.6.3
node scripts/dump_parts.mjs "$POLYTRACK_ROOT"   # once, fills sim/cache/track_parts.json
python scripts/train.py --stop-time 25
python scripts/play.py --episodes 5
```

## What's in the repo

- `polytrack-0.6.3/` — the polytrack fork (simulation_worker.bundle.js,
  main.bundle.js, tracks/, models/, lib/, audio/)
- `polytrack_agent/sim/host.mjs` — Node host that boots the worker and patches it
- `polytrack_agent/sim/client.py` — JSON-line client the Python env talks to
- `polytrack_agent/rl/` — env, PPO trainer, greedy episodes, checkpoints
- `scripts/train.py`, `scripts/play.py`, `scripts/dump_parts.mjs`
