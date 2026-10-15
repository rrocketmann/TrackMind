# PolyTrack vision agent

A vision-only controller for the **official** PolyTrack window at [polytrack.dev](https://polytrack.dev) or [kodub.itch.io/polytrack](https://kodub.itch.io/polytrack). It captures the screen and presses keys. It does not fork the game, read its memory, scrape the page, or open its sockets.

The policy is GRPO on lap numbers: progress along a reference lap, how much that progress just moved, how sure the match is, time, stall, and the previous key chord. It picks one of 27 key holds. It does not see the picture. A group of real laps is scored together, and each lap is judged against the others in that group. The picture is only the progress sensor.

The game is the official site in Chromium, `https://polytrack.dev`. Train and play look for a Chromium window titled PolyTrack and open that URL in your normal Chromium profile when it is not already open. That profile is what carries the nickname, so a real finish can be submitted by the game.

Air pitch and roll in stock PolyTrack reuse WASD while the car is airborne. `q` and `e` are optional extra outputs for a custom bind.

A real finish in the official window is submitted by the game, as its own recording, when you are logged in with a nickname and the client’s determinism check passes. The agent waits a few seconds after a finish so that submit can complete. It does not POST a time itself. The unofficial local copy of the game cannot access the leaderboard. A time that is outside the loaded top 10 and is not a personal best is skipped by the client.

## Layout

```
polytrack_agent/     vision capture, dataset, DINOv3 policy, control, play loop
configs/default.yaml same defaults as polytrack_agent/default_config.yaml
scripts/             record, train, play, calibrate, make_reference
```

`PolyTrackVisionEnv` in `polytrack_agent/env.py` is the later RL hook: `reset` / `step` return stacked RGB frames and a reward built from visual lap progress. It still does not press the restart key for you.

## Setup

Linux needs an X11 session. `pynput` cannot read or send global keys on Wayland. Windows uses the same code; `dxcam` is the fast capture backend there.

```bash
cd polytrack-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Windows, optional faster capture:
pip install -r requirements-windows.txt
python -m pytest
```

Chromium is opened for you if it is not already on the game:

```bash
chromium --app=https://polytrack.dev --class=polytrack-official
```

Use a borderless or fullscreen app window so the capture is the game, not the browser toolbar. Stay logged in with a nickname in that same profile.

Use the same window size, zoom (100%), and graphics settings for recording and play. Hide the cursor over the game if it sits on the road.

Official driving keys are WASD or the arrow keys. `R` or Enter restarts. This agent will not press `R`, Enter, or Escape. You restart the track.

## Calibrate

```bash
python scripts/calibrate.py --preview shots/preview.png --write config.local.yaml
```

Open `shots/preview.png` and confirm it is the game view, not your desktop. If the browser toolbar is in the shot, set `capture.crop` or fullscreen the game and run calibrate again.

`config.local.yaml` is a small overlay (fixed region, `follow_window: false`). A fixed region is the low-latency choice. Leaving `region: null` makes the agent follow a window whose title contains `PolyTrack`.

Pass `--config config.local.yaml` to every later command.

## Record

Focus the game. Release WASD, start the recorder, then drive. Keys that were already held when the listener starts are invisible until you press them again.

```bash
python scripts/record.py --config config.local.yaml
```

| Key | Action |
| --- | --- |
| F6 | Pause or resume sampling |
| F7 | New episode. Press this on every restart so laps stay separate |
| F8 | Save and quit |

The recorder only listens. It does not press keys. At the default 320×180 RGB and 30 fps, expect about 5 MB per minute. Sessions land in `data/recordings/session-*` as numpy chunks of frames plus a multi-hot key mask.

Record one track per dataset. Include:

- several clean laps at the pace you want copied
- misses and the recovery back onto the racing line
- jumps, so airborne WASD is in the pictures

Ten to thirty minutes of that mix is a reasonable first set. More clean laps pull the policy toward your fast line. Recoveries are what let it finish after a mistake.

Arrow keys, or `space` if your build actually uses it, go in `record.keys`. Do not add `r`.

## Train

Load Summer 1 in the official window, build a reference lap, then:

```bash
python scripts/train.py --config config.local.yaml --reference data/summer1-reference.npz
```

GRPO drives groups of 8 laps, about 10 decisions per second, and the actor update is the small numeric policy only. Checkpoints go to `runs/grpo/`. `best.pt` stores that actor. `device: auto` uses CUDA when a GPU is present and CPU otherwise. The RTX 3050 in the custom PC has 8 GB, and this laptop has no 3050, so training stays on CPU here.

The agent taps `R` between episodes. After a finish it releases the driving keys and waits `rl.finish_settle_s` (3 seconds) before restarting, so Chromium can submit the recording. Keep that Chromium window focused. Ctrl+C releases every key and exits.

One reference video is required. It is the progress sensor, not a set of action labels. Record one clean Summer 1 lap, then `scripts/make_reference.py`.

## Offline agreement

```bash
python scripts/play.py --config config.local.yaml \
  --checkpoint runs/bc/best.pt \
  --offline data/recordings/session-YYYYMMDD-HHMMSS
```

This replays recorded pictures through the policy and prints per-key accuracy and F1. A DINOv3 checkpoint encodes every window again here, so it is slower than live play, which encodes only the newest frame. It does not open the game and does not press keys. Steer F1 around 0.7 or better on held-out laps is a sane bar before live play. If offline steering is wrong, live play will be wrong.

## Visual progress

Progress is a match between the live picture and a strip of tiny grayscale frames from one of your laps. A jump back toward the start of that strip is a restart. Reaching the end and staying there is a finish. There is no OCR and no timer read.

Mark one clean lap with F7 so it is its own episode, then:

```bash
python scripts/make_reference.py --config config.local.yaml \
  --session data/recordings/session-... \
  --out data/reference.npz
```

Pass `--episode N` to pin a lap. Otherwise the longest episode is used. Point `play.reference` at the file, or pass `--reference data/reference.npz` when you play.

The play log stores progress and the lap reward (forward progress, a time cost, a restart penalty, a finish bonus). GRPO compares the sum of that reward across the laps in a group. `PolyTrackVisionEnv.step` returns the per-decision reward.

## Play

```bash
python scripts/play.py --config config.local.yaml --checkpoint runs/grpo/best.pt --episodes 5 --reference data/summer1-reference.npz
```

That is five episodes on whatever track is already loaded. Each row is finish, seconds, and max visual progress. A finish waits so the official client can submit, then the next episode taps `R`.

F8 or Ctrl+C releases every key and exits. Keys are sent only while the foreground window title contains `PolyTrack` (`play.require_focus`). Alt-tab away and the agent lets go.

Watch a dry run first. It prints the keys it would hold and presses nothing:

```bash
python scripts/play.py --config config.local.yaml --checkpoint runs/grpo/best.pt --dry-run
```

`play.mode` defaults to `discrete`: a key presses above `threshold_on` (0.62) and releases below `threshold_off` (0.38). Opposite steer, throttle, and air keys cannot be held together. `continuous` duty-cycles a key for part of the 30 Hz tick. PolyTrack reads digital keys, and very short taps are easy for the browser to miss, so leave `discrete` on unless you are experimenting.

A GRPO checkpoint reads the lap numbers and presses one chord. It does not run a vision network. A vision checkpoint still grabs a frame, infers, and turns per-key probabilities into holds. The small CNN is a few milliseconds on CPU. A fixed capture region avoids extra window lookups.

## Config

Edit `config.local.yaml` or `configs/default.yaml`. Important fields:

| Field | Role |
| --- | --- |
| `capture.backend` | `auto` (dxcam on Windows, mss elsewhere), `mss`, or `dxcam` |
| `capture.region` | Screen rectangle in pixels. Fixed is faster and steadier |
| `capture.fps` | Record and play rate. 30 is the default |
| `record.keys` | Output order. Default `w a s d q e` |
| `record.size` | Stored frame size `[H, W]` |
| `train.backbone` | `dinov3` (default) or `cnn` |
| `train.dino_model` | Frozen Hugging Face id. Default is ViT-S/16 |
| `train.image_size` | Network input `[H, W]`. DINOv3 needs multiples of 16. Saved in the checkpoint |
| `train.pool` | Coarse token grid kept from the patch features, default `[4, 6]` |
| `train.stack` | How many past frames the policy sees |
| `rl.action_repeat` | Capture frames each discrete hold is kept. 3 is about 10 Hz |
| `rl.finish_settle_s` | Pause after a finish so the official client can submit |
| `capture.browser` | `chromium`. Capture ignores a PolyTrack window that is not Chromium |
| `capture.url` | Official page opened when Chromium is not already on the game |
| `rl.group_size` | Laps compared in one GRPO update. Default 8 |
| `play.threshold_on` / `threshold_off` | Hysteresis. Raise `threshold_on` if it steers too much |
| `play.min_key_freq` | Keys rarer than this in the demos are forced off |

The checkpoint's image size, stack, and key list win at play time. Record and play at the same resolution and crop.

## Adding RL later

`polytrack_agent/env.py` is the seam:

```python
env = PolyTrackVisionEnv(cfg, dry_run=False)
obs = env.reset()          # uint8 (stack, H, W, 3)
obs, reward, done, info = env.step(action)  # action[i] >= 0.5 presses key i
```

`info["progress"]` comes from the visual reference. `done` is a detected restart or finish. Wire that env to an RL algorithm in a new module. Keep the policy able to consume the same stacked frames so a behavior-cloning checkpoint can initialize it. Do not add a game-memory observation next to the pixels if you want the agent to stay vision-based.

## Troubleshooting

| Symptom | What to change |
| --- | --- |
| Keys go to the wrong window | `play.require_focus: true` and a title that matches the game. F8 releases keys |
| A key sticks after a crash | Ctrl+C or F8. If the process was killed, click the game and tap that key once |
| Preview shows the desktop or a toolbar | Fullscreen the game, set `capture.crop`, calibrate again |
| Offline steer F1 is poor | More laps, same graphics settings, one track per model, train longer |
| Live car only holds W | Steering F1 was the wrong number to ignore. Thresholds may also be too high to press A/D |
| Car oscillates in a turn | Raise `threshold_on` a little, or lower `threshold_off` so a steer releases less eagerly. Discrete mode should already be on |
| Recorder sees no keys | X11 session, not Wayland. Press the keys after the recorder starts |
| `dxcam` import error on Linux | Leave `capture.backend` on `auto` or `mss` |
| Progress never moves | Build `reference.npz` from a lap of this same track and window size |

## Tests

```bash
python -m pytest
```

`tests/test_core.py` covers resize, key hysteresis, visual restart detection, the dataset, the temporal transformer, and a headless env step. `tests/test_train_smoke.py` trains the small CNN on a synthetic video and checks that it imitates left and right from pixels. Neither test downloads DINOv3 or needs the game.
