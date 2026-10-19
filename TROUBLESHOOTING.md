# Troubleshooting (for next session on your Fedora machine)

You lose SSH access to this machine, but everything is pushed to `main` at
`https://github.com/rrocketmann/TrackMind.git`. Continue on your Fedora box:

```bash
cd ~/dev/polytrack-agent
git pull origin main
export POLYTRACK_ROOT=$PWD/polytrack-0.6.3
```

## Known remaining error

`play.py` hits `TypeError: Cannot read properties of undefined (reading 'length')`
inside `sw.fromSaveString` in the fork's `simulation_worker.bundle.js`. The host's
`patchWorker` intercepts `X_.CreateCar` and rebuilds the track from `r.trackData`,
but the fork expects a different save format than `trackText` (the raw `.track` file).

### Where to fix

`polytrack_agent/sim/host.mjs`, function `patchWorker`, the `X_.CreateCar` case.
Currently it sends `save: r.trackData` — the fork may need a parsed/saved track
object instead of the raw text. Also `boot()` sends `messageType: 3` with
`trackData: trackText` — verify that's what `X_.CreateCar` expects.

Look at how the fork's worker deserializes the track:
```
sw.fromSaveString(e.data.trackData)
```

## Commands to run on Fedora (when fixed)

```bash
export POLYTRACK_ROOT=$PWD/polytrack-0.6.3
node scripts/dump_parts.mjs "$POLYTRACK_ROOT"   # already done once
python scripts/play.py --checkpoint runs/summer1/last.pt --episodes 5 --record-video runs/summer1/ai_pass.mp4
git add -f runs/summer1/ai_pass.mp4
git commit -m "Record AI gameplay on polytrack fork"
git push origin main
```

## What's in the repo

- `polytrack-0.6.3/` — full polytrack fork (simulation_worker.bundle.js, main.bundle.js,
  tracks/, models/, lib/, audio/)
- `runs/summer1/gate.mp4` — AI gate pass video (30s)
- `runs/summer1/last.pt` — trained 24-input actor (loads with current observation)
- `runs/summer1/best.pt` — older 21-input actor (won't load, observation changed)
- `runs/real_polytrack.mp4` — real PolyTrack page capture (15s, error screen)
- `scripts/record.mjs` — record real PolyTrack page (needs display+GPU)
- `scripts/play.py --record-video` — record AI on local fork (headless via Xvfb+ffmpeg)
- `polytrack_agent/sim/host.mjs` — simulation host with patchWorker (needs the fix above)
- `polytrack_agent/sim/client.py` — Python client with start_recording/stop_recording
- `polytrack_agent/rl/episodes.py` — video recording in run_episodes
