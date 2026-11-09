# Universal PolyTrack world model

This branch learns one model of how a PolyTrack 0.6.3 car moves, on any official track, so a policy can train inside that model.

The real environment is the local worker and `polytrack_physics.wasm`. A decision holds W, A, S, and D for 50 physics frames, and the worker ticks at 1 ms. The network that picks the chord is small. Almost all of the wait is the physics. Stepping several cars at once still pays that cost on every decision. A world model is a learned step: current car, keys held, and the road around the car in, the next car out. Policy search then happens in that step. The worker is where the model gets its data, and where a policy is checked.

That is the efficiency. The same hour of wasm time covers many more attempts, because most of those attempts never call the worker.

## One model for every track

The car is the same physics on every official track in the 0.6.3 checkout. The roads are not. The model takes the piece of track the car is on, the edge of the road, and the next gate, so the same weights can leave Winter 4 and drive another track. A model that only replays one layout is a specialist, and that work stays on `specialist`.

The prediction is numeric. It is the car state the worker already returns:

- position, velocity, and acceleration
- heading on the ground, uprightness, and how many wheels are down
- meters of road left to the side, and which side the edge is on
- where the next gate sits in the car frame, and how far it is
- the next checkpoint divided by the checkpoint count

That is 19 numbers. Position, acceleration, and gate distance are part of the dynamics, so they stay. The action is one of the nine exclusive chords of W, A, S, and D, held for the same 50 ms the worker uses.

TrackData is the store of real laps across tracks. Those rows are the first training set: a state, a chord, and the state that followed. Worker rollouts add the cases a clean lap never shows, such as a car off the road or stopped against a wall.

## Training a policy in the model

Fit the model on those transitions. Then train the policy by rolling the model forward from real states. The score is the driving reward: forward travel along the track, speed toward the next gate, and a cost when chassis speed falls off suddenly. Discount and the episode rules stay the ones the worker uses, so a lap imagined in the model is aiming at the same thing as a lap in the game.

Every so often, play the policy in the worker on the official tracks. Where the real lap and the imagined lap disagree, add those steps to the model and fit it again. The worker remains the check. A policy that only looks good inside the model is not done.

A finished lap at an archived time is the job of `specialist`. This branch is the shared dynamics, and a policy that can use them on more than one track.

## Boundaries

Checks use the local 0.6.3 worker. This branch does not open an external page, read its memory, or submit a time.

## Status

No model is in this tree. `main` and `specialist` keep the worker, the trainer, and the game. The commit before this README, `41e3123`, is the last copy of that tree on this branch.
