# World model

This branch learns how the car moves, then trains a policy that can drive more than one track.

The environment is the same polytrack fork 0.6.3 worker. The world model predicts the next numeric state from the current state and the key hold. The general policy trains against that model and is checked on the worker across the official tracks in `polytrack-0.6.3/tracks/official`.

A single-track perfect lap stays on `specialist`. This branch does not chase a Summer 1 time.

Nothing here is a world model yet. The worker, the observation, and the PPO trainer on `main` are the starting point. The model and the general policy land in this branch.

The program steps the local worker. It does not open an external page, read its memory, or submit a time.
