# Specialist

This branch trains one policy until it finishes one track cleanly.

The track is Summer 1 (`tracks/official/summer1.track`). The environment is the polytrack fork 0.6.3 worker in `polytrack-0.6.3`. The learner is the PPO actor and critic already in `polytrack_agent/rl`. A perfect run is a finished lap at or under the seconds passed to `python scripts/train.py --stop-time`.

Work here stays on that lap: the Summer 1 observation, the progress reward, and the policy that drives it. A dynamics model and a policy meant to transfer to other tracks belong on `world-model`.

The program steps the local worker. It does not open an external page, read its memory, or submit a time.
