# Specialist

This branch trains one policy until it matches the best archived lap of one track.

The track is whichever official track has the most laps in TrackData. That is Winter 4. PPO starts from a behavioral clone of the clean laps within 0.5 s of the fastest time there. The actor sees velocity, heading, the road edge, the next gate, and the previous chord. It does not see world position, acceleration, or gate distance.

The environment is the polytrack fork 0.6.3 worker. A perfect run is a finished lap at or under that best archived time. A dynamics model and a policy meant to transfer belong on `world-model`.

The program steps the local worker. It does not open an external page, read its memory, or submit a time.
