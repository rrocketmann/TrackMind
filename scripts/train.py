#!/usr/bin/env python3
"""Train PPO on the PolyTrack 0.6.3 fork simulation.

Graphs of the run: python scripts/live.py --run runs/<name>
A chase-camera video of a checkpoint: python scripts/record.py --checkpoint <pt> --out <mp4>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
