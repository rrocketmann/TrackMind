#!/usr/bin/env python3
"""Train PPO on the schoolagy PolyTrack 0.5.2 simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
