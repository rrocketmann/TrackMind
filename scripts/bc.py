#!/usr/bin/env python3
"""Clone numeric TrackData laps into the PPO actor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["bc", *sys.argv[1:]]))
