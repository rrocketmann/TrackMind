#!/usr/bin/env python3
"""Record PolyTrack frames and the keys you are holding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["record", *sys.argv[1:]]))
