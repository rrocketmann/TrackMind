#!/usr/bin/env python3
"""Save a preview frame and a fixed capture rectangle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["calibrate", *sys.argv[1:]]))
