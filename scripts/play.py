#!/usr/bin/env python3
"""Run greedy episodes in the schoolagy PolyTrack 0.6.3 simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["play", *sys.argv[1:]]))
