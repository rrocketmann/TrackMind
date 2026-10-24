#!/usr/bin/env python3
"""Race each button hold and keep the one that goes farthest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["search", *sys.argv[1:]]))
