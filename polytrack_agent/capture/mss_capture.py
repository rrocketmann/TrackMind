"""mss capture. Works on Linux and Windows. Frames are copied out as RGB."""

from __future__ import annotations

import numpy as np

from polytrack_agent.capture.region import Rect
from polytrack_agent.obs.preprocess import bgra_to_rgb


class MssCapture:
    def __init__(self, region: Rect):
        self.region = region
        self._sct = None

    def set_region(self, region: Rect) -> None:
        self.region = region

    def grab(self) -> np.ndarray | None:
        if self._sct is None:
            try:
                import mss
            except ImportError as exc:
                raise RuntimeError(
                    "mss is not installed. From the project directory: pip install -r requirements.txt"
                ) from exc
            self._sct = mss.mss()
        shot = self._sct.grab(self.region.as_mss())
        bgra = np.array(shot, dtype=np.uint8, copy=True)
        return bgra_to_rgb(bgra)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
