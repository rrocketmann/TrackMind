"""Windows DXGI capture via dxcam. Region coordinates are relative to one output."""

from __future__ import annotations

import logging

import numpy as np

from polytrack_agent.capture.region import Rect

log = logging.getLogger(__name__)


class DxcamCapture:
    def __init__(
        self,
        region: Rect,
        origin: tuple[int, int],
        device_idx: int = 0,
        output_idx: int = 0,
    ):
        self.region = region
        self.origin = origin
        self.device_idx = int(device_idx)
        self.output_idx = int(output_idx)
        self._cam = None
        self._warned = False

    def set_region(self, region: Rect) -> None:
        self.region = region

    def _region_tuple(self) -> tuple[int, int, int, int]:
        left = int(self.region.left - self.origin[0])
        top = int(self.region.top - self.origin[1])
        right = left + int(self.region.width)
        bottom = top + int(self.region.height)
        if (left < 0 or top < 0) and not self._warned:
            log.warning(
                "Capture region starts outside the dxcam output at %s. "
                "Set capture.monitor to the display that shows the game.",
                self.origin,
            )
            self._warned = True
        left = max(0, left)
        top = max(0, top)
        right = max(left + 16, right)
        bottom = max(top + 16, bottom)
        return (left, top, right, bottom)

    def grab(self) -> np.ndarray | None:
        if self._cam is None:
            try:
                import dxcam
            except ImportError as exc:
                raise RuntimeError(
                    "dxcam is not installed. On Windows: pip install -r requirements-windows.txt"
                ) from exc

            self._cam = dxcam.create(
                device_idx=self.device_idx,
                output_idx=self.output_idx,
                output_color="RGB",
            )
        frame = self._cam.grab(region=self._region_tuple(), new_frame_only=False)
        if frame is None:
            return None
        return np.array(frame[:, :, :3], dtype=np.uint8, copy=True)

    def close(self) -> None:
        if self._cam is None:
            return
        try:
            release = getattr(self._cam, "release", None)
            if callable(release):
                release()
            else:
                self._cam.stop()
        except Exception:
            log.debug("dxcam shutdown failed", exc_info=True)
        self._cam = None
