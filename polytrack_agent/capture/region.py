"""Screen rectangles shared by the capture backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_mss(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    def cropped(self, crop: dict | None) -> Rect:
        crop = crop or {}
        inset_left = int(crop.get("left", 0) or 0)
        inset_top = int(crop.get("top", 0) or 0)
        inset_right = int(crop.get("right", 0) or 0)
        inset_bottom = int(crop.get("bottom", 0) or 0)
        width = self.width - inset_left - inset_right
        height = self.height - inset_top - inset_bottom
        if width < 16 or height < 16:
            raise ValueError(
                f"Region {self.width}x{self.height} with crop "
                f"left={inset_left} top={inset_top} right={inset_right} "
                f"bottom={inset_bottom} leaves {width}x{height}"
            )
        return Rect(self.left + inset_left, self.top + inset_top, width, height)
