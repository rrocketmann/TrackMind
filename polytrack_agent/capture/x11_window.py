"""Grab an X11 window's own pixels.

On XWayland the root pixmap is black, so a screen grab misses the game.
Reading the window drawable still returns the frame Chromium painted.
"""

from __future__ import annotations

import ctypes
import logging

import numpy as np

from polytrack_agent.capture.region import Rect
from polytrack_agent.obs.preprocess import bgra_to_rgb

log = logging.getLogger(__name__)

_ZPIXMAP = 2
_REVERT_TO_PARENT = 2
_DATA = 16
_BYTES_PER_LINE = 44
_BITS_PER_PIXEL = 48
_RED_MASK = 56
_BLUE_MASK = 72
_DESTROY = 96


def window_crop(window: Rect, request: Rect, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """Map a screen rectangle onto pixels inside the window image."""
    if image_w < 1 or image_h < 1:
        return 0, 0, 0, 0
    if window.width <= 0 or window.height <= 0:
        return 0, 0, image_w, image_h
    scale_x = image_w / float(window.width)
    scale_y = image_h / float(window.height)
    x = int(round((request.left - window.left) * scale_x))
    y = int(round((request.top - window.top) * scale_y))
    width = int(round(request.width * scale_x))
    height = int(round(request.height * scale_y))
    x = max(0, min(image_w - 1, x))
    y = max(0, min(image_h - 1, y))
    width = max(1, min(image_w - x, width))
    height = max(1, min(image_h - y, height))
    return x, y, width, height


def _load_x11():
    lib = ctypes.CDLL("libX11.so.6")
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    lib.XFlush.argtypes = [ctypes.c_void_p]
    lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XGetGeometry.restype = ctypes.c_int
    lib.XGetGeometry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.XTranslateCoordinates.restype = ctypes.c_int
    lib.XTranslateCoordinates.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
    ]
    lib.XGetImage.restype = ctypes.c_void_p
    lib.XGetImage.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_ulong,
        ctypes.c_int,
    ]
    lib.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    return lib


def _i32(addr: int, offset: int) -> int:
    return int(ctypes.c_int32.from_address(addr + offset).value)


def _u64(addr: int, offset: int) -> int:
    return int(ctypes.c_ulong.from_address(addr + offset).value)


def _ptr(addr: int, offset: int) -> int:
    return int(ctypes.c_void_p.from_address(addr + offset).value or 0)


def _destroy_image(image: int) -> None:
    fn = _ptr(image, _DESTROY)
    if not fn:
        return
    destroy = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
    ctypes.cast(fn, destroy)(ctypes.c_void_p(image))


def _image_rgb(image: int) -> np.ndarray:
    width = _i32(image, 0)
    height = _i32(image, 4)
    data = _ptr(image, _DATA)
    stride = _i32(image, _BYTES_PER_LINE)
    bits = _i32(image, _BITS_PER_PIXEL)
    red = _u64(image, _RED_MASK)
    blue = _u64(image, _BLUE_MASK)
    if width < 1 or height < 1 or data == 0 or stride < 1:
        raise RuntimeError("XGetImage returned an empty frame")
    raw = np.frombuffer(ctypes.string_at(data, stride * height), dtype=np.uint8).reshape(height, stride)
    if bits == 32 and red == 0xFF0000 and blue == 0xFF:
        pixels = raw[:, : width * 4].reshape(height, width, 4)
        return bgra_to_rgb(pixels)
    if bits == 32 and red == 0xFF and blue == 0xFF0000:
        pixels = raw[:, : width * 4].reshape(height, width, 4)
        return np.ascontiguousarray(pixels[:, :, :3])
    if bits == 24:
        pixels = raw[:, : width * 3].reshape(height, width, 3)
        if red == 0xFF:
            return np.ascontiguousarray(pixels)
        return np.ascontiguousarray(pixels[:, :, ::-1])
    raise RuntimeError(f"Unsupported X image {width}x{height} {bits}bpp masks {red:#x}/{blue:#x}")


class X11WindowCapture:
    """Copy RGB pixels from one X11 window. `region` is in screen pixels."""

    def __init__(self, window_id: int, region: Rect):
        self.window_id = int(window_id)
        self.region = region
        self._lib = None
        self._display = ctypes.c_void_p()

    def set_region(self, region: Rect) -> None:
        self.region = region

    def grab(self) -> np.ndarray | None:
        lib, display = self._open()
        rect = self._geometry(lib, display)
        if rect is None or rect.width < 1 or rect.height < 1:
            return None
        image = lib.XGetImage(
            display,
            ctypes.c_ulong(self.window_id),
            0,
            0,
            ctypes.c_uint(rect.width),
            ctypes.c_uint(rect.height),
            ctypes.c_ulong((1 << 64) - 1),
            _ZPIXMAP,
        )
        if not image:
            log.warning("XGetImage failed for window %s", hex(self.window_id))
            return None
        try:
            frame = _image_rgb(int(image))
        finally:
            _destroy_image(int(image))
        x, y, width, height = window_crop(rect, self.region, frame.shape[1], frame.shape[0])
        if width < 1 or height < 1:
            return None
        if x == 0 and y == 0 and width == frame.shape[1] and height == frame.shape[0]:
            return frame
        return np.ascontiguousarray(frame[y : y + height, x : x + width])

    def close(self) -> None:
        if self._lib is not None and self._display:
            self._lib.XCloseDisplay(self._display)
        self._display = ctypes.c_void_p()
        self._lib = None

    def _open(self):
        if self._lib is None or not self._display:
            lib = _load_x11()
            display = lib.XOpenDisplay(None)
            if not display:
                raise RuntimeError("XOpenDisplay failed. Is DISPLAY set?")
            self._lib = lib
            self._display = ctypes.c_void_p(display)
        return self._lib, self._display

    def _geometry(self, lib, display) -> Rect | None:
        root = ctypes.c_ulong()
        x = ctypes.c_int()
        y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        depth = ctypes.c_uint()
        ok = lib.XGetGeometry(
            display,
            ctypes.c_ulong(self.window_id),
            ctypes.byref(root),
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(border),
            ctypes.byref(depth),
        )
        if not ok or int(width.value) < 1 or int(height.value) < 1:
            return None
        root_window = lib.XDefaultRootWindow(display)
        screen_x = ctypes.c_int()
        screen_y = ctypes.c_int()
        child = ctypes.c_ulong()
        lib.XTranslateCoordinates(
            display,
            ctypes.c_ulong(self.window_id),
            root_window,
            0,
            0,
            ctypes.byref(screen_x),
            ctypes.byref(screen_y),
            ctypes.byref(child),
        )
        return Rect(int(screen_x.value), int(screen_y.value), int(width.value), int(height.value))


def focus_window(window_id: int) -> None:
    """Raise one X11 window and give it keyboard focus. Does not send keys."""
    lib = _load_x11()
    display = lib.XOpenDisplay(None)
    if not display:
        raise RuntimeError("XOpenDisplay failed. Is DISPLAY set?")
    try:
        window = ctypes.c_ulong(int(window_id))
        lib.XRaiseWindow(display, window)
        lib.XSetInputFocus(display, window, _REVERT_TO_PARENT, ctypes.c_ulong(0))
        lib.XFlush(display)
    finally:
        lib.XCloseDisplay(display)
