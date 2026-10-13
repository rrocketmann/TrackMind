"""Listen to human keys while recording, and press keys while playing.

Injected keys are limited to the configured driving set. F8 is the stop
hotkey and is never part of the action vector.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time

import numpy as np

from polytrack_agent.input.keys import key_name, pynput_key

log = logging.getLogger(__name__)


class KeyObserver:
    """Background listener. Keys already held when it starts are missed until re-pressed."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self._watch = set(self.keys)
        self.state = {key: False for key in self.keys}
        self._lock = threading.Lock()
        self._mark = False
        self._hot_down: set[str] = set()
        self.paused = False
        self.stop_event = threading.Event()
        self._listener = None

    def start(self) -> None:
        try:
            from pynput.keyboard import Listener
        except ImportError as exc:
            raise RuntimeError(
                "pynput is not installed. From the project directory: pip install -r requirements.txt"
            ) from exc

        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.start()

    def stop_listening(self) -> None:
        self.stop_event.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def consume_mark(self) -> bool:
        with self._lock:
            marked = self._mark
            self._mark = False
            return marked

    def snapshot(self, keys: list[str] | None = None) -> np.ndarray:
        order = self.keys if keys is None else keys
        with self._lock:
            return np.array([1 if self.state.get(key, False) else 0 for key in order], dtype=np.uint8)

    def _on_press(self, key) -> None:
        name = key_name(key)
        if name is None:
            return
        if name in {"f6", "f7", "f8"}:
            if name in self._hot_down:
                return
            self._hot_down.add(name)
            if name == "f6":
                self.paused = not self.paused
                log.info("Recording %s", "paused" if self.paused else "resumed")
            elif name == "f7":
                with self._lock:
                    self._mark = True
            else:
                self.stop_event.set()
            return
        if name in self._watch:
            with self._lock:
                self.state[name] = True

    def _on_release(self, key) -> None:
        name = key_name(key)
        if name is None:
            return
        self._hot_down.discard(name)
        if name in self._watch:
            with self._lock:
                self.state[name] = False


class StopHotkey:
    """F8 stop listener used while the agent is driving."""

    def __init__(self):
        self.event = threading.Event()
        self._listener = None

    def start(self) -> None:
        try:
            from pynput.keyboard import Listener
        except ImportError as exc:
            raise RuntimeError(
                "pynput is not installed. From the project directory: pip install -r requirements.txt"
            ) from exc

        self._listener = Listener(on_press=self._on_press, suppress=False)
        self._listener.start()

    def _on_press(self, key) -> None:
        if key_name(key) == "f8":
            self.event.set()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class KeyOutput:
    """Press and release a fixed set of keys. `release_all` is idempotent."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.held: set[str] = set()
        self._controller = None
        self._cache: dict[str, object] = {}
        if enabled:
            atexit.register(self.release_all)

    def apply_mask(self, mask: dict[str, bool]) -> None:
        if not self.enabled:
            return
        for name, down in mask.items():
            if down and name not in self.held:
                self._press(name)
                self.held.add(name)
            elif not down and name in self.held:
                self._release(name)
                self.held.discard(name)

    def tap(self, name: str, hold_s: float = 0.08) -> None:
        """Press and release one key. It is not left in the driving hold set."""
        if not self.enabled:
            return
        self._press(name)
        time.sleep(max(0.0, float(hold_s)))
        self._release(name)
        self.held.discard(name)

    def release_all(self) -> None:
        for name in list(self.held):
            try:
                self._release(name)
            except Exception:
                log.exception("Failed to release %s", name)
            self.held.discard(name)

    def _key(self, name: str):
        cached = self._cache.get(name)
        if cached is None:
            cached = pynput_key(name)
            self._cache[name] = cached
        return cached

    def _controller_get(self):
        if self._controller is None:
            from pynput.keyboard import Controller

            self._controller = Controller()
        return self._controller

    def _press(self, name: str) -> None:
        self._controller_get().press(self._key(name))

    def _release(self, name: str) -> None:
        if self._controller is None:
            return
        self._controller.release(self._key(name))
