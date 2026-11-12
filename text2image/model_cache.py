"""A single-model in-process cache with idle-time eviction."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from typing import Any


class PipelineCache:
    def __init__(self, idle_timeout_seconds: int, enabled: bool = True) -> None:
        self.idle_timeout_seconds = idle_timeout_seconds
        self.enabled = enabled
        self._lock = threading.RLock()
        self._pipeline: Any | None = None
        self._model_key: str | None = None
        self._cleanup: Callable[[], None] | None = None
        self._timer: threading.Timer | None = None
        self._generation = 0

    def get_or_load(
        self,
        model_key: str,
        loader: Callable[[], Any],
        cleanup: Callable[[], None] | None = None,
    ) -> tuple[Any, bool]:
        with self._lock:
            self._cancel_timer_locked()
            if self._pipeline is not None and self._model_key == model_key:
                return self._pipeline, True

            self._evict_locked()
            pipeline = loader()
            self._pipeline = pipeline
            self._model_key = model_key
            self._cleanup = cleanup
            return pipeline, False

    def mark_idle(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            if not self.enabled or self.idle_timeout_seconds == 0:
                self._evict_locked()
                return

            self._cancel_timer_locked()
            generation = self._generation
            self._timer = threading.Timer(
                self.idle_timeout_seconds,
                self._expire,
                args=(generation,),
            )
            self._timer.daemon = True
            self._timer.start()

    def close(self) -> None:
        with self._lock:
            self._cancel_timer_locked()
            self._evict_locked()

    def _expire(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._pipeline is None:
                return
            model_key = self._model_key
            self._evict_locked()
        print(f"Idle model cache expired; unloaded {model_key}.")

    def _cancel_timer_locked(self) -> None:
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _evict_locked(self) -> None:
        had_pipeline = self._pipeline is not None
        cleanup = self._cleanup
        self._pipeline = None
        self._model_key = None
        self._cleanup = None
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if had_pipeline:
            gc.collect()
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as exc:  # Cleanup should never crash the CLI.
                    print(f"WARNING: Could not release cached device memory: {exc}")
