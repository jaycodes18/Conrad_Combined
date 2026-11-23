"""A small publish/subscribe hub that fans studio events out to every open browser tab (SSE)."""

from __future__ import annotations

import contextlib
import json
import math
import queue
import threading
from typing import Any

MAX_QUEUED_EVENTS = 2000


def _clean(value: Any) -> Any:
    """JSON-safe copy: NaN and infinity become None, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class EventHub:
    """Thread-safe broadcaster. Slow subscribers drop their oldest events instead of blocking producers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[str]] = []

    def subscribe(self) -> queue.Queue[str]:
        inbox: queue.Queue[str] = queue.Queue(maxsize=MAX_QUEUED_EVENTS)
        with self._lock:
            self._subscribers.append(inbox)
        return inbox

    def unsubscribe(self, inbox: queue.Queue[str]) -> None:
        with self._lock:
            if inbox in self._subscribers:
                self._subscribers.remove(inbox)

    def publish(self, kind: str, payload: dict[str, Any]) -> None:
        message = json.dumps({"type": kind, **_clean(payload)}, allow_nan=False)
        with self._lock:
            subscribers = list(self._subscribers)
        for inbox in subscribers:
            if inbox.full():
                with contextlib.suppress(queue.Empty):
                    inbox.get_nowait()
            try:
                inbox.put_nowait(message)
            except queue.Full:
                continue
