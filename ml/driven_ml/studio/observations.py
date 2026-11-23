"""Time-stamped sensor readings for multimodal fusion.

Every reading carries confidence, validity, and whether it came from hardware, a
prerecorded file, or a simulation. Invalid or missing sensors are not treated as
evidence of impairment.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SourceKind(StrEnum):
    HARDWARE = "hardware"
    PRERECORDED = "prerecorded"
    SIMULATED = "simulated"


class SensorStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    FAULT = "fault"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class SensorObservation:
    """One measurement at one time. ``value`` is None when the sensor cannot measure."""

    name: str
    t_seconds: float
    value: float | str | None
    confidence: float
    status: SensorStatus
    valid: bool
    source_kind: SourceKind
    unit: str = ""
    duration_seconds: float = 0.0
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation name must not be empty")
        if not math.isfinite(self.t_seconds):
            raise ValueError("observation timestamp must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("observation confidence must be between 0 and 1")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("observation duration must be finite and non-negative")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("observation value must be finite or None")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "t": self.t_seconds,
            "value": self.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "valid": self.valid,
            "source_kind": self.source_kind.value,
            "unit": self.unit,
            "duration_seconds": self.duration_seconds,
            "detail": self.detail,
        }


@dataclass
class ObservationBus:
    """Short history per metric name for the Studio dashboard."""

    maxlen: int = 60
    _history: dict[str, deque[SensorObservation]] = field(default_factory=dict)
    _latest: dict[str, SensorObservation] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def publish(self, observation: SensorObservation) -> None:
        with self._lock:
            latest = self._latest.get(observation.name)
            if latest is not None and observation.t_seconds < latest.t_seconds:
                raise ValueError(f"{observation.name} observation timestamp moved backwards")
            self._latest[observation.name] = observation
            bucket = self._history.setdefault(observation.name, deque(maxlen=self.maxlen))
            bucket.append(observation)

    def latest(self, name: str) -> SensorObservation | None:
        with self._lock:
            return self._latest.get(name)

    def all_latest(self) -> dict[str, SensorObservation]:
        with self._lock:
            return dict(self._latest)

    def history(self, name: str) -> list[SensorObservation]:
        with self._lock:
            return list(self._history.get(name, ()))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "metrics": {name: obs.to_json() for name, obs in self._latest.items()},
                "history": {name: [item.to_json() for item in items] for name, items in self._history.items()},
            }

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._latest.clear()

    def expire(self, t_seconds: float, max_age_seconds: float) -> list[SensorObservation]:
        """Replace stale valid readings with unavailable states instead of showing old values as live."""
        if not math.isfinite(t_seconds) or max_age_seconds <= 0:
            raise ValueError("expiration timestamp and age must be valid")
        expired: list[SensorObservation] = []
        with self._lock:
            for name, observation in tuple(self._latest.items()):
                age_limit = max(max_age_seconds, 15.0) if name == "drowsiness_probability" else max_age_seconds
                if observation.valid and t_seconds - observation.t_seconds > age_limit:
                    current = unavailable(
                        name,
                        t_seconds,
                        detail="reading is stale; waiting for a fresh sensor update",
                        source_kind=observation.source_kind,
                    )
                    self.publish(current)
                    expired.append(current)
        return expired


def unavailable(
    name: str,
    t_seconds: float,
    *,
    detail: str,
    source_kind: SourceKind = SourceKind.HARDWARE,
) -> SensorObservation:
    return SensorObservation(
        name=name,
        t_seconds=t_seconds,
        value=None,
        confidence=0.0,
        status=SensorStatus.UNAVAILABLE,
        valid=False,
        source_kind=source_kind,
        detail=detail,
    )
